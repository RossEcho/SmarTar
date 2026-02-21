import argparse
import base64
import hashlib
import json
import os
import re
import socket
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Optional

from src.config import get_config

NUM_RE = re.compile(r"^(?:0(?:\.\d+)?|1(?:\.0+)?)$")

RERANK_GRAMMAR = r'''
root ::= score
score ::= "0" | "1" | "0." frac
frac ::= digit | digit frac
digit ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
'''.strip()


def pack_f32(vec: list[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def unpack_f32(blob: bytes, dim: Optional[int] = None) -> list[float]:
    if not blob:
        return []
    if dim is None:
        dim = len(blob) // 4
    return list(struct.unpack(f"<{dim}f", blob[: dim * 4]))


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _http_json(method: str, url: str, payload: Optional[dict[str, Any]] = None, timeout: float = 30.0) -> dict[str, Any]:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url=url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else {}


class LlamaServerProcess:
    def __init__(self, bin_path: str, model_path: str, embedding: bool, timeout_s: float = 60.0) -> None:
        self.bin_path = bin_path
        self.model_path = model_path
        self.embedding = embedding
        self.timeout_s = timeout_s
        self.port = _find_free_port()
        self.proc: Optional[subprocess.Popen] = None
        self.start_time_s: float = 0.0

    def start(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            return

        cmd = [
            self.bin_path,
            "-m",
            self.model_path,
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--n-gpu-layers",
            "0",
            "--ctx-size",
            "2048",
            "--temp",
            "0",
            "--top-k",
            "1",
            "--top-p",
            "1",
            "--repeat-penalty",
            "1",
            "--mirostat",
            "0",
        ]
        if self.embedding:
            cmd.append("--embedding")

        t0 = time.perf_counter()
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
        self._wait_ready()
        self.start_time_s = time.perf_counter() - t0

    def stop(self) -> None:
        if not self.proc:
            return
        try:
            if self.proc.poll() is None:
                self.proc.terminate()
                self.proc.wait(timeout=2)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        self.proc = None

    def _wait_ready(self) -> None:
        deadline = time.time() + self.timeout_s
        last_err: Optional[Exception] = None
        while time.time() < deadline:
            if self.proc and self.proc.poll() is not None:
                raise RuntimeError("llama-server terminated during startup")
            try:
                _http_json("GET", f"http://127.0.0.1:{self.port}/health", timeout=2.0)
                return
            except Exception as exc:
                last_err = exc
                time.sleep(0.2)
        raise RuntimeError(f"llama-server did not become ready: {last_err}")

    def embedding(self, text: str) -> list[float]:
        res = _http_json(
            "POST",
            f"http://127.0.0.1:{self.port}/v1/embeddings",
            payload={"input": text},
            timeout=60.0,
        )
        data = res.get("data") or []
        if not data:
            return []
        emb = data[0].get("embedding")
        if isinstance(emb, list):
            return [float(v) for v in emb]
        return []

    def rerank_score(self, query: str, snippet: str) -> float:
        prompt = (
            "Return only a relevance number in [0..1].\\n"
            "Query: " + (query or "") + "\\n"
            "Snippet: " + (snippet or "")[:800] + "\\n"
            "Score:"
        )
        body = {
            "prompt": prompt,
            "temperature": 0,
            "top_k": 1,
            "top_p": 1,
            "repeat_penalty": 1,
            "mirostat": 0,
            "n_predict": 8,
            "grammar": RERANK_GRAMMAR,
        }
        res = _http_json(
            "POST",
            f"http://127.0.0.1:{self.port}/completion",
            payload=body,
            timeout=30.0,
        )
        txt = str(res.get("content") or "").strip()
        if NUM_RE.match(txt):
            value = float(txt)
            if 0.0 <= value <= 1.0:
                return value
        return -1.0


class AIWorker:
    """
    Persistent JSONL worker client.
    Requests: {"op":"embed","text":"..."}, {"op":"rerank","query":"...","snippet":"..."}
    """

    def __init__(self, mode: str = "embeddings", timeout_s: float = 120.0, debug: bool = False) -> None:
        self.mode = mode
        self.timeout_s = timeout_s
        self.debug = debug
        self.proc: Optional[subprocess.Popen] = None
        self.startup_time_s: float = 0.0
        self.fallback_local: bool = False

    def start(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            return
        t0 = time.perf_counter()
        cmd = [sys.executable, "-u", "-m", "src.ai_worker", "--worker-mode", self.mode]
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
            self._rpc({"op": "health"})
            self.startup_time_s = time.perf_counter() - t0
            self.fallback_local = False
        except Exception:
            self.fallback_local = True
            self.startup_time_s = time.perf_counter() - t0
            if self.proc is not None:
                try:
                    self.proc.kill()
                except Exception:
                    pass
                self.proc = None

    def stop(self) -> None:
        if not self.proc:
            return
        try:
            self._rpc({"op": "stop"})
        except Exception:
            pass
        try:
            if self.proc.poll() is None:
                self.proc.terminate()
                self.proc.wait(timeout=2)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        self.proc = None

    def __enter__(self) -> "AIWorker":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def embed(self, text: str) -> list[float]:
        if self.fallback_local:
            return _deterministic_embed(text or "")
        res = self._rpc({"op": "embed", "text": text or ""})
        vec_b64 = str(res.get("vec_b64") or "")
        if not vec_b64:
            return []
        dim = int(res.get("dim") or 0)
        blob = base64.b64decode(vec_b64.encode("ascii"))
        return unpack_f32(blob, dim=dim if dim > 0 else None)

    def rerank(self, query: str, snippet: str, fallback: float = 0.0) -> float:
        if self.fallback_local:
            return fallback
        res = self._rpc({"op": "rerank", "query": query or "", "snippet": snippet or "", "fallback": float(fallback)})
        value = float(res.get("score", fallback))
        if 0.0 <= value <= 1.0:
            return value
        return fallback

    def _rpc(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.proc is None or self.proc.poll() is not None:
            self.start()
        if not self.proc or not self.proc.stdin or not self.proc.stdout:
            raise RuntimeError("AIWorker process not ready")

        self.proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()

        t0 = time.time()
        while True:
            if time.time() - t0 > self.timeout_s:
                raise TimeoutError("AI worker timeout")
            line = self.proc.stdout.readline()
            if line == "":
                raise RuntimeError("AI worker exited")
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("ok") is False:
                raise RuntimeError(str(obj.get("error") or "worker error"))
            return obj


def _deterministic_embed(text: str, dim: int = 384) -> list[float]:
    values: list[float] = []
    seed = text.encode("utf-8", errors="ignore")
    counter = 0
    while len(values) < dim:
        h = hashlib.blake2b(seed + b"|" + str(counter).encode("ascii"), digest_size=32).digest()
        for i in range(0, len(h), 4):
            u = int.from_bytes(h[i : i + 4], "little")
            values.append((u / 4294967295.0) * 2.0 - 1.0)
            if len(values) >= dim:
                break
        counter += 1
    return values


def _worker_main(worker_mode: str) -> int:
    cfg = get_config()
    embed_server: Optional[LlamaServerProcess] = None
    rerank_server: Optional[LlamaServerProcess] = None

    try:
        if worker_mode in ("embeddings", "hybrid"):
            embed_server = LlamaServerProcess(
                bin_path=cfg.llama_bin,
                model_path=cfg.embed_model,
                embedding=True,
            )
            embed_server.start()

        if worker_mode == "hybrid":
            rerank_server = LlamaServerProcess(
                bin_path=cfg.llama_bin,
                model_path=cfg.rerank_model,
                embedding=False,
            )
            rerank_server.start()

        startup = {
            "op": "startup",
            "ok": True,
            "embed_load_s": embed_server.start_time_s if embed_server else 0.0,
            "rerank_load_s": rerank_server.start_time_s if rerank_server else 0.0,
        }
        print(json.dumps(startup), flush=True)

        for raw in sys.stdin:
            raw = raw.strip()
            if not raw:
                continue
            try:
                req = json.loads(raw)
                op = req.get("op")
                if op == "stop":
                    print(json.dumps({"op": "stop", "ok": True}), flush=True)
                    break
                if op == "health":
                    print(json.dumps({"op": "health", "ok": True}), flush=True)
                    continue
                if op == "embed":
                    if embed_server is None:
                        raise RuntimeError("embedding server not enabled")
                    vec = embed_server.embedding(str(req.get("text") or ""))
                    blob = pack_f32(vec)
                    print(
                        json.dumps(
                            {
                                "op": "embed",
                                "ok": True,
                                "vec_b64": base64.b64encode(blob).decode("ascii"),
                                "dim": len(vec),
                            }
                        ),
                        flush=True,
                    )
                    continue
                if op == "rerank":
                    fallback = float(req.get("fallback") or 0.0)
                    if rerank_server is None:
                        score = fallback
                    else:
                        score = rerank_server.rerank_score(
                            str(req.get("query") or ""),
                            str(req.get("snippet") or ""),
                        )
                        if not (0.0 <= score <= 1.0):
                            score = fallback
                    print(json.dumps({"op": "rerank", "ok": True, "score": score}), flush=True)
                    continue

                print(json.dumps({"ok": False, "error": f"unsupported op: {op}"}), flush=True)
            except Exception as exc:
                print(json.dumps({"ok": False, "error": str(exc)}), flush=True)
    finally:
        if rerank_server:
            rerank_server.stop()
        if embed_server:
            embed_server.stop()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="AI worker (client + JSONL server mode)")
    parser.add_argument("--worker-mode", choices=["embeddings", "hybrid"], help="Run as JSONL worker")
    args = parser.parse_args()

    if args.worker_mode:
        return _worker_main(args.worker_mode)

    worker = AIWorker(mode="embeddings")
    worker.start()
    vec = worker.embed("quick check")
    print(f"dim={len(vec)}")
    worker.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
