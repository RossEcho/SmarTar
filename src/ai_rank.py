import math
import time
from typing import Any, Optional

from src.ai_worker import AIWorker, unpack_f32

try:
    import numpy as np
except Exception:
    np = None

_workers: dict[str, AIWorker] = {}


def _get_worker(mode: str = "embeddings", debug: bool = False) -> AIWorker:
    key = mode
    if key not in _workers:
        worker = AIWorker(mode=mode, debug=debug)
        worker.start()
        _workers[key] = worker
    return _workers[key]


def _l2_norm(vec: list[float]) -> float:
    if np is not None:
        arr = np.asarray(vec, dtype=np.float32)
        return float(np.linalg.norm(arr))
    return math.sqrt(sum(v * v for v in vec))


def _cosine_from_blob(query_vec: list[float], query_norm: float, blob: bytes, norm: Optional[float]) -> float:
    if not blob or not query_vec:
        return -1.0
    cand_vec = unpack_f32(blob)
    if len(cand_vec) != len(query_vec):
        return -1.0

    cand_norm = float(norm or 0.0)
    if cand_norm <= 0.0:
        cand_norm = _l2_norm(cand_vec)
    if query_norm <= 0.0 or cand_norm <= 0.0:
        return -1.0

    if np is not None:
        q = np.asarray(query_vec, dtype=np.float32)
        c = np.asarray(cand_vec, dtype=np.float32)
        cos = float(np.dot(q, c) / (query_norm * cand_norm))
    else:
        dot = sum(a * b for a, b in zip(query_vec, cand_vec))
        cos = dot / (query_norm * cand_norm)

    return max(-1.0, min(1.0, cos))


def cosine_to_score(cosine: float) -> float:
    return max(0.0, min(1.0, (cosine + 1.0) / 2.0))


def score(query: str, snippet: str, debug: bool = False) -> float:
    worker = _get_worker(mode="embeddings", debug=debug)
    q = worker.embed(query or "")
    s = worker.embed(snippet or "")
    if not q or not s or len(q) != len(s):
        return 0.0
    qn = _l2_norm(q)
    sn = _l2_norm(s)
    if qn <= 0.0 or sn <= 0.0:
        return 0.0
    if np is not None:
        cos = float(np.dot(np.asarray(q, dtype=np.float32), np.asarray(s, dtype=np.float32)) / (qn * sn))
    else:
        cos = sum(a * b for a, b in zip(q, s)) / (qn * sn)
    return cosine_to_score(cos)


def score_str(query: str, snippet: str, debug: bool = False) -> str:
    return f"{score(query, snippet, debug=debug):.4f}"


def rank_candidates(
    query: str,
    candidates: list[dict[str, Any]],
    mode: str = "embeddings",
    topk: int = 10,
    epsilon: float = 0.02,
    debug: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    t0 = time.perf_counter()
    worker = _get_worker(mode="hybrid" if mode == "hybrid" else "embeddings", debug=debug)
    query_vec = worker.embed(query or "")
    t_embed = time.perf_counter() - t0
    if not query_vec:
        for item in candidates:
            item["score"] = 0.0
            item["base_score"] = 0.0
        return candidates, {"embed_s": t_embed, "cosine_s": 0.0, "rerank_s": 0.0}

    qn = _l2_norm(query_vec)

    t1 = time.perf_counter()
    for item in candidates:
        cos = _cosine_from_blob(query_vec, qn, item.get("vec") or b"", item.get("norm"))
        item["base_score"] = cosine_to_score(cos) if cos >= -1.0 else 0.0
        item["score"] = item["base_score"]
    candidates.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
    t_cos = time.perf_counter() - t1

    rerank_s = 0.0
    if mode == "hybrid" and candidates:
        query_short = len((query or "").split()) <= 3
        top_n = min(max(1, topk), 10, len(candidates))
        spread = candidates[0]["base_score"] - candidates[top_n - 1]["base_score"]
        should_rerank = query_short or (spread <= epsilon)
        if should_rerank:
            t2 = time.perf_counter()
            for i in range(top_n):
                item = candidates[i]
                rr = worker.rerank(query=query, snippet=str(item.get("snippet") or ""), fallback=float(item["base_score"]))
                item["score"] = (float(item["base_score"]) + float(rr)) / 2.0
            candidates.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
            rerank_s = time.perf_counter() - t2

    return candidates, {"embed_s": t_embed, "cosine_s": t_cos, "rerank_s": rerank_s}


def close() -> None:
    for worker in _workers.values():
        worker.stop()
    _workers.clear()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m src.ai_rank <query> <snippet>")
        raise SystemExit(1)
    try:
        print(score_str(sys.argv[1], sys.argv[2]))
    finally:
        close()
