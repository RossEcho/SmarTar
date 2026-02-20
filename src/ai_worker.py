import os
import re
import subprocess
import time
from typing import Optional

# Paths (hardcoded for your Termux setup)
MODEL = "/data/data/com.termux/files/home/.cache/llama.cpp/lmstudio-community_gemma-3n-E2B-it-text-GGUF_gemma-3n-E2B-it-Q4_K_M.gguf"
BIN   = "/data/data/com.termux/files/home/llama.cpp/build/bin/llama-cli"

END_TOKEN = "<END>"

# accept decimals; allow "1.0" too
NUM_RE = re.compile(r"(0\.\d+|1\.0+)")


class AIWorker:
    """
    Persistent llama-cli worker.
    - Starts llama-cli once
    - For each request: clears context, sends prompt, reads until END_TOKEN
    - Returns float score in [0,1]
    """

    def __init__(
        self,
        model_path: str = MODEL,
        bin_path: str = BIN,
        timeout_s: float = 30.0,
        debug: bool = False,
    ) -> None:
        self.model_path = model_path
        self.bin_path = bin_path
        self.timeout_s = timeout_s
        self.debug = debug

        self.proc: Optional[subprocess.Popen] = None

    def start(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            return

        # llama-cli interactive, simple-io reduces decorations, log-disable reduces noise
        # temp/top-k/top-p to keep deterministic formatting
        cmd = [
            self.bin_path,
            "-m", self.model_path,
            "--simple-io",
            "--log-disable",
            "--temp", "0",
            "--top-k", "1",
            "--top-p", "0.1",
        ]

        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

        # optional: try to drain any banner lines quickly (non-fatal if not)
        self._drain_startup(seconds=2.0)

    def stop(self) -> None:
        if not self.proc:
            return
        try:
            if self.proc.poll() is None:
                self.proc.kill()
        except Exception:
            pass
        self.proc = None

    def __enter__(self) -> "AIWorker":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def score(self, query: str, snippet: str) -> float:
        """
        Score relevance between query and snippet.
        Returns float in [0,1].
        """
        self._ensure_running()

        snippet = (snippet or "")[:600]

        prompt = f"""Score how relevant the SNIPPET is to the QUERY.

Rules:
- exact match → 1.0
- strong match → 0.7-1.0
- partial match → 0.3-0.7
- unrelated → 0.0-0.3

Return ONLY a decimal number between 0 and 1, then on a new line write {END_TOKEN}.
No explanation.

QUERY: {query}
SNIPPET:
{snippet}

Score:
"""

        # IMPORTANT: clear context so prior requests don't contaminate
        self._send("/clear\n")
        # Send prompt
        self._send(prompt.strip() + "\n")

        text = self._read_until_end_token()
        if self.debug:
            print("\n=== WORKER RAW ===")
            print(text)
            print("=== /WORKER RAW ===")

        # Only parse before END token, in case model repeats after it
        clean = text.split(END_TOKEN, 1)[0]
        matches = NUM_RE.findall(clean)
        if matches:
            try:
                v = float(matches[-1])
                if 0.0 <= v <= 1.0:
                    return v
            except Exception:
                return 0.0
        return 0.0

    # ---------------- internals ----------------

    def _ensure_running(self) -> None:
        if not self.proc or self.proc.poll() is not None:
            self.stop()
            self.start()
        if not self.proc or not self.proc.stdin or not self.proc.stdout:
            raise RuntimeError("AIWorker failed to start llama-cli")

    def _send(self, s: str) -> None:
        assert self.proc and self.proc.stdin
        self.proc.stdin.write(s)
        self.proc.stdin.flush()

    def _read_until_end_token(self) -> str:
        assert self.proc and self.proc.stdout
        start = time.time()
        out_lines = []

        while True:
            if time.time() - start > self.timeout_s:
                # Timeout: return what we got (caller will parse or treat as 0)
                return "".join(out_lines)

            line = self.proc.stdout.readline()
            if line == "":
                # process ended
                return "".join(out_lines)

            out_lines.append(line)

            # Stop condition
            if END_TOKEN in line:
                return "".join(out_lines)

    def _drain_startup(self, seconds: float = 2.0) -> None:
        """
        llama-cli may print a banner; we don't want it to pollute first response.
        We drain for a short time, best effort.
        """
        if not self.proc or not self.proc.stdout:
            return

        start = time.time()
        while time.time() - start < seconds:
            # Non-blocking-ish: readline can block; so we only do this if data is ready
            # But Python stdio doesn't expose readiness easily without select on fd;
            # so we keep it minimal: short timeout window, and break on empty.
            try:
                self.proc.stdout.flush()
            except Exception:
                pass
            # If there's nothing, break quickly by peeking using poll + sleep
            if self.proc.poll() is not None:
                break
            # Read a line if it comes quickly; otherwise stop draining
            self.proc.stdout.readline()
            # tiny sleep to avoid tight loop
            time.sleep(0.05)
