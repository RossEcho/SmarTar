from typing import Optional
from src.ai_worker import AIWorker

# global singleton worker (lazy init)
_worker: Optional[AIWorker] = None


def _get_worker(debug: bool = False) -> AIWorker:
    global _worker
    if _worker is None:
        _worker = AIWorker(debug=debug)
        _worker.start()
    return _worker


def score(query: str, snippet: str, debug: bool = False) -> float:
    """
    Thin client wrapper.
    Delegates scoring to persistent AIWorker.
    """
    worker = _get_worker(debug=debug)
    return worker.score(query, snippet)


def close():
    """
    Clean shutdown (optional)
    """
    global _worker
    if _worker is not None:
        _worker.stop()
        _worker = None


# CLI for quick testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python src/ai_rank.py <query> <snippet> [--debug]")
        sys.exit(1)

    q = sys.argv[1]
    sn = sys.argv[2]
    dbg = "--debug" in sys.argv[3:]

    try:
        print(score(q, sn, debug=dbg))
    finally:
        close()
