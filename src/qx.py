import argparse
import sqlite3
import re
from typing import List, Tuple
import time

from src.extract import extract_paths
from src.ai_rank import rank_candidates, close as ai_close
from src.config import get_config, normalize_ai_mode


def _tokens(q: str) -> List[str]:
    # split on whitespace and punctuation, keep simple alnum/_ tokens
    toks = re.findall(r"[A-Za-z0-9_]+", q or "")
    return toks[:16] if toks else []


def load_candidates(db_path: str, query: str, limit: int, fallback_all: bool = False) -> List[dict]:
    """
        Return list of candidate dicts using tokenized LIKE over:
      rel_path, tar_path, snippet (if exists)
    If fallback_all is True and token search returns nothing, returns up to `limit`
    rows (best effort) so AI can rank anyway.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # detect if 'snippet' column exists
    cur.execute("PRAGMA table_info(files)")
    cols = [r[1] for r in cur.fetchall()]
    has_snippet = "snippet" in cols

    toks = _tokens(query)
    rows: List[Tuple[int, str, str, bytes, float]] = []

    if toks:
        # Build OR conditions for each token across columns
        # (tokenized search fixes "jwt secret" not matching "JWT_SECRET")
        conds = []
        params: List[str] = []
        for t in toks:
            like = f"%{t}%"
            if has_snippet:
                conds.append("(rel_path LIKE ? OR tar_path LIKE ? OR snippet LIKE ?)")
                params.extend([like, like, like])
            else:
                conds.append("(rel_path LIKE ? OR tar_path LIKE ?)")
                params.extend([like, like])

        where = " OR ".join(conds)

        if has_snippet:
            sql = f"""
            SELECT f.rowid, f.tar_path, COALESCE(f.snippet, ''), e.vec, e.norm
            FROM files f
            LEFT JOIN embeddings e ON e.file_id = f.rowid
            WHERE {where}
            LIMIT ?
            """
        else:
            sql = f"""
            SELECT f.rowid, f.tar_path, '', e.vec, e.norm
            FROM files f
            LEFT JOIN embeddings e ON e.file_id = f.rowid
            WHERE {where}
            LIMIT ?
            """

        params.append(str(limit))
        cur.execute(sql, params)
        rows = cur.fetchall()

    # AI fallback: if no token hits, give AI something to rank
    if (not rows) and fallback_all:
        if has_snippet:
            sql = """
            SELECT f.rowid, f.tar_path, COALESCE(f.snippet, ''), e.vec, e.norm
            FROM files f
            LEFT JOIN embeddings e ON e.file_id = f.rowid
            WHERE f.tar_path IS NOT NULL
            LIMIT ?
            """
            cur.execute(sql, (limit,))
        else:
            sql = """
            SELECT f.rowid, f.tar_path, '', e.vec, e.norm
            FROM files f
            LEFT JOIN embeddings e ON e.file_id = f.rowid
            WHERE f.tar_path IS NOT NULL
            LIMIT ?
            """
            cur.execute(sql, (limit,))
        rows = cur.fetchall()

    conn.close()
    return [
        {
            "file_id": row[0],
            "tar_path": row[1],
            "snippet": row[2] or "",
            "vec": row[3],
            "norm": row[4],
            "score": 0.0,
            "base_score": 0.0,
        }
        for row in rows
    ]


def main():
    parser = argparse.ArgumentParser(description="Query archive with optional AI ranking")
    parser.add_argument("db", help="Path to index .db")
    parser.add_argument("archive", help="Path to archive .tar")
    parser.add_argument("query", help="Query string")
    parser.add_argument("--ai", default=None, help="AI mode: off|emb|hybrid")
    parser.add_argument("--limit", type=int, default=50, help="Max candidates to consider")
    parser.add_argument("--topk", type=int, default=1, help="How many results to extract")
    parser.add_argument("--epsilon", type=float, default=0.02, help="Hybrid rerank trigger spread")
    parser.add_argument("--out", default="extracted", help="Output directory")
    parser.add_argument("--debug", action="store_true", help="Debug output")

    args = parser.parse_args()

    cfg = get_config()
    ai_mode = normalize_ai_mode(args.ai if args.ai is not None else cfg.ai_mode)

    candidates = load_candidates(args.db, args.query, args.limit, fallback_all=(ai_mode != "off"))

    if not candidates:
        print("(no matches)")
        return

    # If no AI, just extract topK by DB order
    if ai_mode == "off":
        paths = [item["tar_path"] for item in candidates[: args.topk]]
        for p in paths:
            print(p)
        extract_paths(args.archive, paths, out_dir=args.out)
        return

    t0 = time.perf_counter()
    ranked, perf = rank_candidates(
        query=args.query,
        candidates=candidates,
        mode=ai_mode,
        topk=args.topk,
        epsilon=args.epsilon,
        debug=args.debug,
    )
    total_s = time.perf_counter() - t0

    try:
        selected_items = ranked[: max(1, args.topk)]
        selected = [it["tar_path"] for it in selected_items]

        for it in selected_items:
            print(f"{it['tar_path']}\t{float(it.get('score', 0.0)):.4f}")

        print(
            f"[perf] embed_s={perf['embed_s']:.4f} cosine_s={perf['cosine_s']:.4f} "
            f"rerank_s={perf['rerank_s']:.4f} total_rank_s={total_s:.4f}"
        )

        extract_paths(args.archive, selected, out_dir=args.out)
    finally:
        ai_close()


if __name__ == "__main__":
    main()
