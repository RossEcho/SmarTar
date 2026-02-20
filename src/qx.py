import argparse
import sqlite3
import re
from typing import List, Tuple

from src.extract import extract_paths
from src.ai_rank import score, close as ai_close


def _tokens(q: str) -> List[str]:
    # split on whitespace and punctuation, keep simple alnum/_ tokens
    toks = re.findall(r"[A-Za-z0-9_]+", q or "")
    return toks[:16] if toks else []


def load_candidates(db_path: str, query: str, limit: int, fallback_all: bool = False) -> List[Tuple[str, str]]:
    """
    Return list of (tar_path, snippet) candidates using tokenized LIKE over:
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
    rows: List[Tuple[str, str]] = []

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
            SELECT tar_path, COALESCE(snippet, '')
            FROM files
            WHERE {where}
            LIMIT ?
            """
        else:
            sql = f"""
            SELECT tar_path, ''
            FROM files
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
            SELECT tar_path, COALESCE(snippet, '')
            FROM files
            WHERE tar_path IS NOT NULL
            LIMIT ?
            """
            cur.execute(sql, (limit,))
        else:
            sql = """
            SELECT tar_path, ''
            FROM files
            WHERE tar_path IS NOT NULL
            LIMIT ?
            """
            cur.execute(sql, (limit,))
        rows = cur.fetchall()

    conn.close()
    return rows


def main():
    parser = argparse.ArgumentParser(description="Query archive with optional AI ranking")
    parser.add_argument("db", help="Path to index .db")
    parser.add_argument("archive", help="Path to archive .tar.zst")
    parser.add_argument("query", help="Query string")
    parser.add_argument("--ai", action="store_true", help="Use AI ranking")
    parser.add_argument("--limit", type=int, default=50, help="Max candidates to consider")
    parser.add_argument("--topk", type=int, default=3, help="How many results to extract")
    parser.add_argument("--threshold", type=float, default=0.7, help="AI score threshold")
    parser.add_argument("--out", default="extracted", help="Output directory")
    parser.add_argument("--debug", action="store_true", help="Debug output")

    args = parser.parse_args()

    candidates = load_candidates(args.db, args.query, args.limit, fallback_all=args.ai)

    if not candidates:
        print("(no matches)")
        return

    # If no AI, just extract topK by DB order
    if not args.ai:
        paths = [p for p, _ in candidates[:args.topk]]
        for p in paths:
            print(p)
        extract_paths(args.archive, paths, out_dir=args.out)
        return

    # AI ranking
    scored = []
    best_score = 0.0
    best_path = None

    try:
        for path, snip in candidates:
            s = score(args.query, snip, debug=args.debug)
            scored.append((s, path))

            if s > best_score:
                best_score = s
                best_path = path

            if args.debug:
                print(f"[AI] {s:.3f} {path}")

        scored.sort(key=lambda x: x[0], reverse=True)

        selected = [p for s, p in scored if s >= args.threshold][: args.topk]

        if selected:
            for p in selected:
                print(p)
            extract_paths(args.archive, selected, out_dir=args.out)
        else:
            print(f"(no match confident enough) best={best_score:.2f} path={best_path}")
    finally:
        ai_close()


if __name__ == "__main__":
    main()
