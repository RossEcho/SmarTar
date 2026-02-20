import sys
import os
import sqlite3


def usage():
    print("Usage:")
    print("  python src/search.py <index.db> <query> [--paths] [--limit N]")
    print("")
    print("Query examples:")
    print("  ext=.json")
    print("  name=config")
    print("  path=logs/")
    print('  content=jwt')
    print('  content="api key"')
    print('  size>100000   (quote >/< in shell)')
    print("")
    print("Flags:")
    print("  --paths        print only tar_path (one per line)")
    print("  --limit N      limit number of results")
    print("")


def parse_args(argv):
    if len(argv) < 3:
        return None

    db_path = argv[1]

    paths_only = False
    limit = None

    tokens = argv[2:]
    query_parts = []

    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == "--paths":
            paths_only = True
            i += 1
            continue
        if t == "--limit":
            if i == len(tokens) - 1:
                raise ValueError("--limit requires a number")
            limit = int(tokens[i + 1])
            i += 2
            continue
        query_parts.append(t)
        i += 1

    query = " ".join(query_parts).strip()
    return db_path, query, paths_only, limit


def build_sql(query: str, limit: int | None):
    q = query.strip()

    base_select = "SELECT tar_path, size, mtime, ext FROM files"
    params = []

    if q.startswith("ext="):
        ext = q.split("=", 1)[1].strip().lower()
        sql = base_select + " WHERE ext = ? ORDER BY tar_path"
        params = [ext]

    elif q.startswith("name="):
        needle = q.split("=", 1)[1].strip()
        sql = base_select + " WHERE tar_path LIKE ? ORDER BY tar_path"
        params = [f"%{needle}%"]

    elif q.startswith("path="):
        needle = q.split("=", 1)[1].strip()
        sql = base_select + " WHERE tar_path LIKE ? ORDER BY tar_path"
        params = [f"%{needle}%"]

    elif q.startswith("content="):
        needle = q.split("=", 1)[1].strip().strip('"').strip("'")
        sql = base_select + " WHERE snippet IS NOT NULL AND snippet LIKE ? ORDER BY tar_path"
        params = [f"%{needle}%"]

    elif q.startswith("size>"):
        n = int(q[5:].strip())
        sql = base_select + " WHERE size > ? ORDER BY size DESC"
        params = [n]

    elif q.startswith("size<"):
        n = int(q[5:].strip())
        sql = base_select + " WHERE size < ? ORDER BY size ASC"
        params = [n]

    elif q.startswith("mtime>"):
        n = int(q[6:].strip())
        sql = base_select + " WHERE mtime > ? ORDER BY mtime DESC"
        params = [n]

    elif q.startswith("mtime<"):
        n = int(q[6:].strip())
        sql = base_select + " WHERE mtime < ? ORDER BY mtime ASC"
        params = [n]

    else:
        # fallback: search by path OR snippet (best POC UX)
        if q:
            sql = base_select + " WHERE tar_path LIKE ? OR (snippet IS NOT NULL AND snippet LIKE ?) ORDER BY tar_path"
            params = [f"%{q}%", f"%{q}%"]
        else:
            sql = base_select + " ORDER BY tar_path"

    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)

    return sql, params


def main():
    try:
        parsed = parse_args(sys.argv)
    except Exception as e:
        print(f"Error: {e}")
        usage()
        return

    if not parsed:
        usage()
        return

    db_path, query, paths_only, limit = parsed

    if not os.path.isfile(db_path):
        print(f"Error: index not found: {db_path}")
        return

    sql, params = build_sql(query, limit)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(sql, params)
    rows = cur.fetchall()
    conn.close()

    if not rows:
        print("(no matches)")
        return

    if paths_only:
        for tar_path, *_ in rows:
            print(tar_path)
        return

    for tar_path, size, mtime, ext in rows:
        print(f"{tar_path}\t{size}\t{mtime}\t{ext}")


if __name__ == "__main__":
    main()
