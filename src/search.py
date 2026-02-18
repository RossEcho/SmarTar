import sys
import os
import sqlite3


def usage():
    print("Usage:")
    print("  python src/search.py <index.db> <query>")
    print("")
    print("Query examples:")
    print("  ext=.json")
    print("  name=config")
    print("  path=logs/")
    print("  size>100000")
    print("  mtime<1700000000   (unix epoch)")
    print("  any text -> substring match on path")
    print("")


def build_sql(query: str):
    q = query.strip()

    # operators
    if q.startswith("ext="):
        ext = q.split("=", 1)[1].strip().lower()
        return ("SELECT path, size, mtime, ext FROM files WHERE ext = ? ORDER BY path", [ext])

    if q.startswith("name="):
        needle = q.split("=", 1)[1].strip()
        return ("SELECT path, size, mtime, ext FROM files WHERE path LIKE ? ORDER BY path", [f"%{needle}%"])

    if q.startswith("path="):
        needle = q.split("=", 1)[1].strip()
        return ("SELECT path, size, mtime, ext FROM files WHERE path LIKE ? ORDER BY path", [f"%{needle}%"])

    if q.startswith("size>"):
        n = int(q[5:].strip())
        return ("SELECT path, size, mtime, ext FROM files WHERE size > ? ORDER BY size DESC", [n])

    if q.startswith("size<"):
        n = int(q[5:].strip())
        return ("SELECT path, size, mtime, ext FROM files WHERE size < ? ORDER BY size ASC", [n])

    if q.startswith("mtime>"):
        n = int(q[6:].strip())
        return ("SELECT path, size, mtime, ext FROM files WHERE mtime > ? ORDER BY mtime DESC", [n])

    if q.startswith("mtime<"):
        n = int(q[6:].strip())
        return ("SELECT path, size, mtime, ext FROM files WHERE mtime < ? ORDER BY mtime ASC", [n])

    # fallback: substring match on path
    return ("SELECT path, size, mtime, ext FROM files WHERE path LIKE ? ORDER BY path", [f"%{q}%"])


def main():
    if len(sys.argv) < 3:
        usage()
        return

    db_path = sys.argv[1]
    query = " ".join(sys.argv[2:])

    if not os.path.isfile(db_path):
        print(f"Error: index not found: {db_path}")
        return

    sql, params = build_sql(query)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(sql, params)
    rows = cur.fetchall()
    conn.close()

    if not rows:
        print("(no matches)")
        return

    for path, size, mtime, ext in rows:
        print(f"{path}\t{size}\t{mtime}\t{ext}")


if __name__ == "__main__":
    main()
