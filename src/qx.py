import sys
import subprocess


def main():
    if len(sys.argv) < 4:
        print("Usage: python src/qx.py <index.db> <archive.tar.zst> <query> [--out OUTDIR] [--limit N]")
        print('Example: python src/qx.py out/test.db out/test.tar.zst "config" --out extracted --limit 1')
        return

    db_path = sys.argv[1]
    archive_path = sys.argv[2]
    args = sys.argv[3:]

    out_dir = "extracted"
    limit = "1"

    # parse flags at end; everything else becomes the query
    query_parts = []
    i = 0
    while i < len(args):
        if args[i] == "--out":
            if i == len(args) - 1:
                print("Error: --out requires a directory")
                return
            out_dir = args[i + 1]
            i += 2
            continue
        if args[i] == "--limit":
            if i == len(args) - 1:
                print("Error: --limit requires a number")
                return
            limit = args[i + 1]
            i += 2
            continue
        query_parts.append(args[i])
        i += 1

    query = " ".join(query_parts).strip()
    if not query:
        print("Error: empty query")
        return

    # 1) search paths (no archive touch)
    res = subprocess.run(
        [sys.executable, "src/search.py", db_path, query, "--paths", "--limit", limit],
        capture_output=True,
        text=True
    )

    if res.returncode != 0:
        print(res.stderr.strip() or "search failed")
        return

    paths = [line.strip() for line in res.stdout.splitlines() if line.strip()]
    if not paths or (len(paths) == 1 and paths[0] == "(no matches)"):
        print("(no matches)")
        return

    # 2) extract those paths
    cmd = [sys.executable, "src/extract.py", archive_path, *paths, "--out", out_dir]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
