import sys
import os
import sqlite3
import subprocess
import time


def create_index(db_path, folder):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE files (
        path TEXT,
        size INTEGER,
        mtime INTEGER,
        ext TEXT
    )
    """)

    for root, dirs, files in os.walk(folder):
        for name in files:
            full_path = os.path.join(root, name)
            rel_path = os.path.relpath(full_path, folder)

            try:
                stat = os.stat(full_path)
            except:
                continue

            size = stat.st_size
            mtime = int(stat.st_mtime)
            ext = os.path.splitext(name)[1].lower()

            cur.execute(
                "INSERT INTO files VALUES (?, ?, ?, ?)",
                (rel_path, size, mtime, ext)
            )

    conn.commit()
    conn.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python pack.py <folder>")
        return

    folder = sys.argv[1]

    if not os.path.isdir(folder):
        print(f"Error: {folder} is not a folder")
        return

    name = os.path.basename(os.path.abspath(folder))

    os.makedirs("out", exist_ok=True)

    archive_path = f"out/{name}.tar.zst"
    db_path = f"out/{name}.db"

    print("[*] Creating index...")
    create_index(db_path, folder)

    print("[*] Packing archive...")
    cmd = [
        "tar",
        "--use-compress-program=zstd",
        "-cf",
        archive_path,
        folder
    ]

    subprocess.run(cmd, check=True)

    print("[+] Done")
    print(f"Archive: {archive_path}")
    print(f"Index:   {db_path}")


if __name__ == "__main__":
    main()
