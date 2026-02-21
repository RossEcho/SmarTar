import sys
import os
import sqlite3
import subprocess
import math
import time

from src.ai_worker import AIWorker, pack_f32
from src.config import get_config

SNIPPET_MAX_BYTES = 2048
TEXT_MAX_FILE_BYTES = 100_000

TEXT_EXTS = {
    ".txt", ".md", ".log", ".json", ".yaml", ".yml", ".ini", ".cfg", ".conf",
    ".py", ".js", ".ts", ".html", ".css", ".sh", ".java", ".xml", ".csv", ".env"
}

def looks_like_text(raw: bytes) -> bool:
    if b"\x00" in raw:
        return False
    printable = sum(32 <= b <= 126 or b in (9, 10, 13) for b in raw)
    return printable / max(len(raw), 1) > 0.9

def read_snippet(full_path: str, ext: str, size: int) -> str | None:
    if size > TEXT_MAX_FILE_BYTES:
        return None
    try:
        with open(full_path, "rb") as f:
            raw = f.read(SNIPPET_MAX_BYTES)
    except Exception:
        return None

    if ext in TEXT_EXTS or looks_like_text(raw):
        return raw.decode("utf-8", errors="ignore")
    return None

def create_index(db_path, folder):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS files")
    cur.execute("DROP TABLE IF EXISTS embeddings")
    cur.execute("""
    CREATE TABLE files (
        id INTEGER PRIMARY KEY,
        rel_path TEXT,
        tar_path TEXT,
        size INTEGER,
        mtime INTEGER,
        ext TEXT,
        snippet TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE embeddings (
        file_id INTEGER PRIMARY KEY,
        vec BLOB,
        norm REAL,
        dim INTEGER,
        FOREIGN KEY(file_id) REFERENCES files(id)
    )
    """)

    folder = os.path.normpath(folder)
    tar_prefix = folder.lstrip("./").replace("\\", "/")

    cfg = get_config()
    worker = None
    embed_enabled = cfg.ai_mode != "off"
    embed_count = 0
    embed_total_s = 0.0
    model_load_s = 0.0
    if embed_enabled:
        try:
            worker = AIWorker(mode="embeddings")
            worker.start()
            model_load_s = worker.startup_time_s
        except Exception:
            worker = None

    for root, dirs, files in os.walk(folder):
        for name in files:
            full_path = os.path.join(root, name)
            rel_path = os.path.relpath(full_path, folder)

            try:
                stat = os.stat(full_path)
            except Exception:
                continue

            size = stat.st_size
            mtime = int(stat.st_mtime)
            ext = os.path.splitext(name)[1].lower()
            tar_path = os.path.join(tar_prefix, rel_path).replace("\\", "/")

            snippet = read_snippet(full_path, ext, size)

            cur.execute(
                "INSERT INTO files(rel_path, tar_path, size, mtime, ext, snippet) VALUES (?, ?, ?, ?, ?, ?)",
                (rel_path, tar_path, size, mtime, ext, snippet)
            )
            file_id = cur.lastrowid

            if worker is not None and snippet:
                t0 = time.perf_counter()
                vec = worker.embed(snippet)
                embed_total_s += time.perf_counter() - t0
                if vec:
                    norm = math.sqrt(sum(v * v for v in vec))
                    cur.execute(
                        "INSERT OR REPLACE INTO embeddings(file_id, vec, norm, dim) VALUES (?, ?, ?, ?)",
                        (file_id, pack_f32(vec), norm, len(vec)),
                    )
                    embed_count += 1

    conn.commit()
    conn.close()
    if worker is not None:
        worker.stop()

    print(f"[perf] model_load_s={model_load_s:.3f}")
    if embed_count > 0:
        print(f"[perf] embed_calls={embed_count} avg_embed_s={embed_total_s / embed_count:.4f}")
    else:
        print("[perf] embed_calls=0")

def main():
    if len(sys.argv) < 2:
        print("Usage: python src/pack.py <folder>")
        return

    folder = sys.argv[1]
    if not os.path.isdir(folder):
        print(f"Error: {folder} is not a folder")
        return

    name = os.path.basename(os.path.abspath(folder))
    os.makedirs("out", exist_ok=True)

    archive_path = f"out/{name}.tar"
    db_path = f"out/{name}.db"

    print("[*] Creating index (with snippets)...")
    create_index(db_path, folder)

    print("[*] Packing archive...")
    subprocess.run(["tar", "-cf", archive_path, folder], check=True)

    print("[+] Done")
    print(f"Archive: {archive_path}")
    print(f"Index:   {db_path}")

if __name__ == "__main__":
    main()
