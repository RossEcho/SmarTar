import sys
import os
import subprocess


def main():
    if len(sys.argv) < 3:
        print("Usage: python src/extract.py <archive.tar.zst> <archive_path1> [archive_path2 ...] [--out OUTDIR]")
        print("Example: python src/extract.py out/test.tar.zst data/test/config.json --out extracted")
        return

    archive = sys.argv[1]
    tokens = sys.argv[2:]

    out_dir = "extracted"
    wanted = []

    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == "--out":
            if i == len(tokens) - 1:
                print("Error: --out requires a directory")
                return
            out_dir = tokens[i + 1]
            i += 2
            continue
        wanted.append(t)
        i += 1

    if not wanted:
        print("Error: no paths provided")
        return

    if not os.path.isfile(archive):
        print(f"Error: archive not found: {archive}")
        return

    os.makedirs(out_dir, exist_ok=True)

    cmd = ["tar", "--use-compress-program=zstd", "-xf", archive, "-C", out_dir, *wanted]
    subprocess.run(cmd, check=True)

    print(f"[+] Extracted to: {out_dir}")


if __name__ == "__main__":
    main()
