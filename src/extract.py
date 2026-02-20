import argparse
import os
import subprocess
from typing import List


def extract_paths(archive_path: str, paths: List[str], out_dir: str = "extracted") -> None:
    """
    Extract specific file paths from a .tar.zst archive into out_dir.

    archive_path: path to out/*.tar.zst
    paths: list of tar paths as stored in the archive (e.g. "data/test/config.json")
    out_dir: extraction destination
    """
    if not os.path.isfile(archive_path):
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    os.makedirs(out_dir, exist_ok=True)

    if not paths:
        print("(no paths to extract)")
        return

    cmd = [
        "tar",
        "--use-compress-program=zstd",
        "-xf",
        archive_path,
        "-C",
        out_dir,
        *paths,
    ]

    subprocess.run(cmd, check=True)
    print(f"[+] Extracted to: {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract specific files from a .tar.zst archive")
    ap.add_argument("archive", help="Path to archive .tar.zst")
    ap.add_argument("paths", nargs="+", help="One or more tar paths to extract")
    ap.add_argument("--out", default="extracted", help="Output directory")
    args = ap.parse_args()

    extract_paths(args.archive, args.paths, out_dir=args.out)


if __name__ == "__main__":
    main()
