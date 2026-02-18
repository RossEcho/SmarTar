import sys
import os
import subprocess


def main():
    if len(sys.argv) < 2:
        print("Usage: python pack.py <folder>")
        return

    folder = sys.argv[1]

    if not os.path.isdir(folder):
        print(f"Error: {folder} is not a folder")
        return

    name = os.path.basename(os.path.abspath(folder))
    output = f"out/{name}.tar.zst"

    os.makedirs("out", exist_ok=True)

    print(f"[*] Packing {folder} -> {output}")

    cmd = [
        "tar",
        "--use-compress-program=zstd",
        "-cf",
        output,
        folder
    ]

    subprocess.run(cmd, check=True)

    print("[+] Done")


if __name__ == "__main__":
    main()
