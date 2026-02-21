import os
import shutil

from src.config import get_config


def resolve_zstd_program() -> str:
	cfg = get_config()
	candidate = (cfg.zstd_bin or "zstd").strip()

	if os.path.isabs(candidate) and os.path.isfile(candidate):
		return candidate

	found = shutil.which(candidate)
	if found:
		return found

	if os.name == "nt":
		win_fallback = r"C:\Users\User\Desktop\zstd-v1.5.7-win64\zstd.exe"
		if os.path.isfile(win_fallback):
			return win_fallback

	raise FileNotFoundError(
		"zstd executable not found. Set ARCHIVE_ZSTD_BIN to your zstd binary path "
		"or ensure zstd is available on PATH."
	)

