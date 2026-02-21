import os
from dataclasses import dataclass

WIN_DEFAULT_EMBED = "~/models/BGE/bge-small-en-v1.5-q4_k_m.gguf"
WIN_DEFAULT_RERANK = "~/models/Qwen/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
WIN_DEFAULT_LLAMA_BIN = r"E:\Ai\llama.cpp\build\bin\llama-server.exe"
WIN_DEFAULT_ZSTD_BIN = r"C:\Users\User\Desktop\zstd-v1.5.7-win64\zstd.exe"


def normalize_ai_mode(value: str | None) -> str:
    v = (value or "embeddings").strip().lower()
    if v in ("off", "none"):
        return "off"
    if v in ("emb", "embedding", "embeddings"):
        return "embeddings"
    if v == "hybrid":
        return "hybrid"
    return "embeddings"


@dataclass(frozen=True)
class AppConfig:
    embed_model: str
    rerank_model: str
    llama_bin: str
    zstd_bin: str
    ai_mode: str


def get_config() -> AppConfig:
    embed_model = os.path.expanduser(os.getenv("ARCHIVE_AI_EMBED_MODEL", WIN_DEFAULT_EMBED))
    rerank_model = os.path.expanduser(os.getenv("ARCHIVE_AI_RERANK_MODEL", WIN_DEFAULT_RERANK))
    llama_bin = os.path.expanduser(os.getenv("ARCHIVE_LLAMA_BIN", WIN_DEFAULT_LLAMA_BIN))
    zstd_bin = os.path.expanduser(os.getenv("ARCHIVE_ZSTD_BIN", WIN_DEFAULT_ZSTD_BIN))

    return AppConfig(
        embed_model=embed_model,
        rerank_model=rerank_model,
        llama_bin=llama_bin,
        zstd_bin=zstd_bin,
        ai_mode=normalize_ai_mode(os.getenv("ARCHIVE_AI_MODE", "embeddings")),
    )
