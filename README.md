# SmarTar (Local AI-Powered Archive Prototype)

Packs folders into `.tar` and builds a SQLite index at pack time. Query flow is:

1. SQL shortlist from index
2. Embeddings rank (default)
3. Optional hybrid rerank on top-K
4. Extract only selected file(s)

## Required Environment Variables

These are read by `src/config.py`:

- `ARCHIVE_AI_EMBED_MODEL` (default Windows):
	`~/models/BGE/bge-small-en-v1.5-q4_k_m.gguf`
- `ARCHIVE_AI_RERANK_MODEL` (default Windows):
	`~/models/Qwen/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf`
- `ARCHIVE_LLAMA_BIN` (path to `llama-server` binary)
- `ARCHIVE_AI_MODE`: `off` | `embeddings` | `hybrid`

## Windows Example (PowerShell)

```powershell
$env:ARCHIVE_AI_EMBED_MODEL="~/models/BGE/bge-small-en-v1.5-q4_k_m.gguf"
$env:ARCHIVE_AI_RERANK_MODEL="~/models/Qwen/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
$env:ARCHIVE_LLAMA_BIN="E:\Ai\llama.cpp\build\bin\llama-server.exe"
$env:ARCHIVE_AI_MODE="embeddings"
```

Pack:

```powershell
python -m src.pack data/test
```

Query and extract best match only:

```powershell
python -m src.qx out/test.db out/test.tar config --ai emb --topk 1 --epsilon 0.02 --out extracted
```

Other AI modes:

- `--ai off` (SQL shortlist only)
- `--ai emb` (embeddings ranking)
- `--ai hybrid` (embeddings + optional top-K rerank)

## Termux Example

```bash
export ARCHIVE_AI_EMBED_MODEL="~/models/BGE/bge-small-en-v1.5-q4_k_m.gguf"
export ARCHIVE_AI_RERANK_MODEL="~/models/Qwen/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
export ARCHIVE_LLAMA_BIN="/data/data/com.termux/files/home/llama.cpp/build/bin/llama-server"
export ARCHIVE_AI_MODE="embeddings"
```

Then run:

```bash
python -m src.pack data/test
python -m src.qx out/test.db out/test.tar config --ai emb --topk 1 --epsilon 0.02 --out extracted
```

## Notes

- Ranking output is numeric and deterministic by design.
- Hybrid rerank is optional and only used on top candidates.
- Paths should be configured through env vars for portability.
