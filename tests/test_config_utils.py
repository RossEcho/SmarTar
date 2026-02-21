import os
import tempfile
import unittest
from unittest import mock

from src import config
from src import utils


class TestConfig(unittest.TestCase):
    def test_normalize_ai_mode_variants(self):
        self.assertEqual(config.normalize_ai_mode("off"), "off")
        self.assertEqual(config.normalize_ai_mode("NONE"), "off")
        self.assertEqual(config.normalize_ai_mode("emb"), "embeddings")
        self.assertEqual(config.normalize_ai_mode("EmbeddingS"), "embeddings")
        self.assertEqual(config.normalize_ai_mode("hybrid"), "hybrid")
        self.assertEqual(config.normalize_ai_mode("unknown"), "embeddings")

    def test_get_config_reads_env(self):
        with mock.patch.dict(
            os.environ,
            {
                "ARCHIVE_AI_EMBED_MODEL": "embed.gguf",
                "ARCHIVE_AI_RERANK_MODEL": "rerank.gguf",
                "ARCHIVE_LLAMA_BIN": "llama-server",
                "ARCHIVE_ZSTD_BIN": "zstd",
                "ARCHIVE_AI_MODE": "hybrid",
            },
            clear=False,
        ):
            cfg = config.get_config()
        self.assertEqual(cfg.embed_model, "embed.gguf")
        self.assertEqual(cfg.rerank_model, "rerank.gguf")
        self.assertEqual(cfg.llama_bin, "llama-server")
        self.assertEqual(cfg.zstd_bin, "zstd")
        self.assertEqual(cfg.ai_mode, "hybrid")


class TestUtils(unittest.TestCase):
    def test_resolve_zstd_absolute_path(self):
        with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as tmp:
            path = tmp.name
        try:
            with mock.patch.dict(os.environ, {"ARCHIVE_ZSTD_BIN": path}, clear=False):
                resolved = utils.resolve_zstd_program()
            self.assertEqual(resolved, path)
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_resolve_zstd_which(self):
        with mock.patch.dict(os.environ, {"ARCHIVE_ZSTD_BIN": "zstd"}, clear=False):
            with mock.patch("src.utils.shutil.which", return_value="/usr/bin/zstd"):
                resolved = utils.resolve_zstd_program()
        self.assertEqual(resolved, "/usr/bin/zstd")

    def test_resolve_zstd_not_found_raises(self):
        with mock.patch.dict(os.environ, {"ARCHIVE_ZSTD_BIN": "missing-zstd"}, clear=False):
            with mock.patch("src.utils.shutil.which", return_value=None):
                with mock.patch("src.utils.os.path.isfile", return_value=False):
                    with self.assertRaises(FileNotFoundError):
                        utils.resolve_zstd_program()


if __name__ == "__main__":
    unittest.main()
