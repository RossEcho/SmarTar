import os
import sqlite3
import tempfile
import unittest
from unittest import mock

from src import qx
from src.pack import create_index


class TestQx(unittest.TestCase):
    def _make_db(self):
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("CREATE TABLE files(rel_path TEXT, tar_path TEXT, snippet TEXT)")
        cur.execute("CREATE TABLE embeddings(file_id INTEGER PRIMARY KEY, vec BLOB, norm REAL)")
        cur.execute(
            "INSERT INTO files(rel_path, tar_path, snippet) VALUES (?, ?, ?)",
            ("test/config.json", "data/test/config.json", "contains jwt secret"),
        )
        conn.commit()
        conn.close()
        return db_path

    def test_tokens(self):
        self.assertEqual(qx._tokens("jwt_secret token-1"), ["jwt_secret", "token", "1"])

    def test_load_candidates_from_token_search(self):
        db_path = self._make_db()
        try:
            rows = qx.load_candidates(db_path, "jwt", limit=10, fallback_all=False)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["tar_path"], "data/test/config.json")
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_load_candidates_fallback_all(self):
        db_path = self._make_db()
        try:
            rows = qx.load_candidates(db_path, "nomatch", limit=10, fallback_all=True)
            self.assertEqual(len(rows), 1)
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)


class TestPack(unittest.TestCase):
    def test_create_index_normalizes_tar_path_and_schema(self):
        with tempfile.TemporaryDirectory() as root:
            data_dir = os.path.join(root, "data", "test")
            os.makedirs(data_dir, exist_ok=True)
            fp = os.path.join(data_dir, "config.json")
            with open(fp, "w", encoding="utf-8") as f:
                f.write('{"k":"v"}')

            db_path = os.path.join(root, "idx.db")
            with mock.patch.dict(os.environ, {"ARCHIVE_AI_MODE": "off"}, clear=False):
                create_index(db_path, os.path.join(root, "data", "test"))

            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("SELECT tar_path FROM files LIMIT 1")
            row = cur.fetchone()
            self.assertIsNotNone(row)
            self.assertIn("/", row[0])
            self.assertNotIn("\\", row[0])

            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings'")
            emb_tbl = cur.fetchone()
            conn.close()
            self.assertIsNotNone(emb_tbl)


if __name__ == "__main__":
    unittest.main()
