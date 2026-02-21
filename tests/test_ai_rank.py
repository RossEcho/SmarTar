import unittest
from unittest import mock

from src import ai_rank
from src.ai_worker import pack_f32


class _FakeWorker:
    def __init__(self, query_vec):
        self.query_vec = query_vec
        self.rerank_calls = 0

    def embed(self, text):
        return self.query_vec

    def rerank(self, query, snippet, fallback=0.0):
        self.rerank_calls += 1
        return min(1.0, fallback + 0.1)


class TestAiRank(unittest.TestCase):
    def test_cosine_to_score_bounds(self):
        self.assertEqual(ai_rank.cosine_to_score(-1.0), 0.0)
        self.assertEqual(ai_rank.cosine_to_score(1.0), 1.0)
        self.assertEqual(ai_rank.cosine_to_score(0.0), 0.5)

    def test_cosine_from_blob(self):
        query = [1.0, 0.0]
        blob = pack_f32([1.0, 0.0])
        cos = ai_rank._cosine_from_blob(query, 1.0, blob, 1.0)
        self.assertAlmostEqual(cos, 1.0, places=5)

    def test_rank_candidates_embeddings(self):
        fake = _FakeWorker([1.0, 0.0])
        candidates = [
            {"tar_path": "a", "snippet": "one", "vec": pack_f32([1.0, 0.0]), "norm": 1.0},
            {"tar_path": "b", "snippet": "two", "vec": pack_f32([0.0, 1.0]), "norm": 1.0},
        ]
        with mock.patch("src.ai_rank._get_worker", return_value=fake):
            ranked, perf = ai_rank.rank_candidates("hello", candidates, mode="embeddings")
        self.assertEqual(ranked[0]["tar_path"], "a")
        self.assertGreaterEqual(ranked[0]["score"], ranked[1]["score"])
        self.assertIn("embed_s", perf)
        self.assertIn("cosine_s", perf)
        self.assertIn("rerank_s", perf)

    def test_rank_candidates_hybrid_rerank(self):
        fake = _FakeWorker([1.0, 0.0])
        candidates = [
            {"tar_path": "a", "snippet": "one", "vec": pack_f32([1.0, 0.0]), "norm": 1.0},
            {"tar_path": "b", "snippet": "two", "vec": pack_f32([0.99, 0.01]), "norm": 0.9900505037},
        ]
        with mock.patch("src.ai_rank._get_worker", return_value=fake):
            ranked, perf = ai_rank.rank_candidates("jwt key", candidates, mode="hybrid", topk=2, epsilon=0.5)
        self.assertEqual(len(ranked), 2)
        self.assertGreater(fake.rerank_calls, 0)
        self.assertGreaterEqual(perf["rerank_s"], 0.0)


if __name__ == "__main__":
    unittest.main()
