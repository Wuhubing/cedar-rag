"""Behavior checks for the assignment's exact keyword rules."""

from pathlib import Path
import subprocess
import sys
import unittest

from keyword_retriever import Chunk, load_chunks, rank_chunks, select_chunks, words


Q1 = "May a trained graduate student operate the laser alone at 19:00?"
SCRIPT = Path(__file__).resolve().parents[1] / "keyword_retriever.py"


class KeywordTests(unittest.TestCase):
    def test_exact_stopwords_and_no_stemming(self):
        self.assertEqual(words("a an the is are to of in at for and or may can i it be"), set())
        self.assertEqual(words("Must WITH Student students 19:00"),
                         {"must", "with", "student", "students", "19", "00"})

    def test_repeated_words_do_not_increase_score(self):
        chunk = Chunk("A1", "access.txt", "Laser laser trained trained")
        result = rank_chunks("LASER laser trained trained trained", [chunk])[0]
        self.assertEqual(result.score, 2)
        self.assertEqual(result.matched_words, ("laser", "trained"))

    def test_real_chunks_exclude_metadata_from_search(self):
        chunks = load_chunks()
        self.assertEqual({chunk.id for chunk in chunks}, {"A1", "A2", "T1", "T2", "B1", "B2"})
        self.assertEqual(len(chunks), 6)
        ranked = rank_chunks("A1 A2 T1 T2 B1 B2 txt", chunks)
        self.assertEqual([result.score for result in ranked], [0] * 6)

    def test_q1_scores_and_tiebreaking(self):
        ranked = rank_chunks(Q1, load_chunks())
        self.assertEqual([(r.chunk.id, r.score) for r in ranked],
                         [("A1", 3), ("A2", 2), ("B1", 1), ("T1", 1), ("T2", 1), ("B2", 0)])
        self.assertEqual([r.chunk.id for r in select_chunks(ranked)], ["A1", "A2"])
        self.assertEqual(len(select_chunks(ranked, 10)), 5)

    def test_cli_no_matches(self):
        output = subprocess.check_output([sys.executable, str(SCRIPT), "quasar"], text=True)
        self.assertIn("No matching passages.", output)
        self.assertNotIn("Selected IDs:", output)

    def test_cli_k_argument(self):
        output = subprocess.check_output([sys.executable, str(SCRIPT), Q1, "--k", "1"], text=True)
        self.assertIn("Selected IDs: A1\n", output)
        self.assertNotIn("\nID: A2\n", output)


if __name__ == "__main__":
    unittest.main()
