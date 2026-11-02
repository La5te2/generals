"""Exercise archived evidence and reject damaged pairings/counts before reporting."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from verify import ARCHIVE, check_result, paired_outcomes, verify


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = json.loads((ARCHIVE / "data/baseline/initial/P2-33-vs-ppo20.json").read_text())

    def damaged(self):
        return copy.deepcopy(self.result)

    def test_archive_reproduces_reported_numbers(self):
        report = verify()
        self.assertEqual(report["pairs"]["P2_combined"]["wins"], 349)
        self.assertEqual(report["pairs"]["P2_combined"]["map_pairs"], 300)
        self.assertEqual(report["shared_map_33_vs_39"], {
            "both_win": 78, "only_33_win": 44, "only_39_win": 44,
            "both_nonwin": 34, "mcnemar_exact_two_sided_p": 1.0})

    def test_missing_game_is_rejected(self):
        r = self.damaged()
        r["records"].pop()
        with self.assertRaisesRegex(ValueError, "game IDs"):
            check_result(r, 200, 930001)

    def test_duplicate_game_is_rejected(self):
        r = self.damaged()
        r["records"][1] = copy.deepcopy(r["records"][0])
        with self.assertRaisesRegex(ValueError, "game IDs"):
            check_result(r, 200, 930001)

    def test_swapped_map_identity_is_rejected(self):
        r = self.damaged()
        r["records"][1]["map_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "pairing"):
            check_result(r, 200, 930001)

    def test_forged_totals_and_intervals_are_rejected(self):
        for field, value in [("wins", 123), ("win_rate_ci95", [0.6, 0.7])]:
            with self.subTest(field=field):
                r = self.damaged()
                r[field] = value
                with self.assertRaises(ValueError):
                    check_result(r, 200, 930001)

    def test_unknown_outcome_is_rejected(self):
        r = self.damaged()
        r["records"][0]["outcome"] = "unresolved"
        with self.assertRaisesRegex(ValueError, "unknown outcome"):
            check_result(r, 200, 930001)

    def test_paired_comparison_uses_map_hash_and_seat(self):
        r = self.damaged()
        r["records"][0]["map_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "unmatched"):
            paired_outcomes(self.result["records"], r["records"])

    def test_tampered_evidence_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "changed.json").write_text("{}")
            (root / "manifest.json").write_text(json.dumps({"files": [
                {"path": "changed.json", "sha256": "0" * 64}]}))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify(root)


if __name__ == "__main__":
    unittest.main()
