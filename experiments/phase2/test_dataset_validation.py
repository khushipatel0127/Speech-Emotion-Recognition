"""Regression tests for the generated Phase 2 manifest and frozen split."""

from __future__ import annotations

import csv
import json
import unittest
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports" / "phase2"
MANIFEST = REPORT_DIR / "dataset_manifest.csv"
REPORT = REPORT_DIR / "dataset_validation.json"
SPLITS = REPORT_DIR / "split_assignments.json"
EMOTIONS = {"angry", "disgust", "fearful", "happy", "neutral", "sad"}
OLD_RAVDESS_HOLDOUT = {
    "RAVDESS_01",
    "RAVDESS_09",
    "RAVDESS_12",
    "RAVDESS_17",
    "RAVDESS_19",
}


class DatasetManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with MANIFEST.open("r", encoding="utf-8", newline="") as handle:
            cls.rows = list(csv.DictReader(handle))
        cls.report = json.loads(REPORT.read_text(encoding="utf-8"))
        cls.splits = json.loads(SPLITS.read_text(encoding="utf-8"))["assignments"]

    def test_expected_inventory(self) -> None:
        self.assertEqual(len(self.rows), 8_498)
        self.assertEqual(sum(row["model_eligible"] == "true" for row in self.rows), 8_491)
        self.assertEqual({row["emotion"] for row in self.rows}, EMOTIONS)
        self.assertEqual(len({row["speaker_id"] for row in self.rows}), 115)

    def test_all_files_exist_and_metadata_is_valid(self) -> None:
        for row in self.rows:
            self.assertTrue((ROOT / row["audio_path"]).is_file())
            self.assertEqual(int(row["sample_rate"]), 16_000)
            self.assertEqual(int(row["channels"]), 1)
            self.assertGreater(float(row["duration_seconds"]), 0.0)

    def test_speakers_never_cross_splits(self) -> None:
        observed: dict[str, set[str]] = defaultdict(set)
        for row in self.rows:
            observed[row["speaker_id"]].add(row["data_split"])
        self.assertTrue(all(len(splits) == 1 for splits in observed.values()))
        self.assertEqual(set(self.splits), set(observed))

    def test_old_ravdess_holdout_is_not_reused_as_final_test(self) -> None:
        test_speakers = {speaker for speaker, split in self.splits.items() if split == "test"}
        self.assertFalse(test_speakers & OLD_RAVDESS_HOLDOUT)

    def test_duplicate_content_is_not_leaked_or_used_ambiguously(self) -> None:
        hashes: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in self.rows:
            hashes[row["audio_sha256"]].append(row)
        for group in hashes.values():
            if len(group) < 2:
                continue
            self.assertEqual(len({row["data_split"] for row in group}), 1)
            if len({row["emotion"] for row in group}) > 1:
                self.assertTrue(all(row["model_eligible"] == "false" for row in group))
            else:
                self.assertEqual(sum(row["model_eligible"] == "true" for row in group), 1)

    def test_report_passed(self) -> None:
        self.assertEqual(self.report["status"], "passed")
        self.assertEqual(self.report["validation_errors"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
