"""Regression checks for the manuscript-alignment patch; no molecular data needed."""

import csv
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("release_validator", ROOT / "scripts/validate_release.py")
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)
WEIGHTING = "molecule_specific_weighting_14_rows.csv"


class WeightingSummaryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.tables = Path(self.directory.name)
        for name in (WEIGHTING, "main_evaluation_effects_12_rows.csv"):
            shutil.copyfile(ROOT / "supplementary/machine_readable" / name, self.tables / name)

    def check_rows(self, mutation=None):
        path = self.tables / WEIGHTING
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            rows = list(reader)
        if mutation is not None:
            mutation(rows)
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        errors = []
        with patch.object(VALIDATOR, "TABLE_DIR", self.tables):
            VALIDATOR.validate_weighting_summary(errors)
        return errors

    def test_current_fourteen_rows_pass(self):
        self.assertEqual(self.check_rows(), [])

    def test_opposite_delta_sign_is_rejected(self):
        self.assertTrue(self.check_rows(lambda rows: rows[0].update(delta_rmse=-float(rows[0]["delta_rmse"]))))

    def test_missing_biogen_row_is_rejected(self):
        self.assertTrue(self.check_rows(lambda rows: rows.pop()))

    def test_duplicate_panel_policy_is_rejected(self):
        self.assertTrue(self.check_rows(lambda rows: rows[0].update(rows[1])))

    def test_wrong_sample_count_is_rejected(self):
        self.assertTrue(self.check_rows(lambda rows: rows[0].update(n="8048")))

    def test_changed_bootstrap_count_is_rejected(self):
        self.assertTrue(self.check_rows(lambda rows: rows[0].update(n_boot="2000")))

    def test_nonfinite_value_is_rejected(self):
        self.assertTrue(self.check_rows(lambda rows: rows[0].update(gate_rmse="nan")))

    def test_reversed_interval_is_rejected(self):
        self.assertTrue(self.check_rows(lambda rows: rows[0].update(ci95_low="1", ci95_high="-1")))

    def test_fixed_baseline_drift_is_rejected_even_with_consistent_delta(self):
        def change(rows):
            rows[0]["fixed_rmse"] = str(float(rows[0]["fixed_rmse"]) + 0.01)
            rows[0]["delta_rmse"] = str(float(rows[0]["gate_rmse"]) - float(rows[0]["fixed_rmse"]))
        self.assertTrue(self.check_rows(change))


class ReleaseMetadataTests(unittest.TestCase):
    def test_title_and_version_match(self):
        errors = []
        VALIDATOR.validate_release_policy(errors)
        self.assertEqual(errors, [])
        citation = (ROOT / "CITATION.cff").read_text()
        title = next(line for line in citation.splitlines() if line.startswith("title: ")).split('"')[1]
        self.assertIn(title, (ROOT / "README.md").read_text())

    def test_fourteen_csv_files_are_registered(self):
        errors = []
        VALIDATOR.validate_tables(errors)
        self.assertEqual(errors, [])
        spec = json.loads((ROOT / "configs/machine_readable_tables.json").read_text())
        self.assertEqual(len(spec["tables"]), 14)

    def test_released_consensus_display_is_not_retrained_value(self):
        contract = json.loads((ROOT / "configs/article_table_3_contract.json").read_text())
        row = next(row for row in contract["rows"] if row["context_id"] == "C14")
        self.assertEqual(row["main_rmse"], "0.6571")
        self.assertEqual(row["n"], 980)

    def test_current_si_cross_references(self):
        self.assertIn("Supplementary Table S17", (ROOT / "figures/README.md").read_text())
        text = (ROOT / "docs/TRAINING_EXECUTION_WORKFLOW.md").read_text()
        self.assertIn("Supplementary Table S6", text)
        self.assertNotIn("Supplementary Note S7", text)


if __name__ == "__main__":
    unittest.main()
