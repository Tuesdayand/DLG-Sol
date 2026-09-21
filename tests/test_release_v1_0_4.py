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


class TableFourAnnotationTests(unittest.TestCase):
    def setUp(self):
        self.contract = json.loads(VALIDATOR.ARTICLE_TABLE_4_CONTRACT.read_text())
        with (ROOT / "supplementary/machine_readable/external_comparator_metrics_31_rows.csv").open(newline="") as handle:
            self.rows = list(csv.DictReader(handle))

    def test_current_annotations_match_all_four_comparators(self):
        self.assertEqual(VALIDATOR.article_table_4_stars(self.contract, self.rows),
                         self.contract["significance_annotation"]["display_stars"])
        errors = []
        VALIDATOR.validate_article_table_4(errors)
        self.assertEqual(errors, [])

    def test_released_consensus_is_not_part_of_jchem_annotation(self):
        released = next(row for row in self.rows if row["model_id"] == "consensus_gnn_author")
        self.assertGreater(float(released["delta_rmse_ci99_high"]), 0)
        self.assertEqual(VALIDATOR.article_table_4_stars(self.contract, self.rows)["R05"], "***")
        self.rows.remove(released)
        self.assertEqual(VALIDATOR.article_table_4_stars(self.contract, self.rows)["R05"], "***")

    def test_released_consensus_cannot_replace_retrained_model(self):
        self.contract["models"][-1].update(source_model_id="consensus_gnn_author", source_variant="released")
        with self.assertRaises(ValueError):
            VALIDATOR.article_table_4_stars(self.contract, self.rows)

    def test_missing_or_duplicate_comparator_is_rejected(self):
        for rows in (self.rows[1:], self.rows + [self.rows[0]]):
            with self.subTest(rows=len(rows)), self.assertRaises(ValueError):
                VALIDATOR.article_table_4_stars(self.contract, rows)

    def test_zero_upper_bound_requires_lower_confidence_annotation(self):
        self.rows[0]["delta_rmse_ci99_high"] = "0"
        self.assertEqual(VALIDATOR.article_table_4_stars(self.contract, self.rows)["R01"], "**")
        self.rows[0]["delta_rmse_ci95_high"] = "0"
        self.assertEqual(VALIDATOR.article_table_4_stars(self.contract, self.rows)["R01"], "")

    def test_nonfinite_or_reversed_intervals_are_rejected(self):
        for value in ("nan", "inf", "-1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.rows[0]["delta_rmse_ci99_high"] = value
                VALIDATOR.article_table_4_stars(self.contract, self.rows)

    def test_incorrect_jchem_display_annotation_is_rejected(self):
        self.contract["significance_annotation"]["display_stars"]["R05"] = "**"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.json"
            path.write_text(json.dumps(self.contract))
            errors = []
            with patch.object(VALIDATOR, "ARTICLE_TABLE_4_CONTRACT", path):
                VALIDATOR.validate_article_table_4(errors)
            self.assertTrue(any("asterisks" in error for error in errors), errors)


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
