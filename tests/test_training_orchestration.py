from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dlg_sol.descriptors import DescriptorFilterConfig
from dlg_sol.training import (
    aggregate_member_predictions,
    evaluate_validation_candidates,
    prepare_descriptor_unit,
    select_nested_trial,
    select_validation_trial,
    validate_oof_predictions,
    validate_training_partitions,
)


class PartitionContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fit = pd.DataFrame({"record_id": ["f1", "f2", "f3"], "logS": [0.1, 0.2, 0.3]})
        self.selection = pd.DataFrame({"record_id": ["v1", "v2"], "logS": [0.4, 0.5]})
        self.scored = pd.DataFrame({"record_id": ["s1", "s2"]})

    def test_disjoint_partition_contract(self) -> None:
        result = validate_training_partitions(self.fit, self.selection, self.scored)
        self.assertEqual(result.fit, ("f1", "f2", "f3"))
        self.assertEqual(result.selection, ("v1", "v2"))
        self.assertEqual(result.scored, ("s1", "s2"))

    def test_scored_labels_are_rejected(self) -> None:
        scored = self.scored.assign(logS=[1.0, 2.0])
        with self.assertRaises(ValueError):
            validate_training_partitions(self.fit, self.selection, scored)

    def test_partition_overlap_is_rejected(self) -> None:
        scored = pd.DataFrame({"record_id": ["s1", "f2"]})
        with self.assertRaises(ValueError):
            validate_training_partitions(self.fit, self.selection, scored)

    def test_record_identifiers_must_remain_unique_after_normalization(self) -> None:
        fit = pd.DataFrame({"record_id": [1, "1"], "logS": [0.1, 0.2]})
        with self.assertRaises(ValueError):
            validate_training_partitions(fit, self.selection, self.scored)

    def test_schema_is_fitted_on_fit_rows_only(self) -> None:
        descriptors = pd.DataFrame(
            {
                "record_id": ["f1", "f2", "f3", "v1", "v2", "s1", "s2"],
                "fit_variable": [0.0, 1.0, 2.0, 8.0, 9.0, 10.0, 11.0],
                "fit_constant": [3.0, 3.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            }
        )
        config = DescriptorFilterConfig(variance_ddof=0, correlation_threshold=None, correlation_comparison=None)
        unit = prepare_descriptor_unit(descriptors, self.fit, self.selection, self.scored, config)
        self.assertEqual(unit.schema.retained_columns, ("fit_variable",))
        self.assertEqual(unit.schema.dropped_constant, ("fit_constant",))
        np.testing.assert_array_equal(unit.x_scored[:, 0], [10.0, 11.0])


class CandidateSelectionTests(unittest.TestCase):
    def test_validation_selection_uses_rmse_then_trial(self) -> None:
        records = pd.DataFrame(
            {
                "trial": [4, 2, 7],
                "val_rmse": [0.8, 0.8, 0.9],
                "best_iteration": [10, 20, 30],
            }
        )
        selected = select_validation_trial(records)
        self.assertEqual(selected.trial, 2)
        self.assertEqual(selected.best_iteration, 20)

    def test_nested_selection_matches_pooled_contract(self) -> None:
        records = pd.DataFrame(
            [
                {"trial": 0, "inner_validation_fold": 1, "n": 1, "sse": 0.01, "rmse": 0.1, "mae": 0.1, "best_iteration": 9},
                {"trial": 0, "inner_validation_fold": 2, "n": 9, "sse": 8.91, "rmse": np.sqrt(0.99), "mae": 0.8, "best_iteration": 19},
                {"trial": 1, "inner_validation_fold": 1, "n": 1, "sse": 0.81, "rmse": 0.9, "mae": 0.9, "best_iteration": 29},
                {"trial": 1, "inner_validation_fold": 2, "n": 9, "sse": 0.81, "rmse": 0.3, "mae": 0.3, "best_iteration": 39},
            ]
        )
        selected = select_nested_trial(records)
        self.assertEqual(selected.trial, 1)
        self.assertAlmostEqual(selected.pooled_inner_rmse, np.sqrt(0.162))
        self.assertEqual(selected.final_tree_count, 35)
        self.assertEqual(selected.inner_best_iterations, (29, 39))

    def test_fractional_trial_identifier_is_rejected(self) -> None:
        records = pd.DataFrame({"trial": [0.5], "val_rmse": [0.8], "best_iteration": [10]})
        with self.assertRaises(ValueError):
            select_validation_trial(records)


class PredictionAggregationTests(unittest.TestCase):
    def test_oof_requires_one_prediction_per_row(self) -> None:
        frame = pd.DataFrame({"record_id": ["b", "a"], "prediction": [2.0, 1.0]})
        result = validate_oof_predictions(frame, expected_rows=2)
        self.assertEqual(result.record_id.tolist(), ["b", "a"])

    def test_member_predictions_are_averaged_in_first_seen_order(self) -> None:
        frame = pd.DataFrame(
            {
                "record_id": ["b", "a", "b", "a"],
                "member": [0, 0, 1, 1],
                "prediction": [2.0, 1.0, 4.0, 5.0],
            }
        )
        result = aggregate_member_predictions(frame, expected_members=2, expected_rows=2)
        self.assertEqual(result.record_id.tolist(), ["b", "a"])
        np.testing.assert_allclose(result.prediction, [3.0, 3.0])

    def test_member_set_mismatch_is_rejected(self) -> None:
        frame = pd.DataFrame(
            {
                "record_id": ["a", "a", "b", "b"],
                "member": [0, 1, 0, 2],
                "prediction": [1.0, 2.0, 3.0, 4.0],
            }
        )
        with self.assertRaises(ValueError):
            aggregate_member_predictions(frame, expected_members=2, expected_rows=2)


@unittest.skipUnless(importlib.util.find_spec("xgboost"), "XGBoost is not installed")
class CandidateRunnerTests(unittest.TestCase):
    def test_candidate_search_never_requires_scored_labels(self) -> None:
        rng = np.random.default_rng(27)
        identifiers = [f"r{i}" for i in range(36)]
        x = rng.normal(size=(36, 3))
        descriptors = pd.DataFrame(x, columns=["a", "b", "c"])
        descriptors.insert(0, "record_id", identifiers)
        y = x[:, 0] - 0.5 * x[:, 1]
        fit = pd.DataFrame({"record_id": identifiers[:24], "logS": y[:24]})
        selection = pd.DataFrame({"record_id": identifiers[24:30], "logS": y[24:30]})
        scored = pd.DataFrame({"record_id": identifiers[30:]})
        config = DescriptorFilterConfig(variance_ddof=0, correlation_threshold=None, correlation_comparison=None)
        unit = prepare_descriptor_unit(descriptors, fit, selection, scored, config)
        candidates = [
            {"trial": 0, "n_estimators": 4, "learning_rate": 0.05, "max_depth": 2},
            {"trial": 1, "n_estimators": 30, "learning_rate": 0.1, "max_depth": 2},
        ]
        result = evaluate_validation_candidates(unit, candidates, lambda trial: 100 + trial, n_jobs=1)
        self.assertIn(result.selected_trial, {0, 1})
        self.assertEqual(result.scored_predictions.shape, (6,))
        self.assertEqual(len(result.trial_records), 2)


if __name__ == "__main__":
    unittest.main()
