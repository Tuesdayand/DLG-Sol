from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dlg_sol.blend import apply_fixed_blend, fit_convex_weight
from dlg_sol.evaluations import load_evaluation_registry, registry_by_id
from dlg_sol.label_firewall import (
    LabelOperation,
    PartitionRole,
    bind_labels_for_scoring,
    enforce_label_access,
    ensure_label_free,
)
from dlg_sol.metrics import paired_rmse_bootstrap, regression_metrics


class EvaluationRegistryTests(unittest.TestCase):
    def test_registry_contract(self) -> None:
        specs = load_evaluation_registry()
        self.assertEqual([spec.evaluation_id for spec in specs], ["R01", "R02", "R03", "R04", "R05", "R09"])
        self.assertEqual(sum(spec.expected_rows for spec in specs), 38228)
        self.assertEqual(registry_by_id()["R01"].coefficient_policy, "prespecified 0.5 in every outer fold")


class LabelFirewallTests(unittest.TestCase):
    def test_scored_labels_cannot_fit(self) -> None:
        with self.assertRaises(PermissionError):
            enforce_label_access(PartitionRole.SCORED, LabelOperation.COEFFICIENT_FITTING)

    def test_label_column_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ensure_label_free(["record_id", "prediction", "logS"])

    def test_scoring_binding_is_exact_and_ordered(self) -> None:
        predictions = pd.DataFrame({"record_id": ["b", "a"], "prediction": [2.0, 1.0]})
        labels = pd.DataFrame({"record_id": ["a", "b"], "logS": [1.5, 2.5]})
        bound = bind_labels_for_scoring(predictions, labels, ["prediction"])
        self.assertEqual(bound.record_id.tolist(), ["b", "a"])
        np.testing.assert_allclose(bound.logS, [2.5, 1.5])

    def test_scoring_binding_rejects_incomplete_rows(self) -> None:
        predictions = pd.DataFrame({"record_id": ["a"], "prediction": [1.0]})
        labels = pd.DataFrame({"record_id": ["a", "b"], "logS": [1.0, 2.0]})
        with self.assertRaises(ValueError):
            bind_labels_for_scoring(predictions, labels, ["prediction"])


class BlendAndMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = json.loads((ROOT / "tests" / "fixtures" / "core_regression_fixture.json").read_text())

    def test_golden_blend_and_metrics(self) -> None:
        fixture = self.fixture
        alpha = fit_convex_weight(
            fixture["observed"],
            fixture["descriptor"],
            fixture["neural"],
            partition=PartitionRole.COEFFICIENT_DEVELOPMENT,
        )
        prediction = apply_fixed_blend(fixture["descriptor"], fixture["neural"], alpha)
        metrics = regression_metrics(fixture["observed"], prediction)
        self.assertAlmostEqual(alpha, fixture["expected"]["alpha"], places=15)
        np.testing.assert_allclose(prediction, fixture["expected"]["blend"], rtol=0, atol=1e-15)
        self.assertAlmostEqual(metrics.rmse, fixture["expected"]["blend_rmse"], places=15)
        self.assertAlmostEqual(metrics.mae, fixture["expected"]["blend_mae"], places=15)
        self.assertAlmostEqual(metrics.bias_prediction_minus_observed, fixture["expected"]["blend_bias"], places=15)

    def test_weight_fit_rejects_scored_partition(self) -> None:
        fixture = self.fixture
        with self.assertRaises(PermissionError):
            fit_convex_weight(
                fixture["observed"],
                fixture["descriptor"],
                fixture["neural"],
                partition=PartitionRole.SCORED,
            )

    def test_weight_clipping_and_default(self) -> None:
        self.assertEqual(
            fit_convex_weight(
                [2.0, 2.0],
                [0.0, 0.0],
                [1.0, 1.0],
                partition=PartitionRole.COEFFICIENT_DEVELOPMENT,
            ),
            1.0,
        )
        self.assertEqual(
            fit_convex_weight(
                [1.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
                partition=PartitionRole.COEFFICIENT_DEVELOPMENT,
                default=0.25,
            ),
            0.25,
        )

    def test_bootstrap_is_deterministic_and_paired(self) -> None:
        fixture = self.fixture
        blend = apply_fixed_blend(fixture["descriptor"], fixture["neural"], fixture["expected"]["alpha"])
        first = paired_rmse_bootstrap(fixture["observed"], blend, fixture["descriptor"], replicates=1000, seed=17, batch_size=73)
        second = paired_rmse_bootstrap(fixture["observed"], blend, fixture["descriptor"], replicates=1000, seed=17, batch_size=73)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first.delta_rmse_a_minus_b, -0.32411575975518725, places=15)
        self.assertEqual(first.inference, "resolved")


if __name__ == "__main__":
    unittest.main()
