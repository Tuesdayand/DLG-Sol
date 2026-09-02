from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dlg_sol.descriptors import (
    DescriptorFilterConfig,
    build_xgb_regressor,
    calculate_mordred_2d,
    filter_config_for,
    fit_descriptor_schema,
    generate_candidates,
    load_descriptor_protocols,
    transform_descriptor_frame,
)


def candidate_hash(candidates: list[dict[str, int | float]]) -> str:
    values = [{key: value for key, value in candidate.items() if key != "trial"} for candidate in candidates]
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


class DescriptorFilteringTests(unittest.TestCase):
    def test_pair_rule_retains_earlier_column(self) -> None:
        frame = pd.DataFrame(
            {
                "earlier": [0.0, 1.0, 2.0, 3.0, 4.0],
                "later": [0.0, 2.0, 4.0, 6.0, 8.0],
                "independent": [1.0, 0.0, 1.0, 0.0, 2.0],
            }
        )
        schema = fit_descriptor_schema(frame, DescriptorFilterConfig(correlation_threshold=0.98, correlation_comparison="ge"))
        self.assertEqual(schema.retained_columns, ("earlier", "independent"))
        self.assertEqual(schema.dropped_high_correlation, ("later",))

    def test_filter_stages_and_transform(self) -> None:
        frame = pd.DataFrame(
            {
                "missing": [1.0, np.nan, 2.0, 3.0],
                "constant": [5.0, 5.0, 5.0, 5.0],
                "low_variance": [0.0, 1e-8, 0.0, 1e-8],
                "retained": [0.0, 1.0, 0.0, 2.0],
            }
        )
        config = DescriptorFilterConfig(variance_threshold=1e-12, variance_ddof=0, correlation_threshold=None, correlation_comparison=None)
        schema = fit_descriptor_schema(frame, config)
        self.assertEqual(schema.dropped_missing, ("missing",))
        self.assertEqual(schema.dropped_constant, ("constant",))
        self.assertEqual(schema.dropped_low_variance, ("low_variance",))
        np.testing.assert_array_equal(transform_descriptor_frame(frame, schema), np.array([[0], [1], [0], [2]], dtype=np.float32))

    def test_complat_disables_correlation_filter(self) -> None:
        frame = pd.DataFrame({"a": [0.0, 1.0, 2.0], "b": [0.0, 2.0, 4.0]})
        schema = fit_descriptor_schema(frame, filter_config_for("complat"))
        self.assertEqual(schema.retained_columns, ("a", "b"))


class DescriptorProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.golden = json.loads((ROOT / "tests" / "fixtures" / "descriptor_protocol_golden.json").read_text())

    def test_protocol_registry(self) -> None:
        protocols = load_descriptor_protocols()["protocols"]
        self.assertEqual(protocols["aqsoldbc"]["filtering"]["correlated_pair_rule"], "retain the earlier input column and remove the later input column")
        self.assertEqual(protocols["complat"]["filtering"]["correlation_threshold"], None)
        self.assertEqual(protocols["jchem"]["expected_split_feature_counts"], self.golden["feature_counts"]["jchem"])

    def test_candidate_grid_hashes(self) -> None:
        observed = {
            "aqsoldbc": candidate_hash(generate_candidates("aqsoldbc")),
            "complat": candidate_hash(generate_candidates("complat")),
        }
        for fold in range(5):
            observed[f"tdc_{fold}"] = candidate_hash(generate_candidates("tdc", fold))
        for fold in range(1, 6):
            observed[f"jchem_{fold}"] = candidate_hash(generate_candidates("jchem", fold))
        self.assertEqual(observed, self.golden["candidate_hashes"])


@unittest.skipUnless(importlib.util.find_spec("mordred") and importlib.util.find_spec("rdkit"), "Mordred or RDKit is not installed")
class MordredCalculationTests(unittest.TestCase):
    def test_raw_descriptor_contract(self) -> None:
        frame = calculate_mordred_2d(["CCO", "invalid"], ["valid", "invalid"])
        self.assertEqual(frame.shape, (2, 1614))
        self.assertEqual(frame.columns[:3].tolist(), ["record_id", "ABC", "ABCGG"])
        self.assertEqual(frame.columns[-1], "mZagreb2")
        self.assertFalse(frame.iloc[0, 1:].isna().all())
        self.assertTrue(frame.iloc[1, 1:].isna().all())


@unittest.skipUnless(importlib.util.find_spec("xgboost"), "XGBoost is not installed")
class XGBoostInterfaceTests(unittest.TestCase):
    def test_builder_matches_direct_constructor(self) -> None:
        from xgboost import XGBRegressor

        rng = np.random.default_rng(41)
        x = rng.normal(size=(48, 5)).astype(np.float32)
        y = (x[:, 0] - 0.4 * x[:, 1] + rng.normal(scale=0.05, size=48)).astype(np.float32)
        parameters = {
            "n_estimators": 17,
            "learning_rate": 0.05,
            "max_depth": 3,
            "min_child_weight": 1.0,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.01,
            "reg_lambda": 1.2,
        }
        canonical = build_xgb_regressor(parameters, random_state=19, device="cpu", n_jobs=1)
        direct = XGBRegressor(objective="reg:squarederror", tree_method="hist", device="cpu", n_jobs=1, random_state=19, verbosity=0, **parameters)
        canonical.fit(x, y)
        direct.fit(x, y)
        np.testing.assert_array_equal(canonical.predict(x), direct.predict(x))


if __name__ == "__main__":
    unittest.main()
