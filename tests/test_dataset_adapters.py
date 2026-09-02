from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dlg_sol.fusion import adapter_head_schedule, aggregate_scored_predictions, build_adapter_masks, fit_adapter_preprocessor, load_adapter_registry, load_fusion_config


ROOT = Path(__file__).resolve().parents[1]


class DatasetAdapterTests(unittest.TestCase):
    def setUp(self):
        self.registry = load_adapter_registry(ROOT / "configs" / "dataset_adapters.json")
        self.fusion = load_fusion_config(ROOT / "configs" / "fusion_ensemble.json")

    def test_exact_evaluation_contract(self):
        self.assertEqual(set(self.registry.adapters), {"R01", "R02", "R03", "R04", "R05", "R09"})
        self.assertEqual(self.registry.adapters["R03"].fit_values, ("fit",))
        self.assertEqual(self.registry.adapters["R03"].reserved_values, ("selection", "official_test"))
        self.assertEqual(self.registry.adapters["R05"].coefficient_values, ("Val",))
        self.assertFalse(self.registry.adapters["R05"].winsorization)
        self.assertEqual(self.registry.adapters["R09"].historical_neural_column, "polarh_aug2_head4")

    def test_tdc_masks_keep_selection_out_of_head_fit(self):
        metadata = pd.DataFrame({"record_id": ["a", "b", "c", "d"], "partition": ["fit", "selection", "oof_holdout", "official_test"]})
        masks = build_adapter_masks(metadata, self.registry.adapters["R03"])
        self.assertTrue(np.array_equal(masks.fit, [True, False, False, False]))
        self.assertTrue(np.array_equal(masks.scored, [False, False, True, False]))
        self.assertTrue(np.array_equal(masks.reserved, [False, True, False, True]))

    def test_jchem_masks_separate_coefficient_rows(self):
        metadata = pd.DataFrame({"record_id": ["a", "b", "c"], "split": ["Train", "Val", "External"]})
        masks = build_adapter_masks(metadata, self.registry.adapters["R05"])
        self.assertTrue(np.array_equal(masks.fit, [True, False, False]))
        self.assertTrue(np.array_equal(masks.coefficient, [False, True, False]))
        self.assertTrue(np.array_equal(masks.scored, [False, False, True]))

    def test_adapter_metadata_rejects_labels_and_unknown_roles(self):
        with self.assertRaises(ValueError):
            build_adapter_masks(pd.DataFrame({"record_id": ["a", "b"], "split": ["Train", "External"], "logS": [0.0, 1.0]}), self.registry.adapters["R02"])
        with self.assertRaises(ValueError):
            build_adapter_masks(pd.DataFrame({"record_id": ["a", "b"], "split": ["Train", "Test"]}), self.registry.adapters["R02"])

    def test_head_schedule_preserves_zero_and_one_based_folds(self):
        aqsol = adapter_head_schedule(self.registry.adapters["R01"], self.fusion, 0)
        jchem = adapter_head_schedule(self.registry.adapters["R05"], self.fusion, 1)
        self.assertEqual(aqsol, ((0, 260901, 29), (1, 260902, 82), (2, 260903, 40), (3, 260904, 21)))
        self.assertEqual(jchem, ((0, 360901, 29), (1, 360902, 82), (2, 360903, 40), (3, 360904, 21)))
        with self.assertRaises(ValueError):
            adapter_head_schedule(self.registry.adapters["R05"], self.fusion, 0)

    def test_adapter_preprocessing_applies_only_declared_winsorization(self):
        language = np.array([[0.0], [10.0], [1000.0]], dtype=np.float32)
        geometry = np.array([[0.0], [20.0], [-1000.0]], dtype=np.float32)
        masks = build_adapter_masks(pd.DataFrame({"record_id": ["a", "b", "c"], "split": ["Train", "Train", "External"]}), self.registry.adapters["R02"])
        clipped = fit_adapter_preprocessor(language, geometry, masks, self.registry.adapters["R02"], 7).transform(language, geometry)
        raw = fit_adapter_preprocessor(language, geometry, masks, self.registry.adapters["R05"], 7).transform(language, geometry)
        self.assertTrue(np.allclose(clipped[2], [1.0, -1.0]))
        self.assertGreater(raw[2, 0], 100.0)
        self.assertLess(raw[2, 1], -50.0)

    def test_oof_aggregation_requires_one_prediction_per_record(self):
        adapter = self.registry.adapters["R01"]
        adapter = type(adapter)(**{**adapter.__dict__, "expected_scored_rows": 2})
        frame = pd.DataFrame({"record_id": ["a", "b"], "fold_index": [0, 4], "aug2_head4": np.array([1.0, 3.0], dtype=np.float32)})
        result = aggregate_scored_predictions(frame, adapter)
        self.assertEqual(result.to_dict("list"), {"record_id": ["a", "b"], "aug2_head4": [1.0, 3.0]})
        with self.assertRaises(ValueError):
            aggregate_scored_predictions(pd.concat([frame, frame.iloc[[0]]], ignore_index=True), adapter)

    def test_split_model_aggregation_requires_every_declared_fold(self):
        adapter = self.registry.adapters["R04"]
        adapter = type(adapter)(**{**adapter.__dict__, "expected_scored_rows": 2})
        frame = pd.DataFrame({"record_id": ["a"] * 5 + ["b"] * 5, "fold_index": [0, 1, 2, 3, 4] * 2, "aug2_head4": np.arange(10, dtype=np.float32)})
        result = aggregate_scored_predictions(frame, adapter)
        self.assertTrue(np.array_equal(result["aug2_head4"].to_numpy(), np.array([2.0, 7.0], dtype=np.float32)))
        with self.assertRaises(ValueError):
            aggregate_scored_predictions(frame.iloc[:-1], adapter)

    def test_full_refit_aggregation_rejects_multiple_models(self):
        adapter = self.registry.adapters["R09"]
        adapter = type(adapter)(**{**adapter.__dict__, "expected_scored_rows": 1})
        frame = pd.DataFrame({"record_id": ["a"], "fold_index": [0], "aug2_head4": [1.0]})
        self.assertEqual(len(aggregate_scored_predictions(frame, adapter)), 1)
        with self.assertRaises(ValueError):
            aggregate_scored_predictions(pd.concat([frame, frame.assign(fold_index=1)]), adapter)

    def test_registry_rejects_winsorization_or_role_drift(self):
        payload = json.loads((ROOT / "configs" / "dataset_adapters.json").read_text(encoding="utf-8"))
        payload["evaluation_adapters"][4]["winsorization"] = True
        payload["evaluation_adapters"][4]["winsor_quantile"] = 0.001
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapters.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_adapter_registry(path)


if __name__ == "__main__":
    unittest.main()
