from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dlg_sol.fusion import fit_fusion_preprocessor, fit_training_mean_fallback, load_fusion_config, mean_member_predictions


ROOT = Path(__file__).resolve().parents[1]
HAS_TORCH = importlib.util.find_spec("torch") is not None


class FusionContractTests(unittest.TestCase):
    def test_configuration_contract(self):
        config = load_fusion_config(ROOT / "configs" / "fusion_ensemble.json")
        observed = [
            (item.member_index, item.language_variant, item.preprocessing, item.regressor, item.base_seed, item.fixed_epochs)
            for item in config.members
        ]
        expected = [
            (0, "randomized_single_task", "raw_standardized_concat", "residual_mlp", 260901, 29),
            (1, "randomized_single_task", "raw_standardized_concat", "residual_mlp", 260902, 82),
            (2, "randomized_single_task", "raw_standardized_concat", "residual_mlp", 260903, 40),
            (3, "randomized_single_task", "raw_standardized_concat", "residual_mlp", 260904, 21),
        ]
        self.assertEqual(observed, expected)
        self.assertEqual(config.model_identity, "aug2_head4")
        self.assertEqual(config.adapter_winsorization, {"aqsoldbc": True, "complat": True, "tdc": True, "jchem": False})
        self.assertEqual(config.head_parameters["hidden"], 384)
        self.assertEqual(config.expected_members, 4)

    def test_configuration_rejects_universal_winsorization_claim(self):
        payload = json.loads((ROOT / "configs" / "fusion_ensemble.json").read_text(encoding="utf-8"))
        payload["preprocessing"]["adapter_winsorization"]["jchem"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fusion.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_fusion_config(path)

    def test_configuration_rejects_mixed_final_member(self):
        payload = json.loads((ROOT / "configs" / "fusion_ensemble.json").read_text(encoding="utf-8"))
        payload["members"][0]["language_variant"] = "canonical_multitask"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fusion.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_fusion_config(path)

    def test_raw_preprocessing_is_fit_only_and_blockwise(self):
        language = np.array([[0.0, 10.0], [2.0, 14.0], [100.0, 1000.0]], dtype=np.float32)
        geometry = np.array([[1.0], [5.0], [-100.0]], dtype=np.float32)
        fit = np.array([True, True, False])
        preprocessor = fit_fusion_preprocessor(language, geometry, fit, "raw_standardized_concat", 7)
        transformed = preprocessor.transform(language, geometry)
        self.assertTrue(np.allclose(transformed[:2, :2].mean(axis=0), 0.0))
        self.assertTrue(np.allclose(transformed[:2, 2:].mean(axis=0), 0.0))
        self.assertGreater(transformed[2, 0], 50.0)
        self.assertLess(transformed[2, 2], -40.0)

    def test_pca_l2_preprocessing_has_recorded_shape_and_norm(self):
        generator = np.random.default_rng(31)
        language = generator.normal(size=(12, 8)).astype(np.float32)
        geometry = generator.normal(size=(12, 5)).astype(np.float32)
        fit = np.array([True] * 9 + [False] * 3)
        preprocessor = fit_fusion_preprocessor(language, geometry, fit, "separate_pca384_l2_concat", 17)
        transformed = preprocessor.transform(language, geometry)
        self.assertEqual(transformed.shape, (12, 768))
        self.assertTrue(np.allclose(np.linalg.norm(transformed, axis=1), 1.0, atol=1e-6))

    def test_optional_winsorization_uses_fit_rows_only(self):
        language = np.array([[0.0], [10.0], [1000.0]], dtype=np.float32)
        geometry = np.array([[0.0], [20.0], [-1000.0]], dtype=np.float32)
        fit = np.array([True, True, False])
        preprocessor = fit_fusion_preprocessor(language, geometry, fit, "raw_standardized_concat", 3, winsor_quantile=0.0)
        transformed = preprocessor.transform(language, geometry)
        self.assertTrue(np.allclose(transformed[2], [1.0, -1.0]))

    def test_training_mean_fallback_is_label_free(self):
        embeddings = np.array([[1.0, 3.0], [3.0, 5.0], [np.nan, np.nan], [100.0, 100.0]], dtype=np.float32)
        fit = np.array([True, True, False, False])
        fallback = fit_training_mean_fallback(embeddings, fit)
        transformed = fallback.transform(embeddings, [False, False, True, False])
        self.assertTrue(np.array_equal(fallback.values, np.array([2.0, 4.0], dtype=np.float32)))
        self.assertTrue(np.array_equal(transformed[2], fallback.values))

    def test_member_average_uses_float32_arithmetic_mean(self):
        values = np.array([[1.0, 2.0, 3.0, 4.0], [-1.0, 0.0, 1.0, 2.0]], dtype=np.float32)
        self.assertTrue(np.array_equal(mean_member_predictions(values), np.array([2.5, 0.5], dtype=np.float32)))
        with self.assertRaises(ValueError):
            mean_member_predictions(values[:, :3])

    @unittest.skipUnless(HAS_TORCH, "fusion modelling dependency is unavailable")
    def test_plain_and_residual_checkpoint_compatibility(self):
        import torch

        from dlg_sol.fusion.modeling import build_fusion_model, load_fusion_checkpoint

        parameters = {"hidden": 8, "depth": 2, "dropout": 0.2}
        with tempfile.TemporaryDirectory() as directory:
            for regressor in ("plain_mlp", "residual_mlp"):
                original = build_fusion_model(regressor, 6, parameters)
                path = Path(directory) / f"{regressor}.pt"
                torch.save({"state_dict": original.state_dict()}, path)
                loaded = load_fusion_checkpoint(path, regressor, 6, parameters)
                self.assertEqual(set(original.state_dict()), set(loaded.state_dict()))

    @unittest.skipUnless(HAS_TORCH, "fusion modelling dependency is unavailable")
    def test_fit_and_label_free_prediction(self):
        import torch

        from dlg_sol.fusion.training import fit_fusion_model, predict_fusion

        generator = np.random.default_rng(9)
        fit_x = generator.normal(size=(24, 4)).astype(np.float32)
        fit_y = (fit_x[:, 0] - fit_x[:, 1]).astype(np.float32)
        selection_x = generator.normal(size=(8, 4)).astype(np.float32)
        selection_y = (selection_x[:, 0] - selection_x[:, 1]).astype(np.float32)
        parameters = {"hidden": 8, "depth": 1, "dropout": 0.0, "learning_rate": 0.01, "weight_decay": 0.0, "batch_size": 8, "maximum_epochs": 3}
        result = fit_fusion_model(fit_x, fit_y, selection_x, selection_y, "plain_mlp", parameters, 2, 5.0, 1e-6, 11, torch.device("cpu"))
        predictions = predict_fusion(result.model, selection_x, torch.device("cpu"))
        self.assertEqual(predictions.shape, (8,))
        self.assertTrue(np.isfinite(predictions).all())

    @unittest.skipUnless(HAS_TORCH, "fusion modelling dependency is unavailable")
    def test_fixed_epoch_residual_head(self):
        import torch

        from dlg_sol.fusion.training import fit_fixed_epoch_fusion_head, predict_fusion

        generator = np.random.default_rng(13)
        fit_x = generator.normal(size=(24, 4)).astype(np.float32)
        fit_y = (fit_x[:, 0] + 0.5 * fit_x[:, 2]).astype(np.float32)
        parameters = {"hidden": 8, "depth": 1, "dropout": 0.0, "learning_rate": 0.01, "weight_decay": 0.0, "batch_size": 8}
        result = fit_fixed_epoch_fusion_head(fit_x, fit_y, parameters, 3, 5.0, 260901, torch.device("cpu"))
        predictions = predict_fusion(result.model, fit_x, torch.device("cpu"))
        self.assertEqual((result.epochs, result.seed), (3, 260901))
        self.assertEqual(predictions.shape, (24,))
        self.assertTrue(np.isfinite(predictions).all())


if __name__ == "__main__":
    unittest.main()
