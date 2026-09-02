from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dlg_sol.language import AUXILIARY_TARGETS, canonicalize_smiles, compute_auxiliary_targets, fit_auxiliary_scaler, load_chemberta_variants, masked_mean_pool, randomized_smiles, training_record_count, transform_auxiliary_targets


ROOT = Path(__file__).resolve().parents[1]
HAS_NEURAL_STACK = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None


class LanguageContractTests(unittest.TestCase):
    def test_variant_registry(self):
        variants = load_chemberta_variants(ROOT / "configs" / "chemberta_variants.json")
        self.assertEqual(set(variants), {"canonical_multitask", "randomized_single_task", "canonical_single_task"})
        self.assertEqual(variants["canonical_multitask"].auxiliary_targets, AUXILIARY_TARGETS)
        self.assertEqual(variants["randomized_single_task"].randomized_smiles_per_molecule, 2)
        self.assertEqual(variants["canonical_single_task"].learning_rate, 5e-5)
        self.assertIsNone(variants["canonical_multitask"].model_revision)

    def test_variant_registry_rejects_policy_change(self):
        source = json.loads((ROOT / "configs" / "chemberta_variants.json").read_text(encoding="utf-8"))
        source["shared"]["inference_smiles"] = "randomized"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "variants.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_chemberta_variants(path)

    def test_auxiliary_targets(self):
        targets = compute_auxiliary_targets(["CCO"])
        self.assertEqual(tuple(targets.columns), AUXILIARY_TARGETS)
        self.assertAlmostEqual(targets.loc[0, "logp"], -0.0014000000000000123, places=12)
        self.assertAlmostEqual(targets.loc[0, "tpsa"], 20.23, places=12)
        self.assertAlmostEqual(targets.loc[0, "molwt"], 46.069, places=12)
        self.assertEqual(targets.loc[0, "hbd"], 1.0)
        self.assertEqual(targets.loc[0, "hba"], 1.0)

    def test_auxiliary_scaling_uses_sample_standard_deviation(self):
        frame = pd.DataFrame({column: [1.0, 3.0, 5.0] for column in AUXILIARY_TARGETS})
        scaler = fit_auxiliary_scaler(frame)
        transformed = transform_auxiliary_targets(frame, scaler)
        self.assertTrue(np.allclose(transformed[:, 0], [-1.0, 0.0, 1.0]))

    def test_auxiliary_scaling_rejects_invalid_rows(self):
        frame = compute_auxiliary_targets(["not-a-smiles"])
        with self.assertRaises(ValueError):
            fit_auxiliary_scaler(frame)

    def test_smiles_contract(self):
        canonical = canonicalize_smiles("OCC")
        randomized = randomized_smiles(canonical)
        self.assertEqual(canonical, "CCO")
        self.assertEqual(canonicalize_smiles(randomized), canonical)
        self.assertEqual(training_record_count(7, 2), 21)

    def test_masked_mean_pool(self):
        hidden = np.array([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
        mask = np.array([[1, 1, 0]])
        pooled = masked_mean_pool(hidden, mask)
        self.assertTrue(np.array_equal(pooled, np.array([[2.0, 3.0]])))

    def test_masked_mean_pool_rejects_empty_sequence(self):
        with self.assertRaises(ValueError):
            masked_mean_pool(np.zeros((1, 2, 3)), np.zeros((1, 2)))

    @unittest.skipUnless(HAS_NEURAL_STACK, "language modelling dependencies are unavailable")
    def test_dataset_augmentation_and_label_free_inference(self):
        import torch

        from dlg_sol.language.modeling import ChemBertaDataset

        class Tokenizer:
            def __call__(self, value, **kwargs):
                return {"input_ids": torch.ones((1, 4), dtype=torch.long), "attention_mask": torch.ones((1, 4), dtype=torch.long)}

        dataset = ChemBertaDataset(["CCO", "CCN"], Tokenizer(), 4, randomized_per_molecule=2)
        self.assertEqual(len(dataset), 6)
        self.assertNotIn("label", dataset[0])
        self.assertNotIn("auxiliary", dataset[0])


if __name__ == "__main__":
    unittest.main()
