from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from dlg_sol.data import prepare_tdc_input_package, reconstruct_tdc_native_partitions


ROOT = Path(__file__).resolve().parents[1]


class TDCPreparationTests(unittest.TestCase):
    def test_native_partition_reconstruction_is_deterministic(self):
        targets = [float(index // 10) for index in range(200)]
        first = reconstruct_tdc_native_partitions(targets, 50, 260722, 10, 5, 0.125)
        second = reconstruct_tdc_native_partitions(targets, 50, 260722, 10, 5, 0.125)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 1250)
        self.assertEqual(Counter(role for _, _, role in first), {"fit": 700, "selection": 100, "oof_holdout": 200, "official_test": 250})
        held_out = [record_id for record_id, _, role in first if role == "oof_holdout"]
        self.assertEqual(len(held_out), 200)
        self.assertEqual(len(set(held_out)), 200)

    def test_recipe_records_both_evaluation_role_boundaries(self):
        recipe = json.loads((ROOT / "configs" / "tdc_partition_recipe.json").read_text(encoding="utf-8"))
        self.assertEqual(recipe["evaluations"]["R03"]["native_role_map"]["oof_holdout"], "scored")
        self.assertEqual(recipe["evaluations"]["R03"]["native_role_map"]["official_test"], "reserved")
        self.assertEqual(recipe["evaluations"]["R04"]["native_role_map"]["oof_holdout"], "reserved")
        self.assertEqual(recipe["evaluations"]["R04"]["native_role_map"]["official_test"], "scored")

    def test_unpinned_runtime_exports_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train_val.csv"
            test = root / "test.csv"
            train.write_text("SMILES,Solubility\nCCO,-1\n", encoding="utf-8")
            test.write_text("SMILES,Solubility\nCCC,-2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pinned historical object"):
                prepare_tdc_input_package(train, test, root / "output", "R03", "2026-09-03", ROOT / "configs" / "tdc_partition_recipe.json")


if __name__ == "__main__":
    unittest.main()
