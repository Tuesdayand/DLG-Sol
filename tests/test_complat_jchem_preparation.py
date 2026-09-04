from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from dlg_sol.data import prepare_complat_input_package, prepare_jchem_input_package, reconstruct_complat_oof_assignments


ROOT = Path(__file__).resolve().parents[1]


class ComPlatPreparationTests(unittest.TestCase):
    def test_small_stratified_reconstruction_is_deterministic(self):
        targets = [float(index // 5) for index in range(100)]
        first = reconstruct_complat_oof_assignments(targets, 5, 20, 260722)
        second = reconstruct_complat_oof_assignments(targets, 5, 20, 260722)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 500)
        self.assertEqual(Counter(role for _, _, role in first), {"fit": 400, "scored": 100})
        self.assertEqual(len({index for index, _, role in first if role == "scored"}), 100)

    def test_unpinned_sources_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.csv"
            test = root / "test.csv"
            train.write_text("C_ID,smiles_canon,LogS\nA,CCO,-1\n", encoding="utf-8")
            test.write_text("C_ID,smiles_canon,LogS\nB,CCC,-2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pinned upstream object"):
                prepare_complat_input_package(train, test, root / "output", "R02", "2026-09-03", ROOT / "configs" / "complat_partition_recipe.json")


class JCheMPreparationTests(unittest.TestCase):
    def test_recipe_records_fixed_test_rounding_boundary(self):
        recipe = json.loads((ROOT / "configs" / "jchem_partition_recipe.json").read_text(encoding="utf-8"))
        self.assertEqual(recipe["label_policy"]["scored"], "SExp from predictions_test_set")
        self.assertEqual(recipe["source"]["invalid_source_rows"], [8340, 8483])
        self.assertEqual(recipe["partition"]["expected_role_counts"]["1"], {"fit": 6860, "coefficient": 1958, "scored": 980})

    def test_unpinned_workbook_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workbook = root / "dataset.xlsx"
            workbook.write_bytes(b"not the pinned workbook")
            with self.assertRaisesRegex(ValueError, "pinned upstream object"):
                prepare_jchem_input_package(workbook, root / "output", "2026-09-03", ROOT / "configs" / "jchem_partition_recipe.json")


if __name__ == "__main__":
    unittest.main()
