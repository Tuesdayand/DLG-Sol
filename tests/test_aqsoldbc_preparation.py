from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from dlg_sol.data import prepare_aqsoldbc_input_package, reconstruct_aqsoldbc_partitions


ROOT = Path(__file__).resolve().parents[1]


class AqSolDBcPreparationTests(unittest.TestCase):
    def test_recorded_partition_identity_and_counts(self):
        recipe = json.loads((ROOT / "configs" / "aqsoldbc_partition_recipe.json").read_text(encoding="utf-8"))
        outer = recipe["partition"]["outer"]
        rows = reconstruct_aqsoldbc_partitions(recipe["source"]["rows"], outer["n_splits"], outer["random_state"], recipe["partition"]["inner_reserved"]["test_fraction"])
        identity = hashlib.sha256("".join(f"{index},{fold},{role}\n" for index, fold, role in rows).encode()).hexdigest()
        self.assertEqual(identity, recipe["expected_output"]["assignment_index_sha256"])
        for fold in range(outer["n_splits"]):
            counts = Counter(role for _, fold_index, role in rows if fold_index == fold)
            self.assertEqual(dict(counts), recipe["expected_role_counts"][str(fold)])

    def test_unpinned_source_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "AqSolDBc.csv"
            source.write_text("ID,SmilesCurated,ExperimentalLogS\nA-1,CCO,-1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pinned original object"):
                prepare_aqsoldbc_input_package(source, root / "output", "2026-09-03", ROOT / "configs" / "aqsoldbc_partition_recipe.json")


if __name__ == "__main__":
    unittest.main()
