from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from dlg_sol.workflow import load_materialized_branch_input, load_training_execution_recipes, load_training_role_contracts
from dlg_sol.workflow.descriptor_execution import run_descriptor_unit


ROOT = Path(__file__).resolve().parents[1]


def _identity(path):
    return {"rows": len(pd.read_csv(path)), "columns": pd.read_csv(path, nrows=0).columns.tolist(), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _materialized(root, scored_label=False):
    stages = {
        "parameter_fit": [(f"f{index}", "C" * (index + 1), -0.2 * index) for index in range(8)],
        "selection_score": [(f"v{index}", "N" + "C" * (index + 1), -0.3 * index) for index in range(3)],
        "scored_prediction": [(f"s{index}", "O" + "C" * (index + 1)) for index in range(2)],
    }
    files = {}
    for stage in ("parameter_fit", "selection_fit", "selection_score", "final_refit", "coefficient_fit", "scored_prediction"):
        path = root / f"{stage}.csv"
        rows = stages.get(stage, [])
        columns = ["record_id", "smiles"] if stage == "scored_prediction" and not scored_label else ["record_id", "smiles", "logS"]
        if stage == "scored_prediction" and scored_label:
            rows = [(*row, -1.0) for row in rows]
        pd.DataFrame(rows, columns=columns).to_csv(path, index=False, lineterminator="\n")
        files[path.name] = _identity(path)
    manifest = {"schema_version": 1, "evaluation_id": "R05", "branch_id": "descriptor", "fold_index": 1, "redistribution_status": "user_local_do_not_redistribute", "files": files, "nested_files": {}}
    (root / "materialization_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _registries():
    roles = load_training_role_contracts(ROOT / "configs/training_role_contracts.json")
    return load_training_execution_recipes(ROOT / "configs/training_execution_recipes.json", roles)


class BranchInputTests(unittest.TestCase):
    def test_recorded_language_and_geometry_execution_profiles(self):
        language = json.loads((ROOT / "configs/chemberta_variants.json").read_text(encoding="utf-8"))["execution_profiles"]
        geometry = json.loads((ROOT / "configs/geometry_branch.json").read_text(encoding="utf-8"))["execution_profiles"]
        self.assertEqual(set(language), {"R01", "R02", "R03", "R04", "R05", "R09"})
        self.assertEqual(language["R02"]["selected_epoch"], 14)
        self.assertEqual(language["R03"]["seed_base"], 260762)
        self.assertEqual(language["R03"]["fold_seed_multiplier"], 100)
        self.assertEqual(geometry["R01"]["selected_trial"], 2)
        self.assertEqual(geometry["R02"]["selected_trial"], 0)
        self.assertEqual(geometry["R02"]["best_epoch"], 19)
        self.assertEqual(geometry["R03"]["candidate_count"], 4)
        self.assertEqual(geometry["R03"]["maximum_epochs"], 140)
        self.assertEqual(geometry["R03"]["patience"], 20)
        self.assertEqual(geometry["R05"]["parameters"], geometry["R01"]["parameters"])

    def test_scored_label_injection_is_rejected_even_with_updated_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _materialized(root, scored_label=True)
            with self.assertRaises(ValueError):
                load_materialized_branch_input(root, _registries())

    def test_materialization_hash_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _materialized(root)
            with (root / "selection_score.csv").open("a", encoding="utf-8") as handle:
                handle.write("extra,CC,-1\n")
            with self.assertRaises(ValueError):
                load_materialized_branch_input(root, _registries())

    def test_descriptor_unit_fits_and_writes_label_free_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            materialized = base / "materialized"
            materialized.mkdir()
            _materialized(materialized)
            branch_input = load_materialized_branch_input(materialized, _registries())
            rows = pd.concat([frame[["record_id"]] for frame in branch_input.stages.values() if len(frame)], ignore_index=True)
            rows["feature_a"] = range(len(rows))
            rows["feature_b"] = [value * value for value in range(len(rows))]
            cache = base / "descriptors.csv"
            rows.to_csv(cache, index=False)
            output = base / "output"
            candidate = {"trial": 0, "n_estimators": 8, "learning_rate": 0.1, "max_depth": 2, "min_child_weight": 1.0, "subsample": 1.0, "colsample_bytree": 1.0, "reg_alpha": 0.0, "reg_lambda": 1.0}
            with patch("dlg_sol.workflow.descriptor_execution.generate_candidates", return_value=[candidate]):
                manifest = run_descriptor_unit(branch_input, output, cache, "cpu", 1)
            predictions = pd.read_csv(output / "predictions.csv")
            self.assertEqual(predictions.columns.tolist(), ["record_id", "prediction"])
            self.assertEqual(predictions["record_id"].tolist(), ["s0", "s1"])
            self.assertFalse(set(predictions.columns) & {"logS", "label", "target", "y"})
            self.assertFalse(manifest["scored_labels_accessed"])
            self.assertEqual(manifest["selection_mode"], "reported_split_validation")

    def test_run_manifest_rejects_private_output_details(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            materialized = base / "materialized"
            materialized.mkdir()
            _materialized(materialized)
            branch_input = load_materialized_branch_input(materialized, _registries())
            from dlg_sol.workflow.branch_io import write_prediction_output

            with self.assertRaises(ValueError):
                write_prediction_output(base / "output", branch_input, [0.0, 0.0], {}, {"source": "/data/" + "koo/private.csv"})


if __name__ == "__main__":
    unittest.main()
