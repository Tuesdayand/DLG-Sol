from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from dlg_sol.workflow import (
    load_training_execution_recipes,
    load_training_role_contracts,
    require_primary_r02_selection,
)


ROOT = Path(__file__).resolve().parents[1]


def _load():
    roles = load_training_role_contracts(ROOT / "configs/training_role_contracts.json")
    recipes = load_training_execution_recipes(ROOT / "configs/training_execution_recipes.json", roles)
    return roles, recipes


def _mutated_config(mutator):
    directory = tempfile.TemporaryDirectory()
    root = Path(directory.name)
    shutil.copytree(ROOT / "configs", root / "configs")
    path = root / "configs/training_execution_recipes.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutator(payload)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return directory, path


class TrainingExecutionRecipeTests(unittest.TestCase):
    def test_all_role_contract_units_are_referenced_without_row_duplication(self):
        roles, registry = _load()
        self.assertEqual(len(registry.recipes), 30)
        referenced_units = 0
        for (evaluation_id, branch_id), recipe in registry.recipes.items():
            units = roles.evaluations[evaluation_id]["branches"][branch_id]["expected_units"]
            referenced_units += len(units)
            self.assertEqual(set(recipe.input_stages), set(units[0]["stages"]))
        self.assertEqual(referenced_units, 126)
        text = (ROOT / "configs/training_execution_recipes.json").read_text(encoding="utf-8")
        for forbidden in ('"rows"', '"row_count"', '"record_id_sha256"', '"stage_assignment_sha256"'):
            self.assertNotIn(forbidden, text)

    def test_r02_primary_rejects_fold_local_reselection(self):
        _, registry = _load()
        for branch in ("descriptor", "language", "geometry"):
            recipe = registry.get("R02", branch)
            require_primary_r02_selection(recipe, "frozen_global_preselection")
            with self.assertRaises(ValueError):
                require_primary_r02_selection(recipe, "fold_local_reselection")

    def test_r04_components_reuse_matching_r03_models(self):
        _, registry = _load()
        for branch in ("descriptor", "language", "geometry", "fusion_head"):
            recipe = registry.get("R04", branch)
            self.assertEqual(recipe.execution_type, "reuse_companion_model")
            self.assertEqual(recipe.dependency, {"evaluation_id": "R03", "branch_id": branch, "fold_mapping": "same_fold"})

    def test_r09_reuses_r02_selected_representation_settings(self):
        _, registry = _load()
        for branch in ("descriptor", "language", "geometry"):
            recipe = registry.get("R09", branch)
            self.assertEqual(recipe.selection_mode, "reuse_r02_frozen_global_preselection")
            self.assertEqual(recipe.dependency, {"evaluation_id": "R02", "branch_id": branch, "fold_mapping": "selected_settings"})

    def test_row_contract_duplication_is_rejected(self):
        directory, path = _mutated_config(lambda payload: payload["evaluations"]["R01"]["descriptor"].update({"rows": 8047}))
        with directory:
            roles = load_training_role_contracts(path.parent / "training_role_contracts.json")
            with self.assertRaises(ValueError):
                load_training_execution_recipes(path, roles)

    def test_stage_drift_is_rejected(self):
        directory, path = _mutated_config(lambda payload: payload["evaluations"]["R03"]["language"].update({"input_stages": ["parameter_fit", "scored_prediction"]}))
        with directory:
            roles = load_training_role_contracts(path.parent / "training_role_contracts.json")
            with self.assertRaises(ValueError):
                load_training_execution_recipes(path, roles)

    def test_unknown_configuration_reference_is_rejected(self):
        directory, path = _mutated_config(lambda payload: payload["evaluations"]["R05"]["geometry"].update({"configuration_refs": ["configs/missing.json"]}))
        with directory:
            roles = load_training_role_contracts(path.parent / "training_role_contracts.json")
            with self.assertRaises(ValueError):
                load_training_execution_recipes(path, roles)

    def test_wrong_r02_selection_mode_is_rejected(self):
        directory, path = _mutated_config(lambda payload: payload["evaluations"]["R02"]["descriptor"].update({"selection_mode": "fold_local_reselection"}))
        with directory:
            roles = load_training_role_contracts(path.parent / "training_role_contracts.json")
            with self.assertRaises(ValueError):
                load_training_execution_recipes(path, roles)


if __name__ == "__main__":
    unittest.main()
