from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

from dlg_sol.data import BenchmarkInputFile, BenchmarkInputPackage
from dlg_sol.workflow import (
    MaterializedTrainingUnit,
    TrainingRoleContractRegistry,
    materialize_training_unit,
    write_materialized_training_unit,
)


def _hash_ids(values: set[str]) -> str:
    return hashlib.sha256("".join(f"{value}\n" for value in sorted(values)).encode()).hexdigest()


def _hash_assignments(stages: dict[str, set[str]]) -> str:
    rows = [f"{stage}\t{record_id}" for stage in sorted(stages) for record_id in sorted(stages[stage])]
    return hashlib.sha256("".join(f"{row}\n" for row in rows).encode()).hexdigest()


def _unit(stages: dict[str, set[str]]) -> dict:
    scored = stages["scored_prediction"]
    overlaps = {
        stage: len(scored & stages.get(stage, set()))
        for stage in ("parameter_fit", "selection_fit", "selection_score", "final_refit", "coefficient_fit")
    }
    return {
        "fold_index": 0,
        "stages": {stage: {"rows": len(values), "record_id_sha256": _hash_ids(values)} for stage, values in stages.items()},
        "stage_assignment_sha256": _hash_assignments(stages),
        "scored_overlap_policy": "zero_for_all_training_stages",
        "scored_label_overlap_rows": overlaps,
    }


def _package(root: Path) -> BenchmarkInputPackage:
    molecules = root / "molecules.csv"
    labels = root / "labels.csv"
    partitions = root / "partitions.csv"
    molecules.write_text("record_id,smiles\nf,CC\nv,CCC\ns,CCCC\n", encoding="utf-8")
    labels.write_text("record_id,logS\nf,-1.0\nv,-2.0\ns,-3.0\n", encoding="utf-8")
    partitions.write_text(
        "record_id,fold_index,role\nf,0,fit\nv,0,reserved\ns,0,scored\n"
        "f,1,scored\nv,1,fit\ns,1,reserved\n",
        encoding="utf-8",
    )
    files = {}
    for role, path, columns in (
        ("molecules", molecules, ("record_id", "smiles")),
        ("labels", labels, ("record_id", "logS")),
        ("partitions", partitions, ("record_id", "fold_index", "role")),
    ):
        files[role] = BenchmarkInputFile(role, path, hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size, None, columns)
    return BenchmarkInputPackage(root, "R01", "aqsoldbc", "source", "revision", "2026-09-04", "policy", "origin", files, 3, 6, 3)


def _registry(root: Path, branch: str = "language") -> TrainingRoleContractRegistry:
    stages = {"parameter_fit": {"f"}, "selection_score": {"v"}, "scored_prediction": {"s"}}
    evaluations = {"R01": {"branches": {branch: {"role_resolution": "fixture", "expected_units": [_unit(stages)]}}}}
    return TrainingRoleContractRegistry(root / "configs" / "training_role_contracts.json", "aug2_head4", "AUDITED_MATERIALIZER_AVAILABLE", evaluations, {})


class TrainingRoleMaterializationTests(unittest.TestCase):
    def test_stage_rows_are_materialized_and_scored_labels_are_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = _package(root)
            unit = materialize_training_unit(package, _registry(root), "R01", "language", 0)
            self.assertEqual(unit.stage_rows["parameter_fit"]["record_id"].tolist(), ["f"])
            self.assertEqual(unit.stage_rows["selection_score"]["record_id"].tolist(), ["v"])
            self.assertEqual(unit.stage_rows["scored_prediction"].columns.tolist(), ["record_id", "smiles"])
            self.assertEqual(unit.stage_rows["scored_prediction"]["record_id"].tolist(), ["s"])

    def test_contract_hash_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = _package(root)
            registry = _registry(root)
            registry.evaluations["R01"]["branches"]["language"]["expected_units"][0]["stages"]["selection_score"]["record_id_sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                materialize_training_unit(package, registry, "R01", "language", 0)

    def test_wrong_evaluation_branch_and_fold_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = _package(root)
            registry = _registry(root)
            for evaluation, branch, fold in (("R02", "language", 0), ("R01", "unknown", 0), ("R01", "language", 9)):
                with self.subTest(evaluation=evaluation, branch=branch, fold=fold), self.assertRaises(ValueError):
                    materialize_training_unit(package, registry, evaluation, branch, fold)

    def test_writer_rejects_scored_label_injection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unit = materialize_training_unit(_package(root), _registry(root), "R01", "language", 0)
            unsafe = dict(unit.stage_rows)
            unsafe["scored_prediction"] = unsafe["scored_prediction"].assign(logS=-3.0)
            mutated = MaterializedTrainingUnit(
                unit.evaluation_id,
                unit.branch_id,
                unit.fold_index,
                unit.role_resolution,
                unit.scored_overlap_policy,
                unsafe,
                unit.nested_selection_units,
            )
            with self.assertRaises(ValueError):
                write_materialized_training_unit(mutated, root / "unsafe")

    def test_writer_creates_local_only_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unit = materialize_training_unit(_package(root), _registry(root), "R01", "language", 0)
            output = root / "output"
            manifest = write_materialized_training_unit(unit, output)
            self.assertEqual(manifest["redistribution_status"], "user_local_do_not_redistribute")
            self.assertEqual(manifest["files"]["scored_prediction.csv"]["columns"], ["record_id", "smiles"])
            observed = json.loads((output / "materialization_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(observed, manifest)

    def test_complat_representation_requires_pinned_role_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = replace(_package(root), evaluation_id="R02", dataset_id="complat")
            stages = {"selection_fit": {"f"}, "selection_score": {"v"}, "final_refit": {"f", "v"}, "scored_prediction": {"s"}}
            evaluations = {"R02": {"branches": {"descriptor": {"role_resolution": "fixture", "expected_units": [_unit(stages)]}}}}
            registry = TrainingRoleContractRegistry(root / "configs/training_role_contracts.json", "aug2_head4", "AUDITED_MATERIALIZER_AVAILABLE", evaluations, {})
            with self.assertRaises(ValueError):
                materialize_training_unit(package, registry, "R02", "descriptor", 0)

    def test_r09_coefficient_requires_r02_dependency_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = replace(_package(root), evaluation_id="R09", dataset_id="complat")
            stages = {"coefficient_fit": {"f", "v"}, "scored_prediction": {"s"}}
            evaluations = {"R09": {"branches": {"blend_coefficient": {"role_resolution": "fixture", "expected_units": [_unit(stages)]}}}}
            registry = TrainingRoleContractRegistry(root / "configs/training_role_contracts.json", "aug2_head4", "AUDITED_MATERIALIZER_AVAILABLE", evaluations, {})
            with self.assertRaises(ValueError):
                materialize_training_unit(package, registry, "R09", "blend_coefficient", 0)


if __name__ == "__main__":
    unittest.main()
