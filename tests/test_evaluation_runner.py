from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from dlg_sol.evaluation import assemble_evaluation_predictions, fit_evaluation_coefficients, load_acquisition_registry, load_evaluation_package, score_evaluation_predictions
from dlg_sol.evaluation.__main__ import _output_directory
from dlg_sol.fusion import load_adapter_registry


ROOT = Path(__file__).resolve().parents[1]


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _adapter(source, *, folds, rows, policy, aggregation):
    return replace(source, fold_indices=tuple(folds), expected_scored_rows=int(rows), coefficient_policy=policy, scored_prediction_aggregation=aggregation)


def _coefficient_rows(folds, alphas):
    predictions = []
    labels = []
    for fold, alpha in zip(folds, alphas):
        for index, direction in enumerate((1.0, 2.0)):
            record_id = f"c{fold}_{index}"
            predictions.append({"deployment_fold": fold, "record_id": record_id, "descriptor_prediction": 0.0, "aug2_head4": direction})
            labels.append({"deployment_fold": fold, "record_id": record_id, "logS": alpha * direction})
    return pd.DataFrame(predictions), pd.DataFrame(labels)


class EvaluationAssemblyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = load_adapter_registry(ROOT / "configs" / "dataset_adapters.json")

    def test_r01_uses_fixed_half_and_rejects_coefficient_files(self):
        adapter = _adapter(self.registry.adapters["R01"], folds=(0, 1), rows=2, policy="prespecified_0.5", aggregation="one_held_out_prediction")
        scored = pd.DataFrame({"record_id": ["a", "b"], "fold_index": [0, 1], "descriptor_prediction": [0.0, 2.0], "aug2_head4": [2.0, 4.0]})
        result = assemble_evaluation_predictions(adapter, scored)
        np.testing.assert_allclose(result.predictions["dlg_sol"], [1.0, 3.0])
        self.assertEqual(result.coefficients["alpha"].tolist(), [0.5, 0.5])
        with self.assertRaises(ValueError):
            fit_evaluation_coefficients(adapter, scored, pd.DataFrame(), pd.DataFrame())

    def test_crossfit_coefficients_are_fold_specific(self):
        adapter = _adapter(self.registry.adapters["R02"], folds=(0, 1), rows=2, policy="cross_fitted_from_training_side_oof", aggregation="one_held_out_prediction")
        scored = pd.DataFrame({"record_id": ["a", "b"], "fold_index": [0, 1], "descriptor_prediction": [0.0, 0.0], "aug2_head4": [10.0, 10.0]})
        coefficient_predictions, coefficient_labels = _coefficient_rows((0, 1), (0.25, 0.75))
        result = assemble_evaluation_predictions(adapter, scored, coefficient_predictions, coefficient_labels)
        np.testing.assert_allclose(result.coefficients["alpha"], [0.25, 0.75])
        np.testing.assert_allclose(result.predictions["dlg_sol"], [2.5, 7.5])

    def test_crossfit_allows_other_fold_records_but_rejects_same_fold_overlap(self):
        adapter = _adapter(self.registry.adapters["R03"], folds=(0, 1), rows=2, policy="cross_fitted_from_training_side_oof", aggregation="one_held_out_prediction")
        scored = pd.DataFrame({"record_id": ["a", "b"], "fold_index": [0, 1], "descriptor_prediction": [0.0, 0.0], "aug2_head4": [1.0, 1.0]})
        predictions = pd.DataFrame({"deployment_fold": [0, 1], "record_id": ["b", "a"], "descriptor_prediction": [0.0, 0.0], "aug2_head4": [1.0, 1.0]})
        labels = pd.DataFrame({"deployment_fold": [0, 1], "record_id": ["b", "a"], "logS": [0.4, 0.6]})
        result = fit_evaluation_coefficients(adapter, scored, predictions, labels)
        np.testing.assert_allclose(result["alpha"], [0.4, 0.6])
        bad = predictions.copy()
        bad.loc[0, "record_id"] = "a"
        bad_labels = labels.copy()
        bad_labels.loc[0, "record_id"] = "a"
        with self.assertRaises(ValueError):
            fit_evaluation_coefficients(adapter, scored, bad, bad_labels)

    def test_missing_crossfit_deployment_fold_is_rejected(self):
        adapter = _adapter(self.registry.adapters["R02"], folds=(0, 1), rows=2, policy="cross_fitted_from_training_side_oof", aggregation="one_held_out_prediction")
        scored = pd.DataFrame({"record_id": ["a", "b"], "fold_index": [0, 1], "descriptor_prediction": [0.0, 0.0], "aug2_head4": [1.0, 1.0]})
        predictions, labels = _coefficient_rows((0,), (0.5,))
        with self.assertRaises(ValueError):
            fit_evaluation_coefficients(adapter, scored, predictions, labels)

    def test_r04_uses_one_coefficient_before_fold_mean(self):
        adapter = _adapter(self.registry.adapters["R04"], folds=(0, 1), rows=1, policy="one_coefficient_from_complete_tdc_oof", aggregation="mean_across_split_models")
        scored = pd.DataFrame({"record_id": ["a", "a"], "fold_index": [0, 1], "descriptor_prediction": [0.0, 4.0], "aug2_head4": [4.0, 8.0]})
        predictions, labels = _coefficient_rows((0,), (0.25,))
        result = assemble_evaluation_predictions(adapter, scored, predictions, labels)
        self.assertAlmostEqual(result.predictions.loc[0, "descriptor_prediction"], 2.0)
        self.assertAlmostEqual(result.predictions.loc[0, "aug2_head4"], 6.0)
        self.assertAlmostEqual(result.predictions.loc[0, "dlg_sol"], 3.0)

    def test_r05_blends_per_fold_before_averaging(self):
        adapter = _adapter(self.registry.adapters["R05"], folds=(1, 2), rows=1, policy="validation_only_within_each_reported_split", aggregation="mean_across_split_models")
        scored = pd.DataFrame({"record_id": ["a", "a"], "fold_index": [1, 2], "descriptor_prediction": [0.0, 0.0], "aug2_head4": [10.0, 30.0]})
        predictions, labels = _coefficient_rows((1, 2), (0.0, 1.0))
        result = assemble_evaluation_predictions(adapter, scored, predictions, labels)
        self.assertAlmostEqual(result.predictions.loc[0, "descriptor_prediction"], 0.0)
        self.assertAlmostEqual(result.predictions.loc[0, "aug2_head4"], 20.0)
        self.assertAlmostEqual(result.predictions.loc[0, "dlg_sol"], 15.0)
        self.assertNotEqual(result.predictions.loc[0, "dlg_sol"], 10.0)

    def test_r09_full_refit(self):
        adapter = _adapter(self.registry.adapters["R09"], folds=(0,), rows=1, policy="one_coefficient_from_complete_complat_oof", aggregation="single_full_refit")
        scored = pd.DataFrame({"record_id": ["a"], "fold_index": [0], "descriptor_prediction": [2.0], "aug2_head4": [6.0]})
        predictions, labels = _coefficient_rows((0,), (0.75,))
        result = assemble_evaluation_predictions(adapter, scored, predictions, labels)
        self.assertAlmostEqual(result.predictions.loc[0, "dlg_sol"], 5.0)

    def test_nonfinite_duplicate_and_incomplete_folds_are_rejected(self):
        adapter = _adapter(self.registry.adapters["R05"], folds=(1, 2), rows=1, policy="validation_only_within_each_reported_split", aggregation="mean_across_split_models")
        predictions, labels = _coefficient_rows((1, 2), (0.5, 0.5))
        incomplete = pd.DataFrame({"record_id": ["a"], "fold_index": [1], "descriptor_prediction": [0.0], "aug2_head4": [1.0]})
        with self.assertRaises(ValueError):
            assemble_evaluation_predictions(adapter, incomplete, predictions, labels)
        duplicate = pd.concat([incomplete, incomplete], ignore_index=True)
        with self.assertRaises(ValueError):
            assemble_evaluation_predictions(adapter, duplicate, predictions, labels)
        invalid = pd.concat([incomplete, incomplete.assign(fold_index=2)], ignore_index=True)
        invalid.loc[1, "aug2_head4"] = np.nan
        with self.assertRaises(ValueError):
            assemble_evaluation_predictions(adapter, invalid, predictions, labels)

    def test_scoring_is_exact_and_deterministic(self):
        predictions = pd.DataFrame({"record_id": ["b", "a"], "descriptor_prediction": [2.0, 0.0], "aug2_head4": [0.0, 2.0], "dlg_sol": [1.0, 1.0]})
        labels = pd.DataFrame({"record_id": ["a", "b"], "logS": [1.0, 1.0]})
        first = score_evaluation_predictions(predictions, labels, seed=17, replicates=100)
        second = score_evaluation_predictions(predictions, labels, seed=17, replicates=100)
        self.assertEqual(first, second)
        self.assertEqual(first.metrics["dlg_sol"]["rmse"], 0.0)


class EvaluationPackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapters = load_adapter_registry(ROOT / "configs" / "dataset_adapters.json")
        cls.acquisition = load_acquisition_registry(ROOT / "configs" / "data_acquisition.json")

    def _write_package(self, root: Path, evaluation_id="R01", invalid_labels=False):
        adapter = replace(self.adapters.adapters[evaluation_id], expected_scored_rows=1)
        source_id = self.acquisition.evaluation_source_map[evaluation_id]
        source = self.acquisition.sources[source_id]
        frames = {
            "scored_component_predictions": pd.DataFrame({"record_id": ["s"], "fold_index": [adapter.fold_indices[0]], "descriptor_prediction": [0.0], "aug2_head4": [1.0]}),
            "scored_labels": pd.DataFrame({"record_id": ["s"], "logS": [0.5]}),
        }
        if adapter.coefficient_policy != "prespecified_0.5":
            folds = (0,) if adapter.coefficient_policy.startswith("one_coefficient") else adapter.fold_indices
            predictions, labels = _coefficient_rows(folds, [0.5] * len(folds))
            frames["coefficient_component_predictions"] = predictions
            frames["coefficient_labels"] = labels
        files = {}
        for role, frame in frames.items():
            path = root / f"{role}.csv"
            if role == "scored_labels" and invalid_labels:
                path.write_text("not,a,csv,matching,the,declaration\n", encoding="utf-8")
                rows = 1
                columns = ["record_id", "logS"]
            else:
                frame.to_csv(path, index=False)
                rows = len(frame)
                columns = list(frame.columns)
            files[role] = {"path": path.name, "sha256": _hash(path), "bytes": path.stat().st_size, "rows": rows, "columns": columns}
        payload = {
            "schema_version": 1,
            "evaluation_id": evaluation_id,
            "dataset_id": adapter.dataset_id,
            "model_identity": "aug2_head4",
            "acquisition_mode": source.acquisition_mode,
            "upstream_source_id": source_id,
            "upstream_revision": source.revision,
            "retrieval_date": "2026-09-02",
            "derivation_record": "synthetic-fixture-v1",
            "contains_scored_labels": True,
            "redistribution_status": "not_bundled_local_use_only",
            "files": files,
        }
        (root / "package_manifest.json").write_text(json.dumps(payload), encoding="utf-8")
        return adapter, payload

    def test_acquisition_registry_denies_automatic_download(self):
        self.assertTrue(all(not source.automatic_download for source in self.acquisition.sources.values()))

    def test_all_six_evaluation_packages_accept_the_declared_layout(self):
        for evaluation_id in ("R01", "R02", "R03", "R04", "R05", "R09"):
            with self.subTest(evaluation_id=evaluation_id), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                adapter, _ = self._write_package(root, evaluation_id=evaluation_id)
                package = load_evaluation_package(root, adapter, self.acquisition, "assemble")
                self.assertEqual(package.evaluation_id, evaluation_id)
                self.assertFalse(package.scored_labels_content_accessed)

    def test_assembly_does_not_inspect_scored_label_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter, _ = self._write_package(root, invalid_labels=True)
            package = load_evaluation_package(root, adapter, self.acquisition, "assemble")
            self.assertFalse(package.scored_labels_content_accessed)
            with self.assertRaises(ValueError):
                load_evaluation_package(root, adapter, self.acquisition, "score")

    def test_hash_size_schema_and_identity_drift_are_rejected(self):
        for mutation in ("hash", "rows", "identity"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                adapter, payload = self._write_package(root)
                if mutation == "hash":
                    payload["files"]["scored_component_predictions"]["sha256"] = "0" * 64
                elif mutation == "rows":
                    payload["files"]["scored_component_predictions"]["rows"] = 2
                else:
                    payload["model_identity"] = "mixed_candidate"
                (root / "package_manifest.json").write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_evaluation_package(root, adapter, self.acquisition, "assemble")

    def test_absolute_traversal_and_symlink_paths_are_rejected(self):
        for relative in ("../scored_component_predictions.csv", "/tmp/scored_component_predictions.csv"):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                adapter, payload = self._write_package(root)
                payload["files"]["scored_component_predictions"]["path"] = relative
                (root / "package_manifest.json").write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_evaluation_package(root, adapter, self.acquisition, "assemble")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter, payload = self._write_package(root)
            source = root / "scored_component_predictions.csv"
            target = root / "real.csv"
            source.rename(target)
            source.symlink_to(target.name)
            payload["files"]["scored_component_predictions"].update({"sha256": _hash(target), "bytes": target.stat().st_size})
            (root / "package_manifest.json").write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_evaluation_package(root, adapter, self.acquisition, "assemble")

    def test_package_rejects_label_in_prediction_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter, payload = self._write_package(root)
            path = root / "scored_component_predictions.csv"
            frame = pd.read_csv(path)
            frame["logS"] = 0.5
            frame.to_csv(path, index=False)
            item = payload["files"]["scored_component_predictions"]
            item.update({"sha256": _hash(path), "bytes": path.stat().st_size, "columns": list(frame.columns)})
            (root / "package_manifest.json").write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_evaluation_package(root, adapter, self.acquisition, "assemble")

    def test_nonempty_output_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "existing.txt").write_text("x", encoding="utf-8")
            with self.assertRaises(ValueError):
                _output_directory(root)


if __name__ == "__main__":
    unittest.main()
