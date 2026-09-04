from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from dlg_sol.data import InputContractRegistry, load_benchmark_input_package, load_input_contract_registry
from dlg_sol.evaluation import load_acquisition_registry


ROOT = Path(__file__).resolve().parents[1]


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for _ in handle) - 1


class BenchmarkInputPackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = load_input_contract_registry(ROOT / "configs" / "benchmark_input_contracts.json")
        cls.acquisition = load_acquisition_registry(ROOT / "configs" / "data_acquisition.json")

    def _write_package(self, root: Path, evaluation_id: str = "R01"):
        original = self.registry.contracts[evaluation_id]
        scored_ids = [f"s{index}" for index in range(max(2, len(original.fold_indices)))]
        contract = replace(original, expected_scored_rows=len(scored_ids))
        contracts = dict(self.registry.contracts)
        contracts[evaluation_id] = contract
        registry = InputContractRegistry(self.registry.files, contracts, self.registry.redistribution_status)
        partition_rows = []
        extra_ids = []
        for position, fold in enumerate(contract.fold_indices):
            fit_id = f"f{fold}"
            extra_ids.append(fit_id)
            partition_rows.append((fit_id, fold, "fit"))
            if "coefficient" in contract.required_roles:
                coefficient_id = f"c{fold}"
                extra_ids.append(coefficient_id)
                partition_rows.append((coefficient_id, fold, "coefficient"))
            if "reserved" in contract.required_roles:
                reserved_id = f"r{fold}"
                extra_ids.append(reserved_id)
                partition_rows.append((reserved_id, fold, "reserved"))
            if contract.scored_assignment == "same_rows_in_every_fold":
                assigned_scored = scored_ids
            elif contract.scored_assignment == "one_held_out_prediction":
                assigned_scored = [scored_ids[position]]
            else:
                assigned_scored = scored_ids
            partition_rows.extend((record_id, fold, "scored") for record_id in assigned_scored)
        molecule_ids = list(dict.fromkeys(scored_ids + extra_ids))
        molecules = root / "molecules.csv"
        labels = root / "labels.csv"
        partitions = root / "partitions.csv"
        derivation = root / "derivation.json"
        molecules.write_text("record_id,smiles\n" + "".join(f"{record_id},CCO\n" for record_id in molecule_ids), encoding="utf-8")
        labels.write_text("record_id,logS\n" + "".join(f"{record_id},-1.0\n" for record_id in molecule_ids), encoding="utf-8")
        partitions.write_text("record_id,fold_index,role\n" + "".join(f"{record_id},{fold},{role}\n" for record_id, fold, role in partition_rows), encoding="utf-8")
        source = self.acquisition.sources[contract.upstream_source_id]
        derivation_payload = {
            "schema_version": 1,
            "source_objects": [{
                "source_id": contract.upstream_source_id,
                "locator": "user-supplied/source.csv",
                "revision": source.revision,
                "retrieval_date": "2026-09-03",
                "sha256": "a" * 64,
                "bytes": 100,
            }],
            "transformations": [{"step": 1, "operation": "normalize source columns", "software": "documented local script v1"}],
            "record_id_policy": "stable synthetic identifier",
            "canonicalization_policy": "fixture-canonicalization-v1",
            "split_assignment_origin": "fixture split",
        }
        derivation.write_text(json.dumps(derivation_payload), encoding="utf-8")
        files = {}
        for role, path in (("molecules", molecules), ("labels", labels), ("partitions", partitions)):
            columns = self.registry.files[role]["columns"]
            rows = _rows(path)
            files[role] = {"path": path.name, "sha256": _hash(path), "bytes": path.stat().st_size, "rows": rows, "columns": columns}
        files["derivation"] = {"path": derivation.name, "sha256": _hash(derivation), "bytes": derivation.stat().st_size}
        manifest = {
            "schema_version": 1,
            "evaluation_id": evaluation_id,
            "dataset_id": contract.dataset_id,
            "upstream_source_id": contract.upstream_source_id,
            "upstream_revision": source.revision,
            "retrieval_date": "2026-09-03",
            "acquisition_mode": source.acquisition_mode,
            "redistribution_status": "not_bundled_local_use_only",
            "canonicalization_policy": "fixture-canonicalization-v1",
            "split_assignment_origin": "fixture split",
            "files": files,
        }
        manifest_path = root / "package_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return registry, manifest, derivation_payload

    def test_all_primary_evaluation_layouts_are_supported(self):
        for evaluation_id in ("R01", "R02", "R03", "R04", "R05", "R09"):
            with self.subTest(evaluation_id=evaluation_id), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                registry, _, _ = self._write_package(root, evaluation_id)
                package = load_benchmark_input_package(root, evaluation_id, registry, self.acquisition)
                self.assertEqual(package.evaluation_id, evaluation_id)
                self.assertGreater(package.molecule_rows, 0)

    def test_hash_schema_and_undeclared_files_are_rejected(self):
        for mutation in ("hash", "columns", "extra"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                registry, manifest, _ = self._write_package(root)
                if mutation == "hash":
                    manifest["files"]["molecules"]["sha256"] = "0" * 64
                elif mutation == "columns":
                    manifest["files"]["molecules"]["columns"] = ["record_id", "structure"]
                else:
                    (root / "extra.csv").write_text("x\n", encoding="utf-8")
                (root / "package_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_benchmark_input_package(root, "R01", registry, self.acquisition)

    def test_duplicate_nonfinite_and_missing_identifiers_are_rejected(self):
        for mutation in ("duplicate", "nonfinite", "missing"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                registry, manifest, _ = self._write_package(root)
                if mutation == "duplicate":
                    path = root / "molecules.csv"
                    path.write_text(path.read_text(encoding="utf-8") + "s0,CCC\n", encoding="utf-8")
                elif mutation == "nonfinite":
                    path = root / "labels.csv"
                    path.write_text(path.read_text(encoding="utf-8").replace("s0,-1.0", "s0,nan"), encoding="utf-8")
                else:
                    path = root / "partitions.csv"
                    path.write_text(path.read_text(encoding="utf-8").replace("s0,0,scored", "unknown,0,scored"), encoding="utf-8")
                role = path.stem
                declaration = manifest["files"][role]
                declaration["sha256"] = _hash(path)
                declaration["bytes"] = path.stat().st_size
                declaration["rows"] = _rows(path)
                (root / "package_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_benchmark_input_package(root, "R01", registry, self.acquisition)

    def test_scored_fold_pattern_and_required_roles_are_rejected_on_drift(self):
        for mutation in ("scored_repeat", "missing_role"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                registry, manifest, _ = self._write_package(root)
                path = root / "partitions.csv"
                text = path.read_text(encoding="utf-8")
                if mutation == "scored_repeat":
                    text += "s0,1,scored\n"
                else:
                    text = text.replace("r0,0,reserved\n", "")
                path.write_text(text, encoding="utf-8")
                declaration = manifest["files"]["partitions"]
                declaration["sha256"] = _hash(path)
                declaration["bytes"] = path.stat().st_size
                declaration["rows"] = _rows(path)
                (root / "package_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_benchmark_input_package(root, "R01", registry, self.acquisition)

    def test_derivation_must_bind_source_hash_revision_and_ordered_steps(self):
        for mutation in ("hash", "revision", "step", "path", "source"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                registry, manifest, derivation = self._write_package(root)
                if mutation == "hash":
                    derivation["source_objects"][0]["sha256"] = "unavailable"
                elif mutation == "revision":
                    derivation["source_objects"][0]["revision"] = "unpinned"
                elif mutation == "step":
                    derivation["transformations"][0]["step"] = 2
                elif mutation == "path":
                    derivation["source_objects"][0]["locator"] = "/private/source.csv"
                else:
                    derivation["source_objects"][0]["source_id"] = "aqsoldb"
                    derivation["source_objects"][0]["revision"] = self.acquisition.sources["aqsoldb"].revision
                path = root / "derivation.json"
                path.write_text(json.dumps(derivation), encoding="utf-8")
                manifest["files"]["derivation"].update({"sha256": _hash(path), "bytes": path.stat().st_size})
                (root / "package_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_benchmark_input_package(root, "R01", registry, self.acquisition)

    def test_traversal_and_symlink_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, manifest, _ = self._write_package(root)
            manifest["files"]["molecules"]["path"] = "../molecules.csv"
            (root / "package_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_benchmark_input_package(root, "R01", registry, self.acquisition)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, manifest, _ = self._write_package(root)
            source = root / "molecules.csv"
            target = root / "real.csv"
            source.rename(target)
            source.symlink_to(target.name)
            manifest["files"]["molecules"].update({"sha256": _hash(target), "bytes": target.stat().st_size})
            (root / "package_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_benchmark_input_package(root, "R01", registry, self.acquisition)


if __name__ == "__main__":
    unittest.main()
