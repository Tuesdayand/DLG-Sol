from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from ..fusion.adapters import load_adapter_registry
from .assembly import assemble_evaluation_predictions
from .package import load_acquisition_registry, load_evaluation_package
from .scoring import score_evaluation_predictions


ROOT = Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _output_directory(path: str | Path) -> Path:
    output = Path(path)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output directory must not exist or must be empty")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _assemble(arguments) -> None:
    adapters = load_adapter_registry(arguments.adapters)
    acquisition = load_acquisition_registry(arguments.acquisition)
    if arguments.evaluation_id not in adapters.adapters:
        raise ValueError("evaluation identifier is not declared")
    adapter = adapters.adapters[arguments.evaluation_id]
    package = load_evaluation_package(arguments.package, adapter, acquisition, "assemble")
    scored = pd.read_csv(package.files["scored_component_predictions"].path)
    coefficient_predictions = None
    coefficient_labels = None
    if "coefficient_component_predictions" in package.files:
        coefficient_predictions = pd.read_csv(package.files["coefficient_component_predictions"].path)
        coefficient_labels = pd.read_csv(package.files["coefficient_labels"].path)
    assembly = assemble_evaluation_predictions(adapter, scored, coefficient_predictions, coefficient_labels)
    output = _output_directory(arguments.output)
    coefficients_path = output / "coefficients.csv"
    predictions_path = output / "row_predictions.csv"
    assembly.coefficients.to_csv(coefficients_path, index=False)
    assembly.predictions.to_csv(predictions_path, index=False)
    labels = package.files["scored_labels"]
    manifest = {
        "schema_version": 1,
        "evaluation_id": package.evaluation_id,
        "dataset_id": package.dataset_id,
        "model_identity": package.model_identity,
        "coefficient_policy": adapter.coefficient_policy,
        "aggregation_policy": "fit_specific_blend_then_record_aggregation",
        "package_manifest_sha256": _sha256(package.root / "package_manifest.json"),
        "input_files": {role: {"sha256": item.sha256, "bytes": item.bytes, "rows": item.rows, "columns": list(item.columns)} for role, item in sorted(package.files.items())},
        "scored_labels_content_accessed": False,
        "scored_labels_declared": {"sha256": labels.sha256, "bytes": labels.bytes, "rows": labels.rows, "columns": list(labels.columns)},
        "outputs": {
            "coefficients.csv": {"sha256": _sha256(coefficients_path), "bytes": coefficients_path.stat().st_size, "rows": len(assembly.coefficients)},
            "row_predictions.csv": {"sha256": _sha256(predictions_path), "bytes": predictions_path.stat().st_size, "rows": len(assembly.predictions)},
        },
    }
    _write_json(output / "assembly_manifest.json", manifest)


def _score(arguments) -> None:
    assembly_root = Path(arguments.assembly)
    manifest_path = assembly_root / "assembly_manifest.json"
    predictions_path = assembly_root / "row_predictions.csv"
    if manifest_path.is_symlink() or predictions_path.is_symlink() or not manifest_path.is_file() or not predictions_path.is_file():
        raise ValueError("assembly directory is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("model_identity") != "aug2_head4" or manifest.get("scored_labels_content_accessed") is not False:
        raise ValueError("assembly manifest differs from the scoring contract")
    declared_prediction = manifest.get("outputs", {}).get("row_predictions.csv", {})
    if _sha256(predictions_path) != declared_prediction.get("sha256") or predictions_path.stat().st_size != declared_prediction.get("bytes"):
        raise ValueError("assembled prediction file differs from its manifest")
    labels_path = Path(arguments.labels)
    if labels_path.is_symlink() or not labels_path.is_file():
        raise ValueError("scored-label file is missing or symbolic")
    declared_labels = manifest.get("scored_labels_declared", {})
    if _sha256(labels_path) != declared_labels.get("sha256") or labels_path.stat().st_size != declared_labels.get("bytes"):
        raise ValueError("scored-label file differs from the package declaration")
    predictions = pd.read_csv(predictions_path)
    labels = pd.read_csv(labels_path)
    score = score_evaluation_predictions(predictions, labels, seed=arguments.seed, replicates=arguments.replicates)
    output = _output_directory(arguments.output)
    metrics_path = output / "metrics.json"
    paired_path = output / "paired_rmse.json"
    _write_json(metrics_path, score.metrics)
    _write_json(paired_path, score.paired_rmse)
    scoring_manifest = {
        "schema_version": 1,
        "evaluation_id": manifest["evaluation_id"],
        "model_identity": "aug2_head4",
        "assembly_manifest_sha256": _sha256(manifest_path),
        "predictions_sha256": _sha256(predictions_path),
        "scored_labels_sha256": _sha256(labels_path),
        "scored_labels_content_accessed": True,
        "labels_written_to_prediction_output": False,
        "bootstrap_replicates": int(arguments.replicates),
        "bootstrap_seed_descriptor": int(arguments.seed),
        "bootstrap_seed_neural": int(arguments.seed) + 1,
        "outputs": {
            "metrics.json": {"sha256": _sha256(metrics_path), "bytes": metrics_path.stat().st_size},
            "paired_rmse.json": {"sha256": _sha256(paired_path), "bytes": paired_path.stat().st_size},
        },
    }
    _write_json(output / "scoring_manifest.json", scoring_manifest)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m dlg_sol.evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    assemble = subparsers.add_parser("assemble")
    assemble.add_argument("--evaluation-id", required=True)
    assemble.add_argument("--package", required=True)
    assemble.add_argument("--output", required=True)
    assemble.add_argument("--adapters", default=str(ROOT / "configs" / "dataset_adapters.json"))
    assemble.add_argument("--acquisition", default=str(ROOT / "configs" / "data_acquisition.json"))
    assemble.set_defaults(function=_assemble)
    score = subparsers.add_parser("score")
    score.add_argument("--assembly", required=True)
    score.add_argument("--labels", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--replicates", type=int, default=10000)
    score.add_argument("--seed", type=int, default=20260821)
    score.set_defaults(function=_score)
    arguments = parser.parse_args()
    arguments.function(arguments)


if __name__ == "__main__":
    main()
