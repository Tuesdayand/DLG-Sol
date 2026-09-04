from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import inchi
from sklearn.model_selection import StratifiedShuffleSplit

from ..data import BenchmarkInputPackage, reconstruct_tdc_native_partitions


STAGES = (
    "parameter_fit",
    "selection_fit",
    "selection_score",
    "final_refit",
    "coefficient_fit",
    "scored_prediction",
)
TRAINING_STAGES = STAGES[:-1]
BRANCHES = {"descriptor", "language", "geometry", "fusion_head", "blend_coefficient"}
FORBIDDEN_SCORED_COLUMNS = {"logs", "label", "observed", "target", "y"}
COMPLAT_SOURCE_COLUMNS = ("", "C_ID", "Name", "InChIKey", "SMILES", "CAS", "LogS", "smiles_canon")
COMPLAT_ROLE_RDKIT_VERSION = "2023.09.6"


@dataclass(frozen=True)
class TrainingRoleContractRegistry:
    path: Path
    model_identity: str
    status: str
    evaluations: dict[str, dict]
    complat_global_selection_recipe: dict


@dataclass(frozen=True)
class MaterializedNestedSelectionUnit:
    inner_fold: int
    stage_rows: dict[str, pd.DataFrame]


@dataclass(frozen=True)
class MaterializedTrainingUnit:
    evaluation_id: str
    branch_id: str
    fold_index: int
    role_resolution: str
    scored_overlap_policy: str
    stage_rows: dict[str, pd.DataFrame]
    nested_selection_units: tuple[MaterializedNestedSelectionUnit, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_ids(values: set[str]) -> str:
    return hashlib.sha256("".join(f"{value}\n" for value in sorted(values)).encode()).hexdigest()


def _hash_assignments(stages: Mapping[str, set[str]]) -> str:
    rows = [f"{stage}\t{record_id}" for stage in sorted(stages) for record_id in sorted(stages[stage])]
    return hashlib.sha256("".join(f"{row}\n" for row in rows).encode()).hexdigest()


def _valid_stage(stage: object) -> bool:
    if not isinstance(stage, dict) or set(stage) != {"rows", "record_id_sha256"}:
        return False
    digest = str(stage["record_id_sha256"])
    return isinstance(stage["rows"], int) and stage["rows"] >= 0 and len(digest) == 64 and all(value in "0123456789abcdef" for value in digest)


def load_training_role_contracts(path: str | Path) -> TrainingRoleContractRegistry:
    contract_path = Path(path).resolve()
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "model_identity",
        "status",
        "stage_definitions",
        "overlap_policy",
        "role_resolution_inputs",
        "complat_global_selection_recipe",
        "evaluations",
    }
    if not isinstance(payload, dict) or set(payload) != required or payload.get("schema_version") != 1:
        raise ValueError("training-role contract schema mismatch")
    if payload.get("model_identity") != "aug2_head4" or payload.get("status") not in {
        "AUDITED_PENDING_MATERIALIZER",
        "AUDITED_MATERIALIZER_AVAILABLE",
    }:
        raise ValueError("training-role contract identity or status mismatch")
    if set(payload.get("stage_definitions", {})) != set(STAGES):
        raise ValueError("training-role stage definitions are incomplete")
    evaluations = payload.get("evaluations")
    if not isinstance(evaluations, dict) or set(evaluations) != {"R01", "R02", "R03", "R04", "R05", "R09"}:
        raise ValueError("training-role evaluation set mismatch")
    for evaluation_id, evaluation in evaluations.items():
        branches = evaluation.get("branches")
        if not isinstance(branches, dict) or set(branches) != BRANCHES:
            raise ValueError(f"training-role branch set mismatch: {evaluation_id}")
        for branch_id, branch in branches.items():
            if branch.get("scored_label_permission") != "scoring_only" or not str(branch.get("role_resolution", "")):
                raise ValueError(f"training-role branch declaration mismatch: {evaluation_id}/{branch_id}")
            units = branch.get("expected_units")
            if not isinstance(units, list) or not units:
                raise ValueError(f"training-role units are missing: {evaluation_id}/{branch_id}")
            folds = []
            for unit in units:
                required_unit = {"fold_index", "stages", "stage_assignment_sha256", "scored_overlap_policy", "scored_label_overlap_rows"}
                unit_keys = frozenset(unit) if isinstance(unit, dict) else frozenset()
                if not isinstance(unit, dict) or unit_keys not in {frozenset(required_unit), frozenset(required_unit | {"nested_selection_units"})} or not isinstance(unit.get("fold_index"), int):
                    raise ValueError("training-role unit is malformed")
                folds.append(unit["fold_index"])
                stages = unit.get("stages")
                if not isinstance(stages, dict) or set(stages) - set(STAGES) or not all(_valid_stage(value) for value in stages.values()):
                    raise ValueError("training-role stage declaration is malformed")
                if unit.get("scored_overlap_policy") not in {
                    "zero_for_all_training_stages",
                    "documented_non_nested_global_preselection",
                }:
                    raise ValueError("training-role overlap policy is unsupported")
                overlaps = unit.get("scored_label_overlap_rows")
                if not isinstance(overlaps, dict) or set(overlaps) != set(TRAINING_STAGES) or not all(isinstance(value, int) and value >= 0 for value in overlaps.values()):
                    raise ValueError("training-role overlap declaration is malformed")
            if len(folds) != len(set(folds)):
                raise ValueError("training-role fold identifiers must be unique within a branch")
    return TrainingRoleContractRegistry(
        contract_path,
        str(payload["model_identity"]),
        str(payload["status"]),
        evaluations,
        dict(payload["complat_global_selection_recipe"]),
    )


def _read_package(package: BenchmarkInputPackage) -> tuple[pd.DataFrame, pd.DataFrame]:
    molecules = pd.read_csv(package.files["molecules"].path, dtype={"record_id": str, "smiles": str})
    labels = pd.read_csv(package.files["labels"].path, dtype={"record_id": str})
    partitions = pd.read_csv(package.files["partitions"].path, dtype={"record_id": str})
    if molecules["record_id"].duplicated().any() or labels["record_id"].duplicated().any():
        raise ValueError("materialization requires unique molecule and label identifiers")
    base = molecules.merge(labels, on="record_id", how="outer", validate="one_to_one", indicator=True)
    if not base["_merge"].eq("both").all():
        raise ValueError("materialization package molecule and label rows differ")
    base = base.drop(columns="_merge")
    base["logS"] = pd.to_numeric(base["logS"], errors="raise")
    if not np.isfinite(base["logS"].to_numpy(float)).all():
        raise ValueError("materialization labels must be finite")
    partitions["fold_index"] = pd.to_numeric(partitions["fold_index"], errors="raise").astype(int)
    return base, partitions


def _partition_ids(partitions: pd.DataFrame, fold_index: int) -> dict[str, set[str]]:
    rows = partitions.loc[partitions["fold_index"].eq(fold_index)]
    if rows.empty or rows["record_id"].duplicated().any():
        raise ValueError("materialization fold is absent or has duplicate role assignments")
    return {role: set(group["record_id"].astype(str)) for role, group in rows.groupby("role", sort=False)}


def _tdc_native_ids(base: pd.DataFrame, registry: TrainingRoleContractRegistry, fold_index: int) -> dict[str, set[str]]:
    recipe_path = registry.path.parent / "tdc_partition_recipe.json"
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    partition = recipe["partition"]
    train = base.loc[base["record_id"].str.startswith("tdc_train_")].copy()
    test = base.loc[base["record_id"].str.startswith("tdc_test_")].copy()
    train["source_index"] = train["record_id"].str.rsplit("_", n=1).str[-1].astype(int)
    train = train.sort_values("source_index")
    if train["source_index"].tolist() != list(range(len(train))):
        raise ValueError("TDC training identifiers do not retain source-row order")
    assignments = reconstruct_tdc_native_partitions(
        train["logS"].astype(float).tolist(),
        len(test),
        int(partition["seed"]),
        int(partition["target_quantiles"]),
        int(partition["outer_folds"]),
        float(partition["selection_fraction_of_outer_training"]),
    )
    return {
        role: {record_id for record_id, fold, native_role in assignments if fold == fold_index and native_role == role}
        for role in ("fit", "selection", "oof_holdout", "official_test")
    }


@lru_cache(maxsize=4)
def _cached_complat_global_ids(
    source_text: str,
    expected_bytes: int,
    expected_sha256: str,
    expected_rows: int,
    validation_fraction: float,
    random_state: int,
) -> tuple[frozenset[str], frozenset[str], tuple[tuple[str, float], ...]]:
    source = Path(source_text)
    if not source.is_file() or source.stat().st_size != expected_bytes or _sha256(source) != expected_sha256:
        raise ValueError("ComPlat training-role source does not match the pinned upstream object")
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != COMPLAT_SOURCE_COLUMNS:
            raise ValueError("ComPlat training-role source schema mismatch")
        rows = list(reader)
    if len(rows) != expected_rows:
        raise ValueError("ComPlat training-role source row count mismatch")
    RDLogger.DisableLog("rdApp.*")
    records = []
    for row in rows:
        molecule = Chem.MolFromSmiles(row["SMILES"])
        if molecule is None:
            raise ValueError("ComPlat training-role source contains an unparsable historical structure")
        records.append((row["C_ID"].strip(), inchi.MolToInchiKey(molecule), float(row["LogS"])))
    frame = pd.DataFrame(records, columns=["record_id", "group_key", "logS"])
    grouped = frame.groupby("group_key", dropna=False).agg(group_logS=("logS", "mean")).reset_index()
    labels = None
    for bins in range(20, 3, -1):
        candidate = pd.qcut(grouped["group_logS"], q=bins, labels=False, duplicates="drop")
        counts = pd.Series(candidate).value_counts(dropna=False)
        if counts.min() >= 5 and counts.size >= 4:
            labels = candidate.astype(int).to_numpy()
            break
    if labels is None:
        raise ValueError("ComPlat global selection strata could not be reconstructed")
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=validation_fraction,
        random_state=random_state,
    )
    fit_index, score_index = next(splitter.split(np.zeros(len(grouped)), labels))
    fit_keys = set(grouped.iloc[fit_index]["group_key"].astype(str))
    score_keys = set(grouped.iloc[score_index]["group_key"].astype(str))
    keys = frame["group_key"].astype(str)
    fit_ids = set(frame.loc[keys.isin(fit_keys), "record_id"])
    score_ids = set(frame.loc[keys.isin(score_keys), "record_id"])
    source_labels = tuple((str(row.record_id), float(row.logS)) for row in frame.itertuples(index=False))
    return frozenset(fit_ids), frozenset(score_ids), source_labels


def _complat_global_ids(
    package_base: pd.DataFrame,
    source_path: str | Path | None,
    registry: TrainingRoleContractRegistry,
) -> tuple[set[str], set[str]]:
    if source_path is None:
        raise ValueError("R02 and R09 representation materialization requires the pinned ComPlat train source")
    if rdBase.rdkitVersion != COMPLAT_ROLE_RDKIT_VERSION:
        raise ValueError(f"ComPlat training-role reconstruction requires RDKit {COMPLAT_ROLE_RDKIT_VERSION}")
    recipe = json.loads((registry.path.parent / "complat_partition_recipe.json").read_text(encoding="utf-8"))
    source_contract = recipe["source"]["train"]
    requested = registry.complat_global_selection_recipe
    fit_ids, score_ids, source_labels = _cached_complat_global_ids(
        str(Path(source_path).resolve()),
        int(source_contract["bytes"]),
        str(source_contract["sha256"]),
        int(source_contract["rows"]),
        float(requested["validation_fraction"]),
        int(requested["random_state"]),
    )
    source_frame = pd.DataFrame(source_labels, columns=["record_id", "logS"])
    package_labels = package_base[["record_id", "logS"]].merge(source_frame, on="record_id", how="inner", suffixes=("_package", "_source"), validate="one_to_one")
    if len(package_labels) != len(source_frame) or not np.allclose(package_labels["logS_package"], package_labels["logS_source"], rtol=0.0, atol=1e-12):
        raise ValueError("ComPlat training-role source does not match the package population or labels")
    return set(fit_ids), set(score_ids)


def _resolved_stage_ids(
    package: BenchmarkInputPackage,
    base: pd.DataFrame,
    partitions: pd.DataFrame,
    registry: TrainingRoleContractRegistry,
    evaluation_id: str,
    branch_id: str,
    fold_index: int,
    dependency_packages: Mapping[str, BenchmarkInputPackage],
    complat_train_source: str | Path | None,
) -> tuple[dict[str, set[str]], list[tuple[int, dict[str, set[str]]]]]:
    roles = _partition_ids(partitions, fold_index)
    stages: dict[str, set[str]] = {}
    nested: list[tuple[int, dict[str, set[str]]]] = []
    if evaluation_id == "R01":
        development = roles.get("fit", set()) | roles.get("reserved", set())
        if branch_id == "descriptor":
            stages.update(selection_fit=development, selection_score=development, final_refit=development)
            outer_scored = roles.get("scored", set())
            for inner_fold in sorted(set(partitions["fold_index"])):
                if inner_fold == fold_index:
                    continue
                inner_roles = _partition_ids(partitions, int(inner_fold))
                inner_score = inner_roles.get("scored", set())
                nested.append((int(inner_fold), {"selection_fit": development - inner_score, "selection_score": inner_score}))
            stages["scored_prediction"] = outer_scored
        elif branch_id in {"language", "geometry"}:
            stages.update(parameter_fit=roles.get("fit", set()), selection_score=roles.get("reserved", set()), scored_prediction=roles.get("scored", set()))
        elif branch_id == "fusion_head":
            stages.update(parameter_fit=roles.get("fit", set()), scored_prediction=roles.get("scored", set()))
        else:
            stages["scored_prediction"] = roles.get("scored", set())
    elif evaluation_id == "R02":
        if branch_id in {"descriptor", "language", "geometry"}:
            selection_fit, selection_score = _complat_global_ids(base, complat_train_source, registry)
            stages.update(selection_fit=selection_fit, selection_score=selection_score, final_refit=roles.get("fit", set()), scored_prediction=roles.get("scored", set()))
        elif branch_id == "fusion_head":
            stages.update(parameter_fit=roles.get("fit", set()), scored_prediction=roles.get("scored", set()))
        else:
            stages.update(coefficient_fit=roles.get("fit", set()), scored_prediction=roles.get("scored", set()))
    elif evaluation_id in {"R03", "R04"}:
        native = _tdc_native_ids(base, registry, fold_index)
        scored_role = "oof_holdout" if evaluation_id == "R03" else "official_test"
        if branch_id in {"descriptor", "language", "geometry"}:
            stages.update(parameter_fit=native["fit"], selection_score=native["selection"], scored_prediction=native[scored_role])
        elif branch_id == "fusion_head":
            stages.update(parameter_fit=native["fit"], scored_prediction=native[scored_role])
        elif evaluation_id == "R03":
            stages.update(coefficient_fit=native["fit"] | native["selection"], scored_prediction=native["oof_holdout"])
        else:
            if fold_index != 0:
                raise ValueError("R04 has one evaluation-wide coefficient unit at fold 0")
            dependency = dependency_packages.get("R03")
            if dependency is None:
                raise ValueError("R04 coefficient materialization requires the R03 input package")
            dependency_base, dependency_partitions = _read_package(dependency)
            aligned = base[["record_id", "smiles", "logS"]].merge(
                dependency_base[["record_id", "smiles", "logS"]],
                on="record_id",
                how="outer",
                suffixes=("_r04", "_r03"),
                validate="one_to_one",
                indicator=True,
            )
            if not aligned["_merge"].eq("both").all() or not aligned["smiles_r04"].eq(aligned["smiles_r03"]).all() or not np.allclose(aligned["logS_r04"], aligned["logS_r03"], rtol=0.0, atol=0.0):
                raise ValueError("R04 coefficient dependency does not match the R03 population")
            all_oof = set(dependency_partitions.loc[dependency_partitions["role"].eq("scored"), "record_id"].astype(str))
            stages.update(coefficient_fit=all_oof, scored_prediction=native["official_test"])
    elif evaluation_id == "R05":
        if branch_id in {"descriptor", "language", "geometry"}:
            stages.update(parameter_fit=roles.get("fit", set()), selection_score=roles.get("coefficient", set()), scored_prediction=roles.get("scored", set()))
        elif branch_id == "fusion_head":
            stages.update(parameter_fit=roles.get("fit", set()), scored_prediction=roles.get("scored", set()))
        else:
            stages.update(coefficient_fit=roles.get("coefficient", set()), scored_prediction=roles.get("scored", set()))
    elif evaluation_id == "R09":
        if branch_id in {"descriptor", "language", "geometry"}:
            selection_fit, selection_score = _complat_global_ids(base.loc[base["record_id"].isin(roles.get("fit", set()))], complat_train_source, registry)
            stages.update(selection_fit=selection_fit, selection_score=selection_score, final_refit=roles.get("fit", set()), scored_prediction=roles.get("scored", set()))
        elif branch_id == "fusion_head":
            stages.update(final_refit=roles.get("fit", set()), scored_prediction=roles.get("scored", set()))
        else:
            dependency = dependency_packages.get("R02")
            if dependency is None:
                raise ValueError("R09 coefficient materialization requires the R02 input package")
            dependency_base, _ = _read_package(dependency)
            coefficient_ids = set(dependency_base["record_id"])
            if coefficient_ids != roles.get("fit", set()):
                raise ValueError("R09 coefficient dependency does not match the R02 population")
            stages.update(coefficient_fit=coefficient_ids, scored_prediction=roles.get("scored", set()))
    else:
        raise ValueError("unsupported materialization evaluation")
    return stages, nested


def _frame(base: pd.DataFrame, record_ids: set[str], scored: bool) -> pd.DataFrame:
    selected = base.loc[base["record_id"].isin(record_ids), ["record_id", "smiles", "logS"]]
    if len(selected) != len(record_ids):
        raise ValueError("materialized stage refers to records absent from its input package")
    if scored:
        selected = selected[["record_id", "smiles"]]
        if {str(column).lower() for column in selected.columns} & FORBIDDEN_SCORED_COLUMNS:
            raise ValueError("scored materialization contains a forbidden label column")
    return selected.reset_index(drop=True)


def _verify(expected: dict, stages: dict[str, set[str]], nested: list[tuple[int, dict[str, set[str]]]]) -> None:
    expected_stages = expected["stages"]
    observed = {stage: ids for stage, ids in stages.items() if ids}
    if set(observed) != set(expected_stages):
        raise ValueError("materialized stage set differs from the historical contract")
    for stage, ids in observed.items():
        declaration = expected_stages[stage]
        if len(ids) != declaration["rows"] or _hash_ids(ids) != declaration["record_id_sha256"]:
            raise ValueError(f"materialized row identity differs from the historical contract: {stage}")
    if _hash_assignments(observed) != expected["stage_assignment_sha256"]:
        raise ValueError("materialized stage-assignment identity differs from the historical contract")
    scored = observed["scored_prediction"]
    overlaps = {stage: len(scored & observed.get(stage, set())) for stage in TRAINING_STAGES}
    if overlaps != expected["scored_label_overlap_rows"]:
        raise ValueError("materialized scored-label overlap differs from the historical contract")
    if expected["scored_overlap_policy"] == "zero_for_all_training_stages" and any(overlaps.values()):
        raise ValueError("materialized training stages overlap scored rows")
    expected_nested = expected.get("nested_selection_units", [])
    if [inner for inner, _ in nested] != [item["inner_fold"] for item in expected_nested]:
        raise ValueError("nested materialization fold set differs from the historical contract")
    for (_, observed_stages), declaration in zip(nested, expected_nested):
        for stage, ids in observed_stages.items():
            expected_stage = declaration["stages"][stage]
            if len(ids) != expected_stage["rows"] or _hash_ids(ids) != expected_stage["record_id_sha256"]:
                raise ValueError("nested materialized row identity differs from the historical contract")
        if _hash_assignments(observed_stages) != declaration["stage_assignment_sha256"]:
            raise ValueError("nested materialized stage-assignment identity differs from the historical contract")


def materialize_training_unit(
    package: BenchmarkInputPackage,
    role_contract: TrainingRoleContractRegistry,
    evaluation_id: str,
    branch_id: str,
    fold_index: int,
    *,
    dependency_packages: Mapping[str, BenchmarkInputPackage] | None = None,
    complat_train_source: str | Path | None = None,
) -> MaterializedTrainingUnit:
    if package.evaluation_id != evaluation_id or package.dataset_id == "":
        raise ValueError("materialization package and requested evaluation differ")
    if evaluation_id not in role_contract.evaluations:
        raise ValueError("materialization evaluation is absent from the role contract")
    if branch_id not in BRANCHES:
        raise ValueError("materialization branch identifier is unsupported")
    evaluation = role_contract.evaluations[evaluation_id]
    branch = evaluation["branches"][branch_id]
    matches = [unit for unit in branch["expected_units"] if unit["fold_index"] == fold_index]
    if len(matches) != 1:
        raise ValueError("materialization fold is absent or ambiguous in the role contract")
    base, partitions = _read_package(package)
    stages, nested = _resolved_stage_ids(
        package,
        base,
        partitions,
        role_contract,
        evaluation_id,
        branch_id,
        fold_index,
        dependency_packages or {},
        complat_train_source,
    )
    _verify(matches[0], stages, nested)
    frames = {stage: _frame(base, stages.get(stage, set()), stage == "scored_prediction") for stage in STAGES}
    nested_frames = tuple(
        MaterializedNestedSelectionUnit(inner_fold, {stage: _frame(base, ids, False) for stage, ids in nested_stages.items()})
        for inner_fold, nested_stages in nested
    )
    return MaterializedTrainingUnit(
        evaluation_id,
        branch_id,
        fold_index,
        str(branch["role_resolution"]),
        str(matches[0]["scored_overlap_policy"]),
        frames,
        nested_frames,
    )


def write_materialized_training_unit(unit: MaterializedTrainingUnit, output: str | Path) -> dict:
    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise ValueError("materialization output directory must be absent or empty")
    root.mkdir(parents=True, exist_ok=True)
    files = {}
    for stage in STAGES:
        frame = unit.stage_rows[stage]
        expected_columns = ["record_id", "smiles"] if stage == "scored_prediction" else ["record_id", "smiles", "logS"]
        if list(frame.columns) != expected_columns:
            raise ValueError(f"materialized output schema is unsafe or incomplete: {stage}")
        if frame["record_id"].astype(str).duplicated().any():
            raise ValueError(f"materialized output contains duplicate identifiers: {stage}")
        if stage != "scored_prediction" and not np.isfinite(pd.to_numeric(frame["logS"], errors="coerce").to_numpy(float)).all():
            raise ValueError(f"materialized output contains non-finite labels: {stage}")
        path = root / f"{stage}.csv"
        frame.to_csv(path, index=False, lineterminator="\n")
        files[path.name] = {
            "rows": len(frame),
            "columns": list(frame.columns),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    nested_files = {}
    for nested in unit.nested_selection_units:
        folder = root / f"nested_{nested.inner_fold}"
        folder.mkdir()
        nested_files[str(nested.inner_fold)] = {}
        for stage, frame in nested.stage_rows.items():
            path = folder / f"{stage}.csv"
            frame.to_csv(path, index=False, lineterminator="\n")
            nested_files[str(nested.inner_fold)][f"{folder.name}/{path.name}"] = {
                "rows": len(frame),
                "columns": list(frame.columns),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    manifest = {
        "schema_version": 1,
        "evaluation_id": unit.evaluation_id,
        "branch_id": unit.branch_id,
        "fold_index": unit.fold_index,
        "role_resolution": unit.role_resolution,
        "scored_overlap_policy": unit.scored_overlap_policy,
        "redistribution_status": "user_local_do_not_redistribute",
        "files": files,
        "nested_files": nested_files,
    }
    manifest_path = root / "materialization_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
