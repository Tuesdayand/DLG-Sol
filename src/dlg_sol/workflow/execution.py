from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .materialization import TrainingRoleContractRegistry


BRANCHES = {"descriptor", "language", "geometry", "fusion_head", "blend_coefficient"}
EXECUTION_TYPES = {
    "train_new",
    "reuse_companion_model",
    "final_refit",
    "use_prespecified_coefficient",
    "fit_cross_fitted_coefficient",
    "fit_from_companion_oof",
    "fit_within_reported_split",
}
OUTPUT_KINDS = {
    "component_prediction",
    "prediction_and_embedding",
    "four_member_and_mean_prediction",
    "fixed_blend_prediction",
}
FORBIDDEN_DUPLICATE_KEYS = {
    "rows",
    "row_count",
    "record_id_sha256",
    "stage_assignment_sha256",
}


@dataclass(frozen=True)
class TrainingExecutionRecipe:
    evaluation_id: str
    branch_id: str
    execution_type: str
    selection_mode: str
    input_stages: tuple[str, ...]
    configuration_refs: tuple[str, ...]
    dependency: dict | None
    output_kind: str


@dataclass(frozen=True)
class TrainingExecutionRegistry:
    path: Path
    model_identity: str
    status: str
    output_contract: dict
    scope: dict
    recipes: dict[tuple[str, str], TrainingExecutionRecipe]

    def get(self, evaluation_id: str, branch_id: str) -> TrainingExecutionRecipe:
        key = (str(evaluation_id), str(branch_id))
        if key not in self.recipes:
            raise ValueError("undeclared training-execution recipe")
        return self.recipes[key]


def _resolve_json_reference(root: Path, reference: str) -> object:
    file_name, separator, pointer = str(reference).partition("#")
    path = root / file_name
    if not path.is_file():
        raise ValueError(f"training-execution configuration reference is missing: {file_name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not separator or pointer in {"", "/"}:
        return value
    if not pointer.startswith("/"):
        raise ValueError(f"invalid JSON pointer in training-execution reference: {reference}")
    for raw in pointer[1:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            value = value[int(token)]
        elif isinstance(value, dict) and token in value:
            value = value[token]
        else:
            raise ValueError(f"unresolved training-execution configuration reference: {reference}")
    return value


def _contains_forbidden_duplicate_key(value: object) -> bool:
    if isinstance(value, dict):
        if set(value) & FORBIDDEN_DUPLICATE_KEYS:
            return True
        return any(_contains_forbidden_duplicate_key(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_forbidden_duplicate_key(item) for item in value)
    return False


def _validate_special_cases(recipes: dict[tuple[str, str], TrainingExecutionRecipe]) -> None:
    representations = {"descriptor", "language", "geometry"}
    for branch in representations:
        r02 = recipes[("R02", branch)]
        if r02.selection_mode != "frozen_global_preselection" or r02.execution_type != "train_new":
            raise ValueError("R02 primary representations must use frozen global preselection")
        r09 = recipes[("R09", branch)]
        expected_dependency = {"evaluation_id": "R02", "branch_id": branch, "fold_mapping": "selected_settings"}
        if r09.selection_mode != "reuse_r02_frozen_global_preselection" or r09.dependency != expected_dependency:
            raise ValueError("R09 representation settings must come from R02 global preselection")
    for branch in {"descriptor", "language", "geometry", "fusion_head"}:
        r04 = recipes[("R04", branch)]
        expected_dependency = {"evaluation_id": "R03", "branch_id": branch, "fold_mapping": "same_fold"}
        if r04.execution_type != "reuse_companion_model" or r04.dependency != expected_dependency:
            raise ValueError("R04 component recipes must reuse the same-fold R03 models")
    if recipes[("R01", "descriptor")].selection_mode != "nested_inner_cv":
        raise ValueError("R01 descriptor selection must remain nested")
    if recipes[("R01", "blend_coefficient")].selection_mode != "prespecified_alpha_0.5":
        raise ValueError("R01 blend coefficient must remain prespecified")


def load_training_execution_recipes(
    path: str | Path,
    role_contracts: TrainingRoleContractRegistry,
) -> TrainingExecutionRegistry:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("model_identity") != "aug2_head4":
        raise ValueError("unsupported training-execution recipe identity")
    if payload.get("status") != "AUDITED_EXECUTION_RECIPES":
        raise ValueError("training-execution recipe status is not auditable")
    role_reference = payload.get("training_role_contract", {})
    if role_reference != {
        "path": "configs/training_role_contracts.json",
        "reference_method": "evaluation_id_branch_id_and_all_declared_folds",
        "duplicates_row_counts_or_hashes": False,
    }:
        raise ValueError("training-execution recipes must reference rather than duplicate row contracts")
    if _contains_forbidden_duplicate_key(payload.get("evaluations", {})):
        raise ValueError("training-execution recipes duplicate row counts or hashes")
    output = payload.get("output_contract", {})
    expected_output = {
        "prediction_columns": ["record_id", "prediction"],
        "scored_labels_permitted": False,
        "run_manifest_required": True,
        "private_absolute_paths_permitted": False,
        "historical_weight_identity_claimed": False,
        "historical_metric_identity_claimed": False,
    }
    if output != expected_output:
        raise ValueError("training-execution output contract is unsafe or unsupported")
    scope = payload.get("scope", {})
    if scope != {
        "unit_level_branch_execution": True,
        "single_command_six_evaluation_orchestration": False,
        "automatic_data_acquisition": False,
        "gpu_cluster_scheduling": False,
        "fresh_training_is_protocol_reproduction": True,
    }:
        raise ValueError("training-execution scope is misstated")
    evaluations = payload.get("evaluations", {})
    if set(evaluations) != set(role_contracts.evaluations):
        raise ValueError("training-execution evaluation set differs from the role contract")
    recipes: dict[tuple[str, str], TrainingExecutionRecipe] = {}
    root = source.resolve().parents[1]
    stage_definitions = {
        "parameter_fit",
        "selection_fit",
        "selection_score",
        "final_refit",
        "coefficient_fit",
        "scored_prediction",
    }
    for evaluation_id, records in evaluations.items():
        contract_branches = role_contracts.evaluations[evaluation_id]["branches"]
        if set(records) != set(contract_branches) or set(records) != BRANCHES:
            raise ValueError(f"training-execution branch set mismatch: {evaluation_id}")
        for branch_id, record in records.items():
            reference = record.get("unit_reference")
            if reference != {"evaluation_id": evaluation_id, "branch_id": branch_id, "folds": "all_declared"}:
                raise ValueError(f"training-execution unit reference mismatch: {evaluation_id}/{branch_id}")
            execution_type = str(record.get("execution_type"))
            if execution_type not in EXECUTION_TYPES:
                raise ValueError(f"unsupported execution type: {evaluation_id}/{branch_id}")
            selection_mode = str(record.get("selection_mode", ""))
            if not selection_mode:
                raise ValueError(f"missing selection mode: {evaluation_id}/{branch_id}")
            input_stages = tuple(str(item) for item in record.get("input_stages", []))
            if not input_stages or len(set(input_stages)) != len(input_stages) or set(input_stages) - stage_definitions:
                raise ValueError(f"invalid input stages: {evaluation_id}/{branch_id}")
            declared_stage_sets = [set(unit["stages"]) for unit in contract_branches[branch_id]["expected_units"]]
            if any(set(input_stages) != stages for stages in declared_stage_sets):
                raise ValueError(f"recipe stages differ from the role contract: {evaluation_id}/{branch_id}")
            references = tuple(str(item) for item in record.get("configuration_refs", []))
            if not references:
                raise ValueError(f"missing configuration reference: {evaluation_id}/{branch_id}")
            for item in references:
                _resolve_json_reference(root, item)
            output_kind = str(record.get("output_kind"))
            if output_kind not in OUTPUT_KINDS:
                raise ValueError(f"unsupported output kind: {evaluation_id}/{branch_id}")
            dependency = record.get("dependency")
            if dependency is not None:
                dependency_key = (str(dependency.get("evaluation_id")), str(dependency.get("branch_id")))
                if dependency_key[0] not in role_contracts.evaluations:
                    raise ValueError(f"unknown dependency evaluation: {evaluation_id}/{branch_id}")
                if dependency_key[1] not in role_contracts.evaluations[dependency_key[0]]["branches"]:
                    raise ValueError(f"unknown dependency branch: {evaluation_id}/{branch_id}")
            recipes[(evaluation_id, branch_id)] = TrainingExecutionRecipe(
                evaluation_id,
                branch_id,
                execution_type,
                selection_mode,
                input_stages,
                references,
                dependency,
                output_kind,
            )
    _validate_special_cases(recipes)
    return TrainingExecutionRegistry(source, str(payload["model_identity"]), str(payload["status"]), output, scope, recipes)


def require_primary_r02_selection(recipe: TrainingExecutionRecipe, requested_mode: str) -> None:
    if recipe.evaluation_id != "R02" or recipe.branch_id not in {"descriptor", "language", "geometry"}:
        raise ValueError("R02 primary-selection guard applies only to representation branches")
    if recipe.selection_mode != "frozen_global_preselection" or str(requested_mode) != "frozen_global_preselection":
        raise ValueError("R02 primary execution rejects fold-local reselection")
