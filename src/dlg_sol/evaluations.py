from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


DEFAULT_REGISTRY = Path(__file__).resolve().parents[2] / "configs" / "evaluations.json"
VALID_DESIGNS = {"out_of_fold", "supplied_test"}


@dataclass(frozen=True)
class EvaluationSpec:
    evaluation_id: str
    dataset_id: str
    display_name: str
    design: str
    expected_rows: int
    outer_folds: int | None
    prediction_aggregation: str
    coefficient_policy: str
    selection_label_access: str
    sensitivity_status: str | None = None


def _validate(specs: tuple[EvaluationSpec, ...]) -> None:
    identifiers = [spec.evaluation_id for spec in specs]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("evaluation identifiers must be unique")
    if any(spec.design not in VALID_DESIGNS for spec in specs):
        raise ValueError("unsupported evaluation design")
    if any(spec.expected_rows < 1 for spec in specs):
        raise ValueError("expected row counts must be positive")
    for spec in specs:
        if spec.design == "out_of_fold" and (spec.outer_folds is None or spec.outer_folds < 2):
            raise ValueError(f"{spec.evaluation_id}: out-of-fold evaluation requires outer folds")
        if spec.design == "supplied_test" and spec.outer_folds is not None:
            raise ValueError(f"{spec.evaluation_id}: supplied-test evaluation cannot declare outer folds")


def load_evaluation_registry(path: str | Path = DEFAULT_REGISTRY) -> tuple[EvaluationSpec, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2:
        raise ValueError("unsupported evaluation-registry schema")
    specs = tuple(EvaluationSpec(**record) for record in payload["evaluations"])
    _validate(specs)
    return specs


def registry_by_id(path: str | Path = DEFAULT_REGISTRY) -> dict[str, EvaluationSpec]:
    return {spec.evaluation_id: spec for spec in load_evaluation_registry(path)}
