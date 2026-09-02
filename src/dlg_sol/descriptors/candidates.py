from __future__ import annotations

import json
from pathlib import Path

import numpy as np


AQ_BASE_SEED = 260701
TDC_BASE_SEED = 260722
COMPLAT_BASE_SEED = 260722
JCHEM_BASE_SEED = 260725
AQ_CANDIDATE_GRID = Path(__file__).resolve().parents[3] / "configs" / "aqsoldbc_candidate_grid.json"


def _aqsoldbc_candidates() -> list[dict[str, int | float]]:
    payload = json.loads(AQ_CANDIDATE_GRID.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("protocol") != "aqsoldbc":
        raise ValueError("invalid AqSolDBc candidate grid")
    candidates = payload.get("candidates", [])
    if len(candidates) != 40 or [row.get("trial") for row in candidates] != list(range(40)):
        raise ValueError("AqSolDBc candidate grid is incomplete")
    return [dict(row) for row in candidates]


def _tdc_candidates(fold: int) -> list[dict[str, int | float]]:
    rng = np.random.default_rng(TDC_BASE_SEED + fold)
    candidates = []
    for trial in range(24):
        candidates.append(
            {
                "trial": trial,
                "n_estimators": 3500,
                "learning_rate": float(10 ** rng.uniform(-2.2, -0.8)),
                "max_depth": int(rng.integers(3, 9)),
                "min_child_weight": float(10 ** rng.uniform(-0.2, 1.25)),
                "subsample": float(rng.uniform(0.62, 1.0)),
                "colsample_bytree": float(rng.uniform(0.45, 1.0)),
                "reg_alpha": float(10 ** rng.uniform(-5, 0.0)),
                "reg_lambda": float(10 ** rng.uniform(-1.0, 1.35)),
            }
        )
    return candidates


def _complat_candidates() -> list[dict[str, int | float]]:
    rng = np.random.default_rng(COMPLAT_BASE_SEED)
    candidates = []
    for trial in range(36):
        candidates.append(
            {
                "trial": trial,
                "n_estimators": int(rng.choice([700, 1000, 1400, 2000, 2800, 3800])),
                "learning_rate": float(10 ** rng.uniform(-2.1, -1.05)),
                "max_depth": int(rng.choice([3, 4, 5, 6, 7, 8])),
                "min_child_weight": float(10 ** rng.uniform(-0.25, 1.2)),
                "subsample": float(rng.uniform(0.6, 1.0)),
                "colsample_bytree": float(rng.uniform(0.45, 1.0)),
                "reg_alpha": float(10 ** rng.uniform(-6, 0.3)),
                "reg_lambda": float(10 ** rng.uniform(-0.7, 1.4)),
            }
        )
    return candidates


def _jchem_candidates(fold: int) -> list[dict[str, int | float]]:
    rng = np.random.default_rng(JCHEM_BASE_SEED + fold)
    candidates = [
        {
            "trial": 0,
            "n_estimators": 1200,
            "learning_rate": 0.020761167753872403,
            "max_depth": 6,
            "min_child_weight": 1.1844087221152417,
            "subsample": 0.9726784864053988,
            "colsample_bytree": 0.7101588968244845,
            "reg_alpha": 6.762219934792396e-05,
            "reg_lambda": 0.7854488818845954,
        }
    ]
    for trial in range(1, 16):
        candidates.append(
            {
                "trial": trial,
                "n_estimators": int(rng.integers(600, 1201)),
                "learning_rate": float(10 ** rng.uniform(-2.05, -0.75)),
                "max_depth": int(rng.integers(4, 9)),
                "min_child_weight": float(10 ** rng.uniform(-0.25, 1.15)),
                "subsample": float(rng.uniform(0.65, 1.0)),
                "colsample_bytree": float(rng.uniform(0.45, 1.0)),
                "reg_alpha": float(10 ** rng.uniform(-5, -0.15)),
                "reg_lambda": float(10 ** rng.uniform(-2, 1.2)),
            }
        )
    return candidates


def generate_candidates(protocol: str, fold: int | None = None) -> list[dict[str, int | float]]:
    if protocol == "aqsoldbc":
        return _aqsoldbc_candidates()
    if protocol == "complat":
        return _complat_candidates()
    if protocol == "tdc":
        if fold is None:
            raise ValueError("TDC candidate generation requires a fold")
        return _tdc_candidates(fold)
    if protocol == "jchem":
        if fold is None or fold not in range(1, 6):
            raise ValueError("JCheM candidate generation requires a fold from one through five")
        return _jchem_candidates(fold)
    raise ValueError("unsupported descriptor protocol")
