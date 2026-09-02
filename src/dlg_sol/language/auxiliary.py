from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors


AUXILIARY_TARGETS = ("logp", "tpsa", "molwt", "hbd", "hba")


@dataclass(frozen=True)
class AuxiliaryScaler:
    mean: tuple[float, ...]
    standard_deviation: tuple[float, ...]
    columns: tuple[str, ...] = AUXILIARY_TARGETS


def compute_auxiliary_targets(smiles: Sequence[str]) -> pd.DataFrame:
    rows = []
    for value in smiles:
        molecule = Chem.MolFromSmiles(str(value))
        if molecule is None:
            rows.append({column: np.nan for column in AUXILIARY_TARGETS})
            continue
        rows.append(
            {
                "logp": float(Crippen.MolLogP(molecule)),
                "tpsa": float(rdMolDescriptors.CalcTPSA(molecule)),
                "molwt": float(Descriptors.MolWt(molecule)),
                "hbd": float(Lipinski.NumHDonors(molecule)),
                "hba": float(Lipinski.NumHAcceptors(molecule)),
            }
        )
    return pd.DataFrame(rows, columns=AUXILIARY_TARGETS)


def fit_auxiliary_scaler(targets: pd.DataFrame) -> AuxiliaryScaler:
    if tuple(targets.columns) != AUXILIARY_TARGETS:
        raise ValueError("auxiliary target columns do not match the released contract")
    values = targets.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("fit-partition auxiliary targets must be finite")
    means = targets.mean(axis=0)
    standard_deviations = targets.std(axis=0, ddof=1).replace(0, 1.0)
    if not np.isfinite(means.to_numpy(dtype=float)).all() or not np.isfinite(standard_deviations.to_numpy(dtype=float)).all():
        raise ValueError("auxiliary scaling statistics must be finite")
    return AuxiliaryScaler(tuple(float(value) for value in means), tuple(float(value) for value in standard_deviations))


def transform_auxiliary_targets(targets: pd.DataFrame, scaler: AuxiliaryScaler) -> np.ndarray:
    if tuple(targets.columns) != scaler.columns:
        raise ValueError("auxiliary target columns do not match the fitted scaler")
    values = targets.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("auxiliary targets must be finite")
    transformed = (values - np.asarray(scaler.mean)) / np.asarray(scaler.standard_deviation)
    return transformed.astype(np.float32)
