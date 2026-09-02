from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def calculate_mordred_2d(smiles: Sequence[str], record_ids: Sequence[str] | None = None) -> pd.DataFrame:
    from mordred import Calculator, descriptors
    from rdkit import Chem

    structures = [str(value) for value in smiles]
    if record_ids is not None and len(record_ids) != len(structures):
        raise ValueError("record identifiers and SMILES must have equal length")
    calculator = Calculator(descriptors, ignore_3D=True)
    names = [str(descriptor) for descriptor in calculator.descriptors]
    rows = []
    for structure in structures:
        molecule = Chem.MolFromSmiles(structure)
        if molecule is None:
            rows.append([np.nan] * len(names))
        else:
            rows.append(list(calculator(molecule).fill_missing(np.nan)))
    frame = pd.DataFrame(rows, columns=names)
    if record_ids is not None:
        identifiers = pd.Series(record_ids, dtype="string")
        if identifiers.isna().any() or identifiers.duplicated().any():
            raise ValueError("record identifiers must be non-missing and unique")
        frame.insert(0, "record_id", identifiers.astype(str).to_numpy())
    return frame
