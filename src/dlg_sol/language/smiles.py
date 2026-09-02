from __future__ import annotations

from rdkit import Chem


def canonicalize_smiles(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError("invalid SMILES")
    return str(Chem.MolToSmiles(molecule, canonical=True))


def randomized_smiles(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError("invalid SMILES")
    value = Chem.MolToSmiles(molecule, doRandom=True, canonical=False)
    if not value:
        raise ValueError("randomized SMILES generation failed")
    return str(value)


def training_record_count(molecules: int, randomized_per_molecule: int) -> int:
    if molecules < 1 or randomized_per_molecule < 0:
        raise ValueError("molecule and augmentation counts are invalid")
    return molecules * (1 + randomized_per_molecule)
