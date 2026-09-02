from __future__ import annotations

import math

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors


ATOM_NUMBERS = (1, 5, 6, 7, 8, 9, 11, 12, 15, 16, 17, 19, 20, 30, 35, 53)
HYBRIDIZATIONS = (
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
)
BOND_TYPES = (Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC)


def one_hot(value, choices):
    values = tuple(choices)
    return [1.0 if value == choice else 0.0 for choice in values] + [0.0 if value in values else 1.0]


def safe_float(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else float(default)
    except Exception:
        return float(default)


def _atom_matches(molecule, pattern):
    query = Chem.MolFromSmarts(pattern)
    if query is None:
        return set()
    result = set()
    for match in molecule.GetSubstructMatches(query):
        result.update(int(index) for index in match)
    return result


def _gasteiger_charges(molecule):
    try:
        AllChem.ComputeGasteigerCharges(molecule)
    except Exception:
        return [0.0] * molecule.GetNumAtoms()
    return [safe_float(atom.GetProp("_GasteigerCharge") if atom.HasProp("_GasteigerCharge") else 0.0) for atom in molecule.GetAtoms()]


def atom_feature_matrix(molecule):
    charges = _gasteiger_charges(molecule)
    acceptors = _atom_matches(molecule, "[$([O,S;H1;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])])]")
    donors = _atom_matches(molecule, "[$([N,O,S;H1,H2,H3;+0]),$([nH])]")
    carboxyl = _atom_matches(molecule, "C(=O)[O-,$([OH])]")
    sulfonyl = _atom_matches(molecule, "S(=O)(=O)[O-,$([OH])]")
    phosphoryl = _atom_matches(molecule, "P(=O)([O-,$([OH])])[O-,$([OH])]")
    quaternary_nitrogen = _atom_matches(molecule, "[N+X4]")
    rows = []
    for index, atom in enumerate(molecule.GetAtoms()):
        rows.append(
            one_hot(atom.GetAtomicNum(), ATOM_NUMBERS)
            + one_hot(min(atom.GetTotalDegree(), 5), (0, 1, 2, 3, 4, 5))
            + one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
            + [
                float(atom.GetFormalCharge()),
                float(charges[index]),
                float(atom.GetTotalNumHs()),
                float(atom.GetIsAromatic()),
                float(atom.IsInRing()),
                float(atom.GetMass()) / 200.0,
                float(index in donors),
                float(index in acceptors),
                float(index in carboxyl),
                float(index in sulfonyl),
                float(index in phosphoryl),
                float(index in quaternary_nitrogen),
            ]
        )
    result = np.asarray(rows, dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != 42 or not np.isfinite(result).all():
        raise ValueError("node-feature construction violated the 42-feature contract")
    return result


def bond_base_features(bond):
    if bond is None:
        return [0.0] * 7
    return [1.0] + [1.0 if bond.GetBondType() == value else 0.0 for value in BOND_TYPES] + [float(bond.GetIsConjugated()), float(bond.IsInRing())]


def _descriptor(function, molecule, default=0.0):
    try:
        return safe_float(function(molecule), default)
    except Exception:
        molecule.UpdatePropertyCache(strict=False)
        try:
            Chem.GetSymmSSSR(molecule)
        except Exception:
            pass
        try:
            return safe_float(function(molecule), default)
        except Exception:
            return float(default)


def molecule_auxiliary_features(molecule, original, fragment_count, largest_fragment_ratio, has_counterion, conformer_success, energies):
    original_charge = sum(atom.GetFormalCharge() for atom in original.GetAtoms())
    fragment_charge = sum(atom.GetFormalCharge() for atom in molecule.GetAtoms())
    positive = sum(1 for atom in original.GetAtoms() if atom.GetFormalCharge() > 0)
    negative = sum(1 for atom in original.GetAtoms() if atom.GetFormalCharge() < 0)
    energy_values = np.asarray(energies if energies else [0.0], dtype=np.float32)
    result = np.asarray(
        [
            float(original_charge),
            float(fragment_charge),
            float(abs(original_charge)),
            float(positive),
            float(negative),
            _descriptor(rdMolDescriptors.CalcTPSA, molecule) / 250.0,
            _descriptor(Crippen.MolLogP, molecule) / 10.0,
            _descriptor(Lipinski.NumHDonors, molecule) / 10.0,
            _descriptor(Lipinski.NumHAcceptors, molecule) / 20.0,
            _descriptor(Lipinski.NumRotatableBonds, molecule) / 30.0,
            _descriptor(rdMolDescriptors.CalcNumRings, molecule) / 10.0,
            _descriptor(Descriptors.MolWt, molecule) / 1000.0,
            float(fragment_count),
            float(largest_fragment_ratio),
            float(has_counterion),
            float(conformer_success),
            float(np.min(energy_values)) / 500.0,
            float(np.mean(energy_values)) / 500.0,
            float(np.std(energy_values)) / 100.0,
        ],
        dtype=np.float32,
    )
    if result.shape != (19,) or not np.isfinite(result).all():
        raise ValueError("molecule-level feature construction violated the 19-feature contract")
    return result
