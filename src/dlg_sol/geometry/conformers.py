from __future__ import annotations

import math

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize

from .features import safe_float


def largest_organic_fragment(molecule):
    fragments = Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=True)
    if len(fragments) <= 1:
        return molecule, 1, 1.0, False
    parent = rdMolStandardize.FragmentParent(molecule)
    if parent is None or parent.GetNumAtoms() == 0:
        parent = max(fragments, key=lambda value: value.GetNumHeavyAtoms())
    try:
        Chem.SanitizeMol(parent)
    except Exception:
        parent.UpdatePropertyCache(strict=False)
        try:
            Chem.GetSymmSSSR(parent)
        except Exception:
            pass
    ratio = parent.GetNumHeavyAtoms() / max(1, molecule.GetNumHeavyAtoms())
    return parent, len(fragments), float(ratio), True


def generate_conformers(molecule, count, seed, prune_rms_threshold=0.5, maximum_iterations=200):
    molecule_h = Chem.AddHs(molecule)
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = int(seed)
    parameters.numThreads = 0
    parameters.pruneRmsThresh = float(prune_rms_threshold)
    try:
        identifiers = list(AllChem.EmbedMultipleConfs(molecule_h, numConfs=int(count), params=parameters))
    except Exception:
        identifiers = []
    if not identifiers:
        return None, (), (), "embed_failed"
    energies = []
    method = "unoptimized"
    try:
        has_mmff = bool(AllChem.MMFFHasAllMoleculeParams(molecule_h))
    except Exception:
        has_mmff = False
    for identifier in identifiers:
        energy = np.nan
        if has_mmff:
            try:
                properties = AllChem.MMFFGetMoleculeProperties(molecule_h)
                force_field = AllChem.MMFFGetMoleculeForceField(molecule_h, properties, confId=int(identifier))
                if force_field is not None:
                    force_field.Minimize(maxIts=int(maximum_iterations))
                    energy = force_field.CalcEnergy()
                    method = "MMFF"
            except Exception:
                pass
        if not math.isfinite(float(energy)) if not np.isnan(energy) else True:
            try:
                force_field = AllChem.UFFGetMoleculeForceField(molecule_h, confId=int(identifier))
                if force_field is not None:
                    force_field.Minimize(maxIts=int(maximum_iterations))
                    energy = force_field.CalcEnergy()
                    if method == "unoptimized":
                        method = "UFF"
            except Exception:
                pass
        energies.append(safe_float(energy, 0.0))
    order = np.argsort(np.asarray(energies, dtype=np.float32)).tolist()
    return molecule_h, tuple(identifiers[index] for index in order), tuple(energies[index] for index in order), method
