from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data

from .config import GeometryConfig
from .conformers import generate_conformers, largest_organic_fragment
from .features import atom_feature_matrix, bond_base_features, molecule_auxiliary_features


@dataclass(frozen=True)
class GeometryBuildResult:
    record_id: str
    canonical_smiles: str
    status: str
    graphs: tuple[Data, ...]
    fragment_count: int
    largest_fragment_ratio: float
    has_counterion: bool
    conformers_generated: int
    conformers_retained: int
    optimization_method: str


def _is_polar_hydrogen(atom):
    if atom.GetAtomicNum() != 1:
        return False
    neighbors = atom.GetNeighbors()
    return bool(neighbors) and neighbors[0].GetAtomicNum() in (7, 8)


def graph_from_conformer(molecule_h, conformer_id, auxiliary, record_id, canonical_smiles, energy, rank, config):
    molecule_h.UpdatePropertyCache(strict=False)
    try:
        Chem.GetSymmSSSR(molecule_h)
    except Exception:
        pass
    full_features = atom_feature_matrix(molecule_h)
    full_positions = np.asarray(molecule_h.GetConformer(int(conformer_id)).GetPositions(), dtype=np.float32)
    kept = [atom.GetIdx() for atom in molecule_h.GetAtoms() if atom.GetAtomicNum() != 1 or _is_polar_hydrogen(atom)]
    remapping = {old: new for new, old in enumerate(kept)}
    node_features = full_features[kept]
    positions = full_positions[kept]
    edges = {}
    for bond in molecule_h.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if begin in remapping and end in remapping:
            base = bond_base_features(bond)
            edges[(remapping[begin], remapping[end])] = base
            edges[(remapping[end], remapping[begin])] = base
    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    for source in range(len(kept)):
        for target in range(len(kept)):
            if source != target and distances[source, target] <= config.proximity_cutoff_angstrom and (source, target) not in edges:
                edges[(source, target)] = bond_base_features(None)
    if not edges:
        edges[(0, 0)] = bond_base_features(None)
    centers = np.linspace(config.rbf_start_angstrom, config.rbf_stop_angstrom, config.rbf_count).astype(np.float32)
    edge_indices = []
    edge_features = []
    for (source, target), base in edges.items():
        distance = float(distances[source, target]) if source != target else 0.0
        radial = np.exp(-((distance - centers) ** 2) / config.rbf_denominator).astype(np.float32).tolist()
        edge_indices.append([source, target])
        edge_features.append(base + [distance / config.distance_scale_angstrom] + radial)
    graph = Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        pos=torch.tensor(positions, dtype=torch.float32),
        edge_index=torch.tensor(edge_indices, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_features, dtype=torch.float32),
        aux=torch.tensor(auxiliary, dtype=torch.float32).view(1, -1),
    )
    graph.record_id = str(record_id)
    graph.canonical_smiles = str(canonical_smiles)
    graph.conf_rank = int(rank)
    graph.conf_energy = float(energy)
    graph.n_polar_h = int(sum(1 for index in kept if molecule_h.GetAtomWithIdx(index).GetAtomicNum() == 1))
    if graph.x.shape[1] != config.node_dimension or graph.edge_attr.shape[1] != config.edge_dimension or graph.aux.shape[1] != config.auxiliary_dimension:
        raise ValueError("geometry graph dimensions differ from the released contract")
    return graph


def build_geometry_graphs(smiles, record_id, config, seed_offset=0):
    if not isinstance(config, GeometryConfig):
        raise TypeError("config must be a validated GeometryConfig")
    molecule = Chem.MolFromSmiles(str(smiles), sanitize=True)
    if molecule is None:
        return GeometryBuildResult(str(record_id), str(smiles), "parse_failed", (), 0, 0.0, False, 0, 0, "none")
    canonical = Chem.MolToSmiles(molecule, canonical=True)
    parent, fragment_count, ratio, counterion = largest_organic_fragment(molecule)
    molecule_h, identifiers, energies, method = generate_conformers(
        parent,
        config.conformers_attempted,
        config.base_seed + int(seed_offset),
        config.conformer_prune_rms_threshold,
        config.optimization_max_iterations,
    )
    success = molecule_h is not None and bool(identifiers)
    auxiliary = molecule_auxiliary_features(parent, molecule, fragment_count, ratio, counterion, success, list(energies))
    if not success:
        return GeometryBuildResult(str(record_id), canonical, "embed_failed", (), fragment_count, ratio, counterion, 0, 0, method)
    graphs = []
    for rank, (identifier, energy) in enumerate(zip(identifiers[: config.conformers_retained], energies[: config.conformers_retained])):
        try:
            graphs.append(graph_from_conformer(molecule_h, identifier, auxiliary, record_id, canonical, energy, rank, config))
        except Exception:
            pass
    status = "ok" if graphs else "graph_failed"
    return GeometryBuildResult(str(record_id), canonical, status, tuple(graphs), fragment_count, ratio, counterion, len(identifiers), len(graphs), method)
