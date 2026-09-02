from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GeometryConfig:
    conformers_attempted: int
    conformers_retained: int
    conformer_prune_rms_threshold: float
    optimization_max_iterations: int
    base_seed: int
    proximity_cutoff_angstrom: float
    distance_scale_angstrom: float
    rbf_start_angstrom: float
    rbf_stop_angstrom: float
    rbf_count: int
    rbf_denominator: float
    node_dimension: int
    edge_dimension: int
    auxiliary_dimension: int
    hidden_dimensions: tuple[int, ...]
    message_passing_layers: tuple[int, ...]
    dropout_values: tuple[float, ...]
    batch_sizes: tuple[int, ...]
    learning_rate_range: tuple[float, float]
    weight_decay_range: tuple[float, float]
    maximum_epochs: int
    patience: int
    gradient_clip_norm: float
    checkpoint_selection: str


def _validate(config: GeometryConfig) -> GeometryConfig:
    dimensions = (config.node_dimension, config.edge_dimension, config.auxiliary_dimension)
    if dimensions != (42, 24, 19):
        raise ValueError("geometry feature dimensions differ from the released contract")
    if config.conformers_attempted != 5 or config.conformers_retained != 3:
        raise ValueError("conformer-count policy differs from the released contract")
    if config.proximity_cutoff_angstrom != 4.5 or config.rbf_count != 16:
        raise ValueError("distance-edge policy differs from the released contract")
    if config.checkpoint_selection != "selection_rmse":
        raise ValueError("geometry checkpoint selection must use selection RMSE")
    if config.maximum_epochs <= 0 or config.patience <= 0 or config.gradient_clip_norm <= 0:
        raise ValueError("geometry training controls must be positive")
    return config


def load_geometry_config(path: str | Path) -> GeometryConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported geometry configuration schema")
    graph = payload["graph_construction"]
    model = payload["model"]
    rbf = graph["rbf_centers_angstrom"]
    if graph["fragment_policy"] != "largest_organic_fragment":
        raise ValueError("unsupported fragment policy")
    if graph["explicit_hydrogen_policy"] != "hydrogens_bonded_to_nitrogen_or_oxygen":
        raise ValueError("unsupported explicit-hydrogen policy")
    if model["coordinate_updates"] is not False or model["architecture"] != "distance_aware_scalar_3d_mpnn":
        raise ValueError("geometry architecture is misstated")
    return _validate(
        GeometryConfig(
            conformers_attempted=int(graph["conformers_attempted"]),
            conformers_retained=int(graph["conformers_retained"]),
            conformer_prune_rms_threshold=float(graph["conformer_prune_rms_threshold"]),
            optimization_max_iterations=int(graph["optimization_max_iterations"]),
            base_seed=int(graph["base_seed"]),
            proximity_cutoff_angstrom=float(graph["proximity_cutoff_angstrom"]),
            distance_scale_angstrom=float(graph["distance_scale_angstrom"]),
            rbf_start_angstrom=float(rbf["start"]),
            rbf_stop_angstrom=float(rbf["stop"]),
            rbf_count=int(rbf["count"]),
            rbf_denominator=float(rbf["denominator"]),
            node_dimension=int(graph["node_dimension"]),
            edge_dimension=int(graph["edge_dimension"]),
            auxiliary_dimension=int(graph["auxiliary_dimension"]),
            hidden_dimensions=tuple(int(value) for value in model["hidden_dimensions"]),
            message_passing_layers=tuple(int(value) for value in model["message_passing_layers"]),
            dropout_values=tuple(float(value) for value in model["dropout_values"]),
            batch_sizes=tuple(int(value) for value in model["batch_sizes"]),
            learning_rate_range=tuple(float(value) for value in model["learning_rate_range"]),
            weight_decay_range=tuple(float(value) for value in model["weight_decay_range"]),
            maximum_epochs=int(model["maximum_epochs"]),
            patience=int(model["patience"]),
            gradient_clip_norm=float(model["gradient_clip_norm"]),
            checkpoint_selection=str(model["checkpoint_selection"]),
        )
    )
