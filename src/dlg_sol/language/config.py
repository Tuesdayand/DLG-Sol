from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ChemBertaVariant:
    name: str
    model_identifier: str
    model_revision: str | None
    pooling: str
    embedding_dimension: int
    max_length: int
    dropout: float
    freeze_layers: int
    weight_decay: float
    batch_size: int
    evaluation_batch_size: int
    maximum_epochs: int
    patience: int
    warmup_ratio: float
    gradient_clip_norm: float
    seed: int
    checkpoint_selection: str
    inference_smiles: str
    learning_rate: float
    randomized_smiles_per_molecule: int
    task: str
    auxiliary_weight: float
    auxiliary_targets: tuple[str, ...]


def _validate(config: ChemBertaVariant) -> ChemBertaVariant:
    if config.pooling != "masked_mean":
        raise ValueError("ChemBERTa pooling must be masked_mean")
    if config.embedding_dimension != 768 or config.max_length != 256:
        raise ValueError("ChemBERTa representation shape does not match the released contract")
    if config.freeze_layers < 0 or config.randomized_smiles_per_molecule < 0:
        raise ValueError("layer and augmentation counts cannot be negative")
    if config.task not in {"logS", "logS_plus_auxiliary"}:
        raise ValueError("unsupported ChemBERTa task")
    if config.task == "logS" and (config.auxiliary_weight != 0 or config.auxiliary_targets):
        raise ValueError("single-task variants cannot declare auxiliary supervision")
    if config.task == "logS_plus_auxiliary" and (config.auxiliary_weight <= 0 or not config.auxiliary_targets):
        raise ValueError("multitask variants require auxiliary supervision")
    if config.inference_smiles != "canonical" or config.checkpoint_selection != "selection_logS_rmse":
        raise ValueError("ChemBERTa inference or selection policy differs from the released contract")
    return config


def load_chemberta_variants(path: str | Path) -> dict[str, ChemBertaVariant]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2:
        raise ValueError("unsupported ChemBERTa configuration schema")
    if payload.get("final_model_profile") != "randomized_single_task":
        raise ValueError("final DLG-Sol language profile must be randomized_single_task")
    expected_scope = {
        "randomized_single_task": "final_dlg_sol_language_encoder",
        "canonical_multitask": "development_candidate_not_final_dlg_sol",
        "canonical_single_task": "development_candidate_not_final_dlg_sol",
    }
    if payload.get("variant_scope") != expected_scope:
        raise ValueError("ChemBERTa final and development scopes are misstated")
    pretrained = payload["pretrained_model"]
    shared = payload["shared"]
    variants = {}
    for name, record in payload["variants"].items():
        variants[name] = _validate(
            ChemBertaVariant(
                name=name,
                model_identifier=str(pretrained["identifier"]),
                model_revision=pretrained.get("revision"),
                pooling=str(shared["pooling"]),
                embedding_dimension=int(shared["embedding_dimension"]),
                max_length=int(shared["max_length"]),
                dropout=float(shared["dropout"]),
                freeze_layers=int(shared["freeze_layers"]),
                weight_decay=float(shared["weight_decay"]),
                batch_size=int(shared["batch_size"]),
                evaluation_batch_size=int(shared["evaluation_batch_size"]),
                maximum_epochs=int(shared["maximum_epochs"]),
                patience=int(shared["patience"]),
                warmup_ratio=float(shared["warmup_ratio"]),
                gradient_clip_norm=float(shared["gradient_clip_norm"]),
                seed=int(shared["seed"]),
                checkpoint_selection=str(shared["checkpoint_selection"]),
                inference_smiles=str(shared["inference_smiles"]),
                learning_rate=float(record["learning_rate"]),
                randomized_smiles_per_molecule=int(record["training_randomized_smiles_per_molecule"]),
                task=str(record["task"]),
                auxiliary_weight=float(record["auxiliary_weight"]),
                auxiliary_targets=tuple(str(value) for value in record["auxiliary_targets"]),
            )
        )
    if set(variants) != {"canonical_multitask", "randomized_single_task", "canonical_single_task"}:
        raise ValueError("ChemBERTa variant set is incomplete")
    return variants
