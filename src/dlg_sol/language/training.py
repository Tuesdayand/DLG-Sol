from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from .auxiliary import compute_auxiliary_targets, fit_auxiliary_scaler, transform_auxiliary_targets
from .config import ChemBertaVariant
from .modeling import ChemBertaDataset, build_chemberta_model, load_chemberta_tokenizer


@dataclass(frozen=True)
class LanguageFitResult:
    model: object
    best_epoch: int
    best_selection_rmse: float
    history: tuple[dict[str, float | int], ...]
    auxiliary_scaler: object | None


@dataclass(frozen=True)
class LanguageInferenceResult:
    predictions: np.ndarray
    embeddings: np.ndarray


def _finite_vector(values, name):
    result = np.asarray(values, dtype=np.float32)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a non-empty finite vector")
    return result


def _predict(model, loader, device):
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            output = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            if isinstance(output, tuple):
                output = output[0]
            predictions.append(output.cpu().numpy())
    result = np.concatenate(predictions).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("ChemBERTa produced non-finite predictions")
    return result


def fit_chemberta_variant(config, fit_smiles, fit_labels, selection_smiles, selection_labels, device):
    if not isinstance(config, ChemBertaVariant):
        raise TypeError("config must be a validated ChemBertaVariant")
    fit_values = tuple(str(value) for value in fit_smiles)
    selection_values = tuple(str(value) for value in selection_smiles)
    y_fit = _finite_vector(fit_labels, "fit labels")
    y_selection = _finite_vector(selection_labels, "selection labels")
    if len(fit_values) != len(y_fit) or len(selection_values) != len(y_selection):
        raise ValueError("SMILES and label counts do not match")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    auxiliary_scaler = None
    fit_auxiliary = None
    selection_auxiliary = None
    if config.task == "logS_plus_auxiliary":
        fit_targets = compute_auxiliary_targets(fit_values)
        selection_targets = compute_auxiliary_targets(selection_values)
        auxiliary_scaler = fit_auxiliary_scaler(fit_targets)
        fit_auxiliary = transform_auxiliary_targets(fit_targets, auxiliary_scaler)
        selection_auxiliary = transform_auxiliary_targets(selection_targets, auxiliary_scaler)
    tokenizer = load_chemberta_tokenizer(config)
    fit_dataset = ChemBertaDataset(fit_values, tokenizer, config.max_length, config.randomized_smiles_per_molecule, y_fit, fit_auxiliary)
    selection_dataset = ChemBertaDataset(selection_values, tokenizer, config.max_length, 0, y_selection, selection_auxiliary)
    fit_loader = DataLoader(fit_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    selection_loader = DataLoader(selection_dataset, batch_size=config.evaluation_batch_size, shuffle=False, num_workers=2, pin_memory=True)
    model = build_chemberta_model(config).to(device)
    no_decay = {"bias", "LayerNorm.weight"}
    parameters = [
        {
            "params": [parameter for name, parameter in model.named_parameters() if not any(term in name for term in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [parameter for name, parameter in model.named_parameters() if any(term in name for term in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(parameters, lr=config.learning_rate)
    total_steps = len(fit_loader) * config.maximum_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * config.warmup_ratio), total_steps)
    criterion = nn.MSELoss()
    history = []
    best_rmse = float("inf")
    best_epoch = -1
    best_state = None
    stale = 0
    for epoch in range(1, config.maximum_epochs + 1):
        model.train()
        total_main = 0.0
        total_auxiliary = 0.0
        for batch in fit_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            output = model(input_ids, attention_mask)
            if isinstance(output, tuple):
                main_prediction, auxiliary_prediction = output
                main_loss = criterion(main_prediction, labels)
                auxiliary_loss = criterion(auxiliary_prediction, batch["auxiliary"].to(device))
                loss = main_loss + config.auxiliary_weight * auxiliary_loss
                total_auxiliary += float(auxiliary_loss.item()) * len(labels)
            else:
                main_loss = criterion(output, labels)
                loss = main_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            scheduler.step()
            total_main += float(main_loss.item()) * len(labels)
        selection_prediction = _predict(model, selection_loader, device)
        selection_rmse = float(np.sqrt(np.mean((y_selection - selection_prediction) ** 2)))
        history.append(
            {
                "epoch": epoch,
                "fit_main_mse": total_main / len(fit_loader.dataset),
                "fit_auxiliary_mse": total_auxiliary / len(fit_loader.dataset),
                "selection_logS_rmse": selection_rmse,
            }
        )
        if selection_rmse < best_rmse:
            best_rmse = selection_rmse
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("ChemBERTa training did not produce a checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.to(device)
    return LanguageFitResult(model, best_epoch, best_rmse, tuple(history), auxiliary_scaler)


def predict_chemberta(model, config, scored_smiles, device):
    values = tuple(str(value) for value in scored_smiles)
    tokenizer = load_chemberta_tokenizer(config)
    dataset = ChemBertaDataset(values, tokenizer, config.max_length)
    loader = DataLoader(dataset, batch_size=config.evaluation_batch_size, shuffle=False, num_workers=2, pin_memory=True)
    model.to(device)
    predictions = _predict(model, loader, device)
    embeddings = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            embeddings.append(model.embed(batch["input_ids"].to(device), batch["attention_mask"].to(device)).cpu().numpy())
    matrix = np.concatenate(embeddings).astype(np.float32)
    if matrix.shape != (len(values), config.embedding_dimension) or not np.isfinite(matrix).all():
        raise ValueError("ChemBERTa scored embedding matrix violates the released contract")
    return LanguageInferenceResult(predictions, matrix)
