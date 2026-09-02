from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .modeling import build_fusion_model


@dataclass(frozen=True)
class FusionFitResult:
    model: object
    best_epoch: int
    best_selection_rmse: float


@dataclass(frozen=True)
class FixedEpochFusionFitResult:
    model: object
    epochs: int
    seed: int


def _matrix(values, name):
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or len(matrix) == 0 or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a non-empty finite matrix")
    return matrix


def _labels(values, rows, name):
    labels = np.asarray(values, dtype=np.float32).reshape(-1)
    if labels.shape != (rows,) or not np.isfinite(labels).all():
        raise ValueError(f"{name} must contain one finite value per row")
    return labels


def _predict(model, matrix, device, batch_size=4096):
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(matrix), int(batch_size)):
            batch = torch.as_tensor(matrix[start : start + int(batch_size)], dtype=torch.float32, device=device)
            outputs.append(model(batch).detach().cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def fit_fusion_model(fit_features, fit_labels, selection_features, selection_labels, regressor, parameters, patience, gradient_clip_norm, minimum_improvement, seed, device):
    fit_x = _matrix(fit_features, "fit features")
    selection_x = _matrix(selection_features, "selection features")
    if fit_x.shape[1] != selection_x.shape[1]:
        raise ValueError("fit and selection feature dimensions differ")
    fit_y = _labels(fit_labels, len(fit_x), "fit labels")
    selection_y = _labels(selection_labels, len(selection_x), "selection labels")
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    model = build_fusion_model(regressor, fit_x.shape[1], parameters).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(parameters["learning_rate"]), weight_decay=float(parameters["weight_decay"]))
    loss_function = nn.MSELoss()
    fit_tensor = torch.as_tensor(fit_x, dtype=torch.float32)
    label_tensor = torch.as_tensor(fit_y, dtype=torch.float32)
    generator = np.random.default_rng(int(seed))
    best_state = None
    best_mse = float("inf")
    best_epoch = -1
    stale = 0
    for epoch in range(1, int(parameters["maximum_epochs"]) + 1):
        model.train()
        order = generator.permutation(len(fit_x))
        for start in range(0, len(order), int(parameters["batch_size"])):
            indices = order[start : start + int(parameters["batch_size"])]
            features = fit_tensor[indices].to(device)
            labels = label_tensor[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(features), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip_norm))
            optimizer.step()
        predictions = _predict(model, selection_x, device)
        mse = float(np.mean((selection_y - predictions) ** 2))
        if mse < best_mse - float(minimum_improvement):
            best_mse = mse
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(patience):
                break
    if best_state is None:
        raise RuntimeError("fusion training did not produce a checkpoint")
    model.load_state_dict(best_state, strict=True)
    return FusionFitResult(model, best_epoch, float(np.sqrt(best_mse)))


def fit_fixed_epoch_fusion_head(fit_features, fit_labels, parameters, epochs, gradient_clip_norm, seed, device):
    fit_x = _matrix(fit_features, "fit features")
    fit_y = _labels(fit_labels, len(fit_x), "fit labels")
    if int(epochs) <= 0:
        raise ValueError("epochs must be positive")
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    model = build_fusion_model("residual_mlp", fit_x.shape[1], parameters).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(parameters["learning_rate"]),
        weight_decay=float(parameters["weight_decay"]),
    )
    loss_function = nn.MSELoss()
    fit_tensor = torch.as_tensor(fit_x, dtype=torch.float32)
    label_tensor = torch.as_tensor(fit_y, dtype=torch.float32)
    generator = np.random.default_rng(int(seed))
    for _ in range(int(epochs)):
        model.train()
        order = generator.permutation(len(fit_x))
        for start in range(0, len(order), int(parameters["batch_size"])):
            indices = order[start : start + int(parameters["batch_size"])]
            features = fit_tensor[indices].to(device)
            labels = label_tensor[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(features), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip_norm))
            optimizer.step()
    return FixedEpochFusionFitResult(model, int(epochs), int(seed))


def predict_fusion(model, scored_features, device):
    matrix = _matrix(scored_features, "scored features")
    model.to(device)
    predictions = _predict(model, matrix, device)
    if not np.isfinite(predictions).all():
        raise ValueError("fusion inference produced non-finite predictions")
    return predictions
