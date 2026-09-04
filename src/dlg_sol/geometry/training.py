from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from .modeling import DistanceAware3DMPNN


@dataclass(frozen=True)
class GeometryFitResult:
    model: object
    best_epoch: int
    best_selection_rmse: float
    history: tuple[dict[str, float | int], ...]


@dataclass(frozen=True)
class GeometryInferenceResult:
    record_ids: tuple[str, ...]
    predictions: np.ndarray
    graph_embeddings: np.ndarray
    auxiliary_features: np.ndarray


def sample_geometry_parameters(seed, trial):
    generator = np.random.default_rng(int(seed) + int(trial) * 997)
    return {
        "hidden_dim": int(generator.choice([96, 128, 192, 256])),
        "num_layers": int(generator.choice([2, 3, 4])),
        "dropout": float(generator.choice([0.05, 0.10, 0.15, 0.25, 0.35])),
        "batch_size": int(generator.choice([64, 96, 128])),
        "lr": float(10 ** generator.uniform(np.log10(2e-4), np.log10(3e-3))),
        "weight_decay": float(10 ** generator.uniform(np.log10(1e-7), np.log10(5e-4))),
        "pool": str(generator.choice(["mean", "mean_aux"])),
    }


def _set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = True


def _attach_labels(graphs, labels):
    label_map = {str(key): float(value) for key, value in labels.items()}
    if not label_map or not np.isfinite(tuple(label_map.values())).all():
        raise ValueError("development labels must be finite and non-empty")
    result = []
    for graph in graphs:
        identifier = str(graph.record_id)
        if identifier not in label_map:
            raise ValueError(f"missing development label for record {identifier}")
        item = graph.clone()
        item.record_id = identifier
        item.y = torch.tensor([label_map[identifier]], dtype=torch.float32)
        result.append(item)
    if not result:
        raise ValueError("geometry graph collection cannot be empty")
    return result


def _record_predictions(model, graphs, batch_size, device, include_labels):
    loader = DataLoader(graphs, batch_size=int(batch_size), shuffle=False, num_workers=0)
    by_record = {}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            identifiers = [str(value) for value in batch.record_id]
            batch = batch.to(device)
            embedding = model.encode_graph(batch)
            prediction = model.forward_from_embedding(embedding, batch.aux if hasattr(batch, "aux") else None)
            label_values = batch.y.detach().cpu().numpy().reshape(-1) if include_labels else None
            for index, identifier in enumerate(identifiers):
                record = by_record.setdefault(identifier, {"predictions": [], "embeddings": [], "auxiliary": [], "labels": []})
                record["predictions"].append(float(prediction[index].detach().cpu()))
                record["embeddings"].append(embedding[index].detach().cpu().numpy())
                record["auxiliary"].append(batch.aux[index].detach().cpu().numpy())
                if include_labels:
                    record["labels"].append(float(label_values[index]))
    identifiers = tuple(by_record)
    predictions = np.asarray([np.mean(by_record[key]["predictions"]) for key in identifiers], dtype=np.float32)
    embeddings = np.asarray([np.mean(by_record[key]["embeddings"], axis=0) for key in identifiers], dtype=np.float32)
    auxiliary = np.asarray([np.mean(by_record[key]["auxiliary"], axis=0) for key in identifiers], dtype=np.float32)
    labels = None if not include_labels else np.asarray([by_record[key]["labels"][0] for key in identifiers], dtype=np.float32)
    return identifiers, predictions, embeddings, auxiliary, labels


def fit_geometry_model(fit_graphs, fit_labels, selection_graphs, selection_labels, parameters, maximum_epochs, patience, gradient_clip_norm, seed, device):
    training = _attach_labels(fit_graphs, fit_labels)
    selection = _attach_labels(selection_graphs, selection_labels)
    _set_seed(seed)
    first = training[0]
    model = DistanceAware3DMPNN(first.x.shape[1], first.edge_attr.shape[1], first.aux.shape[1], parameters["hidden_dim"], parameters["num_layers"], parameters["dropout"], parameters["pool"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=parameters["lr"], weight_decay=parameters["weight_decay"])
    criterion = nn.MSELoss()
    loader = DataLoader(training, batch_size=parameters["batch_size"], shuffle=True, num_workers=0)
    best_state = None
    best_rmse = float("inf")
    best_epoch = -1
    stale = 0
    history = []
    for epoch in range(1, int(maximum_epochs) + 1):
        model.train()
        losses = []
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch), batch.y.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip_norm))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        _, predictions, _, _, labels = _record_predictions(model, selection, parameters["batch_size"], device, True)
        selection_rmse = float(np.sqrt(np.mean((labels - predictions) ** 2)))
        history.append({"epoch": epoch, "fit_conformer_mse": float(np.mean(losses)), "selection_record_rmse": selection_rmse})
        if selection_rmse < best_rmse - 1e-6:
            best_rmse = selection_rmse
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(patience):
                break
    if best_state is None:
        raise RuntimeError("geometry training did not produce a checkpoint")
    model.load_state_dict(best_state, strict=True)
    return GeometryFitResult(model, best_epoch, best_rmse, tuple(history))


def fit_geometry_fixed_epochs(graphs, labels, parameters, epochs, gradient_clip_norm, seed, device):
    training = _attach_labels(graphs, labels)
    if int(epochs) <= 0:
        raise ValueError("geometry fixed-epoch refit requires positive epochs")
    _set_seed(seed)
    first = training[0]
    model = DistanceAware3DMPNN(first.x.shape[1], first.edge_attr.shape[1], first.aux.shape[1], parameters["hidden_dim"], parameters["num_layers"], parameters["dropout"], parameters["pool"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=parameters["lr"], weight_decay=parameters["weight_decay"])
    criterion = nn.MSELoss()
    loader = DataLoader(training, batch_size=parameters["batch_size"], shuffle=True, num_workers=0)
    for _ in range(int(epochs)):
        model.train()
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch), batch.y.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip_norm))
            optimizer.step()
    return model


def predict_geometry(model, graphs, batch_size, device):
    values = tuple(graphs)
    if not values:
        raise ValueError("scored geometry graph collection cannot be empty")
    if any(getattr(graph, "y", None) is not None for graph in values):
        raise ValueError("scored geometry graphs must not contain labels")
    model.to(device)
    identifiers, predictions, embeddings, auxiliary, _ = _record_predictions(model, values, batch_size, device, False)
    if not np.isfinite(predictions).all() or not np.isfinite(embeddings).all() or not np.isfinite(auxiliary).all():
        raise ValueError("geometry inference produced non-finite values")
    return GeometryInferenceResult(identifiers, predictions, embeddings, auxiliary)
