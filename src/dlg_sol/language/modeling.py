from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from .config import ChemBertaVariant
from .smiles import randomized_smiles


class ChemBertaDataset(Dataset):
    def __init__(self, smiles, tokenizer, max_length, randomized_per_molecule=0, labels=None, auxiliary=None):
        self.smiles = tuple(str(value) for value in smiles)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.randomized_per_molecule = int(randomized_per_molecule)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)
        self.auxiliary = None if auxiliary is None else np.asarray(auxiliary, dtype=np.float32)
        if not self.smiles:
            raise ValueError("ChemBERTa dataset cannot be empty")
        if self.randomized_per_molecule < 0:
            raise ValueError("augmentation count cannot be negative")
        if self.labels is not None and len(self.labels) != len(self.smiles):
            raise ValueError("label count does not match molecule count")
        if self.auxiliary is not None and len(self.auxiliary) != len(self.smiles):
            raise ValueError("auxiliary-target count does not match molecule count")

    def __len__(self):
        return len(self.smiles) * (1 + self.randomized_per_molecule)

    def __getitem__(self, index):
        molecule = int(index) % len(self.smiles)
        value = self.smiles[molecule]
        if int(index) // len(self.smiles) > 0:
            value = randomized_smiles(value)
        encoded = self.tokenizer(value, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        record = {"input_ids": encoded["input_ids"].squeeze(0), "attention_mask": encoded["attention_mask"].squeeze(0)}
        if self.labels is not None:
            record["label"] = torch.tensor(float(self.labels[molecule]), dtype=torch.float32)
        if self.auxiliary is not None:
            record["auxiliary"] = torch.tensor(self.auxiliary[molecule], dtype=torch.float32)
        return record


class ChemBertaRegressor(nn.Module):
    def __init__(self, model_identifier, dropout=0.1, freeze_layers=0, revision=None):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_identifier, revision=revision)
        self.drop = nn.Dropout(float(dropout))
        self.head = nn.Linear(int(self.bert.config.hidden_size), 1)
        self._freeze(int(freeze_layers))

    def _freeze(self, freeze_layers):
        if freeze_layers > 0:
            for parameter in self.bert.embeddings.parameters():
                parameter.requires_grad = False
            for layer in self.bert.encoder.layer[:freeze_layers]:
                for parameter in layer.parameters():
                    parameter.requires_grad = False

    def pool(self, hidden, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.pool(output.last_hidden_state, attention_mask)
        return self.head(self.drop(pooled)).squeeze(-1)

    def embed(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.pool(output.last_hidden_state, attention_mask)


class ChemBertaMultiTaskRegressor(ChemBertaRegressor):
    def __init__(self, model_identifier, auxiliary_outputs=5, dropout=0.1, freeze_layers=0, revision=None):
        super().__init__(model_identifier, dropout, freeze_layers, revision)
        self.aux_head = nn.Linear(int(self.bert.config.hidden_size), int(auxiliary_outputs))

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.pool(output.last_hidden_state, attention_mask)
        return self.head(self.drop(pooled)).squeeze(-1), self.aux_head(self.drop(pooled))


def build_chemberta_model(config: ChemBertaVariant):
    if config.task == "logS_plus_auxiliary":
        return ChemBertaMultiTaskRegressor(
            config.model_identifier,
            len(config.auxiliary_targets),
            config.dropout,
            config.freeze_layers,
            config.model_revision,
        )
    return ChemBertaRegressor(config.model_identifier, config.dropout, config.freeze_layers, config.model_revision)


def load_chemberta_tokenizer(config: ChemBertaVariant):
    return AutoTokenizer.from_pretrained(config.model_identifier, revision=config.model_revision)


def extract_embeddings(model, dataset, batch_size, device):
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False)
    model.to(device)
    model.eval()
    parts = []
    with torch.no_grad():
        for batch in loader:
            parts.append(model.embed(batch["input_ids"].to(device), batch["attention_mask"].to(device)).cpu().numpy())
    result = np.concatenate(parts, axis=0).astype(np.float32)
    if result.ndim != 2 or result.shape[0] != len(dataset) or not np.isfinite(result).all():
        raise ValueError("ChemBERTa embedding extraction failed its shape or finite-value contract")
    return result
