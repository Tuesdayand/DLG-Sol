from __future__ import annotations

import torch
import torch.nn as nn


class PlainMLP(nn.Module):
    def __init__(self, input_dimension, hidden_dimension, depth, dropout):
        super().__init__()
        layers = []
        dimension = int(input_dimension)
        for _ in range(int(depth)):
            layers.extend([nn.Linear(dimension, int(hidden_dimension)), nn.ReLU(), nn.Dropout(float(dropout))])
            dimension = int(hidden_dimension)
        layers.append(nn.Linear(dimension, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, values):
        return self.net(values).squeeze(-1)


class ResidualMLP(nn.Module):
    def __init__(self, input_dimension, hidden_dimension, depth, dropout):
        super().__init__()
        hidden = int(hidden_dimension)
        self.inp = nn.Linear(int(input_dimension), hidden)
        self.blocks = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(float(dropout)), nn.Linear(hidden, hidden)) for _ in range(int(depth))]
        )
        self.out = nn.Linear(hidden, 1)

    def forward(self, values):
        hidden = torch.relu(self.inp(values))
        for block in self.blocks:
            hidden = torch.relu(hidden + block(hidden))
        return self.out(hidden).squeeze(-1)


def build_fusion_model(regressor, input_dimension, parameters):
    values = (input_dimension, parameters["hidden"], parameters["depth"], parameters["dropout"])
    if regressor == "plain_mlp":
        return PlainMLP(*values)
    if regressor == "residual_mlp":
        return ResidualMLP(*values)
    raise ValueError("unsupported fusion regressor")


def load_fusion_checkpoint(path, regressor, input_dimension, parameters, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model = build_fusion_model(regressor, input_dimension, parameters)
    model.load_state_dict(state, strict=True)
    return model.to(device)
