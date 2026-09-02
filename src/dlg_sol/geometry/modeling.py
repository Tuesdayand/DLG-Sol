from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool


class DistanceAwareMessagePassingLayer(MessagePassing):
    def __init__(self, hidden_dim, edge_dim, dropout):
        super().__init__(aggr="add")
        self.msg = nn.Sequential(
            nn.Linear(int(hidden_dim) * 2 + int(edge_dim), int(hidden_dim) * 2),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim) * 2, int(hidden_dim)),
            nn.SiLU(),
        )
        self.upd = nn.Sequential(
            nn.Linear(int(hidden_dim) * 2, int(hidden_dim) * 2),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim) * 2, int(hidden_dim)),
        )
        self.norm = nn.LayerNorm(int(hidden_dim))

    def forward(self, x, edge_index, edge_attr):
        messages = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.norm(x + self.upd(torch.cat([x, messages], dim=-1)))

    def message(self, x_i, x_j, edge_attr):
        return self.msg(torch.cat([x_i, x_j, edge_attr], dim=-1))


class DistanceAware3DMPNN(nn.Module):
    def __init__(self, node_dim, edge_dim, aux_dim, hidden_dim, num_layers, dropout, pool):
        super().__init__()
        if pool not in {"mean", "mean_aux"}:
            raise ValueError("pool must be mean or mean_aux")
        self.pool = pool
        self.node_encoder = nn.Sequential(nn.Linear(int(node_dim), int(hidden_dim)), nn.SiLU(), nn.LayerNorm(int(hidden_dim)))
        self.edge_encoder = nn.Sequential(nn.Linear(int(edge_dim), int(hidden_dim)), nn.SiLU(), nn.LayerNorm(int(hidden_dim)))
        self.layers = nn.ModuleList([DistanceAwareMessagePassingLayer(hidden_dim, hidden_dim, dropout) for _ in range(int(num_layers))])
        self.embedding_head = nn.Sequential(nn.Linear(int(hidden_dim), int(hidden_dim)), nn.SiLU(), nn.Dropout(float(dropout)), nn.Linear(int(hidden_dim), int(hidden_dim)))
        prediction_dim = int(hidden_dim) + (int(aux_dim) if pool == "mean_aux" else 0)
        self.regression_head = nn.Sequential(nn.Linear(prediction_dim, int(hidden_dim)), nn.SiLU(), nn.Dropout(float(dropout)), nn.Linear(int(hidden_dim), 1))

    def encode_graph(self, data):
        nodes = self.node_encoder(data.x)
        edges = self.edge_encoder(data.edge_attr)
        for layer in self.layers:
            nodes = F.silu(layer(nodes, data.edge_index, edges))
        return self.embedding_head(global_mean_pool(nodes, data.batch))

    def forward_from_embedding(self, embedding, auxiliary=None):
        if self.pool == "mean_aux":
            if auxiliary is None:
                raise ValueError("auxiliary features are required for mean_aux")
            embedding = torch.cat([embedding, auxiliary], dim=-1)
        return self.regression_head(embedding).view(-1)

    def forward(self, data):
        embedding = self.encode_graph(data)
        auxiliary = data.aux if hasattr(data, "aux") else None
        return self.forward_from_embedding(embedding, auxiliary)


def load_geometry_checkpoint(path, device="cpu", parameters=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    resolved = checkpoint.get("params") if parameters is None else parameters
    if resolved is None:
        raise ValueError("checkpoint parameters must be supplied for legacy checkpoints that did not store them")
    model = DistanceAware3DMPNN(
        checkpoint["node_dim"],
        checkpoint["edge_dim"],
        checkpoint["aux_dim"],
        resolved["hidden_dim"],
        resolved["num_layers"],
        resolved["dropout"],
        resolved["pool"],
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device), dict(resolved)
