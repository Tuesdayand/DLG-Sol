from __future__ import annotations

import numpy as np


def masked_mean_pool(hidden_states: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    hidden = np.asarray(hidden_states)
    mask = np.asarray(attention_mask)
    if hidden.ndim != 3 or mask.ndim != 2 or hidden.shape[:2] != mask.shape:
        raise ValueError("hidden states and attention mask have incompatible shapes")
    if not np.isfinite(hidden).all() or not np.isfinite(mask).all():
        raise ValueError("pooling inputs must be finite")
    weights = mask[..., None].astype(hidden.dtype, copy=False)
    denominator = weights.sum(axis=1)
    if np.any(denominator <= 0):
        raise ValueError("every sequence must contain at least one unmasked token")
    return (hidden * weights).sum(axis=1) / denominator
