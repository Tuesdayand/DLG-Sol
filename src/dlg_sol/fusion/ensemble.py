from __future__ import annotations

import numpy as np


def mean_member_predictions(predictions, expected_members=4):
    matrix = np.asarray(predictions, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != int(expected_members) or len(matrix) == 0:
        raise ValueError("member predictions have an incompatible shape")
    if not np.isfinite(matrix).all():
        raise ValueError("member predictions must be finite")
    return np.mean(matrix, axis=1, dtype=np.float32)
