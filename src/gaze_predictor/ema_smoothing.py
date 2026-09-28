"""Causal one-sided EMA smoothing for gaze predictor outputs.

The predictor emits one (x, y) estimate per frame. A causal (one-sided) EMA
over this stream smooths frame-to-frame variation without look-ahead, so it
runs online at deployment (alpha=0.2 in the paper).

Update rule:
    smoothed[0] = pred[0]
    smoothed[t] = alpha * pred[t] + (1 - alpha) * smoothed[t - 1]   for t >= 1

alpha = 1 -> no smoothing; alpha -> 0 -> infinite-memory smoothing (no update).
"""
from __future__ import annotations

import numpy as np


def ema_smooth(pred_seq: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """Causal EMA over the leading time axis.

    Args:
        pred_seq: float array, shape ``[T, ...]``. For our use, typically
            ``[T, n_anchors, 2]`` of (x, y) in 2160-px space.
        alpha: EMA coefficient in (0, 1]. ``alpha=1`` -> no smoothing.

    Returns:
        Smoothed array with the same shape and dtype as ``pred_seq``.
    """
    if pred_seq.ndim < 1 or pred_seq.shape[0] == 0:
        return pred_seq.copy()
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1]; got {alpha}")
    smoothed = pred_seq.astype(np.float32, copy=True)
    for t in range(1, len(pred_seq)):
        smoothed[t] = alpha * pred_seq[t] + (1.0 - alpha) * smoothed[t - 1]
    return smoothed
