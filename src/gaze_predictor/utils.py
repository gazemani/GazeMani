"""Shared helpers for gaze_predictor.

    - set_seed
    - gaze_to_heatmap_14x14   (used by training CE loss)
    - soft_argmax_2d           (used by --readout=global)
    - windowed_soft_argmax     (used by deployment + --readout=windowed)
    - load_yaml_config
    - slug_for_prompt          (used by cache.py)
    - ensure_dir
"""
from __future__ import annotations

import os
import random
import re

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gaze_to_heatmap_14x14(gaze_xy_2160: torch.Tensor,
                          image_size: int = 2160,
                          grid: int = 14,
                          sigma_in_grid: float = 1.0) -> torch.Tensor:
    """Convert gaze (x,y) in image_size px space to a normalized GxG heatmap.

    Works for any square ``grid`` (224 for the released output head).

    gaze_xy_2160: [..., 2] tensor (or [2]) in pixel coords (x first, then y).
    Returns heatmap of shape gaze.shape[:-1] + (grid, grid) summing to 1 along (-2,-1).
    """
    cell_size = image_size / grid
    cy = (torch.arange(grid, dtype=torch.float32, device=gaze_xy_2160.device) + 0.5) * cell_size
    cx = (torch.arange(grid, dtype=torch.float32, device=gaze_xy_2160.device) + 0.5) * cell_size
    cy_grid, cx_grid = torch.meshgrid(cy, cx, indexing='ij')   # [G,G]
    sigma_px = sigma_in_grid * cell_size
    gx = gaze_xy_2160[..., 0:1].unsqueeze(-1)   # [..., 1, 1]
    gy = gaze_xy_2160[..., 1:2].unsqueeze(-1)
    cy_grid = cy_grid.expand(*gx.shape[:-2], grid, grid)
    cx_grid = cx_grid.expand(*gx.shape[:-2], grid, grid)
    dist_sq = (cx_grid - gx) ** 2 + (cy_grid - gy) ** 2
    heatmap = torch.exp(-dist_sq / (2.0 * sigma_px ** 2))
    s = heatmap.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    return heatmap / s


def soft_argmax_2d(prob: torch.Tensor, image_size: int = 2160) -> torch.Tensor:
    """Fully-differentiable global expected-value soft-argmax over a 2D grid.

    prob: [..., G, G] tensor that sums to 1 over the last two dims.
    Returns: [..., 2] tensor (x, y) in pixel space (image_size scale).
    """
    G = prob.shape[-1]
    cell_size = image_size / G
    coords = (torch.arange(G, device=prob.device, dtype=prob.dtype) + 0.5) * cell_size
    px = (prob.sum(dim=-2) * coords).sum(-1)
    py = (prob.sum(dim=-1) * coords).sum(-1)
    return torch.stack([px, py], dim=-1)


def windowed_soft_argmax(heatmap: torch.Tensor,
                         image_size: int = 2160,
                         k: int = 3) -> torch.Tensor:
    """Take argmax cell, then a kxk window soft-argmax around it. Returns (x,y) in pixels.

    heatmap: [..., G, G] either probabilities or logits (we soft-max if not summing to 1).
    """
    G = heatmap.shape[-1]
    cell_size = image_size / G
    h = heatmap
    # If not normalized, treat as logits
    if not torch.all(torch.isfinite(h)):
        h = torch.where(torch.isfinite(h), h, torch.full_like(h, float('-inf')))
    s = h.sum(dim=(-2, -1), keepdim=True)
    if (s > 1.5).any() or (s < 0.5).any():
        h = h.flatten(-2).softmax(-1).reshape(*h.shape)

    flat = h.flatten(-2)
    am = flat.argmax(-1)              # [...]
    cy = am // G
    cx = am %  G

    half = k // 2
    out = torch.zeros(*h.shape[:-2], 2, dtype=torch.float32, device=h.device)
    for cy_ in torch.unique(cy):
        for cx_ in torch.unique(cx):
            mask = (cy == cy_) & (cx == cx_)
            if not mask.any():
                continue
            y0 = max(int(cy_) - half, 0); y1 = min(int(cy_) + half + 1, G)
            x0 = max(int(cx_) - half, 0); x1 = min(int(cx_) + half + 1, G)
            sub = h[..., y0:y1, x0:x1][mask]   # [N, dy, dx]
            yy = torch.arange(y0, y1, dtype=torch.float32, device=h.device).view(1, -1, 1)
            xx = torch.arange(x0, x1, dtype=torch.float32, device=h.device).view(1, 1, -1)
            wsum = sub.sum(dim=(-2, -1)).clamp_min(1e-12)
            yc = (sub * yy).sum(dim=(-2, -1)) / wsum
            xc = (sub * xx).sum(dim=(-2, -1)) / wsum
            xy = torch.stack([(xc + 0.5) * cell_size, (yc + 0.5) * cell_size], dim=-1)
            out[mask] = xy
    return out


def load_yaml_config(path: str) -> dict:
    import yaml
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def slug_for_prompt(prompt: str) -> str:
    s = re.sub(r'[^a-zA-Z0-9]+', '_', prompt.strip().lower()).strip('_')
    return s[:80]


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p
