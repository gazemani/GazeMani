"""Gaze KL auxiliary loss helpers.

JAX port of ``gaze_predictor.utils.gaze_to_heatmap_14x14`` (same Gaussian
rasterization, configurable grid; 16x16 here for SigLIP). Used to
turn current-frame gaze coordinates (2160 px left-crop space) into a
patch-level probability distribution that lines up with the SigLIP
visual token grid, then compute KL(G || S) between this gaze prior and
the model's language-to-vision attention.
"""
from __future__ import annotations

import jax.numpy as jnp


def gaze_to_patch_distribution(
    gaze_xy: jnp.ndarray,
    grid: int = 16,
    sigma_in_grid: float = 1.0,
    image_size: int = 2160,
) -> jnp.ndarray:
    """Convert ``(x, y)`` pixel coords to a normalized ``GxG`` patch distribution.

    Args:
        gaze_xy: ``[..., 2]`` array, gaze in pixel coords (x first, then y).
        grid: ``G`` along each side. ``16`` matches SigLIP So400m/14 @ 224.
        sigma_in_grid: Gaussian sigma in cells (cell width = ``image_size / grid``).
        image_size: pixel extent the gaze coordinates are reported in.

    Returns:
        ``[..., grid * grid]`` distribution that sums to 1 over the last axis.
    """
    cell_size = image_size / grid
    centers = (jnp.arange(grid, dtype=jnp.float32) + 0.5) * cell_size
    cy_grid, cx_grid = jnp.meshgrid(centers, centers, indexing="ij")
    sigma_px = sigma_in_grid * cell_size

    gx = gaze_xy[..., 0:1, None]  # [..., 1, 1]
    gy = gaze_xy[..., 1:2, None]
    cy_grid = jnp.broadcast_to(cy_grid, gx.shape[:-2] + (grid, grid))
    cx_grid = jnp.broadcast_to(cx_grid, gx.shape[:-2] + (grid, grid))

    dist_sq = (cx_grid - gx) ** 2 + (cy_grid - gy) ** 2
    heatmap = jnp.exp(-dist_sq / (2.0 * sigma_px ** 2))  # [..., G, G]
    flat = heatmap.reshape(*heatmap.shape[:-2], grid * grid)
    return flat / flat.sum(axis=-1, keepdims=True).clip(min=1e-12)


def kl_div(target: jnp.ndarray, pred: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """``sum target * (log target - log pred)`` over the last axis, then mean.

    Args:
        target: ``[..., N]`` reference distribution (e.g. gaze prior).
        pred: ``[..., N]`` model distribution. Must sum to 1.
    """
    t = target.clip(min=eps)
    p = pred.clip(min=eps)
    per_sample = jnp.sum(t * (jnp.log(t) - jnp.log(p)), axis=-1)
    return jnp.mean(per_sample)
