"""Vision-feature pre-caching for the K-frame temporal-context head.

The base ``cache.py`` only caches the input frame ``t`` for each sample plan.
The ``prompt_then_temporal_224`` head with K=3 needs ``[t-2, t-1, t]`` per
sample, so we need cache entries at ``t-1, t-2`` as well. Frames outside the
cache are encoded live (frozen vision encoder, batched on GPU) and saved as
``.npz`` files matching the existing layout
(``vis/<dataset>__ep_NNNNNN__f_NNNNNN.npz``).

Edge handling: when ``t < K - 1``, neighbors ``t - i`` for ``i > t`` get clamped
to frame 0.
"""
from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .data import (
    SamplePlan, preprocess_image_224,
)


def _vis_cache_path(cache_dir: str, dataset_path: str,
                    episode_index: int, t: int) -> str:
    ds_name = os.path.basename(dataset_path.rstrip("/"))
    return os.path.join(cache_dir, "vis",
                        "{}__ep_{:06d}__f_{:06d}.npz".format(
                            ds_name, episode_index, t))


def neighbor_frames_for_plan(p: SamplePlan, K: int) -> List[int]:
    """Indices ``[max(0, t-K+1), ..., t-1, t]`` (length K, clamped at 0)."""
    out = []
    for i in range(K - 1, -1, -1):
        out.append(max(0, p.t - i))
    return out


def collect_required_frames(plans: Sequence[SamplePlan], K: int
                             ) -> List[Tuple[str, int, int]]:
    """Returns the unique ``(dataset_path, episode_index, t)`` triples that the
    K-frame temporal head needs (current + K-1 past neighbors, clamped at 0)."""
    required: set = set()
    for p in plans:
        for f in neighbor_frames_for_plan(p, K):
            required.add((p.dataset_path, int(p.episode_index), int(f)))
    return sorted(required)


@torch.no_grad()
def ensure_temporal_vision_cache(model, plans: Sequence[SamplePlan],
                                  cfg: dict, K: int,
                                  device: str = "cuda",
                                  batch_size: int = 32,
                                  desc: str = "temporal-vis-cache") -> Dict[str, int]:
    """Encode + save any (dataset, ep, t) frames missing from the cache.

    Groups missing frames by (dataset, episode) and batch-decodes all needed
    timesteps from each video via decord ``get_batch`` to avoid per-frame
    random seeks."""
    import decord
    from collections import defaultdict
    from .data import video_path
    cache_dir = cfg["cache_dir"]
    os.makedirs(os.path.join(cache_dir, "vis"), exist_ok=True)
    required = collect_required_frames(plans, K)
    by_ep: Dict[Tuple[str, int], List[Tuple[int, str]]] = defaultdict(list)
    for ds, ep, t in required:
        pth = _vis_cache_path(cache_dir, ds, ep, t)
        if not os.path.exists(pth):
            by_ep[(ds, ep)].append((t, pth))
    n_missing = sum(len(v) for v in by_ep.values())
    if not by_ep:
        return {"required": len(required), "missing_before": 0, "encoded": 0}

    mean, std = cfg["clip_mean"], cfg["clip_std"]
    model = model.to(device).eval()

    pbar = tqdm(total=n_missing, desc=desc)
    encoded = 0
    for (ds, ep), entries in by_ep.items():
        entries.sort()
        ts = [t for t, _ in entries]
        outs = [out for _, out in entries]
        vp = video_path(ds, ep)
        vr = decord.VideoReader(vp, num_threads=4)
        frames = vr.get_batch(ts).asnumpy()  # [N, H, W, 3] uint8 RGB
        if frames.shape[2] >= 4320:
            frames = frames[:, :, :2160, :]
        elif frames.shape[2] >= 2160 and frames.shape[1] == 2160:
            frames = frames[:, :, :2160, :]
        del vr
        for i in range(0, len(frames), batch_size):
            chunk_frames = frames[i:i + batch_size]
            chunk_outs = outs[i:i + batch_size]
            images = [preprocess_image_224(f, mean, std) for f in chunk_frames]
            encoded += _flush_batch(model, images, chunk_outs, device)
            pbar.update(len(images))
    pbar.close()
    return {"required": len(required),
            "missing_before": n_missing,
            "encoded": encoded}


def _flush_batch(model, images: List[torch.Tensor], paths: List[str],
                 device: str) -> int:
    batch = torch.stack(images).to(device)
    feats = model.encode_image_raw(batch).cpu().numpy()           # [B, 196, D]
    for f, p in zip(feats, paths):
        np.savez(p, vis_feat=f.astype(np.float16))
    return len(paths)


def load_temporal_vis_stack(plan: SamplePlan, cache_dir: str, K: int
                             ) -> np.ndarray:
    """Returns ``[K, P, D]`` raw vision features for the K-frame stack of
    ``plan`` (clamped at 0 for early frames). All K paths must be present."""
    frames = neighbor_frames_for_plan(plan, K)
    feats = []
    for f in frames:
        pth = _vis_cache_path(cache_dir, plan.dataset_path,
                              int(plan.episode_index), int(f))
        feats.append(np.load(pth)["vis_feat"])
    return np.stack(feats, axis=0)                                 # [K, P, D]
