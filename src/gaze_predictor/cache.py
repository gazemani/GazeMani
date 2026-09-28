"""Pre-compute frozen-encoder features for the gaze predictor.

We cache:
  * vision: per-(dataset, episode, t) the post-encoder pre-projection token grid.
  * text:   per unique prompt the post-encoder pre-projection sequence.

Files saved as .npz; kept "raw" (i.e., before the trainable proj).
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .data import (
    SamplePlan,
    load_episodes_meta,
    make_sample_plan,
    preprocess_image_224,
    split_episodes_by_ratio,
    video_path,
)
from .model import GazeTrajectoryPredictor
from .utils import ensure_dir, load_yaml_config, slug_for_prompt


def _vis_path(cache_dir: str, dataset_path: str, episode_index: int, t: int) -> str:
    ds_name = os.path.basename(dataset_path.rstrip("/"))
    return os.path.join(cache_dir, "vis",
                        "{}__ep_{:06d}__f_{:06d}.npz".format(ds_name, episode_index, t))


def _text_path(cache_dir: str, prompt: str) -> str:
    return os.path.join(cache_dir, "text",
                        "{}.npz".format(slug_for_prompt(prompt)))


def cache_vision(model: GazeTrajectoryPredictor, plans: List[SamplePlan],
                 cfg: dict, cache_dir: str, device: str = "cuda"):
    """Encode + save vision features for every unique (dataset, episode, t)
    in ``plans``. Groups by (dataset, episode) and batch-decodes all needed
    frames from each video via decord ``get_batch`` — far faster than
    per-frame random seeks."""
    import decord
    from collections import defaultdict
    ensure_dir(os.path.join(cache_dir, "vis"))
    mean, std = cfg["clip_mean"], cfg["clip_std"]
    # Deduplicate (dataset_path, ep, t); only the input frame t is encoded here
    # (past temporal neighbors are cached by temporal_cache.py).
    unique_keys = set((p.dataset_path, p.episode_index, p.t) for p in plans)
    todo: Dict[Tuple[str, int], List[Tuple[int, str]]] = defaultdict(list)
    n_total_unique = len(unique_keys)
    n_new = 0
    for ds, ep, t in unique_keys:
        out = _vis_path(cache_dir, ds, ep, t)
        if not os.path.exists(out):
            todo[(ds, ep)].append((t, out))
            n_new += 1
    print("vision cache: {} unique frames, {} new ({} episodes to visit)".format(
        n_total_unique, n_new, len(todo)))
    model = model.to(device).eval()
    gpu_bs = 32
    pbar = tqdm(total=n_new, desc="vision")
    for (ds, ep), entries in todo.items():
        entries.sort()  # by t
        ts = [t for t, _ in entries]
        outs = [out for _, out in entries]
        vp = video_path(ds, ep)
        # Local reader with multi-threaded decode (avoid the shared cache's
        # num_threads=1).
        vr = decord.VideoReader(vp, num_threads=4)
        # Batch-decode all needed frames at once. decord returns frames in the
        # order requested.
        frames = vr.get_batch(ts).asnumpy()  # [N, H, W, 3] uint8 RGB
        # Crop left half (matches read_frame_left_crop semantics).
        if frames.shape[2] >= 4320:
            frames = frames[:, :, :2160, :]
        elif frames.shape[2] >= 2160 and frames.shape[1] == 2160:
            frames = frames[:, :, :2160, :]
        del vr  # release decoder
        # GPU encode in chunks of gpu_bs.
        for i in range(0, len(frames), gpu_bs):
            chunk_frames = frames[i:i + gpu_bs]
            chunk_outs = outs[i:i + gpu_bs]
            images = [preprocess_image_224(f, mean, std) for f in chunk_frames]
            _flush_vis(model, images, chunk_outs, device)
            pbar.update(len(images))
    pbar.close()


def _flush_vis(model, images, paths, device):
    batch = torch.stack(images).to(device)
    with torch.no_grad():
        feats = model.encode_image_raw(batch).cpu().numpy()  # [B, 196, D]
    for f, p in zip(feats, paths):
        np.savez(p, vis_feat=f.astype(np.float16))


def cache_text(model: GazeTrajectoryPredictor, prompts: List[str],
               cache_dir: str, device: str = "cuda") -> Dict[str, str]:
    ensure_dir(os.path.join(cache_dir, "text"))
    model = model.to(device).eval()
    out_paths = {}
    for prompt in tqdm(sorted(set(prompts)), desc="text"):
        outp = _text_path(cache_dir, prompt)
        out_paths[prompt] = outp
        if os.path.exists(outp):
            continue
        with torch.no_grad():
            raw, mask = model.encode_text_raw([prompt])
        np.savez(outp,
                 text_feat=raw[0].cpu().numpy().astype(np.float32),
                 text_mask=mask[0].cpu().numpy().astype(np.bool_),
                 prompt=prompt)
    return out_paths


def _as_dataset_list(x) -> List[str]:
    """Normalize a ``train_dataset`` field that may be either a single path
    string or a list of paths into a list of strings. ``None``/empty returns
    an empty list (the caller decides whether that's an error)."""
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return list(x)


def _merge_stats(stats_list: List[Dict]) -> Dict:
    """Sum-aggregate per-dataset stats so the upstream fallback-rate guard
    still sees a single rate across the joint dataset."""
    out = {
        "fallback_frames": 0,
        "total_frames": 0,
        "n_kept": 0,
        "n_total": 0,
    }
    for s in stats_list:
        out["fallback_frames"] += int(s.get("fallback_frames", 0))
        out["total_frames"] += int(s.get("total_frames", 0))
        out["n_kept"] += int(s.get("n_kept", 0))
        out["n_total"] += int(s.get("n_total", 0))
    out["fallback_rate"] = out["fallback_frames"] / max(out["total_frames"], 1)
    return out


def build_plans_for_split(cfg: dict, seed_offset: int = 0):
    """Build SamplePlan objects for {train, val, test} splits.

    ``cfg["train_dataset"]`` may be either:
      - a single dataset path (string), or
      - a list of dataset paths (joint training).

    For list inputs:
      - Each dataset is loaded separately, an 80/20 episode-level stratified
        split is applied per-dataset (so each contributes proportionally to
        train and val), and the resulting plans are concatenated.
      - Cache file paths use ``os.path.basename(dataset_path)`` so distinct
        datasets must have distinct basenames; we error out otherwise.

    ``cfg["test_dataset"]`` may also be a list (in which case the test plans
    are concatenated) or None (no held-out test).
    """
    train_paths = _as_dataset_list(cfg["train_dataset"])
    if not train_paths:
        raise ValueError("cfg['train_dataset'] is empty")

    # Sanity: distinct basenames so vis-cache files don't collide.
    bases = [os.path.basename(p.rstrip("/")) for p in train_paths]
    if len(set(bases)) != len(bases):
        raise ValueError(
            "[cache] train_dataset paths have colliding basenames "
            f"({bases}); rename one or use distinct dirs so vis-cache "
            "files don't overwrite each other.")

    ep_filter = cfg.get("train_episode_filter", None)
    if ep_filter is not None and len(train_paths) > 1:
        # The semantics of an episode-filter across multiple source datasets
        # are ambiguous (which dataset's episode 5?). Reject explicitly.
        raise ValueError(
            "[cache] train_episode_filter is only supported for "
            "single-dataset training.")

    import random as _rnd
    fb = tuple(cfg.get("gaze_fallback_value", [0, 0]))

    # samples_per_episode may be a single int (apply to all datasets) or a
    # list/tuple aligned with train_paths for per-dataset frame-balancing.
    spe_cfg = cfg["samples_per_episode"]
    if isinstance(spe_cfg, (list, tuple)):
        if len(spe_cfg) != len(train_paths):
            raise ValueError(
                f"[cache] samples_per_episode list length {len(spe_cfg)} != "
                f"train_dataset count {len(train_paths)}")
        spe_per_ds = [int(x) for x in spe_cfg]
    else:
        spe_per_ds = [int(spe_cfg)] * len(train_paths)

    # Per-dataset 80/20 split + plan build for train/val.
    plans_train: List[SamplePlan] = []
    plans_val: List[SamplePlan] = []
    train_eps_all_concat: List[Dict] = []
    val_eps_all_concat: List[Dict] = []
    train_stats_list: List[Dict] = []
    val_stats_list: List[Dict] = []

    for di, ds_path in enumerate(train_paths):
        train_eps_all = load_episodes_meta(ds_path)
        if ep_filter is not None:
            ep_filter_set = set(int(x) for x in ep_filter)
            train_eps_all = [e for e in train_eps_all
                              if int(e["episode_index"]) in ep_filter_set]
            print("[cache] train_episode_filter applied: {} episodes kept "
                  "(filter size {})".format(len(train_eps_all),
                                             len(ep_filter_set)))
        train_eps, val_eps = split_episodes_by_ratio(
            train_eps_all, cfg["val_split_ratio"],
            seed=cfg["seed"] + 1009 * di)
        rng_train = _rnd.Random(cfg["seed"] + 1 + seed_offset + 9001 * di)
        rng_val = _rnd.Random(cfg["seed"] + 2 + seed_offset + 9001 * di)
        p_tr, st_tr = make_sample_plan(
            ds_path, train_eps,
            anchor_offsets=cfg["anchor_offsets"],
            samples_per_episode=spe_per_ds[di],
            min_episode_length=cfg["min_episode_length"],
            gaze_fallback_value=fb,
            rng=rng_train,
        )
        p_va, st_va = make_sample_plan(
            ds_path, val_eps,
            anchor_offsets=cfg["anchor_offsets"],
            samples_per_episode=spe_per_ds[di],
            min_episode_length=cfg["min_episode_length"],
            gaze_fallback_value=fb,
            rng=rng_val,
        )
        plans_train.extend(p_tr)
        plans_val.extend(p_va)
        train_eps_all_concat.extend(train_eps)
        val_eps_all_concat.extend(val_eps)
        train_stats_list.append(st_tr)
        val_stats_list.append(st_va)
        if len(train_paths) > 1:
            print("[cache] dataset {}: {} (train_eps={} val_eps={} "
                  "n_train_kept={} n_val_kept={} fb_rate={:.4f})".format(
                      di, ds_path, len(train_eps), len(val_eps),
                      st_tr["n_kept"], st_va["n_kept"],
                      st_tr["fallback_rate"]))

    st_train = _merge_stats(train_stats_list)
    st_val = _merge_stats(val_stats_list)

    # Test split (optional; may also be a list).
    test_paths = _as_dataset_list(cfg.get("test_dataset"))
    # For test, use a scalar samples_per_episode (frame-balancing not needed
    # for periodic eval). If the train-side cfg is a list, fall back to 8.
    spe_test = 8 if isinstance(spe_cfg, (list, tuple)) else int(spe_cfg)
    plans_test: List[SamplePlan] = []
    test_eps_concat: List[Dict] = []
    test_stats_list: List[Dict] = []
    for di, ds_path in enumerate(test_paths):
        test_eps = load_episodes_meta(ds_path)
        rng_test = _rnd.Random(cfg["seed"] + 3 + seed_offset + 9001 * di)
        p_te, st_te = make_sample_plan(
            ds_path, test_eps,
            anchor_offsets=cfg["anchor_offsets"],
            samples_per_episode=spe_test,
            min_episode_length=cfg["min_episode_length"],
            gaze_fallback_value=fb,
            rng=rng_test,
        )
        plans_test.extend(p_te)
        test_eps_concat.extend(test_eps)
        test_stats_list.append(st_te)
    if test_stats_list:
        st_test = _merge_stats(test_stats_list)
    else:
        # No test set: empty plans + zero stats so downstream code doesn't break.
        st_test = {"fallback_frames": 0, "total_frames": 0,
                   "n_kept": 0, "n_total": 0, "fallback_rate": 0.0}

    return {"train": (plans_train, st_train, train_eps_all_concat),
            "val":   (plans_val,   st_val,   val_eps_all_concat),
            "test":  (plans_test,  st_test,  test_eps_concat)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = load_yaml_config(args.config)
    print("[cache] config:", args.config)
    cache_dir = cfg["cache_dir"]
    ensure_dir(cache_dir)

    train_paths = _as_dataset_list(cfg["train_dataset"])
    test_paths = _as_dataset_list(cfg.get("test_dataset"))
    print("[cache] train_dataset(s): {}".format(train_paths))
    print("[cache] test_dataset(s):  {}".format(test_paths))

    model = GazeTrajectoryPredictor(
        encoder_kind=cfg["encoder_kind"],
        fusion_dim=cfg["fusion_dim"],
        n_anchors=cfg["n_anchors"],
        grid=cfg["grid"],
        clip_model_id=cfg["clip_model_id"],
        head_kind=cfg.get("head_kind"),
    )

    plans = build_plans_for_split(cfg)
    print("plans: train={} val={} test={}".format(
        len(plans["train"][0]), len(plans["val"][0]), len(plans["test"][0])))
    print("fallback rates: train={:.2%} val={:.2%} test={:.2%}".format(
        plans["train"][1]["fallback_rate"],
        plans["val"][1]["fallback_rate"],
        plans["test"][1]["fallback_rate"]))

    # Vision cache for all splits
    all_plans = plans["train"][0] + plans["val"][0] + plans["test"][0]
    t0 = time.time()
    cache_vision(model, all_plans, cfg, cache_dir, device=args.device)
    print("vision cache done in", time.time() - t0, "s")

    # Text cache: every unique prompt across train+val+test.
    prompts = set(p.prompt for p in all_plans)
    cache_text(model, list(prompts), cache_dir, device=args.device)
    print("text cache done")


if __name__ == "__main__":
    main()
