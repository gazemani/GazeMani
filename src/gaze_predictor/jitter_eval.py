"""Frame-to-frame jitter evaluation for the gaze predictor.

Measures the frame-to-frame stability of the predicted gaze over dense
consecutive frames.

For each valid t in [0, ep_len - max_offset - 1]:
  pred_xy(t, i) = windowed_soft_argmax(model(frames[t-K+1..t], prompt))[i]

  Δ_model(t, i)  = ||pred_xy(t+1, i) - pred_xy(t, i)||  (px in 2160 space)
  Δ_gt(t, i)     = ||gt_xy(T+1) - gt_xy(T)||,  T = t + offsets[i]

If Δ_model ~ Δ_gt, the model just tracks GT motion. If Δ_model >> Δ_gt, the
model is adding jitter on top.

Usage:
    python -m gaze_predictor.jitter_eval \
        --config src/gaze_predictor/configs/gaze_predictor_six_task.yaml \
        --max_frames 5000

Writes ``<out_dir>/jitter_eval.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch

from .data import (
    load_episodes_meta,
    preprocess_image_224,
    read_episode_gaze,
    read_frame_left_crop,
)
from .model import GazeTrajectoryPredictor
from .train_prompts import (
    InMemTextStore,
    encode_prompts_into_store,
)
from .utils import (
    ensure_dir, load_yaml_config, set_seed, soft_argmax_2d,
    windowed_soft_argmax,
)


# ---------------------------------------------------------------------------
# Prompt resolution per (episode, t): the episode's task instruction
# (prompt_mode="default").
# ---------------------------------------------------------------------------

def build_prompt_resolver(cfg: dict):
    mode = cfg.get("prompt_mode", "default")
    if mode == "default":
        def _resolve(ep_prompt: str, ep_idx: int, t: int) -> str:
            return ep_prompt
        return _resolve, ["__identity__"]
    raise ValueError(
        f"unsupported prompt_mode={mode!r}; only 'default' is supported.")


# ---------------------------------------------------------------------------
# Vision encoder live forward (no cache for dense frames)
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_vision_batch(model: GazeTrajectoryPredictor,
                        ds_path: str, ep_idx: int, t_list: List[int],
                        cfg: dict, device: str,
                        batch_size: int = 32) -> np.ndarray:
    """Returns [N, 196, D] vision features for the given (ep, t) frames."""
    mean, std = cfg["clip_mean"], cfg["clip_std"]
    feats = []
    for i0 in range(0, len(t_list), batch_size):
        chunk = t_list[i0:i0 + batch_size]
        imgs = []
        for t in chunk:
            frame = read_frame_left_crop(ds_path, ep_idx, t)
            imgs.append(preprocess_image_224(frame, mean, std))
        x = torch.stack(imgs).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            f = model.encode_image_raw(x)         # [B, 196, D]
        feats.append(f.float().cpu().numpy())
    return np.concatenate(feats, axis=0)


# ---------------------------------------------------------------------------
# Main eval
# ---------------------------------------------------------------------------

def _build_temporal_stacks(vis_raw: np.ndarray, K: int) -> np.ndarray:
    """vis_raw: [N, P, D] dense per-frame features (frame 0 .. frame N-1).
    Returns [N, K, P, D] where row k = stack of frames [max(0,k-K+1), ..., k].
    """
    N, P, D = vis_raw.shape
    out = np.empty((N, K, P, D), dtype=vis_raw.dtype)
    for k in range(N):
        for j in range(K):
            src = max(0, k - (K - 1 - j))
            out[k, j] = vis_raw[src]
    return out


@torch.no_grad()
def run_episode_predictions(model: GazeTrajectoryPredictor,
                            ds_path: str, ep_idx: int, ep_prompt: str,
                            ep_len: int, cfg: dict, device: str,
                            text_store: InMemTextStore,
                            resolver,
                            anchor_offsets: List[int],
                            image_size: int,
                            n_anchors: int,
                            readout: str = "windowed") -> Tuple[np.ndarray, np.ndarray]:
    """Run inference at every valid t in [0, ep_len - max_offset - 1]."""
    horizon = max(anchor_offsets) + 1
    max_t = ep_len - horizon
    if max_t < 0:
        return np.zeros((0, n_anchors, 2), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    t_list = list(range(0, max_t + 1))

    is_temporal = bool(getattr(model, "is_temporal", False))
    K = int(getattr(model, "temporal_K", 1)) if is_temporal else 1

    # 1) Vision encode all dense frames in batches.
    vis_raw = encode_vision_batch(model, ds_path, ep_idx, t_list, cfg, device,
                                  batch_size=32)
    if is_temporal and K > 1:
        vis_stacks_np = _build_temporal_stacks(vis_raw, K)            # [N, K, P, D]
        vis = torch.from_numpy(vis_stacks_np).to(device)
    else:
        vis = torch.from_numpy(vis_raw).to(device)                    # [N, P, D]

    # 2) Per-frame prompt resolution -> text feature.
    prompts = [resolver(ep_prompt, ep_idx, t) for t in t_list]
    text_feats = []
    text_masks = []
    for p in prompts:
        tf = text_store.get(p)
        text_feats.append(torch.from_numpy(tf.feat).to(device))
        text_masks.append(torch.from_numpy(tf.mask).to(device))
    Tmax = max(t.shape[0] for t in text_feats)
    D = text_feats[0].shape[-1]
    N = len(text_feats)
    text_pad = torch.zeros(N, Tmax, D, dtype=torch.float32, device=device)
    mask_pad = torch.zeros(N, Tmax, dtype=torch.bool, device=device)
    for i, (tf, m) in enumerate(zip(text_feats, text_masks)):
        L = tf.shape[0]
        text_pad[i, :L] = tf
        mask_pad[i, :L] = m[:L] if m.shape[0] >= L else torch.cat(
            [m, torch.zeros(L - m.shape[0], dtype=torch.bool, device=device)])

    # 3) Forward in chunks to fit memory.
    bs = 16 if (is_temporal and K > 1) else 32
    pred_xy_all = np.zeros((N, n_anchors, 2), dtype=np.float32)
    for i0 in range(0, N, bs):
        i1 = min(i0 + bs, N)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            out = model(vis[i0:i1], text_pad[i0:i1], text_mask=mask_pad[i0:i1])
        logits = out["logits"] if isinstance(out, dict) else out
        for n in range(n_anchors):
            prob_n = logits[:, n].float().flatten(-2).softmax(-1).reshape(
                logits[:, n].shape)
            if readout == "global":
                xy = soft_argmax_2d(prob_n, image_size=image_size)
            else:
                xy = windowed_soft_argmax(prob_n, image_size=image_size)
            pred_xy_all[i0:i1, n, :] = xy.cpu().numpy()
    return pred_xy_all, np.asarray(t_list, dtype=np.int64)


def compute_deltas(pred_xy: np.ndarray, ts: np.ndarray, gt: np.ndarray,
                   anchor_offsets: List[int]) -> Dict[str, np.ndarray]:
    """Compute per-anchor model deltas (consecutive-t) and matching GT deltas."""
    diff_model = pred_xy[1:] - pred_xy[:-1]
    delta_model = np.linalg.norm(diff_model, axis=-1)

    delta_gt = np.zeros_like(delta_model)
    for i, off in enumerate(anchor_offsets):
        T = ts[:-1] + off
        gt_T = gt[T]
        gt_T1 = gt[T + 1]
        delta_gt[:, i] = np.linalg.norm(gt_T1 - gt_T, axis=-1)
    return {"delta_model": delta_model, "delta_gt": delta_gt}


def summarize_deltas(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"),
                "p90": float("nan"), "p95": float("nan")}
    return {
        "n":      int(arr.size),
        "mean":   float(arr.mean()),
        "median": float(np.median(arr)),
        "p90":    float(np.percentile(arr, 90)),
        "p95":    float(np.percentile(arr, 95)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None,
                    help="default: <out_dir>/best.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_frames", type=int, default=5000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--save_arrays", action="store_true")
    ap.add_argument("--arrays_path", default=None)
    ap.add_argument("--readout", default="windowed",
                    choices=["windowed", "global"])
    ap.add_argument("--ema_alpha", type=float, default=None,
                    help="If set, apply causal EMA with this alpha to each "
                         "episode's pred_xy stream BEFORE computing deltas.")
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    set_seed(cfg["seed"])
    out_dir = ensure_dir(cfg["out_dir"])
    device = args.device
    ckpt = args.ckpt or os.path.join(out_dir, "best.pt")
    image_size = int(cfg.get("left_crop_size", 2160))
    anchor_offsets = list(cfg["anchor_offsets"])
    n_anchors = int(cfg["n_anchors"])
    assert n_anchors == len(anchor_offsets)

    print(f"[jitter] config={args.config}")
    print(f"[jitter] ckpt={ckpt}")
    print(f"[jitter] anchor_offsets={anchor_offsets}")
    print(f"[jitter] max_frames={args.max_frames}")

    print("[jitter] building model...")
    model = GazeTrajectoryPredictor(
        encoder_kind=cfg["encoder_kind"],
        fusion_dim=cfg["fusion_dim"],
        n_anchors=cfg["n_anchors"],
        grid=cfg["grid"],
        clip_model_id=cfg["clip_model_id"],
        head_kind=cfg.get("head_kind"),
        temporal_K=int(cfg.get("temporal_K", 3)),
        n_temporal_layers=int(cfg.get("n_temporal_layers", 2)),
    ).to(device)
    sd = torch.load(ckpt, map_location=device, weights_only=False)["model"]
    model.load_state_dict(sd)
    model.eval()

    print(f"[jitter] readout={args.readout}")

    resolver, prompts_to_encode = build_prompt_resolver(cfg)
    text_store = InMemTextStore()
    eps = load_episodes_meta(cfg["test_dataset"])
    if cfg.get("prompt_mode", "default") == "default":
        prompts_to_encode = sorted(set(e["tasks"][0] for e in eps))
    extra_resolved = set()
    for e in eps:
        ep_prompt = e["tasks"][0]
        try:
            extra_resolved.add(resolver(ep_prompt, int(e["episode_index"]), 0))
        except Exception:
            pass
    extra_resolved.update(e["tasks"][0] for e in eps)
    prompts_to_encode = list(set(list(prompts_to_encode) + list(extra_resolved)))
    print(f"[jitter] encoding {len(prompts_to_encode)} prompts into store...")
    encode_prompts_into_store(model, prompts_to_encode, text_store, device)

    horizon = max(anchor_offsets) + 1
    plan: List[Tuple[int, str, int]] = []
    total = 0
    for e in eps:
        ep_len = int(e["length"])
        if ep_len < int(cfg.get("min_episode_length", horizon)):
            continue
        n_t = max(0, ep_len - horizon + 1)
        if n_t < 2:
            continue
        if total + n_t > args.max_frames and total > 0:
            break
        plan.append((int(e["episode_index"]), e["tasks"][0], ep_len))
        total += n_t
        if total >= args.max_frames:
            break
    print(f"[jitter] plan: n_episodes={len(plan)} total_dense_frames={total}")

    all_dm: List[np.ndarray] = []
    all_dg: List[np.ndarray] = []
    per_ep_diag: List[dict] = []
    t0 = time.time()
    for k, (ep_idx, ep_prompt, ep_len) in enumerate(plan):
        print(f"[jitter] ep {k+1}/{len(plan)} ep_idx={ep_idx} len={ep_len}...",
              flush=True)
        pred_xy, ts = run_episode_predictions(
            model, cfg["test_dataset"], ep_idx, ep_prompt, ep_len, cfg, device,
            text_store, resolver, anchor_offsets, image_size, n_anchors,
            readout=args.readout)
        gt = read_episode_gaze(cfg["test_dataset"], ep_idx)
        if ts.size < 2:
            continue
        if args.ema_alpha is not None:
            from .ema_smoothing import ema_smooth
            pred_xy = ema_smooth(pred_xy, alpha=float(args.ema_alpha))
        d = compute_deltas(pred_xy, ts, gt, anchor_offsets)
        all_dm.append(d["delta_model"])
        all_dg.append(d["delta_gt"])
        per_ep_diag.append({
            "ep_idx": ep_idx, "ep_len": ep_len, "n_t": int(ts.size),
            "model_mean_overall": float(d["delta_model"].mean()),
            "gt_mean_overall":    float(d["delta_gt"].mean()),
        })
    elapsed = time.time() - t0
    print(f"[jitter] inference done in {elapsed:.1f}s "
          f"({total / max(elapsed, 1e-6):.1f} fps)")

    delta_model = np.concatenate(all_dm, axis=0) if all_dm else np.zeros((0, n_anchors))
    delta_gt    = np.concatenate(all_dg, axis=0) if all_dg else np.zeros((0, n_anchors))

    per_anchor: List[Dict[str, dict]] = []
    for i in range(n_anchors):
        sm = summarize_deltas(delta_model[:, i])
        sg = summarize_deltas(delta_gt[:, i])
        ratio = sm["mean"] / max(sg["mean"], 1e-12)
        per_anchor.append({
            "anchor": i,
            "offset": int(anchor_offsets[i]),
            "delta_model": sm,
            "delta_gt": sg,
            "ratio_mean_model_over_gt": float(ratio),
        })
        print(f"  a{i} (off={anchor_offsets[i]:>2}): "
              f"model mean={sm['mean']:6.1f} med={sm['median']:6.1f} "
              f"P90={sm['p90']:6.1f} | gt mean={sg['mean']:6.1f} | "
              f"ratio={ratio:.2f}x")

    overall_m = summarize_deltas(delta_model)
    overall_g = summarize_deltas(delta_gt)
    overall_ratio = overall_m["mean"] / max(overall_g["mean"], 1e-12)
    print(f"  OVERALL: model mean={overall_m['mean']:6.1f} med={overall_m['median']:6.1f} "
          f"P90={overall_m['p90']:6.1f} P95={overall_m['p95']:6.1f} | "
          f"gt mean={overall_g['mean']:6.1f} | ratio={overall_ratio:.2f}x")

    summary = {
        "ckpt": ckpt,
        "config": args.config,
        "test_dataset": cfg["test_dataset"],
        "image_size_px": image_size,
        "n_anchors": n_anchors,
        "anchor_offsets": anchor_offsets,
        "readout": args.readout,
        "ema_alpha": float(args.ema_alpha) if args.ema_alpha is not None else None,
        "n_episodes": len(plan),
        "n_dense_frames_planned": total,
        "n_delta_pairs": int(delta_model.shape[0]),
        "elapsed_seconds": float(elapsed),
        "per_anchor": per_anchor,
        "overall": {
            "delta_model": overall_m,
            "delta_gt": overall_g,
            "ratio_mean_model_over_gt": float(overall_ratio),
        },
        "per_episode": per_ep_diag,
    }
    suffix = "" if args.readout == "windowed" else f"_{args.readout}"
    if args.ema_alpha is not None:
        suffix = suffix + f"_ema_{float(args.ema_alpha):.2f}"
    out_path = args.out or os.path.join(out_dir, f"jitter_eval{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[jitter] wrote {out_path}")

    if args.save_arrays:
        ap_path = args.arrays_path or os.path.join(
            out_dir, f"jitter_eval{suffix}.npz")
        np.savez_compressed(ap_path,
                            delta_model=delta_model.astype(np.float32),
                            delta_gt=delta_gt.astype(np.float32),
                            anchor_offsets=np.asarray(anchor_offsets, dtype=np.int32))
        print(f"[jitter] wrote {ap_path}")


if __name__ == "__main__":
    main()
