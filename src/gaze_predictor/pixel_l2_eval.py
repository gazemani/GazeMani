"""Pixel-space L2 metrics for gaze-predictor checkpoints.

Computes the continuous (x, y) prediction with windowed soft-argmax (3x3
window) and compares to the 2160-px GT, reporting

    L2 = sqrt((px - gx)^2 + (py - gy)^2)

aggregated per anchor and overall. Each sample uses its episode's task
instruction.

Usage:
    python -m gaze_predictor.pixel_l2_eval \
        --config src/gaze_predictor/configs/gaze_predictor_six_task.yaml \
        --split test --readout windowed

Writes JSON to ``cfg['out_dir']/pixel_l2_<split>.json``.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from .model import GazeTrajectoryPredictor
from .train_prompts import (
    InMemTextStore,
    collate_with_text_mask,
    make_loaders_with_prompt_mode,
)
from .utils import (
    ensure_dir, load_yaml_config, set_seed, soft_argmax_2d,
    windowed_soft_argmax,
)


HIT_THRESHOLDS_PX = [50, 154, 308, 500]


@torch.no_grad()
def collect_l2(model, loader, device, max_samples: int = 1024,
               image_size: int = 2160,
               readout: str = "windowed") -> Dict[str, np.ndarray]:
    """Run the model over ``loader``, soft-argmax to (x, y), compute L2 to GT."""
    model.eval()
    l2_per_anchor: List[List[float]] = None
    pred_xys: List[np.ndarray] = []
    gt_xys:   List[np.ndarray] = []
    n_seen = 0
    for batch in loader:
        vis  = batch["vis_feat"].to(device, non_blocking=True)
        text = batch["text_feat"].to(device, non_blocking=True)
        mask = batch["text_mask"].to(device, non_blocking=True)
        gt   = batch["gaze_xy"].to(device, non_blocking=True)   # [B, N, 2]
        out = model(vis, text, text_mask=mask)
        logits = out["logits"] if isinstance(out, dict) else out
        prob = logits.flatten(-2).softmax(-1).reshape(logits.shape)
        B, N, _, _ = prob.shape
        if l2_per_anchor is None:
            l2_per_anchor = [[] for _ in range(N)]
        for n in range(N):
            if readout == "global":
                xy = soft_argmax_2d(prob[:, n].float(), image_size=image_size)
            else:
                xy = windowed_soft_argmax(prob[:, n], image_size=image_size) # [B, 2]
            d = (xy - gt[:, n]).norm(dim=-1)                                  # [B]
            l2_per_anchor[n].extend(d.cpu().tolist())
            pred_xys.append(xy.cpu().numpy())
            gt_xys.append(gt[:, n].cpu().numpy())
        n_seen += B
        if n_seen >= max_samples:
            break

    per_anchor = np.array(l2_per_anchor, dtype=np.float64)   # [N, M]
    overall = per_anchor.reshape(-1)
    return {
        "per_anchor_l2": per_anchor,
        "overall_l2":   overall,
    }


def summarize_l2(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return {
            "n": 0, "mean": float("nan"), "median": float("nan"),
            "p25": float("nan"), "p75": float("nan"),
            "p90": float("nan"), "p95": float("nan"),
            "hit_rate": {f"<{thr}px": float("nan") for thr in HIT_THRESHOLDS_PX},
        }
    return {
        "n": int(arr.size),
        "mean":   float(arr.mean()),
        "median": float(np.median(arr)),
        "p25":    float(np.percentile(arr, 25)),
        "p75":    float(np.percentile(arr, 75)),
        "p90":    float(np.percentile(arr, 90)),
        "p95":    float(np.percentile(arr, 95)),
        "hit_rate": {
            f"<{thr}px": float((arr < thr).mean()) for thr in HIT_THRESHOLDS_PX
        },
    }


@torch.no_grad()
def _collect_l2_dense_ema(*, model, cfg, args, device):
    """Dense per-episode inference + causal EMA + endpoint L2 to GT."""
    from .data import load_episodes_meta, read_episode_gaze
    from .ema_smoothing import ema_smooth
    from .jitter_eval import (
        build_prompt_resolver, run_episode_predictions,
    )
    from .train_prompts import (
        InMemTextStore, encode_prompts_into_store,
    )

    image_size = int(cfg.get("left_crop_size", 2160))
    anchor_offsets = list(cfg["anchor_offsets"])
    n_anchors = int(cfg["n_anchors"])
    assert n_anchors == len(anchor_offsets)

    resolver, prompts_to_encode = build_prompt_resolver(cfg)
    text_store = InMemTextStore()
    eps = load_episodes_meta(cfg["test_dataset"])
    extra_resolved = set()
    for e in eps:
        ep_prompt = e["tasks"][0]
        try:
            extra_resolved.add(resolver(ep_prompt, int(e["episode_index"]), 0))
        except Exception:
            pass
    extra_resolved.update(e["tasks"][0] for e in eps)
    prompts_to_encode = list(set(list(prompts_to_encode) + list(extra_resolved)))
    print(f"[pixel-l2-ema] encoding {len(prompts_to_encode)} prompts...")
    encode_prompts_into_store(model, prompts_to_encode, text_store, device)

    horizon = max(anchor_offsets) + 1
    plan = []
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
    print(f"[pixel-l2-ema] plan: n_episodes={len(plan)} "
          f"total_dense_frames={total}")

    per_anchor_lists = [[] for _ in range(n_anchors)]
    for k, (ep_idx, ep_prompt, ep_len) in enumerate(plan):
        print(f"[pixel-l2-ema] ep {k+1}/{len(plan)} ep_idx={ep_idx} "
              f"len={ep_len}", flush=True)
        pred_xy, ts = run_episode_predictions(
            model, cfg["test_dataset"], ep_idx, ep_prompt, ep_len, cfg, device,
            text_store, resolver, anchor_offsets, image_size, n_anchors,
            readout=args.readout)
        gt = read_episode_gaze(cfg["test_dataset"], ep_idx)
        if ts.size < 1:
            continue
        if args.ema_alpha is not None:
            pred_xy = ema_smooth(pred_xy, alpha=float(args.ema_alpha))
        for i, off in enumerate(anchor_offsets):
            gt_i = gt[ts + off]
            d = np.linalg.norm(pred_xy[:, i] - gt_i, axis=-1)
            per_anchor_lists[i].extend(d.tolist())
    M = min(len(x) for x in per_anchor_lists) if per_anchor_lists else 0
    per_anchor = np.array([np.asarray(x[:M], dtype=np.float64)
                           for x in per_anchor_lists], dtype=np.float64)
    overall = per_anchor.reshape(-1)
    return per_anchor, overall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None,
                    help="default: <out_dir>/best.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_samples", type=int, default=1024)
    ap.add_argument("--out", default=None)
    ap.add_argument("--split", default="test",
                    choices=["train", "val", "test", "eval"])
    ap.add_argument("--readout", default="windowed",
                    choices=["windowed", "global"])
    ap.add_argument("--ema_alpha", type=float, default=None,
                    help="If set, switch to dense-episode mode and apply causal "
                         "EMA(alpha) over each episode's pred_xy stream BEFORE "
                         "L2 to GT. Output JSON filename gets a `_ema_<alpha>` "
                         "suffix.")
    ap.add_argument("--max_frames", type=int, default=5000,
                    help="Cap total dense frames across episodes when "
                         "--ema_alpha is set.")
    args = ap.parse_args()

    if args.split == "eval":
        args.split = "test"

    cfg = load_yaml_config(args.config)
    set_seed(cfg["seed"])
    out_dir = ensure_dir(cfg["out_dir"])
    device = args.device
    ckpt = args.ckpt or os.path.join(out_dir, "best.pt")

    print(f"[pixel-l2] config={args.config}")
    print(f"[pixel-l2] ckpt={ckpt}")
    print(f"[pixel-l2] split={args.split}")

    print("[pixel-l2] building model...")
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

    text_store = InMemTextStore()
    print("[pixel-l2] building loaders + base text store...")
    train_loader, val_loader, test_loader, stats = make_loaders_with_prompt_mode(
        cfg, model, text_store, device)
    for k, v in stats.items():
        print("[pixel-l2] split {}: n_kept={} fallback_rate={:.4f}".format(
            k, v["n_kept"], v["fallback_rate"]))

    src_loader = {"train": train_loader, "val": val_loader,
                  "test": test_loader}[args.split]
    if args.split != "test":
        eval_loader = DataLoader(
            src_loader.dataset,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=int(cfg.get("num_workers", 4)),
            collate_fn=collate_with_text_mask,
            pin_memory=True,
            drop_last=False,
        )
    else:
        eval_loader = src_loader
    print("[pixel-l2] using split={} (n_samples={}, max_samples={})".format(
        args.split, len(eval_loader.dataset), args.max_samples))

    print(f"[pixel-l2] readout={args.readout}")
    if args.ema_alpha is not None:
        print(f"[pixel-l2] EMA mode: dense per-episode inference with "
              f"alpha={args.ema_alpha} (max_frames={args.max_frames})")
        per_anchor, overall = _collect_l2_dense_ema(
            model=model, cfg=cfg, args=args, device=device)
        n_anchors = per_anchor.shape[0]
    else:
        print("[pixel-l2] computing pixel L2...")
        out = collect_l2(model, eval_loader, device, max_samples=args.max_samples,
                         readout=args.readout)
        per_anchor = out["per_anchor_l2"]
        overall = out["overall_l2"]
        n_anchors = per_anchor.shape[0]

    print("[pixel-l2] overall: mean={:.1f} median={:.1f} P90={:.1f}".format(
        float(overall.mean()), float(np.median(overall)),
        float(np.percentile(overall, 90))))
    for n in range(n_anchors):
        print("  anchor {}: mean={:.1f} median={:.1f} P90={:.1f}".format(
            n,
            float(per_anchor[n].mean()),
            float(np.median(per_anchor[n])),
            float(np.percentile(per_anchor[n], 90)),
        ))

    summary = {
        "ckpt": ckpt,
        "config": args.config,
        "split": args.split,
        "max_samples": args.max_samples,
        "readout": args.readout,
        "ema_alpha": float(args.ema_alpha) if args.ema_alpha is not None else None,
        "image_size_px": 2160,
        "grid": cfg["grid"],
        "px_per_cell": 2160.0 / cfg["grid"],
        "hit_thresholds_px": HIT_THRESHOLDS_PX,
        "overall": summarize_l2(overall),
        "per_anchor": [summarize_l2(per_anchor[n]) for n in range(n_anchors)],
    }
    suffix = "" if args.readout == "windowed" else f"_{args.readout}"
    if args.ema_alpha is not None:
        suffix = suffix + f"_ema_{float(args.ema_alpha):.2f}"
    raw_path = os.path.join(out_dir, f"pixel_l2_{args.split}{suffix}.npz")
    np.savez_compressed(raw_path,
                        per_anchor_l2=per_anchor,
                        overall_l2=overall)
    summary["raw_path"] = raw_path

    default_name = f"pixel_l2_{args.split}{suffix}.json"
    out_path = args.out or os.path.join(out_dir, default_name)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("[pixel-l2] wrote", out_path)
    print("[pixel-l2] wrote", raw_path)


if __name__ == "__main__":
    main()
