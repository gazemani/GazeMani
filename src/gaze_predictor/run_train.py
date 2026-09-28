"""One-shot training driver for the gaze predictor (K=3 prompt-then-temporal
head, 14->224 upsampler).

Pass one or more train datasets (and an optional held-out test dataset), and
this driver:

  1. Loads the base config
     (``src/gaze_predictor/configs/gaze_predictor_six_task.yaml``).
  2. Overrides train_dataset / test_dataset / cache_dir / out_dir / wandb*
     / epochs / patience / batch_size from CLI.
  3. Pre-caches frozen-encoder features (vision + text) under a per-run
     cache_dir so multi-dataset launches don't collide.
  4. Runs ``train_prompts.train_loop`` with wandb enabled (per-epoch
     train+val curves, periodic test-loss for monitoring, end-of-train summaries, and N
     test videos uploaded as wandb media).
  5. After training, runs ``pixel_l2_eval`` / ``jitter_eval`` on the test
     set (windowed readout, raw + EMA α=0.2), pushes their summaries to
     wandb, and renders ``num_test_videos`` test episodes, saving under
     ``<out_dir>/qual/test_video_<NN>.mp4``.

Usage example (single dataset):

    python -m gaze_predictor.run_train \\
        --train_datasets /path/to/stack_bowls_train \\
        --out_dir        ./checkpoints/stack_bowls_train/gaze_predictor/default \\
        --exp_name       default

Multi-dataset (joint training):

    python -m gaze_predictor.run_train \\
        --train_datasets /path/to/stack_bowls_train,/path/to/some_other_train \\
        --out_dir        ./checkpoints/multi_task/gaze_predictor/default \\
        --exp_name       default
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from typing import List

import yaml

# We avoid importing torch / cv2 at module import time so --help is fast.


# This file lives at src/gaze_predictor/run_train.py. The default
# base config is a sibling under configs/.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BASE_CONFIG = os.path.join(
    _THIS_DIR, "configs", "gaze_predictor_six_task.yaml")
DEFAULT_CACHE_ROOT = "./cache/gaze_predictor_clip_features"


def _split_paths(s: str) -> List[str]:
    return [p.strip() for p in s.split(",") if p.strip()]


def _hash_paths(paths: List[str]) -> str:
    h = hashlib.sha1()
    for p in sorted(paths):
        h.update(p.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:10]


def _resolve_cfg(args) -> dict:
    with open(args.base_config, "r") as f:
        cfg = yaml.safe_load(f)

    train_paths = _split_paths(args.train_datasets)
    if not train_paths:
        raise SystemExit("--train_datasets must list at least one path")
    cfg["train_dataset"] = train_paths if len(train_paths) > 1 else train_paths[0]

    if args.test_dataset:
        test_paths = _split_paths(args.test_dataset)
        cfg["test_dataset"] = (test_paths if len(test_paths) > 1
                                else test_paths[0])
    else:
        cfg["test_dataset"] = None

    # Per-run cache_dir so multi-dataset feature caches stay isolated.
    if args.cache_dir:
        cfg["cache_dir"] = args.cache_dir
    else:
        all_paths = list(train_paths)
        if args.test_dataset:
            all_paths += _split_paths(args.test_dataset)
        h = _hash_paths(all_paths)
        cfg["cache_dir"] = os.path.join(
            DEFAULT_CACHE_ROOT,
            f"{cfg.get('encoder_kind', 'enc')}_{h}")

    cfg["out_dir"] = args.out_dir

    # Hyperparam overrides (None = keep base config value).
    if args.epochs is not None: cfg["epochs"] = int(args.epochs)
    if args.patience is not None: cfg["patience"] = int(args.patience)
    if args.batch_size is not None: cfg["batch_size"] = int(args.batch_size)
    if args.num_workers is not None: cfg["num_workers"] = int(args.num_workers)

    # wandb knobs.
    cfg["wandb"] = bool(args.wandb)
    if args.wandb_project: cfg["wandb_project"] = args.wandb_project
    cfg["wandb_run_name"] = args.exp_name
    cfg.setdefault("test_eval_every_epochs", int(args.test_eval_every_epochs))

    return cfg


def _write_effective_config(cfg: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "effective_config.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def _run_cache_pass(cfg: dict, device: str) -> None:
    """Encode + save frozen vision/text features for every plan."""
    import torch
    from .cache import (
        build_plans_for_split, cache_text, cache_vision,
    )
    from .model import GazeTrajectoryPredictor
    from .temporal_cache import ensure_temporal_vision_cache
    from .utils import ensure_dir

    ensure_dir(cfg["cache_dir"])
    print(f"[run_train] cache_dir={cfg['cache_dir']}")

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

    plans = build_plans_for_split(cfg)
    n_train = len(plans["train"][0])
    n_val = len(plans["val"][0])
    n_test = len(plans["test"][0])
    print(f"[run_train] plans: train={n_train} val={n_val} test={n_test}")

    all_plans = plans["train"][0] + plans["val"][0] + plans["test"][0]
    cache_vision(model, all_plans, cfg, cfg["cache_dir"], device=device)

    K = int(cfg.get("temporal_K", 1))
    if K > 1:
        info = ensure_temporal_vision_cache(
            model, all_plans, cfg, K=K, device=device)
        print("[run_train] temporal-cache: required={} missing_before={} "
              "encoded={}".format(info["required"],
                                   info["missing_before"], info["encoded"]))

    prompts = set(p.prompt for p in all_plans)
    cache_text(model, sorted(prompts), cfg["cache_dir"], device=device)

    del model
    torch.cuda.empty_cache()


def _maybe_run_eval_subprocess(module: str, cfg_path: str,
                                extra_args: List[str], log_prefix: str) -> int:
    """Run a sub-eval as a child process so its CLI argparse is unchanged."""
    py = sys.executable
    cmd = [py, "-m", module, "--config", cfg_path] + extra_args
    print(f"[{log_prefix}] $ " + " ".join(cmd))
    rc = subprocess.call(cmd)
    print(f"[{log_prefix}] returncode={rc}")
    return rc


def _select_test_episodes(test_dataset: str, n: int, seed: int) -> List[int]:
    """Pick n distinct episode indices from the test dataset."""
    from .data import load_episodes_meta
    eps = load_episodes_meta(test_dataset)
    candidates = sorted(int(e["episode_index"]) for e in eps
                        if int(e["length"]) >= 30)
    if len(candidates) <= n:
        return candidates
    rng = random.Random(seed)
    return sorted(rng.sample(candidates, n))


def _render_test_videos(cfg: dict, ckpt_path: str, test_dataset: str,
                         num_videos: int, device: str,
                         wandb_handle) -> List[str]:
    """Render `num_videos` test-episode MP4s at out_dir/qual/."""
    import torch
    from .model import GazeTrajectoryPredictor
    from .utils import ensure_dir
    from .visualize_video import render_test_episode_video

    qual_dir = ensure_dir(os.path.join(cfg["out_dir"], "qual"))
    eps = _select_test_episodes(test_dataset, num_videos, seed=cfg.get("seed", 42))
    if not eps:
        print("[run_train] no test episodes found; skipping video render")
        return []
    print(f"[run_train] rendering test videos for episodes={eps}")

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
    sd = torch.load(ckpt_path, map_location=device,
                     weights_only=False)["model"]
    model.load_state_dict(sd); model.eval()

    # Draw the predicted fixation at the same geometry as the policy-input
    # crosshair (half-arm 120 px in 2160-px space).
    anchor_sizes_2160 = (120,)
    offsets = tuple(int(o) for o in cfg.get("anchor_offsets", [0]))

    written = []
    for i, ep in enumerate(eps):
        out_mp4 = os.path.join(qual_dir, f"test_video_{i:02d}_ep{ep:04d}.mp4")
        try:
            render_test_episode_video(
                model,
                test_dataset_path=test_dataset,
                cfg=cfg,
                out_path=out_mp4,
                device=device,
                episode_index=int(ep),
                horizon_offsets=offsets,
                fps=30,
                output_size=720,
                anchor_sizes_2160=anchor_sizes_2160,
                ema_alpha=0.2,
            )
            written.append(out_mp4)
            wandb_handle.log_video(f"qual/test_video_{i:02d}", out_mp4,
                                    caption=f"ep {ep}")
        except Exception as e:
            print(f"[run_train] video render failed for ep {ep}: {e}")

    del model
    torch.cuda.empty_cache()
    return written


def _push_eval_summaries_to_wandb(cfg: dict, wandb_handle) -> None:
    """After the eval subprocesses finish, scan their output JSONs and push
    a flat summary to wandb."""
    out_dir = cfg["out_dir"]
    summary = {}

    for name in os.listdir(out_dir):
        if not name.endswith(".json"):
            continue
        if not (name.startswith("pixel_l2_") or name.startswith("jitter_eval")):
            continue
        try:
            with open(os.path.join(out_dir, name), "r") as f:
                blob = json.load(f)
        except Exception:
            continue
        if "overall" in blob and isinstance(blob["overall"], dict):
            tag = name.replace(".json", "")
            for k, v in blob["overall"].items():
                if isinstance(v, (int, float)):
                    summary[f"eval/{tag}/overall/{k}"] = float(v)
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        if isinstance(vv, (int, float)):
                            summary[f"eval/{tag}/overall/{k}/{kk}"] = float(vv)
        if "summary" in blob and isinstance(blob["summary"], dict):
            tag = name.replace(".json", "")
            for grp_name, grp in blob["summary"].items():
                if isinstance(grp, dict):
                    for k, v in grp.items():
                        if isinstance(v, (int, float)):
                            summary[f"eval/{tag}/{grp_name}/{k}"] = float(v)
    if summary:
        wandb_handle.summary_update(summary)
        print(f"[run_train] pushed {len(summary)} eval scalars to wandb summary")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_datasets", required=True,
                    help="Comma-separated list of LeRobot dataset paths.")
    ap.add_argument("--test_dataset", default=None,
                    help="Optional held-out test dataset (single path; "
                         "comma-separated for multi).")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--exp_name", required=True,
                    help="Run name (used for wandb).")
    ap.add_argument("--base_config", default=DEFAULT_BASE_CONFIG)
    ap.add_argument("--cache_dir", default=None,
                    help="Override the auto-derived per-run cache dir.")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--wandb", action="store_true", default=True,
                    help="Enable wandb logging (default on; overrides the config's "
                         "``wandb`` field; pass --no_wandb to disable).")
    ap.add_argument("--no_wandb", dest="wandb", action="store_false")
    ap.add_argument("--wandb_project", default="gaze_predictor")
    ap.add_argument("--num_test_videos", type=int, default=5)
    ap.add_argument("--test_eval_every_epochs", type=int, default=10,
                    help="Periodic test-loss logging cadence during training "
                         "(monitoring only; checkpoint selection uses "
                         "validation loss).")
    ap.add_argument("--skip_eval_suite", action="store_true",
                    help="Skip pixel_l2 / jitter eval after training.")
    ap.add_argument("--skip_videos", action="store_true",
                    help="Skip test-video rendering after training.")
    ap.add_argument("--skip_train", action="store_true",
                    help="Skip training (assume best.pt exists).")
    args = ap.parse_args()

    cfg = _resolve_cfg(args)
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    cfg_path = _write_effective_config(cfg, out_dir)
    print(f"[run_train] effective_config={cfg_path}")
    print(f"[run_train] out_dir={out_dir}")

    # --- Stage 1: feature cache (idempotent) ---
    if not args.skip_train:
        print("[run_train] Stage 1: feature cache pre-pass")
        _run_cache_pass(cfg, device=args.device)
    else:
        print("[run_train] Stage 1: SKIPPED (--skip_train)")

    # --- Stage 2: training ---
    from .train_prompts import _WandbHandle, train_loop

    wb = _WandbHandle()
    wb.init(cfg, out_dir, run_name=args.exp_name)
    try:
        if not args.skip_train:
            print("[run_train] Stage 2: training")
            t0 = time.time()
            res = train_loop(cfg, device=args.device, wandb_handle=wb)
            print("[run_train] training finished in {:.1f}s; "
                  "best_val_loss={}".format(time.time() - t0,
                                             res.get("best_val_loss")))
        else:
            print("[run_train] Stage 2: SKIPPED (--skip_train)")
            res = {"best_path": os.path.join(out_dir, "best.pt")}

        best_path = res["best_path"]
        if not os.path.exists(best_path):
            print(f"[run_train] WARN: best.pt not found at {best_path}; "
                  "skipping eval/videos")
            return

        # --- Stage 3: eval suite (only if test_dataset present) ---
        first_test = (cfg["test_dataset"][0]
                      if isinstance(cfg["test_dataset"], list)
                      else cfg["test_dataset"])
        if first_test and not args.skip_eval_suite:
            print("[run_train] Stage 3: pixel_l2 / jitter")
            _maybe_run_eval_subprocess(
                "gaze_predictor.pixel_l2_eval", cfg_path,
                ["--split", "test", "--readout", "windowed"],
                log_prefix="pixel_l2-raw")
            _maybe_run_eval_subprocess(
                "gaze_predictor.pixel_l2_eval", cfg_path,
                ["--split", "test", "--readout", "windowed",
                 "--ema_alpha", "0.2"],
                log_prefix="pixel_l2-ema")
            _maybe_run_eval_subprocess(
                "gaze_predictor.jitter_eval", cfg_path,
                ["--readout", "windowed"],
                log_prefix="jitter-raw")
            _maybe_run_eval_subprocess(
                "gaze_predictor.jitter_eval", cfg_path,
                ["--readout", "windowed", "--ema_alpha", "0.2"],
                log_prefix="jitter-ema")
            _push_eval_summaries_to_wandb(cfg, wb)
        else:
            print("[run_train] Stage 3: SKIPPED "
                  "(skip_eval_suite or no test_dataset)")

        # --- Stage 4: test videos ---
        if first_test and not args.skip_videos:
            print("[run_train] Stage 4: test-episode video rendering")
            _render_test_videos(cfg, best_path, first_test,
                                 num_videos=int(args.num_test_videos),
                                 device=args.device, wandb_handle=wb)
        else:
            print("[run_train] Stage 4: SKIPPED")

    finally:
        wb.finish()
    print("[run_train] all stages complete.")


if __name__ == "__main__":
    main()
