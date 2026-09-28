"""Training entrypoint for the gaze predictor.

Default configuration:

  prompt_mode = "default"  (each episode keeps its native task instruction)
  head_kind   = "prompt_then_temporal_224"

Reuses the on-disk vision feature cache. Text features for the prompts are
encoded on the fly (frozen CLIP text encoder) and held in memory.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .cache import build_plans_for_split
from .data import SamplePlan, read_episode_gaze
from .model import GazeTrajectoryPredictor
from .temporal_cache import (
    ensure_temporal_vision_cache,
    load_temporal_vis_stack,
)
from .utils import (
    ensure_dir, gaze_to_heatmap_14x14, load_yaml_config, set_seed,
)


# ---------------------------------------------------------------------------
# Loss + scheduler + evaluate.
# ---------------------------------------------------------------------------

def heatmap_ce_loss(logits, gt_xy, image_size=2160, sigma_in_grid=1.0):
    """Cross-entropy of softmax(logits) against a Gaussian-blurred GT heatmap.

    ``sigma_in_grid`` is in output-grid cells (≈ 9.64 px each at 224x224 in
    2160 space); the default 16 ≈ 154 px. Set via the config field
    ``heatmap_sigma_in_grid``.
    """
    B, N, G, _ = logits.shape
    target = gaze_to_heatmap_14x14(gt_xy, image_size=image_size, grid=G,
                                    sigma_in_grid=sigma_in_grid)
    logp = F.log_softmax(logits.flatten(-2), dim=-1).reshape(B, N, G, G)
    ce = -(target * logp).sum(dim=(-2, -1))
    per_anchor = ce.mean(0)
    return per_anchor.mean(), per_anchor


def make_scheduler(optimizer, total_steps, warmup_steps, lr_min):
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(warmup_steps, 1))
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        ratio_min = lr_min / base_lrs[0]
        return ratio_min + (1.0 - ratio_min) * cos
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def base_evaluate(model, loader, device, sigma_in_grid: float = 1.0):
    model.eval()
    n = 0
    per_anc_sum = None
    total = 0.0
    for batch in loader:
        vis = batch["vis_feat"].to(device, non_blocking=True)
        text = batch["text_feat"].to(device, non_blocking=True)
        mask = batch["text_mask"].to(device, non_blocking=True)
        gt = batch["gaze_xy"].to(device, non_blocking=True)
        out = model(vis, text, text_mask=mask)
        logits = out["logits"] if isinstance(out, dict) else out
        loss, per_anc = heatmap_ce_loss(logits, gt, sigma_in_grid=sigma_in_grid)
        bsz = vis.size(0)
        total += loss.item() * bsz
        if per_anc_sum is None:
            per_anc_sum = per_anc.detach().clone() * bsz
        else:
            per_anc_sum += per_anc.detach() * bsz
        n += bsz
    return {"loss": total / max(n, 1),
            "per_anchor": (per_anc_sum / max(n, 1)).cpu().tolist(),
            "n": n}


# ---------------------------------------------------------------------------
# In-memory text-feature lookup
# ---------------------------------------------------------------------------

@dataclass
class TextFeat:
    feat: np.ndarray   # [T, D]
    mask: np.ndarray   # [T] bool


class InMemTextStore:
    """prompt -> TextFeat. Built by encoding a fixed list of prompts via the
    frozen text encoder."""
    def __init__(self):
        self._d: Dict[str, TextFeat] = {}

    def add(self, prompt: str, feat: np.ndarray, mask: np.ndarray):
        self._d[prompt] = TextFeat(feat=feat.astype(np.float32),
                                   mask=mask.astype(bool))

    def __contains__(self, p): return p in self._d

    def get(self, prompt: str) -> TextFeat:
        return self._d[prompt]

    def keys(self) -> List[str]:
        return list(self._d.keys())


@torch.no_grad()
def encode_prompts_into_store(model: GazeTrajectoryPredictor,
                               prompts: Sequence[str],
                               store: InMemTextStore,
                               device: str):
    for prompt in sorted(set(prompts)):
        if prompt in store:
            continue
        raw, mask = model.text_encoder._raw([prompt], device)
        store.add(prompt, raw[0].cpu().numpy(), mask[0].cpu().numpy())


# ---------------------------------------------------------------------------
# Dataset that pulls vision feats from disk and text feats from in-memory store,
# with prompt resolution per sample.
# ---------------------------------------------------------------------------

PromptResolver = Callable[[SamplePlan, int], str]


class PromptOverrideDataset(Dataset):
    """Like CachedFeatureDataset but text features come from an in-memory
    InMemTextStore and the prompt for each sample is provided by a resolver
    callable.

    Args:
        plans: list of SamplePlan (``prompt`` is the episode's task
            instruction; the resolver maps each plan to the string the model sees).
        cache_dir: dir containing ``vis/`` subdir of cached vision features.
        text_store: InMemTextStore prepopulated with all prompts the resolver
            might return.
        resolver: callable (plan, idx) -> str (released config: identity).
        temporal_K: if > 1, ``vis_feat`` becomes ``[K, P, D]`` of consecutive
            frames ``[t-K+1, ..., t]`` (clamped at frame 0 for early frames).
            All K cache files must exist (call
            ``ensure_temporal_vision_cache`` upstream).
    """
    def __init__(self,
                 plans: Sequence[SamplePlan],
                 cache_dir: str,
                 text_store: InMemTextStore,
                 resolver: PromptResolver,
                 temporal_K: int = 1):
        self.plans = list(plans)
        self.cache_dir = cache_dir
        self.text_store = text_store
        self.resolver = resolver
        self.temporal_K = int(temporal_K)

    def __len__(self):
        return len(self.plans)

    def _vis_path(self, p: SamplePlan) -> str:
        ds_name = os.path.basename(p.dataset_path.rstrip('/'))
        return os.path.join(self.cache_dir, 'vis',
                            f'{ds_name}__ep_{p.episode_index:06d}__f_{p.t:06d}.npz')

    def __getitem__(self, idx: int):
        p = self.plans[idx]
        if self.temporal_K > 1:
            vis = load_temporal_vis_stack(p, self.cache_dir, self.temporal_K)
            # vis: [K, P, D]
        else:
            vis = np.load(self._vis_path(p))['vis_feat']            # [P, D]
        gaze = read_episode_gaze(p.dataset_path, p.episode_index)
        gaze_anchors = np.stack([gaze[a] for a in p.anchor_frames]).astype(np.float32)
        prompt = self.resolver(p, idx)
        tf = self.text_store.get(prompt)
        return {
            'vis_feat': torch.from_numpy(vis).float(),
            'text_feat': torch.from_numpy(tf.feat).float(),
            'text_mask': torch.from_numpy(tf.mask).bool(),
            'gaze_xy': torch.from_numpy(gaze_anchors).float(),
            'prompt': prompt,
            'episode_index': p.episode_index,
            't': p.t,
            'anchor_frames': torch.as_tensor(p.anchor_frames, dtype=torch.long),
        }


def collate_with_text_mask(batch):
    # vis_feat per item is either [P, D] (single frame) or [K, P, D]
    # (temporal-context stack). torch.stack handles both: it adds a leading
    # batch dim, giving [B, P, D] or [B, K, P, D].
    vis = torch.stack([b['vis_feat'] for b in batch])
    gaze = torch.stack([b['gaze_xy'] for b in batch])
    text_lens = [b['text_feat'].shape[0] for b in batch]
    Tm = max(text_lens)
    D = batch[0]['text_feat'].shape[-1]
    text = torch.zeros(len(batch), Tm, D, dtype=torch.float32)
    text_mask = torch.zeros(len(batch), Tm, dtype=torch.bool)
    for i, b in enumerate(batch):
        L = b['text_feat'].shape[0]
        text[i, :L] = b['text_feat']
        m = b['text_mask']
        if m.shape[0] < L:
            mm = torch.zeros(L, dtype=torch.bool); mm[:m.shape[0]] = m
            m = mm
        text_mask[i, :L] = m[:L]
    return {
        'vis_feat': vis,
        'text_feat': text,
        'text_mask': text_mask,
        'gaze_xy': gaze,
        'prompt': [b['prompt'] for b in batch],
        'episode_index': torch.as_tensor([b['episode_index'] for b in batch]),
        't': torch.as_tensor([b['t'] for b in batch]),
        'anchor_frames': torch.stack([b['anchor_frames'] for b in batch]),
    }


def _worker_init_fn(worker_id: int):
    """Reseed each dataloader worker."""
    import random as _rnd
    seed = (torch.initial_seed() + worker_id) & 0xFFFFFFFF
    _rnd.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Loader builder.
# ---------------------------------------------------------------------------

def make_loaders_with_prompt_mode(cfg: dict, model: GazeTrajectoryPredictor,
                                  text_store: InMemTextStore, device: str):
    plans = build_plans_for_split(cfg)
    cache_dir = cfg["cache_dir"]
    temporal_K = int(cfg.get("temporal_K", 1))

    # If the model uses a K-frame temporal stack we may need to encode + cache
    # the past (K-1) neighbor frames for every plan. This is a one-time op;
    # subsequent runs hit the disk cache.
    if temporal_K > 1:
        all_plans = []
        for s in plans.values():
            all_plans.extend(s[0])
        info = ensure_temporal_vision_cache(
            model, all_plans, cfg, K=temporal_K, device=device)
        print("[temporal-cache] required={} missing_before={} encoded={}".format(
            info["required"], info["missing_before"], info["encoded"]))

    mode = cfg.get("prompt_mode", "default")
    if mode == "default":
        # Identity resolver: each plan keeps its episode's task instruction.
        # Encode every unique prompt seen across splits.
        prompts_to_encode = sorted(set(
            p.prompt for split in plans.values() for p in split[0]))
        encode_prompts_into_store(model, prompts_to_encode, text_store, device)
        def _identity(p: SamplePlan, idx: int) -> str:
            return p.prompt
        resolver_train = _identity
        resolver_val   = _identity
        resolver_test  = _identity
    else:
        raise NotImplementedError(
            f"unsupported prompt_mode={mode!r}; only 'default' is supported.")

    ds_train = PromptOverrideDataset(plans["train"][0], cache_dir, text_store,
                                     resolver_train, temporal_K=temporal_K)
    ds_val   = PromptOverrideDataset(plans["val"][0],   cache_dir, text_store,
                                     resolver_val,   temporal_K=temporal_K)
    ds_test  = PromptOverrideDataset(plans["test"][0],  cache_dir, text_store,
                                     resolver_test,  temporal_K=temporal_K)

    nw = int(cfg.get("num_workers", 4))
    sampling_mode = cfg.get("sampling_mode", "sparse")
    if sampling_mode == "weighted_balanced":
        # Per-sample weight = 1 / (n_datasets * dataset_sample_count) so each
        # dataset contributes equally to every batch.
        from collections import Counter
        from torch.utils.data import WeightedRandomSampler
        train_plans = plans["train"][0]
        ds_count = Counter(p.dataset_path for p in train_plans)
        n_ds = max(len(ds_count), 1)
        weights = [1.0 / (n_ds * ds_count[p.dataset_path]) for p in train_plans]
        num_samples = int(cfg.get("samples_per_epoch", len(train_plans)))
        sampler = WeightedRandomSampler(weights, num_samples=num_samples,
                                        replacement=True)
        print(f"[train] sampling_mode=weighted_balanced "
              f"n_datasets={n_ds} cached_samples={len(train_plans)} "
              f"draws_per_epoch={num_samples} (with replacement)")
        train_loader = DataLoader(
            ds_train, batch_size=cfg["batch_size"], sampler=sampler,
            num_workers=nw, collate_fn=collate_with_text_mask,
            pin_memory=True, drop_last=True, worker_init_fn=_worker_init_fn)
    else:
        train_loader = DataLoader(
            ds_train, batch_size=cfg["batch_size"], shuffle=True, num_workers=nw,
            collate_fn=collate_with_text_mask, pin_memory=True, drop_last=True,
            worker_init_fn=_worker_init_fn)
    val_loader = DataLoader(
        ds_val, batch_size=cfg["batch_size"], shuffle=False, num_workers=nw,
        collate_fn=collate_with_text_mask, pin_memory=True,
        worker_init_fn=_worker_init_fn)
    test_loader = DataLoader(
        ds_test, batch_size=cfg["batch_size"], shuffle=False, num_workers=nw,
        collate_fn=collate_with_text_mask, pin_memory=True,
        worker_init_fn=_worker_init_fn)
    stats = {k: plans[k][1] for k in plans}
    return train_loader, val_loader, test_loader, stats


# ---------------------------------------------------------------------------
# wandb thin wrapper (no-op when wandb is disabled or unavailable)
# ---------------------------------------------------------------------------


class _WandbHandle:
    """Tiny shim so the train loop can call .log/.summary uniformly whether
    wandb is enabled or not. When disabled, every method is a no-op."""

    def __init__(self):
        self.run = None
        self._wandb = None

    def init(self, cfg: dict, out_dir: str, run_name: Optional[str] = None):
        if not bool(cfg.get("wandb", False)):
            return
        try:
            import wandb as _wandb
        except ImportError:
            print("[wandb] wandb is not installed; skipping init")
            return
        try:
            self.run = _wandb.init(
                project=cfg.get("wandb_project", "gaze_predictor"),
                name=run_name or cfg.get("wandb_run_name") or os.path.basename(
                    out_dir.rstrip("/")) or "run",
                config={k: v for k, v in cfg.items() if not k.startswith("_")},
                dir=out_dir,
                reinit=True,
            )
            self._wandb = _wandb
            print("[wandb] init OK: project={} name={} url={}".format(
                self.run.project, self.run.name, self.run.url))
        except Exception as e:
            print(f"[wandb] init failed: {e}; continuing without wandb")
            self.run = None
            self._wandb = None

    def log(self, payload: dict, step: Optional[int] = None):
        if self.run is None:
            return
        try:
            self.run.log(payload, step=step)
        except Exception as e:
            print(f"[wandb] log failed: {e}")

    def summary_update(self, payload: dict):
        if self.run is None:
            return
        try:
            for k, v in payload.items():
                self.run.summary[k] = v
        except Exception as e:
            print(f"[wandb] summary failed: {e}")

    def log_video(self, key: str, path: str, caption: Optional[str] = None,
                  fps: int = 30):
        if self.run is None or self._wandb is None:
            return
        try:
            self.run.log({key: self._wandb.Video(path, fps=fps,
                                                 caption=caption,
                                                 format="mp4")})
        except Exception as e:
            print(f"[wandb] video upload failed for {path}: {e}")

    def finish(self):
        if self.run is None:
            return
        try:
            self.run.finish()
        except Exception as e:
            print(f"[wandb] finish failed: {e}")
        self.run = None


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------


def train_loop(cfg: dict, *, device: str = "cuda",
               max_epochs: Optional[int] = None,
               wandb_handle: Optional[_WandbHandle] = None) -> dict:
    """Run training end-to-end given a fully-resolved cfg dict.

    Cross-entropy training loop (heatmap CE on the Gaussian gaze target).
    Returns a dict with the final test eval numbers (best ckpt) plus paths to
    ckpts.
    """
    set_seed(cfg["seed"])
    out_dir = ensure_dir(cfg["out_dir"])
    log_path = os.path.join(out_dir, "train.log")
    log_f = open(log_path, "w")
    def log(msg):
        print(msg)
        log_f.write(msg + "\n"); log_f.flush()
    log("[train] cfg={}".format(json.dumps(cfg, indent=2, default=str)))

    if wandb_handle is None:
        wandb_handle = _WandbHandle()
        wandb_handle.init(cfg, out_dir, run_name=cfg.get("wandb_run_name"))

    log("[train] building model...")
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
    log("[train] trainable params: {}".format(model.trainable_param_count()))
    log("[train] head_kind: {}".format(model.head_kind))
    if model.is_temporal:
        log("[train] temporal_K: {} (n_temporal_layers={})".format(
            model.temporal_K, cfg.get("n_temporal_layers", 2)))
    sigma_in_grid = float(cfg.get("heatmap_sigma_in_grid", 1.0))
    log("[train] heatmap_sigma_in_grid={} (grid={})".format(
        sigma_in_grid, cfg.get("grid")))

    text_store = InMemTextStore()
    log("[train] building loaders (mode={})...".format(cfg.get("prompt_mode")))
    train_loader, val_loader, test_loader, stats = make_loaders_with_prompt_mode(
        cfg, model, text_store, device)
    for k, v in stats.items():
        log("[train] split {}: n_kept={} n_total={} fallback_rate={:.4f}".format(
            k, v["n_kept"], v["n_total"], v["fallback_rate"]))
    if any(stats[s]["fallback_rate"] > 0.20 for s in stats):
        log("[train] error: a split has > 20% missing-gaze (fallback) frames; aborting.")
        sys.exit(4)
    log("[train] text store has {} prompts".format(len(text_store.keys())))

    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                              betas=tuple(cfg["betas"]))
    epochs = max_epochs or cfg["epochs"]
    steps_per_epoch = max(1, len(train_loader))
    total_steps = epochs * steps_per_epoch
    sched = make_scheduler(optim, total_steps, cfg["warmup_steps"], cfg["schedule_lr_min"])
    use_bf16 = cfg.get("mixed_precision", "bf16") == "bf16"
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float32
    best_val = float("inf")
    best_path = os.path.join(out_dir, "best.pt")
    last_path = os.path.join(out_dir, "last.pt")
    plateau = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        ep_per_anc = None
        nb = 0
        t_ep = time.time()
        for batch in train_loader:
            vis = batch["vis_feat"].to(device, non_blocking=True)
            text = batch["text_feat"].to(device, non_blocking=True)
            mask = batch["text_mask"].to(device, non_blocking=True)
            gt = batch["gaze_xy"].to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_bf16):
                logits = model(vis, text, text_mask=mask)
                loss, per_anc = heatmap_ce_loss(
                    logits, gt, sigma_in_grid=sigma_in_grid)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optim.step()
            sched.step()
            ep_loss += loss.item()
            nb += 1
            if ep_per_anc is None:
                ep_per_anc = per_anc.detach().clone()
            else:
                ep_per_anc += per_anc.detach()
        ep_loss /= max(nb, 1)
        ep_per_anc = (ep_per_anc / max(nb, 1)).cpu().tolist()
        val = base_evaluate(model, val_loader, device,
                            sigma_in_grid=sigma_in_grid)
        elapsed = time.time() - t_ep
        log("[train] ep {:3d}: train={:.4f} {} | val={:.4f} {} | {:.1f}s".format(
            epoch, ep_loss, [round(x, 3) for x in ep_per_anc],
            val["loss"], [round(x, 3) for x in val["per_anchor"]], elapsed))
        h_entry = {"epoch": epoch, "train_loss": ep_loss,
                    "train_per_anc": ep_per_anc,
                    "val_loss": val["loss"],
                    "val_per_anc": val["per_anchor"]}

        # Optional periodic test-set CE, logged for monitoring only; checkpoint
        # selection and early stopping use the validation loss.
        test_eval_every = int(cfg.get("test_eval_every_epochs", 0))
        if (test_eval_every > 0 and len(test_loader.dataset) > 0
                and (epoch % test_eval_every == 0 or epoch == epochs)):
            t_test = base_evaluate(model, test_loader, device,
                                    sigma_in_grid=sigma_in_grid)
            log("[train]   periodic test_ce={:.4f} per_anc={}".format(
                t_test["loss"], [round(x, 3) for x in t_test["per_anchor"]]))
            h_entry["test_loss"] = t_test["loss"]
            h_entry["test_per_anc"] = t_test["per_anchor"]
        history.append(h_entry)

        # wandb per-epoch log
        wb_payload = {
            "epoch": epoch,
            "train/loss": ep_loss,
            "val/loss": val["loss"],
            "lr": optim.param_groups[0]["lr"],
            "train/elapsed_s": elapsed,
        }
        for i, x in enumerate(ep_per_anc):
            wb_payload[f"train/ce_anchor_{i}"] = float(x)
        for i, x in enumerate(val["per_anchor"]):
            wb_payload[f"val/ce_anchor_{i}"] = float(x)
        if "test_loss" in h_entry:
            wb_payload["test/loss"] = h_entry["test_loss"]
            for i, x in enumerate(h_entry["test_per_anc"]):
                wb_payload[f"test/ce_anchor_{i}"] = float(x)
        wandb_handle.log(wb_payload, step=epoch)

        torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch}, last_path)
        if val["loss"] < best_val - 1e-4:
            best_val = val["loss"]
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch,
                        "val_loss": val["loss"], "val_per_anc": val["per_anchor"]},
                       best_path)
            plateau = 0
            log("[train] new best val_loss={:.4f} saved".format(best_val))
        else:
            plateau += 1
            if plateau >= cfg.get("patience", 10):
                log("[train] early stop after {} plateau epochs".format(plateau))
                break
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    result: Dict[str, object] = {
        "out_dir": out_dir,
        "best_path": best_path,
        "last_path": last_path,
        "best_val_loss": float(best_val) if best_val != float("inf") else None,
        "history_path": os.path.join(out_dir, "history.json"),
    }

    if len(test_loader.dataset) > 0:
        log("[train] loading best ckpt for test eval...")
        sd = torch.load(best_path, map_location=device, weights_only=False)["model"]
        model.load_state_dict(sd)
        test = base_evaluate(model, test_loader, device,
                             sigma_in_grid=sigma_in_grid)
        log("[train] TEST loss={:.4f} per_anc={}".format(
            test["loss"], [round(x, 3) for x in test["per_anchor"]]))
        with open(os.path.join(out_dir, "test_eval.json"), "w") as f:
            json.dump({"loss": test["loss"], "per_anchor": test["per_anchor"]},
                      f, indent=2)
        wandb_handle.summary_update({
            "test/best_loss": float(test["loss"]),
        })
        result["test_loss"] = float(test["loss"])
        result["test_per_anchor"] = list(test["per_anchor"])
    else:
        log("[train] no test split; skipping post-train test eval")

    log("[train] done.")
    log_f.close()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_epochs", type=int, default=None)
    args = ap.parse_args()
    cfg = load_yaml_config(args.config)
    print("[train] config={}".format(args.config))
    wb = _WandbHandle()
    wb.init(cfg, ensure_dir(cfg["out_dir"]),
            run_name=cfg.get("wandb_run_name"))
    try:
        train_loop(cfg, device=args.device, max_epochs=args.max_epochs,
                   wandb_handle=wb)
    finally:
        wb.finish()


if __name__ == "__main__":
    main()
