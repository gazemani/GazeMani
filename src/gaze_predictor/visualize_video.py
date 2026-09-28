"""Video-based qualitative visualizations for the gaze predictor.

  1. ``render_train_episode_video``: One MP4 over a TRAIN episode, showing GT
     crosshairs (green) at anchor frames vs predicted crosshairs (red) from
     the model conditioned on the episode prompt.

  2. ``render_test_episode_video``: One MP4 per test episode using the
     episode's native prompt. Predictions (red) + optional GT (green) with
     causal EMA smoothing.

Colors here are diagnostic only; the policy-input gaze prompt is the fixed
cyan crosshair.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from .data import (
    load_episodes_meta,
    preprocess_image_224,
    read_episode_gaze,
    read_frame_left_crop,
)
from .ema_smoothing import ema_smooth
from .model import GazeTrajectoryPredictor
from .utils import windowed_soft_argmax


# ---------------------------------------------------------------------------
# Crosshair drawing.
# ---------------------------------------------------------------------------

# Default crosshair half-arm per anchor, in 2160-px space (run_train passes
# 120 px, matching the policy-input crosshair).
ANCHOR_CROSSHAIR_SIZES_2160 = (80, 70, 60, 50, 40)
ANCHOR_CROSSHAIR_THICKNESS = 6  # constant


def draw_crosshair(img_bgr: np.ndarray, x: float, y: float, color_bgr,
                   size: int = 60, thickness: int = 6) -> np.ndarray:
    """In-place draw of a gun-sight crosshair on a BGR image."""
    xi, yi = int(round(x)), int(round(y))
    cv2.line(img_bgr, (xi - size, yi), (xi + size, yi), color_bgr, thickness, cv2.LINE_AA)
    cv2.line(img_bgr, (xi, yi - size), (xi, yi + size), color_bgr, thickness, cv2.LINE_AA)
    cv2.circle(img_bgr, (xi, yi), max(size // 3, 4), color_bgr, thickness, cv2.LINE_AA)
    return img_bgr


def draw_anchor_set(img_bgr: np.ndarray, xys_2160: Sequence[Tuple[float, float]],
                    color_bgr, sizes: Sequence[int] = ANCHOR_CROSSHAIR_SIZES_2160,
                    thickness: int = ANCHOR_CROSSHAIR_THICKNESS) -> np.ndarray:
    for i, (x, y) in enumerate(xys_2160):
        s = sizes[min(i, len(sizes) - 1)]
        draw_crosshair(img_bgr, x, y, color_bgr, size=s, thickness=thickness)
    return img_bgr


# ---------------------------------------------------------------------------
# Text overlay helpers.
# ---------------------------------------------------------------------------

def burn_text(img_bgr: np.ndarray, lines: Sequence[str],
              org_xy: Tuple[int, int] = (20, 40),
              font_scale: float = 0.7, color=(255, 255, 255),
              thickness: int = 2, line_height: int = 30,
              shadow: bool = True) -> np.ndarray:
    x, y = org_xy
    for i, line in enumerate(lines):
        py = y + i * line_height
        if shadow:
            cv2.putText(img_bgr, line, (x + 1, py + 1), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(img_bgr, line, (x, py), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness, cv2.LINE_AA)
    return img_bgr


def burn_legend(img_bgr: np.ndarray, entries: Sequence[Tuple[str, Tuple[int, int, int]]],
                org_xy: Tuple[int, int] = (20, 1980),
                font_scale: float = 0.8, line_height: int = 40,
                swatch_size: int = 26) -> np.ndarray:
    x, y = org_xy
    for i, (label, color_bgr) in enumerate(entries):
        py = y + i * line_height
        cv2.rectangle(img_bgr, (x, py - swatch_size + 4),
                      (x + swatch_size, py + 4), color_bgr, -1)
        cv2.rectangle(img_bgr, (x, py - swatch_size + 4),
                      (x + swatch_size, py + 4), (0, 0, 0), 2)
        tx = x + swatch_size + 14
        cv2.putText(img_bgr, label, (tx + 1, py + 1), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img_bgr, label, (tx, py), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), 2, cv2.LINE_AA)
    return img_bgr


# ---------------------------------------------------------------------------
# Model inference helpers.
# ---------------------------------------------------------------------------

class _TextCache:
    """Tokenize+encode text once per prompt; reuse fusion-projected feats."""
    def __init__(self, model: GazeTrajectoryPredictor, device: str):
        self.model = model
        self.device = device
        self.cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    @torch.no_grad()
    def get(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if prompt in self.cache:
            return self.cache[prompt]
        text_proj, mask = self.model.encode_text([prompt])
        self.cache[prompt] = (text_proj.detach(), mask.detach())
        return self.cache[prompt]


class _RollingVisCache:
    """Per-stream rolling cache of the last K projected vision features.

    The temporal head consumes a stack of [B=1, K, P, fusion_dim]. The cache
    is indexed by an arbitrary stream id (e.g. episode index).
    """
    def __init__(self, K: int):
        self.K = int(K)
        self._buf: Dict[str, List[torch.Tensor]] = {}

    def push_get_stack(self, stream_id: str, vis_proj_1pd: torch.Tensor
                       ) -> torch.Tensor:
        """vis_proj_1pd: [1, P, fusion_dim] — current frame's projected feats.
        Returns [1, K, P, fusion_dim] with the past K-1 frames + current,
        clamped to the first frame for early steps.
        """
        buf = self._buf.setdefault(stream_id, [])
        if not buf:
            stacked = vis_proj_1pd.unsqueeze(1).expand(-1, self.K, -1, -1).contiguous()
            buf.append(vis_proj_1pd)
            return stacked
        buf.append(vis_proj_1pd)
        if len(buf) > self.K:
            buf.pop(0)
        if len(buf) < self.K:
            need = self.K - len(buf)
            stacked = torch.cat([buf[0]] * need + buf, dim=0)              # [K, P, D]
        else:
            stacked = torch.cat(buf, dim=0)                                # [K, P, D]
        return stacked.unsqueeze(0)                                         # [1, K, P, D]


@torch.no_grad()
def predict_anchors_from_image(model: GazeTrajectoryPredictor,
                                img2160: np.ndarray,
                                prompts: Sequence[str],
                                text_cache: _TextCache,
                                cfg: dict,
                                device: str,
                                rolling: Optional["_RollingVisCache"] = None,
                                stream_id: str = "default"
                                ) -> Dict[str, List[Tuple[float, float]]]:
    """Encode the (single) image once, then run head per prompt. Returns
    ``{prompt: [(x, y), ...]}``."""
    x224 = preprocess_image_224(img2160, cfg["clip_mean"], cfg["clip_std"]).unsqueeze(0).to(device)
    vis_proj = model.encode_image(x224)  # [1, P, fusion_dim]
    if model.is_temporal:
        if rolling is None:
            vis_in = vis_proj.unsqueeze(1).expand(-1, model.temporal_K, -1, -1).contiguous()
        else:
            vis_in = rolling.push_get_stack(stream_id, vis_proj)
    else:
        vis_in = vis_proj
    out: Dict[str, List[Tuple[float, float]]] = {}
    for prompt in prompts:
        text_proj, mask = text_cache.get(prompt)
        logits = model.head(vis_in, text_proj, text_mask=mask)
        prob = logits.flatten(-2).softmax(-1).reshape(logits.shape)        # [1, N, G, G]
        pts: List[Tuple[float, float]] = []
        N = prob.shape[1]
        for k in range(N):
            xy = windowed_soft_argmax(prob[:, k:k+1], image_size=2160)[0, 0]
            pts.append((float(xy[0].item()), float(xy[1].item())))
        out[prompt] = pts
    return out


# ---------------------------------------------------------------------------
# Video writer wrapper.
# ---------------------------------------------------------------------------

class _Mp4Writer:
    def __init__(self, path: str, fps: float, frame_size_hw: Tuple[int, int]):
        h, w = frame_size_hw
        self.path = path
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.w = cv2.VideoWriter(path, fourcc, fps, (w, h))
        if not self.w.isOpened():
            raise RuntimeError(f"VideoWriter failed to open: {path}")

    def write_bgr(self, frame_bgr: np.ndarray):
        self.w.write(frame_bgr)

    def close(self):
        self.w.release()


# ---------------------------------------------------------------------------
# Train video: GT vs pred on a chosen train episode.
# ---------------------------------------------------------------------------

GT_COLOR_BGR = (0, 255, 0)            # green
PRED_COLOR_BGR = (0, 0, 255)          # red

ANCHOR_OFFSETS_DEFAULT = (0,)


def _pick_train_episode(train_dataset_path: str, prefer_ep: int = 0,
                        min_len: int = 100,
                        gaze_fb_bad: Tuple[Tuple[int, int], ...] = ((0, 0),),
                        max_bad_frac: float = 0.05) -> Tuple[int, str, int]:
    """Return (episode_index, prompt, length). Tries ``prefer_ep`` first."""
    eps = load_episodes_meta(train_dataset_path)
    candidates = [e for e in eps if e["length"] >= min_len]
    ordered = sorted(candidates, key=lambda e: 0 if e["episode_index"] == prefer_ep else 1)
    for e in ordered:
        ep = e["episode_index"]
        prompt = e["tasks"][0]
        try:
            g = read_episode_gaze(train_dataset_path, ep)
        except Exception:
            continue
        n = len(g)
        bad_count = 0
        for fb in gaze_fb_bad:
            bad_count += int(((g[:, 0] == fb[0]) & (g[:, 1] == fb[1])).sum())
        if bad_count / max(n, 1) > max_bad_frac:
            continue
        return ep, prompt, n
    raise RuntimeError(f"no usable train episode in {train_dataset_path}")


def render_train_episode_video(model: GazeTrajectoryPredictor,
                               train_dataset_path: str,
                               cfg: dict,
                               out_path: str,
                               device: str,
                               horizon_offsets: Sequence[int] = ANCHOR_OFFSETS_DEFAULT,
                               fps: int = 30,
                               output_size: int = 720,
                               prefer_ep: int = 0,
                               anchor_sizes_2160: Optional[Sequence[int]] = None) -> str:
    ep, ep_prompt, n_frames = _pick_train_episode(train_dataset_path, prefer_ep=prefer_ep)
    prompt = ep_prompt
    print(f"[train video] dataset={train_dataset_path}  episode={ep}  len={n_frames}  "
          f"prompt={prompt!r}")
    gaze = read_episode_gaze(train_dataset_path, ep)  # [T, 2] in 2160 px
    horizon = max(horizon_offsets)
    last_t_with_gt = n_frames - 1 - horizon
    text_cache = _TextCache(model, device)
    rolling = (_RollingVisCache(model.temporal_K) if model.is_temporal else None)

    writer = _Mp4Writer(out_path, fps=fps, frame_size_hw=(output_size, output_size))
    last_pred: List[Tuple[float, float]] = []
    last_gt: List[Tuple[float, float]] = []

    try:
        for t in range(n_frames):
            img2160 = read_frame_left_crop(train_dataset_path, ep, t)  # RGB 2160x2160
            bgr = cv2.cvtColor(img2160, cv2.COLOR_RGB2BGR)

            if t <= last_t_with_gt:
                preds = predict_anchors_from_image(
                    model, img2160, [prompt], text_cache, cfg, device,
                    rolling=rolling, stream_id=f"train_ep_{ep}")
                pred_xys = preds[prompt]
                gt_xys: List[Tuple[float, float]] = []
                for off in horizon_offsets:
                    g = gaze[t + off]
                    gt_xys.append((float(g[0]), float(g[1])))
                last_pred = pred_xys
                last_gt = gt_xys
                state_str = "OK"
            else:
                pred_xys = last_pred
                gt_xys = last_gt
                state_str = f"FROZEN (t > len - {horizon + 1})"

            sizes_arg = (tuple(int(s) for s in anchor_sizes_2160)
                         if anchor_sizes_2160 is not None
                         else ANCHOR_CROSSHAIR_SIZES_2160)
            if gt_xys:
                draw_anchor_set(bgr, gt_xys, GT_COLOR_BGR, sizes=sizes_arg)
            if pred_xys:
                draw_anchor_set(bgr, pred_xys, PRED_COLOR_BGR, sizes=sizes_arg)

            text_lines = [
                f"TRAIN ep {ep}  frame {t}/{n_frames - 1}  [{state_str}]",
                f"prompt: {prompt}",
                f"anchors offsets: {list(horizon_offsets)}",
            ]
            burn_text(bgr, text_lines, org_xy=(20, 60), font_scale=1.2, thickness=3,
                      line_height=46)
            burn_legend(
                bgr,
                entries=[
                    ("GT (green)", GT_COLOR_BGR),
                    ("pred (red)", PRED_COLOR_BGR),
                ],
                org_xy=(20, 2050),
                font_scale=1.1,
                line_height=50,
                swatch_size=36,
            )

            small = cv2.resize(bgr, (output_size, output_size), interpolation=cv2.INTER_AREA)
            writer.write_bgr(small)
            if (t + 1) % 50 == 0 or t == n_frames - 1:
                print(f"  frame {t + 1}/{n_frames}")
    finally:
        writer.close()
    print(f"[train video] wrote {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Test video: one episode, native prompt, red prediction + (optional) green GT,
# with EMA(alpha) smoothing applied to predictions.
# ---------------------------------------------------------------------------


def _gt_is_broken(gaze_xy: np.ndarray) -> bool:
    """Return True if an episode's GT gaze stream is missing or constant."""
    if gaze_xy is None or gaze_xy.size == 0:
        return True
    if gaze_xy.std(axis=0).max() < 1.0:
        return True
    pairs = [tuple(map(float, p)) for p in gaze_xy[:1000]]
    from collections import Counter
    if not pairs:
        return True
    most = Counter(pairs).most_common(1)[0][1]
    return most / len(pairs) > 0.95


def render_test_episode_video(model: GazeTrajectoryPredictor,
                               test_dataset_path: str,
                               cfg: dict,
                               out_path: str,
                               device: str,
                               episode_index: int,
                               horizon_offsets: Sequence[int] = ANCHOR_OFFSETS_DEFAULT,
                               fps: int = 30,
                               output_size: int = 720,
                               anchor_sizes_2160: Optional[Sequence[int]] = None,
                               ema_alpha: Optional[float] = 0.10) -> str:
    """Render a single test-episode video using the EPISODE'S NATIVE PROMPT.

    - Predictions (red) are computed with the model conditioned on the
      ``tasks[0]`` from ``meta/episodes.jsonl`` for the given test dataset and
      episode.
    - If the test episode has valid GT (``observation.gaze`` not
      degenerate), green crosshairs are also drawn at anchor offsets.
    - Predictions are smoothed with causal EMA (``ema_alpha``) before drawing.
    """
    eps = load_episodes_meta(test_dataset_path)
    by_idx = {int(e["episode_index"]): e for e in eps}
    if episode_index not in by_idx:
        raise KeyError(f"episode {episode_index} not in {test_dataset_path}")
    e = by_idx[episode_index]
    n_frames = int(e["length"])
    prompt = e["tasks"][0]
    print(f"[test video] dataset={test_dataset_path}  episode={episode_index}  "
          f"len={n_frames}  prompt={prompt!r}")

    try:
        gaze = read_episode_gaze(test_dataset_path, episode_index)  # [T, 2]
        has_gt = not _gt_is_broken(gaze)
    except Exception:
        gaze = None
        has_gt = False
    if not has_gt:
        print("[test video]   GT gaze unavailable or constant; rendering prediction only")

    horizon = max(horizon_offsets)
    last_t_with_gt = n_frames - 1 - horizon
    text_cache = _TextCache(model, device)
    rolling = (_RollingVisCache(model.temporal_K) if model.is_temporal else None)

    # Pass 1: collect raw predictions for ALL valid t. We'll EMA-smooth then
    # walk frames again for drawing.
    raw_pred: List[List[Tuple[float, float]]] = []
    t_to_pred_idx: Dict[int, int] = {}
    for t in range(n_frames):
        if t > last_t_with_gt:
            continue
        img2160 = read_frame_left_crop(test_dataset_path, episode_index, t)
        preds = predict_anchors_from_image(
            model, img2160, [prompt], text_cache, cfg, device,
            rolling=rolling, stream_id=f"test_ep_{episode_index}")
        raw_pred.append(preds[prompt])
        t_to_pred_idx[t] = len(raw_pred) - 1
    raw_arr = np.asarray(raw_pred, dtype=np.float32)  # [N_t, n_anchors, 2]
    if ema_alpha is not None and raw_arr.size > 0:
        sm = ema_smooth(raw_arr, alpha=float(ema_alpha))
    else:
        sm = raw_arr

    sizes_arg = (tuple(int(s) for s in anchor_sizes_2160)
                 if anchor_sizes_2160 is not None
                 else ANCHOR_CROSSHAIR_SIZES_2160)

    writer = _Mp4Writer(out_path, fps=fps, frame_size_hw=(output_size, output_size))
    last_pred: List[Tuple[float, float]] = []
    last_gt: List[Tuple[float, float]] = []
    try:
        for t in range(n_frames):
            img2160 = read_frame_left_crop(test_dataset_path, episode_index, t)
            bgr = cv2.cvtColor(img2160, cv2.COLOR_RGB2BGR)
            if t in t_to_pred_idx:
                pi = t_to_pred_idx[t]
                pred_xys = [(float(sm[pi, k, 0]), float(sm[pi, k, 1]))
                            for k in range(sm.shape[1])]
                if has_gt and gaze is not None:
                    gt_xys = []
                    for off in horizon_offsets:
                        g = gaze[t + off]
                        gt_xys.append((float(g[0]), float(g[1])))
                else:
                    gt_xys = []
                last_pred = pred_xys
                last_gt = gt_xys
                state_str = "OK"
            else:
                pred_xys = last_pred
                gt_xys = last_gt
                state_str = f"FROZEN (t > len - {horizon + 1})"

            if gt_xys:
                draw_anchor_set(bgr, gt_xys, GT_COLOR_BGR, sizes=sizes_arg)
            if pred_xys:
                draw_anchor_set(bgr, pred_xys, PRED_COLOR_BGR, sizes=sizes_arg)

            text_lines = [
                f"TEST ep {episode_index}  frame {t}/{n_frames - 1}  [{state_str}]",
                f"prompt: {prompt}",
                f"GT: {'shown (green)' if has_gt else 'unavailable'}",
            ]
            burn_text(bgr, text_lines, org_xy=(20, 60), font_scale=1.2,
                      thickness=3, line_height=46)
            legend_entries = [("pred (red)", PRED_COLOR_BGR)]
            if has_gt:
                legend_entries.insert(0, ("GT (green)", GT_COLOR_BGR))
            burn_legend(bgr, entries=legend_entries,
                        org_xy=(20, 2050), font_scale=1.1,
                        line_height=50, swatch_size=36)

            small = cv2.resize(bgr, (output_size, output_size),
                                interpolation=cv2.INTER_AREA)
            writer.write_bgr(small)
            if (t + 1) % 100 == 0 or t == n_frames - 1:
                print(f"  ep {episode_index} frame {t + 1}/{n_frames}")
    finally:
        writer.close()
    print(f"[test video] wrote {out_path}")
    return out_path
