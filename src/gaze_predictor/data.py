"""LeRobot v2.1 dataset loader for the gaze predictor.

Each "sample" is (image_at_t, prompt, gaze_xy) where ``gaze_xy`` covers
``len(anchor_offsets)`` anchors at fixed offsets from t (released configs:
anchor_offsets=[0], i.e. the gaze at the current frame t). The image is
the LEFT half of the head_camera frame (4320x2160 -> 2160x2160) with NO
rotation.

Vision features are pre-cached by cache.py; the dataset itself returns
either raw images (for the cache pass) or pre-cached features (for training).

Standalone module — no internal package imports.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# meta loading
# ---------------------------------------------------------------------------

def load_episodes_meta(dataset_path: str) -> List[Dict]:
    """Returns list of {episode_index, tasks, length}."""
    out = []
    with open(os.path.join(dataset_path, 'meta', 'episodes.jsonl')) as f:
        for line in f:
            out.append(json.loads(line))
    return out


def parquet_path(dataset_path: str, episode_index: int) -> str:
    return os.path.join(dataset_path, 'data', 'chunk-000',
                        f'episode_{episode_index:06d}.parquet')


def video_path(dataset_path: str, episode_index: int,
               video_key: str = 'observation.images.head_camera') -> str:
    return os.path.join(dataset_path, 'videos', 'chunk-000', video_key,
                        f'episode_{episode_index:06d}.mp4')


def read_episode_gaze(dataset_path: str, episode_index: int) -> np.ndarray:
    """[T, 2] gaze coords in 2160 px left-crop space (x, y)."""
    df = pd.read_parquet(parquet_path(dataset_path, episode_index),
                         columns=['observation.gaze', 'frame_index'])
    col = df['observation.gaze']
    arr = np.stack([np.asarray(x, dtype=np.float32) for x in col])  # [T, 2]
    return arr


# ---------------------------------------------------------------------------
# image i/o (decord)
# ---------------------------------------------------------------------------

class _VideoCache:
    """Simple cache of decord readers keyed by path. Decord readers are heavy."""
    def __init__(self, max_open: int = 4):
        self.max_open = max_open
        self.readers: Dict[str, "decord.VideoReader"] = {}
        self.order: List[str] = []

    def get(self, path: str):
        import decord
        if path in self.readers:
            return self.readers[path]
        if len(self.readers) >= self.max_open:
            oldest = self.order.pop(0)
            self.readers.pop(oldest, None)
        self.readers[path] = decord.VideoReader(path, num_threads=1)
        self.order.append(path)
        return self.readers[path]


_VC = _VideoCache()


def read_frame_left_crop(dataset_path: str, episode_index: int,
                         frame_index: int) -> np.ndarray:
    """Returns 2160x2160x3 uint8 RGB (left half of 4320x2160)."""
    vp = video_path(dataset_path, episode_index)
    vr = _VC.get(vp)
    frame = vr[frame_index].asnumpy()   # HxWx3, RGB
    # left half
    if frame.shape[1] >= 4320:
        frame = frame[:, :2160, :]
    elif frame.shape[1] >= 2160 and frame.shape[0] == 2160:
        frame = frame[:, :2160, :]
    else:
        # fall back: assume already left
        pass
    return frame


def preprocess_image_224(frame_2160: np.ndarray, mean: Sequence[float],
                         std: Sequence[float]) -> torch.Tensor:
    """[2160,2160,3] uint8 RGB -> [3, 224, 224] float32 normalized for CLIP."""
    img = cv2.resize(frame_2160, (224, 224), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = (img - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return torch.from_numpy(img.transpose(2, 0, 1)).contiguous()


# ---------------------------------------------------------------------------
# sampling plan
# ---------------------------------------------------------------------------

@dataclass
class SamplePlan:
    dataset_path: str
    episode_index: int
    prompt: str
    t: int
    anchor_frames: Tuple[int, ...]


def make_sample_plan(dataset_path: str,
                     episodes: Sequence[Dict],
                     anchor_offsets: Sequence[int],
                     samples_per_episode: int,
                     min_episode_length: int,
                     gaze_fallback_value: Tuple[int, int] = (0, 0),
                     rng: Optional[random.Random] = None,
                     verify_fallback: bool = True) -> Tuple[List[SamplePlan], Dict]:
    """Builds a list of sample plans, dropping samples whose anchor frame(s)
    carry the fallback (missing) gaze value.

    Returns (plans, stats) where stats has 'fallback_rate', 'n_kept', 'n_total',
    'fallback_frames', 'total_frames'.
    """
    rng = rng or random.Random(42)
    horizon = max(anchor_offsets) + 1
    fb_x, fb_y = gaze_fallback_value
    plans: List[SamplePlan] = []
    fallback_frames = 0
    total_frames = 0
    n_total = 0
    n_kept = 0
    for ep in episodes:
        ep_len = ep['length']
        if ep_len < min_episode_length:
            continue
        prompt = ep['tasks'][0]
        ep_idx = ep['episode_index']
        gaze = read_episode_gaze(dataset_path, ep_idx)  # [T,2]
        total_frames += ep_len
        is_fb = (gaze[:, 0] == fb_x) & (gaze[:, 1] == fb_y)
        fallback_frames += int(is_fb.sum())
        # Sample s timesteps
        max_t = ep_len - horizon
        if max_t < 0:
            continue
        for _ in range(samples_per_episode):
            t = rng.randint(0, max_t)
            anchors = tuple(t + o for o in anchor_offsets)
            n_total += 1
            if any(is_fb[a] for a in anchors):
                continue
            plans.append(SamplePlan(dataset_path, ep_idx, prompt, t, anchors))
            n_kept += 1
    stats = {
        'fallback_frames': fallback_frames,
        'total_frames': total_frames,
        'fallback_rate': fallback_frames / max(total_frames, 1),
        'n_kept': n_kept,
        'n_total': n_total,
    }
    return plans, stats


def split_episodes_by_ratio(episodes: Sequence[Dict], val_ratio: float,
                            seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """Episode-level split, preserving prompt balance via stratified shuffle."""
    rng = random.Random(seed)
    by_prompt: Dict[str, List[Dict]] = {}
    for e in episodes:
        by_prompt.setdefault(e['tasks'][0], []).append(e)
    train: List[Dict] = []
    val: List[Dict] = []
    for group in by_prompt.values():
        idx = list(range(len(group)))
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_ratio)))
        val_idx = set(idx[:n_val])
        for i, ep in enumerate(group):
            (val if i in val_idx else train).append(ep)
    return train, val


# ---------------------------------------------------------------------------
# Dataset that uses pre-cached frozen-encoder features
# ---------------------------------------------------------------------------

class CachedFeatureDataset(Dataset):
    """Returns (vis_feat[196,D], text_feat[T,D], gaze_xy[N,2])."""

    def __init__(self, plans: Sequence[SamplePlan], cache_dir: str,
                 prompt_to_text_path: Dict[str, str]):
        self.plans = list(plans)
        self.cache_dir = cache_dir
        self.prompt_to_text_path = prompt_to_text_path

    def __len__(self):
        return len(self.plans)

    def _vis_path(self, p: SamplePlan) -> str:
        ds_name = os.path.basename(p.dataset_path.rstrip('/'))
        return os.path.join(self.cache_dir, 'vis',
                            f'{ds_name}__ep_{p.episode_index:06d}__f_{p.t:06d}.npz')

    def __getitem__(self, idx: int):
        p = self.plans[idx]
        vis = np.load(self._vis_path(p))['vis_feat']  # [196, D] (single frame, t)
        gaze = read_episode_gaze(p.dataset_path, p.episode_index)
        gaze_anchors = np.stack([gaze[a] for a in p.anchor_frames]).astype(np.float32)
        txt = np.load(self.prompt_to_text_path[p.prompt])['text_feat']  # [T, D]
        return {
            'vis_feat': torch.from_numpy(vis).float(),
            'text_feat': torch.from_numpy(txt).float(),
            'gaze_xy': torch.from_numpy(gaze_anchors).float(),
            'prompt': p.prompt,
            'episode_index': p.episode_index,
            't': p.t,
            'anchor_frames': torch.as_tensor(p.anchor_frames, dtype=torch.long),
        }
