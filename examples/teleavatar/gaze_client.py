#!/usr/bin/env python3
"""
GazeAnnotator: transparent wrapper that draws a crosshair on the head_camera
image based on a gaze predictor's output, before the image is sent to the
policy server.

Pipeline:
    head_camera (224x224 RGB uint8)
        --> CLIP-normalize
        --> rolling K=3 vision feature cache
        --> head forward (cached text features per prompt)
        --> windowed soft-argmax over heatmap logits (in 2160-px space)
        --> causal EMA smoothing
        --> rescale gaze to 224 px
        --> draw cyan crosshair on a copy of the input image
        --> return annotated 224x224 RGB uint8

Color order: head_camera arrives as RGB (PyAV decoded with format='rgb24' in
ros2_interface._ffmpeg_callback). The annotator preserves RGB on output.
Crosshair color (0, 255, 255) is therefore cyan in the RGB image.

Module imports the gaze_predictor package that ships with this repo at
src/gaze_predictor/.
"""

from __future__ import annotations

import logging
import os
import queue
import sys
import threading
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml

# ``gaze_predictor`` ships as a package under src/. When openpi is installed
# (``uv pip install -e .``) the plain import works; otherwise (e.g. running on
# the robot with the system ROS python) fall back to adding the repo's
# ``src/`` dir to sys.path.
try:
    from gaze_predictor.model import GazeTrajectoryPredictor
    from gaze_predictor.utils import windowed_soft_argmax
except ImportError:
    _SRC_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    if _SRC_DIR not in sys.path:
        sys.path.insert(0, _SRC_DIR)
    from gaze_predictor.model import GazeTrajectoryPredictor  # noqa: E402
    from gaze_predictor.utils import windowed_soft_argmax  # noqa: E402


# CLIP normalization constants — must match training cfg.
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def map_to_gaze_prompt(pi0_prompt: str) -> str:
    """Map a pi0 task prompt to the prompt the gaze predictor was trained on.

    The predictor is trained jointly on all tasks with the same instruction
    strings the policy consumes, so this is an identity mapping. Use
    ``--gaze-prompt-override`` on main.py to feed the predictor a different
    prompt than the policy.
    """
    return pi0_prompt


class AsyncFrameRecorder:
    """Background frame writer for the annotated head_camera stream.

    Designed for the deploy hot path: the producer (`get_observation`) calls
    `push(frame_rgb)` from the control-loop thread, which only has to do a
    bounded ndarray copy + non-blocking enqueue (~1 ms for 224x224x3). Disk
    encode + write happens on the worker thread.

    On queue overflow (worker behind), frames are silently dropped to favor
    control-loop responsiveness over completeness. The drop count is logged.

    File layout:
        <out_dir>/ep_<episode:04d>/<step:06d>.png

    Episode counter is bumped by `reset_episode()`. Step counter resets per
    episode. Both counters are owned by the producer, so no lock is needed
    for them — `push` is the only mutator.
    """

    def __init__(self, out_dir: str, queue_size: int = 64,
                 every_n_frames: int = 30):
        """
        Args:
            out_dir: where to dump png frames.
            queue_size: bounded queue, drops on overflow (favor responsiveness).
            every_n_frames: only save 1 out of every N frames. Default 30 =
                ~1 Hz at 30 Hz control. Set to 1 to save every frame.
        """
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.queue: "queue.Queue[Optional[Tuple[int, int, np.ndarray]]]" = (
            queue.Queue(maxsize=queue_size)
        )
        self._step = 0           # increments every push() call
        self._episode = 0
        self._dropped = 0
        self._written = 0
        self._every_n = max(1, int(every_n_frames))
        self._stop = False
        self.worker = threading.Thread(
            target=self._worker, daemon=True, name="GazeFrameRecorder"
        )
        self.worker.start()
        logging.info(
            f"[AsyncFrameRecorder] started. out_dir={self.out_dir} "
            f"queue_size={queue_size} every_n_frames={self._every_n}"
        )

    def push(self, frame_rgb: np.ndarray) -> None:
        """Non-blocking enqueue of an annotated RGB frame.

        Only every Nth frame is enqueued (subsampling for ~1 Hz disk write).
        Drops on overflow (logs a counter). The producer must not be allowed
        to block on disk I/O — that would stall the control loop.
        """
        if self._stop:
            return
        # Subsample: skip cheaply on non-save frames (no copy, no enqueue).
        if self._step % self._every_n != 0:
            self._step += 1
            return
        try:
            # Copy is required: caller may mutate or reuse the buffer.
            self.queue.put_nowait((self._episode, self._step, frame_rgb.copy()))
        except queue.Full:
            self._dropped += 1
            if self._dropped % 16 == 1:
                logging.warning(
                    f"[AsyncFrameRecorder] queue full, dropped frame "
                    f"(total dropped={self._dropped})"
                )
        self._step += 1

    def reset_episode(self) -> None:
        """Bump episode counter and reset step counter. Call at episode start."""
        self._episode += 1
        self._step = 0

    def _worker(self) -> None:
        while True:
            item = self.queue.get()
            try:
                if item is None:
                    break
                ep, step, frame = item
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                ep_dir = os.path.join(self.out_dir, f"ep_{ep:04d}")
                os.makedirs(ep_dir, exist_ok=True)
                cv2.imwrite(os.path.join(ep_dir, f"{step:06d}.png"), bgr)
                self._written += 1
            except Exception as e:  # noqa: BLE001
                logging.error(f"[AsyncFrameRecorder] write failed: {e}")
            finally:
                self.queue.task_done()

    def shutdown(self, timeout: float = 5.0) -> None:
        """Drain the queue and stop the worker. Safe to call multiple times."""
        if self._stop:
            return
        self._stop = True
        try:
            self.queue.put(None, timeout=1.0)
        except queue.Full:
            pass
        self.worker.join(timeout=timeout)
        logging.info(
            f"[AsyncFrameRecorder] shutdown. written={self._written} "
            f"dropped={self._dropped}"
        )


class GazeAnnotator:
    """Loads a gaze predictor checkpoint and annotates 224x224 head_camera
    frames with a single crosshair at the predicted gaze location.

    Stateful: maintains a rolling K-frame vision feature cache and a causal
    EMA state for the (x, y) output. Call ``reset()`` at the start of each
    episode.
    """

    def __init__(
        self,
        ckpt_path: str,
        cfg_path: str,
        ema_alpha: float = 0.2,
        device: Optional[str] = None,
        crosshair_color_rgb: Tuple[int, int, int] = (0, 255, 255),  # cyan
        crosshair_size_px: int = 12,
        crosshair_thickness: int = 1,
    ):
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"gaze ckpt not found: {ckpt_path}")
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"gaze config not found: {cfg_path}")

        with open(cfg_path, "r") as f:
            self.cfg: Dict = yaml.safe_load(f)

        # Resolve device + dtype.
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        # bf16 if supported (matches training mixed_precision).
        self._use_bf16 = (
            self.device.type == "cuda"
            and torch.cuda.is_bf16_supported()
            and self.cfg.get("mixed_precision", "fp32") == "bf16"
        )
        self._dtype = torch.bfloat16 if self._use_bf16 else torch.float32

        # Build model.
        self.model = GazeTrajectoryPredictor(
            encoder_kind=self.cfg.get("encoder_kind", "clip_clip"),
            fusion_dim=int(self.cfg.get("fusion_dim", 256)),
            n_anchors=int(self.cfg.get("n_anchors", 1)),
            grid=int(self.cfg.get("grid", 224)),
            clip_model_id=self.cfg.get("clip_model_id", "openai/clip-vit-base-patch16"),
            head_kind=self.cfg.get("head_kind"),
            temporal_K=int(self.cfg.get("temporal_K", 3)),
            n_temporal_layers=int(self.cfg.get("n_temporal_layers", 2)),
        )

        # Load weights.
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            logging.warning(
                f"[GazeAnnotator] load_state_dict missing={missing} unexpected={unexpected}"
            )
        self.model.eval()
        self.model.to(self.device)

        # Inference dtype is handled by torch.autocast in predict_xy_2160().

        # K-frame cache and text cache.
        self._K = int(self.cfg.get("temporal_K", 3))
        self._is_temporal = self.model.is_temporal
        self._vis_buf: list[torch.Tensor] = []  # each [1, P, fusion_dim]
        self._text_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

        # EMA state.
        self.ema_alpha = float(ema_alpha)
        self._ema_xy: Optional[np.ndarray] = None  # shape (2,)
        # Latest smoothed (x, y) in 224-px space, written by ``annotate``.
        # ``None`` until the first prediction.
        self.last_xy_224: Optional[tuple] = None

        # Drawing config.
        self._color = tuple(int(c) for c in crosshair_color_rgb)
        self._size = int(crosshair_size_px)
        self._thickness = int(crosshair_thickness)

        # Image scale: predictor reports in 2160-px space; head_camera is 224.
        self._image_scale = 224.0 / 2160.0

        # CLIP normalization constants on GPU (avoid per-frame CPU subtract/divide).
        # Shape [1, 3, 1, 1] for broadcast against [1, 3, 224, 224].
        self._clip_mean_gpu = torch.tensor(
            CLIP_MEAN, device=self.device, dtype=torch.float32
        ).view(1, 3, 1, 1)
        self._clip_std_gpu = torch.tensor(
            CLIP_STD, device=self.device, dtype=torch.float32
        ).view(1, 3, 1, 1)

        logging.info(
            f"[GazeAnnotator] ready. K={self._K} is_temporal={self._is_temporal} "
            f"device={self.device} dtype={self._dtype} ema_alpha={self.ema_alpha} "
            f"output_grid={self.cfg.get('grid')}"
        )

    # ------------------------------------------------------------------ utils

    def _normalize_to_clip(self, image_224_rgb_uint8: np.ndarray) -> torch.Tensor:
        """[224, 224, 3] uint8 RGB -> [1, 3, 224, 224] float32 normalized.
        All heavy ops (cast, /255, normalize) run on GPU to avoid CPU bottleneck."""
        if image_224_rgb_uint8.dtype != np.uint8:
            raise TypeError(
                f"expected uint8, got {image_224_rgb_uint8.dtype}"
            )
        if image_224_rgb_uint8.shape != (224, 224, 3):
            raise ValueError(
                f"expected 224x224x3, got {image_224_rgb_uint8.shape}"
            )
        # uint8 -> GPU asap (small payload, ~150KB), then do float math on GPU.
        t = torch.from_numpy(image_224_rgb_uint8).to(
            self.device, non_blocking=True
        )                                  # [224, 224, 3] uint8 on GPU
        t = t.permute(2, 0, 1).unsqueeze(0)        # [1, 3, 224, 224]
        t = t.to(torch.float32).div_(255.0)        # GPU cast + scale
        t = (t - self._clip_mean_gpu) / self._clip_std_gpu  # GPU normalize
        return t

    @torch.no_grad()
    def _get_text_features(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        cached = self._text_cache.get(prompt)
        if cached is not None:
            return cached
        text_proj, mask = self.model.encode_text([prompt])
        self._text_cache[prompt] = (text_proj.detach(), mask.detach())
        return self._text_cache[prompt]

    def _push_vis_and_get_stack(self, vis_proj_1pd: torch.Tensor) -> torch.Tensor:
        """Mirror gaze_predictor.visualize_video._RollingVisCache.push_get_stack."""
        K = self._K
        if not self._vis_buf:
            self._vis_buf.append(vis_proj_1pd)
            return vis_proj_1pd.unsqueeze(1).expand(-1, K, -1, -1).contiguous()
        self._vis_buf.append(vis_proj_1pd)
        if len(self._vis_buf) > K:
            self._vis_buf.pop(0)
        if len(self._vis_buf) < K:
            need = K - len(self._vis_buf)
            stacked = torch.cat([self._vis_buf[0]] * need + self._vis_buf, dim=0)
        else:
            stacked = torch.cat(self._vis_buf, dim=0)
        return stacked.unsqueeze(0)  # [1, K, P, D]

    def _apply_ema(self, xy: np.ndarray) -> np.ndarray:
        if self._ema_xy is None:
            self._ema_xy = xy.astype(np.float32)
        else:
            self._ema_xy = (
                self.ema_alpha * xy.astype(np.float32)
                + (1.0 - self.ema_alpha) * self._ema_xy
            )
        return self._ema_xy.copy()

    def _draw_crosshair(self, image_rgb: np.ndarray, x_2160: float,
                        y_2160: float) -> np.ndarray:
        """Draw crosshair directly on the 224-px image. Parameters are scaled
        from the training convention used by
        ``openpi.policies.teleavatar_policy.GazeCrosshairOverlay``
        (crosshair_size=120, thickness=6 in 2160-px space) by 224/2160
        (i.e. ÷9.64), giving size≈12, thickness=1 with LINE_AA. This
        approximates the training-time rendering (drawn at 2160 px, then
        resized) directly at 224 px, avoiding a resize round-trip."""
        out = image_rgb.copy()
        # 2160 px coords -> 224 px coords
        scale = 224.0 / 2160.0
        x = x_2160 * scale
        y = y_2160 * scale
        xi = int(round(np.clip(x, 0, 223)))
        yi = int(round(np.clip(y, 0, 223)))
        size = self._size            # 120 in 2160 space -> ~12 in 224 (matches training)
        thickness = self._thickness  # 6 in 2160 space -> ~1 in 224 (after antialiasing)
        c = self._color
        cv2.line(out, (xi - size, yi), (xi + size, yi), c, thickness, cv2.LINE_AA)
        cv2.line(out, (xi, yi - size), (xi, yi + size), c, thickness, cv2.LINE_AA)
        cv2.circle(out, (xi, yi), max(size // 3, 2), c, thickness, cv2.LINE_AA)
        return out

    # ------------------------------------------------------------------ API

    def reset(self) -> None:
        """Clear K-frame cache and EMA state. Call at episode start."""
        self._vis_buf.clear()
        self._ema_xy = None
        self.last_xy_224 = None
        logging.info("[GazeAnnotator] reset")

    @torch.no_grad()
    def predict_xy_2160(self, image_224_rgb_uint8: np.ndarray, gaze_prompt: str
                        ) -> Tuple[float, float]:
        """Run the predictor and return raw (x, y) in 2160-px space (no EMA,
        no scaling). ``annotate()`` calls this and then applies EMA and
        rendering."""
        x224 = self._normalize_to_clip(image_224_rgb_uint8)
        if self._use_bf16:
            ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            ctx = torch.autocast(device_type=self.device.type, enabled=False)
        with ctx:
            vis_proj = self.model.encode_image(x224)  # [1, P, fusion_dim]
            if self._is_temporal:
                vis_in = self._push_vis_and_get_stack(vis_proj)
            else:
                vis_in = vis_proj
            text_proj, mask = self._get_text_features(gaze_prompt)
            logits = self.model.head(vis_in, text_proj, text_mask=mask)
            # logits: [1, n_anchors, G, G]; anchor 0 (n_anchors=1).
            prob = logits.flatten(-2).softmax(-1).reshape(logits.shape)
            xy = windowed_soft_argmax(prob[:, 0:1], image_size=2160)[0, 0]
        x_2160 = float(xy[0].item())
        y_2160 = float(xy[1].item())
        return x_2160, y_2160

    def annotate(self, image_224_rgb_uint8: np.ndarray, gaze_prompt: str
                 ) -> np.ndarray:
        """Forward the predictor on image+prompt, EMA-smooth the output, and
        draw a cyan crosshair on a copy of the input.

        Returns a 224x224x3 uint8 RGB array. The smoothed gaze coordinates
        used to draw the crosshair are also stored on
        ``self.last_xy_224`` (in 224-px image space) for callers that need
        to log / record them separately.
        """
        x_2160, y_2160 = self.predict_xy_2160(image_224_rgb_uint8, gaze_prompt)
        xy = np.array([x_2160, y_2160], dtype=np.float32)
        xy_smooth = self._apply_ema(xy)
        # Convert 2160-space coords to 224-space for callers that want to log
        # the smoothed gaze alongside the 224-px image.
        scale = 224.0 / 2160.0
        self.last_xy_224 = (
            float(xy_smooth[0]) * scale,
            float(xy_smooth[1]) * scale,
        )
        return self._draw_crosshair(
            image_224_rgb_uint8, float(xy_smooth[0]), float(xy_smooth[1])
        )


# ---------------------------------------------------------------------------
# Async wrapper: runs annotation in a background thread so the control loop
# doesn't block on CLIP forward each step.
# ---------------------------------------------------------------------------


class AsyncGazeAnnotator:
    """Runs ``GazeAnnotator.annotate`` in a background worker thread.

    The ROS image callback calls ``on_head_frame(img)`` to deliver each new
    head_camera frame; the worker keeps annotating the latest frame in a tight
    loop and stores the result. The control loop calls ``get_latest()`` —
    constant-time read of the most recent annotated frame + (x, y) in 224-px
    space. Old frames are dropped if the worker is behind (annotation rate is
    capped by GPU forward time, not by ROS callback rate).

    Thread-safety: a single worker thread owns the GPU model. ROS callbacks
    and the main thread only touch shared state under ``self._lock``.
    """

    def __init__(self, annotator: "GazeAnnotator", gaze_prompt: str):
        self._ann = annotator
        self._prompt = gaze_prompt
        # ``_lock`` guards the small queue-like state (pending frame, last
        # result, ready event). Held only briefly.
        self._lock = threading.Lock()
        # ``_annotate_lock`` serializes annotator state mutations: the worker
        # holds it during ``annotate()``, and ``reset()`` waits on it
        # so we never clear EMA / K-frame buffers mid-forward.
        self._annotate_lock = threading.Lock()
        # Latest frame to annotate (set by callback, consumed by worker).
        self._pending: Optional[np.ndarray] = None
        # Most recent annotation result (written by worker, read by main).
        self._latest_annotated: Optional[np.ndarray] = None
        self._latest_xy_224: Optional[Tuple[float, float]] = None
        # Set by the callback to wake the worker. Cleared by the worker.
        self._wake = threading.Event()
        self._stop = threading.Event()
        # First-annotation barrier so the control loop can block briefly on
        # startup until the first real annotation arrives.
        self._first_ready = threading.Event()
        self._thread = threading.Thread(
            target=self._worker, name="AsyncGazeAnnotator", daemon=True
        )
        self._thread.start()
        logging.info(
            f"[AsyncGazeAnnotator] worker thread started  prompt={gaze_prompt!r}"
        )

    def on_head_frame(self, image_224_rgb_uint8: np.ndarray) -> None:
        """Called from the ROS callback thread. Stashes the latest frame and
        wakes the worker. Older pending frames are dropped (we only ever
        annotate the freshest)."""
        with self._lock:
            self._pending = image_224_rgb_uint8  # already a copy from caller
        self._wake.set()

    def get_latest(self) -> Tuple[Optional[np.ndarray], Optional[Tuple[float, float]]]:
        """Called from the main / control-loop thread. Returns (annotated, xy)
        or (None, None) if no annotation has been produced yet."""
        with self._lock:
            return self._latest_annotated, self._latest_xy_224

    def wait_for_first(self, timeout: float = 5.0) -> bool:
        """Block until the worker has produced at least one annotated frame.
        Call this once after ROS image callbacks start so the control loop's
        first ``get_observation`` doesn't see (None, None)."""
        return self._first_ready.wait(timeout=timeout)

    def reset(self) -> None:
        """Forward to the underlying annotator (clears EMA + K-frame cache).
        Also clears the latest annotation buffer. Waits for any in-flight
        annotate() to finish so we don't race the worker's EMA / K-frame
        buffer updates."""
        with self._annotate_lock:
            with self._lock:
                self._ann.reset()
                self._latest_annotated = None
                self._latest_xy_224 = None
                self._first_ready.clear()
                # Drop any stale pending frame from before the reset so the
                # next annotate starts fresh.
                self._pending = None

    def shutdown(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=timeout)

    def _worker(self) -> None:
        while not self._stop.is_set():
            self._wake.wait()
            if self._stop.is_set():
                break
            with self._lock:
                frame = self._pending
                self._pending = None
                # Clear under the lock: a callback arriving after this sets
                # both _pending and _wake again, so no frame is lost.
                self._wake.clear()
            if frame is None:
                continue
            try:
                with self._annotate_lock:
                    annotated = self._ann.annotate(frame, self._prompt)
                    xy = self._ann.last_xy_224
            except Exception:
                logging.exception("[AsyncGazeAnnotator] annotate() failed")
                continue
            with self._lock:
                self._latest_annotated = annotated
                self._latest_xy_224 = xy
            self._first_ready.set()
