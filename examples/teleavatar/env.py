#!/usr/bin/env python3
"""
Environment wrapper for Teleavatar robot using openpi_client.runtime framework.
"""

import logging
import threading
from typing import Optional

import numpy as np
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override

from examples.teleavatar import ros2_interface


class TeleavatarEnvironment(_environment.Environment):
    """Environment for the Teleavatar mobile manipulator."""

    def __init__(
        self,
        prompt: str = "stack red bowls",
        gaze_predictor_ckpt: Optional[str] = None,
        gaze_predictor_config: Optional[str] = None,
        gaze_prompt_override: Optional[str] = None,
        gaze_ema_alpha: float = 0.2,
        gaze_crosshair_color: tuple = (0, 255, 255),
        gaze_record_dir: Optional[str] = None,
        gaze_record_every_n_frames: int = 30,
        arm_offsets: Optional[tuple] = None,
    ):
        """Initialize Teleavatar environment.

        Args:
            prompt: Default language instruction for the policy
            gaze_predictor_ckpt: Path to a gaze predictor checkpoint .pt file.
                If None (default), gaze annotation is disabled and the
                head_camera image passes through unchanged.
            gaze_predictor_config: Path to the matching gaze predictor YAML
                config. Required if gaze_predictor_ckpt is set.
            gaze_prompt_override: Optional explicit prompt to feed the gaze
                predictor. If None, the gaze predictor receives the policy
                prompt (via gaze_client.map_to_gaze_prompt).
            gaze_ema_alpha: Causal EMA smoothing factor on the predicted
                (x, y). Lower = smoother, more lag.
            gaze_record_dir: If set (and gaze annotation is enabled), a
                subsampled stream of annotated head_camera frames (see
                gaze_record_every_n_frames) is written to
                <gaze_record_dir>/ep_<idx>/<step>.png on a background
                writer thread. Disk I/O does NOT block get_observation.
                None = no recording.
            gaze_record_every_n_frames: Save 1 of every N annotated frames
                to gaze_record_dir (default 30 ≈ 1 Hz at a 30 Hz loop).
            gaze_crosshair_color: RGB crosshair color. Default cyan
                (0, 255, 255), matching training.
            arm_offsets: Optional joint-encoder calibration offsets (rad),
                14 values: L1..L7 then R1..R7. None = all zeros.

        Note: Images are decoded by ros2_interface via PyAV (hevc_cuvid GPU decode).
        - left_color, right_color: 480×848×3
        - head_camera: 224×224×3 (GPU hw-resize + upright left-eye crop done in ros2_interface)
        """
        self._prompt = prompt
        self._gaze_prompt_override = gaze_prompt_override
        # Optional joint-encoder calibration offsets (rad). 14 values in
        # order: L1..L7 then R1..R7. State seen by policy is ``raw + offset``,
        # action published to ROS is ``policy_out - offset``. Default: all
        # zeros (no-op).
        offsets = arm_offsets if arm_offsets is not None else (0.0,) * 14
        if len(offsets) != 14:
            raise ValueError(f"arm_offsets must have length 14, got {len(offsets)}")
        self._arm_offsets = np.asarray(offsets, dtype=np.float32)
        if np.any(self._arm_offsets != 0):
            logging.info(
                "Joint calibration offsets ENABLED (L1..L7, R1..R7): %s",
                self._arm_offsets.tolist(),
            )

        # Optional gaze annotator. Lazily import so the openpi venv only
        # needs gaze_predictor deps when this feature is actually used.
        # ``self._gaze`` is the AsyncGazeAnnotator (worker thread); the
        # underlying ``GazeAnnotator`` (GPU model) lives inside it.
        self._gaze = None
        self._gaze_recorder = None  # AsyncFrameRecorder (optional PNG dump of annotated head frames)
        if gaze_predictor_ckpt is not None:
            if gaze_predictor_config is None:
                raise ValueError(
                    "gaze_predictor_config is required when "
                    "gaze_predictor_ckpt is set"
                )
            from examples.teleavatar.gaze_client import (
                AsyncFrameRecorder,
                AsyncGazeAnnotator,
                GazeAnnotator,
                map_to_gaze_prompt,
            )
            inner = GazeAnnotator(
                ckpt_path=gaze_predictor_ckpt,
                cfg_path=gaze_predictor_config,
                ema_alpha=gaze_ema_alpha,
                crosshair_color_rgb=tuple(gaze_crosshair_color),
            )
            # Resolve the gaze prompt once at init: ``prompt`` is fixed for
            # the lifetime of the env, and the worker only sees this string.
            resolved_gaze_prompt = gaze_prompt_override or map_to_gaze_prompt(prompt)
            self._gaze = AsyncGazeAnnotator(inner, resolved_gaze_prompt)
            logging.info(
                f"Gaze annotation ENABLED (async)  ckpt={gaze_predictor_ckpt}  "
                f"ema_alpha={gaze_ema_alpha}  "
                f"gaze_prompt={resolved_gaze_prompt!r}"
            )
            if gaze_record_dir is not None:
                self._gaze_recorder = AsyncFrameRecorder(
                    out_dir=gaze_record_dir,
                    every_n_frames=gaze_record_every_n_frames,
                )
                logging.info(
                    f"Gaze frame recording ENABLED  dir={gaze_record_dir}"
                )
        else:
            logging.info("Gaze annotation DISABLED (no ckpt provided)")
            if gaze_record_dir is not None:
                logging.warning(
                    "gaze_record_dir was set but no gaze predictor was "
                    "configured; recording will be skipped."
                )

        # Initialize ROS2 interface in a separate thread
        self._ros_interface: Optional[ros2_interface.TeleavatarROS2Interface] = None
        self._ros_thread: Optional[threading.Thread] = None
        self._init_ros2()

        # Hook the async gaze worker to the head_camera ROS callback. Once
        # registered, every newly-decoded head frame triggers an annotation
        # forward on the worker thread — get_observation reads the latest
        # annotated frame in constant time, not blocking the control loop.
        if self._gaze is not None:
            self._ros_interface.register_gaze_sink(self._gaze)
            if not self._gaze.wait_for_first(timeout=5.0):
                logging.warning(
                    "AsyncGazeAnnotator did not produce a frame within 5s "
                    "of registration; the first few get_observation calls "
                    "will fall back to the raw head_camera image."
                )

        logging.info(f"TeleavatarEnvironment initialized with prompt: '{prompt}'")

    def _init_ros2(self):
        """Initialize ROS2 in a background thread and wait for initial sensor data."""
        import rclpy
        import time

        # Event to signal when executor starts spinning
        spin_started = threading.Event()

        def ros_spin():
            rclpy.init()
            self._ros_interface = ros2_interface.TeleavatarROS2Interface()

            # Spin in background
            # Explicitly allocate enough threads for all camera + joint callbacks to run concurrently
            executor = rclpy.executors.MultiThreadedExecutor(num_threads=8)
            executor.add_node(self._ros_interface)

            # Signal that spinning is about to start
            spin_started.set()

            try:
                executor.spin()
            finally:
                executor.shutdown()
                self._ros_interface.destroy_node()
                rclpy.shutdown()

        self._ros_thread = threading.Thread(target=ros_spin, daemon=True)
        self._ros_thread.start()

        # Wait for ROS2 interface object to be created
        timeout = 10.0
        start_time = time.time()
        while self._ros_interface is None and time.time() - start_time < timeout:
            time.sleep(0.1)

        if self._ros_interface is None:
            raise RuntimeError("Failed to initialize ROS2 interface object within timeout")

        logging.info("ROS2 interface object created, waiting for executor to start spinning...")

        # Wait for executor to start spinning
        if not spin_started.wait(timeout=5.0):
            raise RuntimeError("ROS2 executor failed to start spinning")

        logging.info("ROS2 executor started, waiting for initial sensor data...")

        # Now wait for initial sensor data (callbacks can now be triggered)
        if not self._ros_interface.wait_for_initial_data(timeout=30.0):
            raise RuntimeError(
                "Failed to receive initial sensor data. "
                "Please check that ROS2 topics are publishing:\n"
                "  ros2 topic list\n"
                "  ros2 topic hz /left/color/image_raw/ffmpeg\n"
                "  ros2 topic echo /left_arm/joint_states --once"
            )

        logging.info("ROS2 interface initialized successfully with sensor data")

    @override
    def reset(self) -> None:
        """Reset the environment.

        No robot motion is commanded here. Resets the gaze annotator's
        K-frame cache and EMA state so each episode starts fresh, and bumps
        the recorder's episode index.
        """
        if self._gaze is not None:
            self._gaze.reset()
        if self._gaze_recorder is not None:
            self._gaze_recorder.reset_episode()
        logging.info("Environment reset called (no-op for Teleavatar)")

    @override
    def is_episode_complete(self) -> bool:
        """Check if episode is complete.

        For Teleavatar, episodes never complete automatically - they must be
        terminated by the user (e.g., Ctrl+C).
        """
        return False

    @override
    def get_observation(self) -> dict:
        """Get current observation from robot sensors.

        Returns:
            Dictionary with the exact keys expected by teleavatar_policy.py:
                - 'observation/state': 48-dim proprioceptive state
                - 'observation/images/left_color',
                  'observation/images/right_color',
                  'observation/images/head_camera': camera images
                - 'prompt': Language instruction

        Note: Images are returned in (H,W,C) uint8 format:
            - left_color, right_color: 480×848×3
            - head_camera: 224×224×3 (already cropped + rotated by ros2_interface)
        """
        if self._ros_interface is None:
            raise RuntimeError("ROS2 interface not initialized")

        # Get raw observation from ROS2
        raw_obs = self._ros_interface.get_observation()
        if raw_obs is None:
            raise RuntimeError("Failed to get observation from ROS2 interface")

        # Joint-encoder calibration offsets. State layout: positions are
        # at indices 0..6 (left arm J1..J7) and 8..14 (right arm J1..J7);
        # indices 7 / 15 hold the gripper position and aren't touched.
        # ``raw_obs['state']`` is a fresh copy from ros2_interface so
        # in-place add is safe.
        if np.any(self._arm_offsets != 0):
            state = raw_obs['state']
            state[0:7]  += self._arm_offsets[0:7]    # left arm
            state[8:15] += self._arm_offsets[7:14]   # right arm

        # Images are passed through as (H, W, C) uint8; the policy's input
        # transforms handle resizing and format conversion.
        # Return with the exact keys expected by teleavatar_policy.py
        obs = {
            'observation/state': raw_obs['state'],
            'observation/images/left_color': image_tools.convert_to_uint8(raw_obs['images']['left_color']),
            'observation/images/right_color': image_tools.convert_to_uint8(raw_obs['images']['right_color']),
            'observation/images/head_camera': image_tools.convert_to_uint8(raw_obs['images']['head_camera']),
            'prompt': self._prompt,
        }

        # Optional gaze annotation: read the latest annotated frame produced
        # by the AsyncGazeAnnotator worker, which is fed from the head-camera
        # callback and runs off the control loop. Constant-time lookup with no
        # CLIP forward on the critical path. Falls back to the raw head image
        # only on the very first frames before the worker has produced a
        # result.
        gaze_xy = None
        if self._gaze is not None:
            annotated, gaze_xy = self._gaze.get_latest()
            if annotated is not None:
                obs['observation/images/head_camera'] = annotated
                # Async dump after annotation. The recorder copies internally
                # and writes on a worker thread, so this call is bounded.
                if self._gaze_recorder is not None:
                    self._gaze_recorder.push(annotated)
            else:
                # Worker hasn't produced its first frame yet — pass the raw
                # head image through and skip recording for this step.
                logging.debug(
                    "AsyncGazeAnnotator has no result yet; passing raw head."
                )

        return obs

    @override
    def apply_action(self, action: dict) -> None:
        """Apply action to the robot.

        Args:
            action: Dictionary containing 'actions' key with 16-dim action array
        """
        if self._ros_interface is None:
            raise RuntimeError("ROS2 interface not initialized")

        if 'actions' not in action:
            raise ValueError(f"Action dict must contain 'actions' key, got: {action.keys()}")

        actions = action['actions']
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions, dtype=np.float32)

        # Ensure correct shape
        if actions.shape != (16,):
            raise ValueError(f"Expected 16-dim action, got shape {actions.shape}")

        # Joint-encoder calibration offsets (inverse of the +offset on
        # observation/state in get_observation). Action layout: arm
        # positions at [0:7] (left) and [8:15] (right); [7] / [15] are
        # gripper effort. Copy first because the 16-dim chunk array is
        # shared with the action_chunk_broker — an in-place subtract
        # would shift every cached frame in the current chunk.
        if np.any(self._arm_offsets != 0):
            actions = actions.copy()
            actions[0:7]  -= self._arm_offsets[0:7]
            actions[8:15] -= self._arm_offsets[7:14]

        # Publish to ROS2
        self._ros_interface.publish_action(actions)

    def __del__(self):
        """Cleanup when environment is destroyed."""
        # Stop the async gaze worker so its background thread doesn't outlive
        # the env (and so the GPU model gets freed promptly).
        try:
            if getattr(self, "_gaze", None) is not None:
                self._gaze.shutdown(timeout=2.0)
        except Exception:  # noqa: BLE001
            pass
        # Drain the gaze frame recorder first so we don't lose buffered frames.
        try:
            if getattr(self, "_gaze_recorder", None) is not None:
                self._gaze_recorder.shutdown(timeout=10.0)
        except Exception:  # noqa: BLE001
            pass
        ros_thread = getattr(self, "_ros_thread", None)
        if ros_thread is not None and ros_thread.is_alive():
            logging.info("Shutting down ROS2 thread...")
