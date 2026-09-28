#!/usr/bin/env python3
"""
ROS2 interface wrapper for the Teleavatar robot.
Handles subscribing to sensor topics and publishing actions.

Image decoding strategy:
- Subscribes directly to FFMPEGPacket (H.265) topics, bypassing ffmpeg_image_transport republish.
- Uses PyAV with hevc_cuvid (GPU) for decoding, falling back to CPU hevc if unavailable.
- head_camera: GPU hw-resize 2160×4320 → 224×448 during decode, then take the right half of
  the upside-down stereo frame + rot180 → upright 224×224 left-eye view (matching the training
  crop), ready for the model with no further processing.
- left_color / right_color: decoded as-is at 480×848.
"""

import logging
import time
from collections import deque
from threading import Lock
from typing import Any, Dict, Optional

import av
import numpy as np
from ffmpeg_image_transport_msgs.msg import FFMPEGPacket
from rclpy.node import Node
from sensor_msgs.msg import JointState

_HZ_WINDOW = 60  # frames to use for Hz estimate


def _make_codec(name: str, options: dict | None = None) -> av.CodecContext:
    """Create and open a PyAV codec context, trying GPU first then CPU."""
    gpu_codec = "hevc_cuvid"
    cpu_codec = "hevc"
    try:
        ctx = av.CodecContext.create(gpu_codec, "r")
        if options:
            ctx.options = options
        ctx.open()
        logging.info(f"[{name}] using {gpu_codec}" + (f" options={options}" if options else ""))
        return ctx
    except Exception as e:
        logging.warning(f"[{name}] {gpu_codec} unavailable ({e}), falling back to CPU")
        ctx = av.CodecContext.create(cpu_codec, "r")
        ctx.open()
        return ctx


class TeleavatarROS2Interface(Node):
    """Thread-safe ROS2 interface for the Teleavatar robot's sensors and actuators."""

    def __init__(self, node_name: str = "teleavatar_openpi_interface"):
        super().__init__(node_name)

        self.logger = self.get_logger()
        self.lock = Lock()

        # Storage for latest sensor data
        self.latest_images: Dict[str, np.ndarray] = {}
        self.latest_joint_states: Dict[str, JointState] = {}
        self.image_timestamps: Dict[str, float] = {}
        self.joint_timestamps: Dict[str, float] = {}

        # Hz and latency tracking per camera
        self._image_recv_times: Dict[str, deque] = {
            "left_color": deque(maxlen=_HZ_WINDOW),
            "right_color": deque(maxlen=_HZ_WINDOW),
            "head_camera": deque(maxlen=_HZ_WINDOW),
        }
        self._image_latencies: Dict[str, deque] = {
            "left_color": deque(maxlen=_HZ_WINDOW),
            "right_color": deque(maxlen=_HZ_WINDOW),
            "head_camera": deque(maxlen=_HZ_WINDOW),
        }
        self._last_hz_log: float = time.time()
        self._hz_log_interval: float = 5.0  # log Hz every N seconds

        self.left_joint_names = ['l_joint1', 'l_joint2', 'l_joint3', 'l_joint4', 'l_joint5', 'l_joint6', 'l_joint7']
        self.right_joint_names = ['r_joint1', 'r_joint2', 'r_joint3', 'r_joint4', 'r_joint5', 'r_joint6', 'r_joint7']
        self.left_gripper_names = ['l_joint8']
        self.right_gripper_names = ['r_joint8']

        # PyAV codec contexts (one per camera)
        # head_camera: hw resize 2160×4320 → 224×448 during GPU decode
        self._codecs: Dict[str, av.CodecContext] = {
            "left_color": _make_codec("left_color"),
            "right_color": _make_codec("right_color"),
            "head_camera": _make_codec("head_camera", options={"resize": "448x224"}),
        }

        # Optional async gaze sink. Duck-typed:
        #   on_head_frame(image_224_rgb_uint8: np.ndarray) -> None
        # Called from the head_camera ffmpeg callback thread the moment a new
        # decoded frame is ready — lets a worker run the gaze predictor off the
        # control-loop critical path.
        self._gaze_sink: Optional[Any] = None

        self._setup_subscribers()
        self._setup_publishers()

        self.logger.info("TeleavatarROS2Interface initialized (waiting for sensor data in background)")

    def _setup_subscribers(self):
        """Setup ROS2 subscribers for images and joint states."""
        # Subscribe directly to H.265 compressed topics (bypasses republish node overhead)
        self.create_subscription(
            FFMPEGPacket,
            '/left/color/image_raw/ffmpeg',
            lambda msg: self._ffmpeg_callback(msg, 'left_color'),
            10,
        )
        self.create_subscription(
            FFMPEGPacket,
            '/right/color/image_raw/ffmpeg',
            lambda msg: self._ffmpeg_callback(msg, 'right_color'),
            10,
        )
        self.create_subscription(
            FFMPEGPacket,
            '/xr_video_topic/ffmpeg',
            lambda msg: self._ffmpeg_callback(msg, 'head_camera'),
            10,
        )

        self.create_subscription(
            JointState,
            '/left_arm/joint_states',
            lambda msg: self._joint_state_callback(msg, 'left_arm'),
            10,
        )
        self.create_subscription(
            JointState,
            '/right_arm/joint_states',
            lambda msg: self._joint_state_callback(msg, 'right_arm'),
            10,
        )
        self.create_subscription(
            JointState,
            '/left_gripper/joint_states',
            lambda msg: self._joint_state_callback(msg, 'left_gripper'),
            10,
        )
        self.create_subscription(
            JointState,
            '/right_gripper/joint_states',
            lambda msg: self._joint_state_callback(msg, 'right_gripper'),
            10,
        )

        self.logger.info("ROS2 subscribers initialized")

    def _setup_publishers(self):
        """Setup ROS2 publishers for action commands."""
        self.action_publishers = {
            'left_arm': self.create_publisher(JointState, '/left_arm/openpi_joint_cmd', 10),
            'right_arm': self.create_publisher(JointState, '/right_arm/openpi_joint_cmd', 10),
            'left_gripper': self.create_publisher(JointState, '/left_gripper/joint_cmd', 10),
            'right_gripper': self.create_publisher(JointState, '/right_gripper/joint_cmd', 10),
        }
        self.logger.info("ROS2 publishers initialized")

    def _ffmpeg_callback(self, msg: FFMPEGPacket, camera_name: str):
        """Decode H.265 FFMPEGPacket directly with PyAV (GPU hevc_cuvid)."""
        try:
            t0 = time.time()
            raw_bytes = bytes(msg.data)

            pkt = av.Packet(raw_bytes)
            pkt.pts = msg.pts
            pkt.dts = msg.pts  # ffmpeg_image_transport sometimes leaves dts unset

            frames = self._codecs[camera_name].decode(pkt)

            for frame in frames:
                if camera_name == "head_camera":
                    # After hw resize (2160×4320 → 224×448): the camera is mounted
                    # upside-down, so take the right half and rotate 180° to get
                    # the upright left-eye view (224×224).
                    img = frame.to_ndarray(format="rgb24")  # 224×448×3
                    img = img[:, 224:, :]                   # 224×224×3
                    img = np.rot90(img, k=2).copy()         # 224×224×3
                else:
                    img = frame.to_ndarray(format="rgb24")  # 480×848×3

                now = time.time()
                with self.lock:
                    self.latest_images[camera_name] = img
                    self.image_timestamps[camera_name] = now
                    self._image_recv_times[camera_name].append(now)
                    self._image_latencies[camera_name].append((now - t0) * 1000)

                # Async gaze sink: push the freshly-decoded head_camera frame
                # to a worker that runs the gaze predictor off-loop. Copy so
                # the model-input path doesn't share the same buffer if the
                # worker holds a reference.
                if camera_name == "head_camera":
                    gaze_sink = self._gaze_sink
                    if gaze_sink is not None:
                        try:
                            gaze_sink.on_head_frame(img.copy())
                        except Exception:
                            self.logger.exception("gaze_sink.on_head_frame failed")

                self._maybe_log_hz()
                break  # one packet → at most one output frame

        except Exception as e:
            self.logger.error(f"Failed to decode {camera_name}: {e}")

    def _maybe_log_hz(self):
        """Periodically log per-camera frame rate and decode latency."""
        now = time.time()
        if now - self._last_hz_log < self._hz_log_interval:
            return
        self._last_hz_log = now

        parts = []
        with self.lock:
            for name in self._image_recv_times:
                times = self._image_recv_times[name]
                lats = self._image_latencies[name]
                hz = (len(times) - 1) / (times[-1] - times[0]) if len(times) >= 2 else 0.0
                lat = float(np.mean(lats)) if lats else 0.0
                parts.append(f"{name}={hz:.1f}Hz/{lat:.1f}ms")

        self.logger.info(f"Cameras: {', '.join(parts)}")

    def _joint_state_callback(self, msg: JointState, joint_group: str):
        """Callback for joint state messages."""
        with self.lock:
            self.latest_joint_states[joint_group] = msg
            self.joint_timestamps[joint_group] = time.time()

    def register_gaze_sink(self, sink: Any) -> None:
        """Register an async gaze sink. Sink must implement
        ``on_head_frame(image_224_rgb_uint8: np.ndarray) -> None``.

        The head_camera ffmpeg callback will push every newly-decoded 224×224
        head frame to this sink on the ROS executor thread, so the sink must
        be thread-safe and non-blocking. Pass ``None`` to unregister.
        """
        self._gaze_sink = sink
        if sink is None:
            self.logger.info("Gaze sink UNREGISTERED")
        else:
            self.logger.info(f"Gaze sink registered: {type(sink).__name__}")

    def wait_for_initial_data(self, timeout: float = 10.0) -> bool:
        """Wait for initial sensor data to arrive.

        NOTE: This should be called AFTER the ROS2 node starts spinning,
        otherwise callbacks will never be triggered!

        Returns:
            True if all data received, False if timeout
        """
        required_images = ['left_color', 'right_color', 'head_camera']
        required_joints = ['left_arm', 'right_arm', 'left_gripper', 'right_gripper']

        start_time = time.time()
        self.logger.info("Waiting for initial sensor data...")

        last_status_time = start_time
        while time.time() - start_time < timeout:
            with self.lock:
                images_ready = all(cam in self.latest_images for cam in required_images)
                joints_ready = all(joint in self.latest_joint_states for joint in required_joints)

                if time.time() - last_status_time > 2.0:
                    have_images = [cam for cam in required_images if cam in self.latest_images]
                    have_joints = [joint for joint in required_joints if joint in self.latest_joint_states]
                    self.logger.info(f"  Progress: images={have_images}, joints={have_joints}")
                    last_status_time = time.time()

                if images_ready and joints_ready:
                    self.logger.info("✓ All sensor data received!")
                    return True

            time.sleep(0.1)

        with self.lock:
            missing_images = [cam for cam in required_images if cam not in self.latest_images]
            missing_joints = [joint for joint in required_joints if joint not in self.latest_joint_states]

        self.logger.error(
            f"✗ Timeout waiting for sensor data after {timeout}s. "
            f"Missing: images={missing_images}, joints={missing_joints}"
        )
        return False

    def get_observation(self) -> Optional[Dict]:
        """Get current observation from all sensors."""
        with self.lock:
            required_images = ['left_color', 'right_color', 'head_camera']
            required_joints = ['left_arm', 'right_arm', 'left_gripper', 'right_gripper']

            if not all(cam in self.latest_images for cam in required_images):
                return None
            if not all(joint in self.latest_joint_states for joint in required_joints):
                return None

            state_48d = np.zeros(48, dtype=np.float32)

            left_arm = self.latest_joint_states['left_arm']
            right_arm = self.latest_joint_states['right_arm']
            left_gripper = self.latest_joint_states['left_gripper']
            right_gripper = self.latest_joint_states['right_gripper']

            state_48d[0:7] = self._extract_joint_field(left_arm, 'position', 7)
            state_48d[7] = self._extract_joint_field(left_gripper, 'position', 1)[0]
            state_48d[8:15] = self._extract_joint_field(right_arm, 'position', 7)
            state_48d[15] = self._extract_joint_field(right_gripper, 'position', 1)[0]

            state_48d[16:23] = self._extract_joint_field(left_arm, 'velocity', 7)
            state_48d[23] = self._extract_joint_field(left_gripper, 'velocity', 1)[0]
            state_48d[24:31] = self._extract_joint_field(right_arm, 'velocity', 7)
            state_48d[31] = self._extract_joint_field(right_gripper, 'velocity', 1)[0]

            state_48d[32:39] = self._extract_joint_field(left_arm, 'effort', 7)
            state_48d[39] = self._extract_joint_field(left_gripper, 'effort', 1)[0]
            state_48d[40:47] = self._extract_joint_field(right_arm, 'effort', 7)
            state_48d[47] = self._extract_joint_field(right_gripper, 'effort', 1)[0]

            return {
                'images': {
                    'left_color': self.latest_images['left_color'].copy(),
                    'right_color': self.latest_images['right_color'].copy(),
                    # head_camera is already 224×224 (cropped + rotated in _ffmpeg_callback)
                    'head_camera': self.latest_images['head_camera'].copy(),
                },
                'state': state_48d,
            }

    def _extract_joint_field(self, msg: JointState, field: str, num_joints: int) -> np.ndarray:
        """Extract joint data field (position/velocity/effort) from JointState message."""
        data = getattr(msg, field, [])
        if len(data) >= num_joints:
            return np.array(data[:num_joints], dtype=np.float32)
        else:
            result = np.zeros(num_joints, dtype=np.float32)
            result[:len(data)] = data
            return result

    def publish_action(self, actions: np.ndarray):
        """Publish 16-dimensional action to ROS topics."""
        if actions.shape != (16,):
            self.logger.error(f"Expected 16-dim action, got shape {actions.shape}")
            return

        now = self.get_clock().now()
        timestamp = now.to_msg()

        left_arm_msg = JointState()
        left_arm_msg.header.stamp = timestamp
        left_arm_msg.header.frame_id = 'left_arm'
        left_arm_msg.name = self.left_joint_names
        left_arm_msg.position = actions[0:7].tolist()
        left_arm_msg.velocity = np.zeros(7).tolist()
        left_arm_msg.effort = np.zeros(7).tolist()
        self.action_publishers['left_arm'].publish(left_arm_msg)

        left_gripper_msg = JointState()
        left_gripper_msg.header.stamp = timestamp
        left_gripper_msg.header.frame_id = 'left_gripper'
        left_gripper_msg.name = self.left_gripper_names
        left_gripper_msg.position = [0.0]
        left_gripper_msg.velocity = [0.0]
        left_gripper_msg.effort = [float(actions[7])]
        self.action_publishers['left_gripper'].publish(left_gripper_msg)

        right_arm_msg = JointState()
        right_arm_msg.header.stamp = timestamp
        right_arm_msg.header.frame_id = 'right_arm'
        right_arm_msg.name = self.right_joint_names
        right_arm_msg.position = actions[8:15].tolist()
        right_arm_msg.velocity = np.zeros(7).tolist()
        right_arm_msg.effort = np.zeros(7).tolist()
        self.action_publishers['right_arm'].publish(right_arm_msg)

        right_gripper_msg = JointState()
        right_gripper_msg.header.stamp = timestamp
        right_gripper_msg.header.frame_id = 'right_gripper'
        right_gripper_msg.name = self.right_gripper_names
        right_gripper_msg.position = [0.0]
        right_gripper_msg.velocity = [0.0]
        right_gripper_msg.effort = [float(actions[15])]
        self.action_publishers['right_gripper'].publish(right_gripper_msg)
