import dataclasses
import logging

import cv2
import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

logger = logging.getLogger(__name__)


def _parse_image(image) -> np.ndarray:
    """Parse image to uint8 (H,W,C) format following LeRobot conventions."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _left_gripper_effort_to_normalized(effort: np.ndarray) -> np.ndarray:
    """Convert left gripper effort to normalized [0, 1] controller range."""
    return np.where(effort > 0, 0.5 - effort / 7.0, 0.5 - effort)


def _right_gripper_effort_to_normalized(effort: np.ndarray) -> np.ndarray:
    """Convert right gripper effort to normalized [0, 1] controller range."""
    return np.where(effort < 0, effort / 7.0 + 0.5, effort + 0.5)


def _left_gripper_normalized_to_effort(data: np.ndarray) -> np.ndarray:
    """Convert normalized [0, 1] controller value to left gripper effort."""
    grip = -(data - 0.5)
    return np.where(grip > 0, grip * 7.0, grip)


def _right_gripper_normalized_to_effort(data: np.ndarray) -> np.ndarray:
    """Convert normalized [0, 1] controller value to right gripper effort."""
    grip = data - 0.5
    return np.where(grip < 0, grip * 7.0, grip)


def _extract_left_head_view(image: np.ndarray, *, rotate: bool = False) -> np.ndarray:
    """Crop the left eye from a side-by-side stereo head image, optionally
    rotating 180°.

    The head camera is mounted upside-down. The orientation handling is
    asymmetric between training and inference because the rotation lives
    in different places:

    * Training: the dataset is built with the 180° rotation already applied
      to BOTH the image and ``observation.gaze``, so the caller
      passes ``rotate=False`` and we only crop the left half.
    * Inference: ``examples/teleavatar/ros2_interface.py`` already resizes,
      crops and rotates the head frame to ``(224, 224, 3)``, so this
      function is a no-op there.
    * Raw upside-down stereo input ``(2160, 4320, 3)``: pass ``rotate=True``
      to crop AND rotate so the model sees the training orientation.

    The crop is gated on the ``width == 2 * height`` shape check so an
    already-cropped square ``(H, H, 3)`` frame is left untouched.
    """
    height, width = image.shape[:2]
    if width == 2 * height:
        if rotate:
            image = np.rot90(image, k=2)
        image = image[:, :height, :]
    return image


@dataclasses.dataclass(frozen=True)
class TeleavatarInputs(transforms.DataTransformFn):
    """
    Converts inputs to the model format for Teleavatar robot.

    **Input format (48-dim observation/state from LeRobot dataset):**
    Layout: [positions(16), velocities(16), efforts(16)]
    - Indices 0-15: Joint positions (7 left arm, 1 left gripper, 7 right arm, 1 right gripper)
    - Indices 16-31: Joint velocities (same layout)
    - Indices 32-47: Joint efforts (same layout)

    **Model state format (14-dim):**
    We extract: [left_arm_pos(7), right_arm_pos(7)]
    - Indices 0-6: Left arm positions (from input[0:7])
    - Indices 7-13: Right arm positions (from input[8:15])

    This matches the layout of the GazeMani LeRobot dataset.
    """
    model_type: _model.ModelType
    use_delta_joint_actions: bool = False
    # Whether to rotate 180° before cropping the head frame. Property of the
    # source data, not of train/inference. The 2:1-width guard inside
    # _extract_left_head_view already no-ops for already-cropped (square)
    # frames, so this only matters when the head input is raw stereo.
    #   True  → upside-down stereo source (camera mounted upside-down)
    #   False → right-side-up stereo source (default; the GazeMani dataset
    #           is stored already rotated)
    rotate_head_camera: bool = False

    def __call__(self, data: dict) -> dict:
        # Parse images to uint8 (H,W,C) format
        # LeRobot stores as float32 (C,H,W) during training, but runtime sends uint8 (H,W,C)
        left_color = _parse_image(data["observation/images/left_color"])
        right_color = _parse_image(data["observation/images/right_color"])
        head_color = _parse_image(data["observation/images/head_camera"])
        # Rotate iff the configured source orientation says so. The width-check
        # guard inside _extract_left_head_view makes this a no-op when the
        # head frame already arrives 224×224 (cropped by ros2_interface).
        head_color = _extract_left_head_view(
            head_color, rotate=self.rotate_head_camera
        )

        # Extract 14-dim state from 48-dim observation
        # Input layout: [positions(0-15), velocities(16-31), efforts(32-47)]
        state_14d = np.concatenate([
            data["observation/state"][0:7],    # Left arm positions (indices 0-6)
            data["observation/state"][8:15],   # Right arm positions (indices 8-14)
        ], axis=0)

        # Create inputs dict. Do not change the keys in the dict below.
        # Pi0 models support three image inputs: one third-person view and two wrist views.
        # Map Teleavatar cameras to the expected model inputs.
        inputs = {
            "state": state_14d,
            "image": {
                "base_0_rgb": head_color,
                "left_wrist_0_rgb": left_color,
                "right_wrist_0_rgb": right_color,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Extract 16-dim actions from 48-dim during training
        # Actions are only available during training, not during inference
        if "action" in data:
            # data["action"] has shape [action_horizon, 48]
            # Layout: [positions(0-15), velocities(16-31), efforts(32-47)]
            action_data = data["action"]

            # Extract 16 dimensions: joint positions (14) + gripper efforts (2)
            # Note: State only uses 14 dims (joint positions), but actions include gripper efforts
            selected_actions = np.concatenate([
                action_data[:, 0:7],    # Left arm positions
                action_data[:, 39:40],  # Left gripper effort (index 39 = 32+7)
                action_data[:, 8:15],   # Right arm positions
                action_data[:, 47:48],  # Right gripper effort (index 47 = 32+15)
            ], axis=1)  # Concatenate along action dimension

            # Convert gripper efforts to normalized [0, 1] controller range
            selected_actions[:, 7] = _left_gripper_effort_to_normalized(selected_actions[:, 7])
            selected_actions[:, 15] = _right_gripper_effort_to_normalized(selected_actions[:, 15])

            # Apply delta to arm joints only (not gripper). State is 14-dim [left_arm(7), right_arm(7)]
            # but actions interleave gripper: [left_arm(7), left_gripper(1), right_arm(7), right_gripper(1)].
            # The generic DeltaActions transform cannot handle this layout mismatch, so we do it here.
            if self.use_delta_joint_actions:
                selected_actions[:, 0:7] -= state_14d[np.newaxis, 0:7]    # left arm delta
                selected_actions[:, 8:15] -= state_14d[np.newaxis, 7:14]  # right arm delta

            inputs["actions"] = selected_actions

        # Pass the prompt (aka language instruction) to the model. During
        # training this should be filled by PromptFromLeRobotTask + the
        # "prompt": "prompt" entry in the repack structure (see
        # LeRobotTeleavatarDataConfig.create). The fallback below only kicks
        # in for inference callers that don't supply a prompt; if it ever
        # triggers during training, every sample shares one fixed string and
        # the language channel is dead — make that visible.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        else:
            inputs["prompt"] = "stack red bowls"
            logger.warning(
                "TeleavatarInputs: no 'prompt' in sample; using hardcoded "
                "fallback. Expected during inference without a client prompt, "
                "but indicates a training-pipeline bug if seen on training data."
            )

        # Forward the current-frame gaze (anchor 0) for the optional KL aux loss.
        # Coordinates remain in 2160 px left-crop space; the JAX side handles
        # the projection onto the visual token grid.
        if "observation/gaze" in data:
            gaze = np.asarray(data["observation/gaze"], dtype=np.float32)
            inputs["gaze_xy_kl"] = gaze[0] if gaze.ndim == 2 else gaze

        return inputs


@dataclasses.dataclass(frozen=True)
class TeleavatarOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back to the dataset specific format. It is
    used for inference only.

    For Teleavatar, we return 16 actions:
    - Joint positions for joints 1-7 for both arms (14 values)
    - Joint efforts for left and right grippers (2 values)

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """
    use_delta_joint_actions: bool = False

    def __call__(self, data: dict) -> dict:
        # Only return the first 16 actions for Teleavatar.
        # Since the model may output more dimensions due to padding, we extract just what we need.
        # For your own dataset, replace `16` with the action dimension of your dataset.
        actions = np.asarray(data["actions"][:, :16])

        # Convert delta arm joints back to absolute using current state.
        # data["state"] is the un-normalized 14-dim state: [left_arm(7), right_arm(7)].
        if self.use_delta_joint_actions:
            state = np.asarray(data["state"])
            actions[:, 0:7] += state[0:7]    # left arm absolute
            actions[:, 8:15] += state[7:14]  # right arm absolute

        # Convert normalized [0, 1] gripper values back to effort for robot execution
        actions[:, 7] = _left_gripper_normalized_to_effort(actions[:, 7])
        actions[:, 15] = _right_gripper_normalized_to_effort(actions[:, 15])
        return {"actions": actions}


# ---------------------------------------------------------------------------
# Gaze visual-prompt (cyan crosshair) overlay
# ---------------------------------------------------------------------------
#
# Used by ``LeRobotTeleavatarDataConfig`` when ``use_gt_crosshair=True``. The
# overlay sits BEFORE ``TeleavatarInputs`` in ``data_transforms.inputs``, so
# it operates on the raw ``(2160, 4320, 3)`` head-camera frame. Gaze
# coordinates from ``observation.gaze`` are in 2160 px left-crop
# space and live in the same coordinate system as the (pre-rotated) image
# from the dataset, so we draw crosshairs directly at the gaze coords; the
# downstream left-half crop in ``_extract_left_head_view`` carries them
# through to the model input.

def _to_uint8_hwc(image) -> np.ndarray:
    """Coerce a LeRobot image leaf to ``uint8 HWC`` numpy.

    LeRobot returns ``(C, H, W) float32 in [0, 1]`` from MP4 decoding; runtime
    inputs are ``(H, W, C) uint8``. We support both.
    """
    arr = np.asarray(image)
    if np.issubdtype(arr.dtype, np.floating):
        arr = (255.0 * arr).clip(0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = einops.rearrange(arr, "c h w -> h w c")
    if not (arr.ndim == 3 and arr.shape[2] == 3):
        raise ValueError(f"unexpected image shape {arr.shape}")
    return np.ascontiguousarray(arr)


def draw_crosshairs(
    img: np.ndarray,
    gaze_xy: np.ndarray,
    sizes,
    color: tuple[int, int, int] = (0, 255, 255),
    thickness: int = 6,
) -> np.ndarray:
    """Draw N crosshairs in-place on ``img`` (uint8 HWC). Returns ``img``."""
    if img.dtype != np.uint8:
        raise TypeError(f"draw_crosshairs expects uint8, got {img.dtype}")
    g = np.asarray(gaze_xy, dtype=np.int64)
    if g.shape != (len(sizes), 2):
        raise ValueError(f"gaze_xy must be ({len(sizes)}, 2); got {g.shape}")
    H, W = img.shape[:2]
    for i in range(g.shape[0]):
        x, y = int(g[i, 0]), int(g[i, 1])
        if not (-W <= x <= 2 * W and -H <= y <= 2 * H):
            continue
        size = int(sizes[i])
        cv2.line(img, (x - size, y), (x + size, y), color, thickness)
        cv2.line(img, (x, y - size), (x, y + size), color, thickness)
        cv2.circle(img, (x, y), max(1, size // 3), color, thickness)
    return img


@dataclasses.dataclass
class GazeCrosshairOverlay(transforms.DataTransformFn):
    """Draws the GT-gaze crosshair onto the head-camera image.

    Place BEFORE ``TeleavatarInputs`` in ``data_transforms.inputs``. The
    transform reads ``observation.gaze`` (``[1, 2]`` int gaze in
    2160 left-crop space) and renders a crosshair onto the left half of
    the raw 4320×2160 head frame. ``LeRobotTeleavatarDataConfig`` adds the
    gaze key to the repack structure when ``use_gt_crosshair=True``.

    ``observation.gaze`` in the GazeMani dataset is filtered during
    preprocessing, so every frame carries a valid gaze location.
    """

    # Slash-separated keys: this runs after RepackTransform, which converts
    # ``a.b.c`` → ``a/b/c``.
    head_camera_key: str = "observation/images/head_camera"
    gaze_key: str = "observation/gaze"
    # Crosshair half-line length in 2160 left-crop space (≈ 12 px after the
    # 224 resize, ~1 SigLIP patch).
    crosshair_size: int = 120
    # Crosshair color (RGB): cyan.
    color_rgb: tuple[int, int, int] = (0, 255, 255)
    thickness: int = 6
    enabled: bool = True

    def __call__(self, data: dict) -> dict:
        if not self.enabled:
            return data
        if self.gaze_key not in data:
            return data  # inference-time call, no gaze available

        head = _to_uint8_hwc(data[self.head_camera_key])
        if head.shape[:2] != (2160, 4320):
            raise ValueError(
                "GazeCrosshairOverlay expects raw (2160, 4320, 3) head camera; "
                f"got {head.shape}. Place this transform BEFORE TeleavatarInputs."
            )

        gazes = np.asarray(data[self.gaze_key], dtype=np.int64).reshape(-1, 2)

        draw_crosshairs(
            head,
            gazes,
            sizes=(self.crosshair_size,) * gazes.shape[0],
            color=self.color_rgb,
            thickness=self.thickness,
        )
        data[self.head_camera_key] = head
        return data
