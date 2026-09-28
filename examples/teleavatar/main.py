#!/usr/bin/env python3
"""
Main entry point for running the Teleavatar robot with an OpenPI policy.

This script uses the standard openpi_client.runtime framework for clean,
modular robot control with remote policy inference.

Usage:
    # Start policy server first (in another terminal):
    uv run scripts/serve_policy.py policy:checkpoint \
        --policy.config=pi0_teleavatar \
        --policy.dir=checkpoints/pi0_teleavatar/my_experiment/19999 \
        --policy.asset-id=/path/to/task_dataset

    # Then run this script:
    python examples/teleavatar/main.py --remote-host 127.0.0.1
"""

import dataclasses
import logging
from typing import Optional

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro

from examples.teleavatar import env as _env


@dataclasses.dataclass
class Args:
    """Command-line arguments for Teleavatar deployment."""

    # Remote policy server settings
    remote_host: str = "0.0.0.0"
    """IP address of the policy server (e.g., '192.168.1.100')"""

    remote_port: int = 8000
    """Port of the policy server"""

    # Control settings
    control_frequency: float = 30.0
    """Control loop frequency in Hz"""

    action_horizon: int = 30
    """Action chunk length of the policy (default: 30). Logged and checked
    against --open-loop-horizon; the chunk length itself is set by the policy config."""

    open_loop_horizon: int = 24
    """Number of actions to execute before querying policy again (default: 24)"""

    # Task settings
    prompt: str = "stack red bowls"
    """Language instruction for the robot"""

    # Episode settings
    num_episodes: int = 100
    """Number of episodes to run"""

    max_episode_steps: int = 0
    """Maximum steps per episode (0 = unlimited)"""

    # Gaze predictor (optional). Default = disabled (vanilla rollout).
    gaze_predictor_ckpt: Optional[str] = None
    """Path to gaze predictor checkpoint .pt. None disables gaze annotation."""

    gaze_predictor_config: Optional[str] = None
    """Path to gaze predictor config YAML. Required if --gaze-predictor-ckpt is set."""

    gaze_prompt_override: Optional[str] = None
    """Force a different prompt for the gaze predictor. None = use --prompt."""

    gaze_ema_alpha: float = 0.2
    """Causal EMA smoothing factor on the predicted (x, y). Lower = smoother but more lag."""

    gaze_crosshair_color: tuple[int, int, int] = (0, 255, 255)
    """RGB color of the crosshair drawn on head_camera. Default cyan,
    matching training. Pass three ints 0-255: ``--gaze-crosshair-color 0 255 255``."""

    gaze_record_dir: Optional[str] = None
    """Directory for dumping annotated head_camera frames (one PNG every
    --gaze-record-every-n-frames steps, organized by episode). Disk I/O
    happens on a background thread so it does not block get_observation.
    None = no recording."""

    gaze_record_every_n_frames: int = 30
    """Save 1 out of every N frames to --gaze-record-dir. Default 30 = ~1 Hz
    at 30 Hz control loop. Set to 1 to save every frame."""

    # Optional joint-encoder calibration offsets (radians) for both arms.
    # 14 floats in order: L1, L2, L3, L4, L5, L6, L7, R1, R2, R3, R4, R5, R6, R7.
    # State seen by policy: ``state[arm_pos_idx] += offset[i]``.
    # Action published to ROS:  ``action[arm_pos_idx] -= offset[i]``.
    # Default: all zeros (no-op).
    arm_offsets: tuple[float, float, float, float, float, float, float,
                       float, float, float, float, float, float, float] = (
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # left  J1..J7
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # right J1..J7
    )
    """Optional joint-encoder calibration offsets (rad), 14 values: L1..L7, R1..R7. Default all zeros."""


def main(args: Args) -> None:
    """Main function to run Teleavatar with policy inference."""

    logging.info("=" * 60)
    logging.info("Teleavatar OpenPI Deployment")
    logging.info("=" * 60)
    logging.info(f"Policy server: ws://{args.remote_host}:{args.remote_port}")
    logging.info(f"Control frequency: {args.control_frequency} Hz")
    logging.info(f"Action horizon: {args.action_horizon} steps")
    logging.info(f"Open-loop horizon: {args.open_loop_horizon} steps")
    logging.info(f"Prompt: '{args.prompt}'")
    logging.info("=" * 60)

    # Validate settings
    if args.open_loop_horizon > args.action_horizon:
        logging.warning(
            f"open_loop_horizon ({args.open_loop_horizon}) > action_horizon ({args.action_horizon}). "
            f"Each policy chunk supplies at most action_horizon actions."
        )

    # Create WebSocket client policy
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.remote_host,
        port=args.remote_port,
    )

    # Get and log server metadata
    metadata = ws_client_policy.get_server_metadata()
    logging.info(f"Connected to policy server. Metadata: {metadata}")

    # Create Teleavatar environment
    # Wrist views are passed at native 480x848; the head view arrives as 224x224.
    environment = _env.TeleavatarEnvironment(
        prompt=args.prompt,
        gaze_predictor_ckpt=args.gaze_predictor_ckpt,
        gaze_predictor_config=args.gaze_predictor_config,
        gaze_prompt_override=args.gaze_prompt_override,
        gaze_ema_alpha=args.gaze_ema_alpha,
        gaze_crosshair_color=args.gaze_crosshair_color,
        gaze_record_dir=args.gaze_record_dir,
        gaze_record_every_n_frames=args.gaze_record_every_n_frames,
        arm_offsets=args.arm_offsets,
    )

    # Create policy agent with action chunking
    agent = _policy_agent.PolicyAgent(
        policy=action_chunk_broker.ActionChunkBroker(
            policy=ws_client_policy,
            action_horizon=args.open_loop_horizon,  # Execute this many actions before querying
        )
    )

    # Create runtime
    runtime = _runtime.Runtime(
        environment=environment,
        agent=agent,
        subscribers=[],
        max_hz=args.control_frequency,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    # Run!
    logging.info("\nStarting robot control loop...")
    logging.info("Press Ctrl+C to stop\n")

    try:
        runtime.run()
    except KeyboardInterrupt:
        logging.info("\n\nStopping robot (Ctrl+C pressed)...")
    finally:
        logging.info("Shutdown complete.")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
        force=True
    )

    # Parse arguments and run
    args: Args = tyro.cli(Args)
    main(args)
