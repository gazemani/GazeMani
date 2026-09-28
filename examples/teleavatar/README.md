# Teleavatar deployment

Run a fine-tuned π₀ policy on the Teleavatar mobile manipulator, optionally with
the gaze-prompt pipeline (online gaze predictor + crosshair rendering on the
head view).

## Components

```
examples/teleavatar/
├── README.md                    (this file)
├── main.py                      # entry point: control loop + remote policy client
├── env.py                       # TeleavatarEnvironment (openpi_client.runtime)
├── ros2_interface.py            # ROS2 node: camera decode (PyAV/HEVC), joint state, action publish
├── gaze_client.py               # GazeAnnotator (predictor + crosshair rendering)
└── arm_pd_controller.py         # proportional joint-velocity controller node (policy cmd -> velocity cmd)
```

The robot publishes three H.265 camera streams (`/left/color/image_raw/ffmpeg`,
`/right/color/image_raw/ffmpeg`, head stereo on `/xr_video_topic/ffmpeg`) plus
arm/gripper `joint_states`; the policy's 16-dim action is published back as
joint position commands, which `arm_pd_controller.py` converts to velocity
commands at 100 Hz with a clipped proportional control law.

## Running a rollout

1. **Start the policy server** (machine with the training GPU):

   ```bash
   uv run scripts/serve_policy.py policy:checkpoint \
       --policy.config=pi0_teleavatar \
       --policy.dir=checkpoints/pi0_teleavatar/gaze-prompt/19999 \
       --policy.asset-id=/path/to/task_dataset
   ```

   `asset_id` must point at the same dataset id used to compute norm stats
   during training.

2. **Start the joint-velocity controller** (robot machine, ROS2 sourced):

   ```bash
   python3 examples/teleavatar/arm_pd_controller.py
   ```

3. **Run the control loop** (robot machine):

   Vanilla policy:

   ```bash
   python3 examples/teleavatar/main.py \
       --remote-host <policy-server-ip> \
       --prompt "stack red bowls"
   ```

   Gaze-prompt policy (ours) — add the gaze predictor so the crosshair is
   rendered at the *predicted* fixation every frame:

   ```bash
   python3 examples/teleavatar/main.py \
       --remote-host <policy-server-ip> \
       --prompt "stack red bowls" \
       --gaze-predictor-ckpt   /path/to/gaze_predictor/best.pt \
       --gaze-predictor-config src/gaze_predictor/configs/gaze_predictor_six_task.yaml
   ```

   Useful flags: `--gaze-ema-alpha` (default 0.2, causal EMA on the predicted
   fixation), `--gaze-record-dir` (dump annotated head frames as PNGs for
   inspection), `--arm-offsets` (optional joint-encoder calibration offsets, default zeros).

The gaze predictor runs on a background thread fed directly from the camera
callback, so its forward pass stays off the control loop's critical path.
