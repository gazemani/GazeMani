# gaze_predictor

Training + deployment code for the GazeMani gaze predictor: a lightweight
model that estimates the operator's fixation on the head-camera view from a
short window of frames and the task instruction.

- **Encoder**: frozen CLIP-B/16 vision + text (`encoder_kind: clip_clip`)
- **Head**: per-frame prompt conditioning + temporal fusion over K=3 frames
  (N=3 in the paper) + 14->224 spatial upsampler
  (`head_kind: prompt_then_temporal_224`)
- **Output**: 224x224 fixation heatmap; sub-cell readout via windowed
  soft-argmax
- **Supervision**: per-frame cross-entropy on a Gaussian-smoothed target
  heatmap (σ=16 grid cells)
- **Inference smoothing**: causal EMA on the (x, y) readout, α=0.2

## Layout

```
src/gaze_predictor/
├── README.md                 (this file)
├── __init__.py
├── model.py                  # CLIP encoders + PromptThenTemporalHead224 +
│                             # GazeTrajectoryPredictor
├── data.py                   # LeRobot v2.1 loader (videos + parquet gaze)
├── cache.py                  # vision/text feature cache; multi-dataset support
├── temporal_cache.py         # K-frame neighbor cache for temporal head
├── utils.py                  # gaze_to_heatmap_14x14, soft_argmax_2d,
│                             # windowed_soft_argmax, set_seed, ...
├── ema_smoothing.py          # streaming causal EMA
├── train_prompts.py          # CE loss + scheduler + train loop + wandb shim
├── pixel_l2_eval.py          # windowed L2 (raw + EMA)
├── jitter_eval.py            # frame-to-frame Δ (raw + EMA)
├── visualize_video.py        # train + test episode MP4 rendering
├── run_train.py              # one-shot driver (cache -> train -> eval -> videos)
└── configs/
    ├── gaze_predictor_six_task.yaml          # paper default (joint training, σ=16)
    └── gaze_predictor_single_task.yaml       # single-dataset template
```

## Train

The gaze predictor additionally requires `decord` for video decoding, which
is not in the main `pyproject.toml`. Install it once before training:

```bash
uv pip install decord
```

The paper's predictor is trained jointly on all six task datasets with a
task-balanced weighted sampler. Pass the datasets in the order the config
expects: T6 Catch Rolling Ball, T5 Insert Toilet Paper, T1 Stack Bowls,
T2 Stack Cubes in Order, T3 Bus Table, T4 Fold Towel. From the repo root:

```bash
uv run python -m gaze_predictor.run_train \
    --train_datasets /path/to/catch_rolling_ball_train,/path/to/insert_toilet_paper_train,/path/to/stack_bowls_train,/path/to/stack_cubes_in_order_train,/path/to/bus_table_train,/path/to/fold_towel_train \
    --base_config    src/gaze_predictor/configs/gaze_predictor_six_task.yaml \
    --out_dir        ./checkpoints/gaze_predictor/six_task \
    --exp_name       six_task
```

Single-dataset training uses the single-task template:

```bash
uv run python -m gaze_predictor.run_train \
    --train_datasets /path/to/stack_bowls_train \
    --base_config    src/gaze_predictor/configs/gaze_predictor_single_task.yaml \
    --out_dir        ./checkpoints/gaze_run/test \
    --exp_name       test
```

The driver caches frozen-encoder features once and trains with early stopping
on the 80/20 episode-level validation split. Passing `--test_dataset` with
held-out episodes additionally runs the pixel-L2 / jitter eval suite and
renders sample test videos.

## Deploy

`examples/teleavatar/gaze_client.py` imports `GazeTrajectoryPredictor`
and `windowed_soft_argmax` from this package and runs the model online from a
ROS2 head_camera stream (rolling K=3 vision cache + per-frame causal EMA
smoothing + crosshair overlay). See `examples/teleavatar/README.md`.
