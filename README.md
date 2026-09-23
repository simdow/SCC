# SCC-Net

Sensorless ultrasound motion estimation with frequency feature injection and bidirectional temporal cross attention.

## Architecture

Five grayscale frames form four consecutive frame pairs. The shared encoder produces image features and the original no-gradient correlation volume. Low/high-frequency channel and spatial attention inject a residual into the image features. Correlation enters the local fusion and global encoder paths. Local/global attention produces two sequences of 4 x 4 spatial tokens. SequenceCross exchanges local/global information across the forward and reversed sequences. Each stream is spatially averaged, processed by its own LSTM and linear pose head, and the two predictions are averaged.

Cross attention remains active during both training stages and inference. After the first training stage, an auxiliary branch masks only forward tokens and uses clean reverse tokens as reference. Both auxiliary inputs are detached; the branch shares the cross-attention module and pose heads. The objective uses pose MAE, correlation, and triplet losses. The default auxiliary coefficient is 0.05. There is no image-reconstruction objective in this training entry point.

## Layout

- `src/model.py`: SCC-Net sequence interaction and single-sided auxiliary branch.
- `src/backbone.py`: MoGLo-derived encoders and local/global attention.
- `src/frequency.py`: frequency decomposition and feature injection.
- `src/correlation.py`: cached, no-gradient correlation volume.
- `src/spatial_attention.py`: equivalent vectorized patch extraction.
- `src/data.py`: five-frame windows, preprocessing and pose labels.
- `src/losses.py`: original pose loss components and head averaging.
- `src/train.py`: fresh training and explicit same-entry-point resume.
- `src/inference.py`: complete scan inference and scan-level summaries.
- `src/metrics.py`: transform integration and metric definitions.

## Data

Dataset and scale paths have no machine-specific defaults. Provide them explicitly. The dataset directory contains numeric subject directories, each with H5 scans:

```text
dataset/
  000/*.h5
  ...
  049/*.h5
```

H5 fields are `frames` (T,H,W grayscale intensities on the original 0-255 scale), `tforms` (T,4,4 tracker transforms), and `T_calib` (4,4 calibration used for evaluation). Training uses subjects 000-029, validation 030-039, and test 040-049. Inputs are divided by 255 and bicubic-antialias resized to 256 x 256. Training labels are raw adjacent tracker transforms, with translation followed by Euler yzx angles in degrees. Calibration is applied to both predicted and reference transforms during evaluation, not to the training labels.

Supply the six positive label scales used by the checkpoint. A scale file may contain a tensor or a dictionary with `scale`, `M_emp_dy`, `My`, or `y_scale`. Do not estimate scale from validation or test subjects. Changing the scale or preprocessing changes the model contract.

## Training

Install dependencies from `requirements.txt` in your own environment. Expose one GPU. Start training from random initialization:

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.train \
  --data-root /path/to/dataset \
  --scale-path /path/to/training_scale.pt \
  --run-dir /path/to/new_run \
  --epochs 200 --mask-start 100 --batch-size 8
```

The main network remains active throughout training. The auxiliary branch is enabled at the configured boundary while retaining the optimizer state. Adam uses a learning rate of 1e-4. Training uses FP32 with TF32 disabled, random training windows, and reverse supervision constructed from inverse relative transforms.

Checkpoints retain model, optimizer, scheduler, and random states. `best.pt` is selected by fixed-window validation pose loss. Resume an interrupted run by adding `--resume /path/to/run/last.pt` with the same experiment configuration.

## Inference

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.inference \
  --data-root /path/to/dataset \
  --checkpoint /path/to/run/best.pt \
  --output-dir /path/to/new_inference
```

Five-frame windows use stride four and retain all adjacent predictions. A final overlapping window covers the tail, with duplicate predictions trimmed. The checkpoint contains label scale; use `--scale-path` only when a separate scale is required. Outputs include per-scan predictions, per-scan metrics and equal-scan mean/SD. SD is population standard deviation across scans (ddof=0), not standard error or variance. RTE is adjacent-step translation error; RRE is geodesic rotation error in degrees. FDR is final-position error divided by reference path length, computed per scan before averaging. No trajectory fitting is applied.

## Attribution

The backbone and pose-loss components derive from MoGLo. Preserve upstream attribution and applicable licensing when redistributing. This release uses LSTM temporal heads.
