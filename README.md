# SCC-Net

SCC-Net estimates adjacent ultrasound motion from short frame sequences with spatial-temporal feature modeling.

## Data

Provide the dataset and label-scale file explicitly. Each subject directory contains H5 scans with:

```text
dataset/
  000/*.h5
  ...
```

Each H5 file must contain `frames`, `tforms`, and `T_calib`. Frames are grayscale images on the 0-255 scale. Inputs are resized to 256 x 256. The scale file must contain the six positive label scales used by the model.

## Training

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.train   --data-root /path/to/dataset   --scale-path /path/to/training_scale.pt   --run-dir /path/to/new_run
```

## Inference

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.inference   --data-root /path/to/dataset   --checkpoint /path/to/run/best.pt   --output-dir /path/to/new_inference
```

Inference uses five-frame windows with stride four and reports scan-level motion metrics.
