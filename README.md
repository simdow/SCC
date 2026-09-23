# SCC-Net

This release contains the reusable model, training loop, dataset adapter, loss definitions, and full-sequence inference entry point for SCC-Net.

## Layout

- `src/model.py`: encoder, correlation, SACD, bidirectional temporal cross interaction, recurrent heads, and optional masked cross architecture.
- `src/train.py`: training and validation entry point.
- `src/inference.py`: sliding-window full-sequence inference.
- `src/data.py`: TUS-REC-style H5 reader and pose conversion.
- `src/losses.py`: pose MAE, correlation, and triplet losses.

## Data and outputs

The release intentionally contains no machine-specific dataset, checkpoint, or output path. Pass the dataset directory explicitly with `--data-root`, pass a writable experiment directory with `--run-dir`, and provide checkpoint and scale paths explicitly for inference. The expected dataset layout is numeric subject directories containing H5 scans; the H5 reader documents the required frame and transform fields in `src/data.py`.

## Training

Run one of the two direct 200-epoch truncations:

```bash
python src/train.py \
  --data-root /path/to/tus_rec_dataset \
  --run-dir /path/to/no_mask_200 \
  --device cuda:0 \
  --devices 0 \
  --batch-size 8 \
  --epochs 200 \
  --enable-local-sacd
```

For the mask truncation, use a separate run directory and enable the mask cross block from the first epoch:

```bash
python src/train.py \
  --data-root /path/to/tus_rec_dataset \
  --run-dir /path/to/mask_200 \
  --device cuda:0 \
  --devices 0 \
  --batch-size 8 \
  --epochs 200 \
  --enable-local-sacd \
  --enable-mask-cross \
  --stage2-start-epoch 0
```

Add `--enable-bidirectional-consistency` to either run when the bidirectional consistency term is part of the experiment. Keep the dataset split, seed, and loss configuration fixed so the two checkpoints are directly comparable.

## Inference

```bash
python src/inference.py \
  --data-root /path/to/tus_rec_dataset \
  --ckpt-path /path/to/run/checkpoints/best_model.pth \
  --scale-path /path/to/run/y_scale_98.pt \
  --output-dir /path/to/inference \
  --device cuda:0
```

Inference uses an N-frame window with stride N-1 by default and retains all adjacent inter-frame predictions for trajectory reconstruction.

## Reproducibility

Pin one visible GPU, record the command line, seed, data split, checkpoint, and scale file for each run. The repository does not assume access to any local server path.
