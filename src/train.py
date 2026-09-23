import argparse
import json
import os
import random
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from scipy.spatial.transform import Rotation as R
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

EXECUTION_CONTRACT = 'stage2_dense_mask_residual_serial_lstm_v1'

from inference import compute_src_style_metrics, evaluate_and_save
from data import (
    DEFAULT_DATA_ROOT,
    DEFAULT_EULER_ORDER,
    DEFAULT_RUN_ROOT,
    CalibratedPoseDataset,
    ensure_dir,
    format_split,
    generate_scale_file,
    list_subjects,
    load_scale_tensor,
    load_scan_arrays,
    relative_poses_from_tforms,
    resolve_scan_path,
    split_subjects,
)
from model import MyNet
from losses import MoGLoLoss_Wrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full progressive training for the local SCC-Net on TUS-REC style H5 scans."
    )
    parser.add_argument("--data-root", default="")
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Directory for checkpoints, logs, scale file, and evaluation snapshots.",
    )
    parser.add_argument(
        "--scale-path",
        default="",
        help="Optional external y_scale path. If empty, it is generated inside run-dir.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--devices",
        default="0",
        help="Comma-separated CUDA device ids for DataParallel, e.g. '0,1,2'. "
        "Ignored when device is CPU.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adamw")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--stage2-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument(
        "--train-alpha",
        type=int,
        default=10,
        help="Random 5-frame windows sampled per scan in each epoch. "
        "Aligned with freehand_us/src default alpha=10.",
    )
    parser.add_argument(
        "--val-windows-per-scan",
        type=int,
        default=2,
        help="Fixed validation windows per scan on eval epochs. Keep this small for fast training feedback.",
    )
    parser.add_argument(
        "--test-windows-per-scan",
        type=int,
        default=2,
        help="Fixed test windows per scan on eval epochs. Keep this small unless you really need per-epoch test monitoring.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.6, help="3:1:1 split -> 0.6")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="3:1:1 split -> 0.2")
    parser.add_argument("--scale-percentile", type=float, default=98.0)
    parser.add_argument("--rot-weight", type=float, default=5.0)
    parser.add_argument("--triplet-samples", type=int, default=2)
    parser.add_argument("--corr-weight", type=float, default=1.0)
    parser.add_argument("--triplet-weight", type=float, default=0.005)
    parser.add_argument("--mask-ratio", type=float, default=0.1)
    parser.add_argument("--mask-depth", type=int, default=2)
    parser.add_argument("--mask-num-heads", type=int, default=8)
    parser.add_argument("--mask-loss-weight", type=float, default=0.0)
    parser.add_argument("--consistency-weight", type=float, default=0.03)
    parser.add_argument(
        "--stage2-start-epoch",
        type=int,
        default=0,
        help="When --enable-mask-cross is set, the mask cross block is active from epoch 1.",
    )
    parser.add_argument(
        "--force-stage2-lr-schedule",
        action="store_true",
        help="Apply the stage-2 LR switch at stage2-start-epoch even when mask cross is disabled.",
    )
    parser.add_argument(
        "--consistency-start-epoch",
        type=int,
        default=0,
        help="Epoch index that activates bidirectional consistency. 0 means enable from epoch 1.",
    )
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument(
        "--window-eval-splits",
        default="val",
        help="Comma-separated splits for fixed-window evaluation on eval epochs. "
        "Supported: 'val', 'test'. Empty disables window eval.",
    )
    parser.add_argument(
        "--scan-eval-every",
        type=int,
        default=25,
        help="How often to run full-sequence evaluation on val/test scans. 0 disables it.",
    )
    parser.add_argument(
        "--fullseq-eval-splits",
        default="",
        help="Comma-separated splits for full-sequence evaluation. "
        "Supported: 'val', 'test'. Empty disables full-sequence eval.",
    )
    parser.add_argument(
        "--scan-eval-max-scans",
        type=int,
        default=0,
        help="Optional cap per split for full-sequence evaluation. 0 means use all scans.",
    )
    parser.add_argument(
        "--scan-eval-batch-size",
        type=int,
        default=16,
        help="Window batch size used inside full-sequence evaluation.",
    )
    parser.add_argument(
        "--save-fullseq-plots",
        type=int,
        default=1,
        help="How many scans per split to save trajectory plots for during full-sequence evaluation.",
    )
    parser.add_argument(
        "--monitor-h5-path",
        default="",
        help="Optional single scan path for periodic full-sequence monitoring. "
        "Supports absolute path, path relative to data-root, or a unique .h5 file name.",
    )
    parser.add_argument(
        "--enable-local-sacd",
        action="store_true",
        help="Keep the SACD local branch active. Recommended.",
    )
    parser.add_argument(
        "--enable-mask-cross",
        action="store_true",
        help="Enable the CroCo-style mask cross block for stage-2 training.",
    )
    parser.add_argument(
        "--enable-convlstm",
        action="store_true",
        help="Use spatial ConvLSTM before the existing vector LSTM heads.",
    )
    parser.add_argument(
        "--enable-bidirectional-consistency",
        action="store_true",
        help="Enable the flipped-sequence branch and consistency prior.",
    )
    parser.add_argument(
        "--disable-bf16",
        action="store_true",
        help="Disable bfloat16 autocast on CUDA.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore latest checkpoint even if it exists.",
    )
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument('--init-checkpoint', default='', help='Model weights only; requires --no-resume and a fresh run directory.')
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--max-test-batches", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        print("⚠️ CUDA 不可用，自动切换到 CPU。")
        return torch.device("cpu")
    return torch.device(device_arg)


def parse_eval_splits(raw: str) -> List[str]:
    if not raw.strip():
        return []
    items = [item.strip().lower() for item in raw.split(",") if item.strip()]
    valid = {"val", "test"}
    invalid = [item for item in items if item not in valid]
    if invalid:
        raise ValueError(f"Unsupported eval splits: {invalid}. Valid values are {sorted(valid)}")
    ordered = []
    for name in items:
        if name not in ordered:
            ordered.append(name)
    return ordered


def parse_device_ids(device_arg: str, devices_arg: str) -> List[int]:
    if not device_arg.startswith("cuda"):
        return []
    device_ids: List[int] = []
    for item in devices_arg.split(","):
        item = item.strip()
        if not item:
            continue
        device_ids.append(int(item))
    return device_ids


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def forward_model(
    model: torch.nn.Module,
    frames: torch.Tensor,
    enable_mask: bool,
    enable_backward_consistency: bool,
):
    if isinstance(model, nn.DataParallel) and frames.shape[0] < len(model.device_ids):
        base_model = unwrap_model(model)
        base_device = next(base_model.parameters()).device
        frames = frames.to(base_device, non_blocking=True)
        return base_model(
            frames,
            enable_mask=enable_mask,
            enable_backward_consistency=enable_backward_consistency,
        )

    return model(
        frames,
        enable_mask=enable_mask,
        enable_backward_consistency=enable_backward_consistency,
    )


def load_flexible_state_dict(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    model_to_load = unwrap_model(model)
    try:
        model_to_load.load_state_dict(state_dict)
        return
    except RuntimeError:
        pass

    if any(key.startswith("module.") for key in state_dict):
        stripped = {key[len("module.") :]: value for key, value in state_dict.items()}
        model_to_load.load_state_dict(stripped)
        return

    prefixed = {f"module.{key}": value for key, value in state_dict.items()}
    model.load_state_dict(prefixed)


def preprocess_window(frames_np: np.ndarray, image_size: int) -> torch.Tensor:
    imgs = TF.resize(
        torch.from_numpy(frames_np).float(),
        [image_size, image_size],
        antialias=True,
    )
    imgs = (imgs - imgs.mean()) / (imgs.std() + 1e-5)
    return imgs


def transform_v2m_convention(
    y: torch.Tensor,
    convention: str = DEFAULT_EULER_ORDER,
    degrees: bool = True,
) -> torch.Tensor:
    batch, steps, _ = y.shape
    y_flat = y.reshape(-1, 6).detach().cpu().numpy()
    rot_mats = R.from_euler(convention, y_flat[:, 3:], degrees=degrees).as_matrix()
    rot_mats = torch.from_numpy(rot_mats).to(y.device).float()

    transforms = torch.eye(4, device=y.device).repeat(batch * steps, 1, 1)
    transforms[:, :3, :3] = rot_mats
    transforms[:, :3, 3] = torch.from_numpy(y_flat[:, :3]).to(y.device).float()
    return transforms.view(batch, steps, 4, 4)


def transform_acum_convention(
    y: torch.Tensor,
    convention: str = DEFAULT_EULER_ORDER,
    degrees: bool = True,
) -> torch.Tensor:
    batch, steps, _ = y.shape
    t_rel = transform_v2m_convention(y, convention=convention, degrees=degrees)
    t_abs = [torch.eye(4, device=y.device).repeat(batch, 1, 1)]
    for step_idx in range(steps):
        t_abs.append(torch.matmul(t_abs[-1], t_rel[:, step_idx]))
    return torch.stack(t_abs, dim=1)


def compute_window_metrics(
    y_pred: torch.Tensor,
    y_gt: torch.Tensor,
    convention: str = DEFAULT_EULER_ORDER,
) -> Dict[str, float]:
    t_pred, t_gt = y_pred[..., :3], y_gt[..., :3]
    rte = torch.norm(t_pred - t_gt, dim=-1).mean().item()

    r_pred = transform_v2m_convention(y_pred, convention=convention)[..., :3, :3]
    r_gt = transform_v2m_convention(y_gt, convention=convention)[..., :3, :3]
    r_diff = torch.matmul(r_gt.transpose(-1, -2), r_pred)
    trace = r_diff[..., 0, 0] + r_diff[..., 1, 1] + r_diff[..., 2, 2]
    trace = torch.clamp(trace, -1.0, 3.0)
    rre = (torch.acos((trace - 1.0) / 2.0) * (180.0 / torch.pi)).mean().item()

    corr = F.cosine_similarity(
        t_pred.reshape(-1, 3),
        t_gt.reshape(-1, 3),
        dim=-1,
    ).mean().item()

    t_pred_abs = transform_acum_convention(y_pred, convention=convention)
    t_gt_abs = transform_acum_convention(y_gt, convention=convention)
    pos_pred, pos_gt = t_pred_abs[..., :3, 3], t_gt_abs[..., :3, 3]

    ate = torch.sqrt(torch.mean(torch.sum((pos_pred - pos_gt) ** 2, dim=-1))).item()
    fd = torch.norm(pos_pred[:, -1, :] - pos_gt[:, -1, :], dim=-1).mean().item()
    path_len = torch.sum(
        torch.norm(pos_gt[:, 1:, :] - pos_gt[:, :-1, :], dim=-1),
        dim=-1,
    )
    fdr = ((fd / (path_len.mean().item() + 1e-6)) * 100.0) if path_len.numel() > 0 else 0.0

    src_metrics = compute_src_style_metrics(y_pred, y_gt)

    return {
        **src_metrics,
        "RTE_mm": float(rte),
        "RRE_deg": float(rre),
        "ATE_mm": float(ate),
        "FD_mm": float(fd),
        "FDR_percent": float(fdr),
        "StepTranslationCosine": float(corr),
    }


def list_scan_paths(data_root: str, subjects: Sequence[str]) -> List[str]:
    paths: List[str] = []
    for subject in subjects:
        subject_dir = os.path.join(data_root, subject)
        if not os.path.isdir(subject_dir):
            continue
        for file_name in sorted(os.listdir(subject_dir)):
            if file_name.endswith(".h5"):
                paths.append(os.path.join(subject_dir, file_name))
    return paths


def average_scan_prediction(
    model: torch.nn.Module,
    frames: np.ndarray,
    y_scale: torch.Tensor,
    seq_len: int,
    image_size: int,
    device: torch.device,
    enable_mask: bool,
    batch_size: int = 16,
) -> torch.Tensor:
    num_frames = len(frames)
    num_steps = num_frames - 1
    max_start = num_frames - seq_len
    sum_pred = torch.zeros(num_steps, 6, dtype=torch.float32)
    count = torch.zeros(num_steps, 1, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        starts = list(range(max_start + 1))
        batch_size = max(int(batch_size), 1)
        for batch_start in range(0, len(starts), batch_size):
            batch_starts = starts[batch_start : batch_start + batch_size]
            imgs = torch.stack(
                [
                    preprocess_window(
                        frames[start_idx : start_idx + seq_len],
                        image_size=image_size,
                    )
                    for start_idx in batch_starts
                ],
                dim=0,
            ).to(device, non_blocking=True)
            with autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                preds, _, _ = forward_model(
                    model=model,
                    frames=imgs,
                    enable_mask=enable_mask,
                    enable_backward_consistency=False,
                )
            preds_real_batch = (preds.float() * y_scale).cpu()
            for item_idx, start_idx in enumerate(batch_starts):
                preds_real = preds_real_batch[item_idx]
                for local_step in range(seq_len - 1):
                    global_step = start_idx + local_step
                    sum_pred[global_step] += preds_real[local_step]
                    count[global_step] += 1.0

    pred_rel = sum_pred / count.clamp_min(1.0)
    return pred_rel.unsqueeze(0).to(device)


def evaluate_split_fullseq(
    model: torch.nn.Module,
    data_root: str,
    subjects: Sequence[str],
    y_scale: torch.Tensor,
    seq_len: int,
    image_size: int,
    device: torch.device,
    enable_mask: bool,
    output_dir: str,
    max_scans: int = 0,
    batch_size: int = 16,
    save_plots: int = 0,
    scan_paths_override: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    if scan_paths_override is not None:
        scan_paths = list(scan_paths_override)
    else:
        scan_paths = list_scan_paths(data_root, subjects)
    if max_scans > 0:
        scan_paths = scan_paths[:max_scans]
    if not scan_paths:
        return {}

    metrics_all: List[Dict[str, float]] = []
    ensure_dir(output_dir)
    for scan_idx, h5_path in enumerate(scan_paths):
        frames, tforms, t_calib = load_scan_arrays(h5_path, data_root=data_root)
        gt_rel_all = relative_poses_from_tforms(
            tforms,
            t_calib=t_calib,
            degrees=True,
            euler_order=DEFAULT_EULER_ORDER,
        )
        y_pred = average_scan_prediction(
            model=model,
            frames=frames,
            y_scale=y_scale,
            seq_len=seq_len,
            image_size=image_size,
            device=device,
            enable_mask=enable_mask,
            batch_size=batch_size,
        )
        y_gt = torch.from_numpy(gt_rel_all).float().unsqueeze(0).to(device)

        if scan_idx < save_plots:
            metrics = evaluate_and_save(
                y_pred=y_pred,
                y_gt=y_gt,
                output_dir=output_dir,
                scan_name=os.path.basename(h5_path),
            )
        else:
            metrics = evaluate_and_save(
                y_pred=y_pred,
                y_gt=y_gt,
                output_dir=os.path.join(output_dir, "_tmp_metrics"),
                scan_name=os.path.basename(h5_path),
            )
            fig_path = metrics.get("figure_path")
            traj_path = metrics.get("trajectory_path")
            metrics_path = metrics.get("metrics_path")
            if fig_path and os.path.exists(fig_path):
                os.remove(fig_path)
            if traj_path and os.path.exists(traj_path):
                os.remove(traj_path)
            if metrics_path and os.path.exists(metrics_path):
                os.remove(metrics_path)

        metrics_all.append(metrics)

    metric_keys = [
        "rAE",
        "aAE",
        "rFE",
        "aFE",
        "Corr",
        "FDR",
        "ATE_mm",
        "FDR_percent",
        "FD_mm",
        "PathLength_mm",
        "RTE_mm",
        "RRE_deg",
        "StepTranslationCosine",
        "StepRotationAbsMAE_deg",
        "PathShapeCorr",
        "TurnAngleMAE_deg",
    ]
    summary: Dict[str, float] = {}
    for key in metric_keys:
        values = [float(item[key]) for item in metrics_all if key in item]
        if values:
            summary[key] = float(np.mean(values))
    summary["num_scans"] = float(len(metrics_all))
    return summary


def load_initial_weights(model: torch.nn.Module, checkpoint_path: str) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = checkpoint.get('model_state', checkpoint)
    unwrap_model(model).load_state_dict(state, strict=True)
    digest = hashlib.sha256()
    with open(checkpoint_path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            digest.update(chunk)
    return {'mode': 'model_weights_only', 'path': os.path.abspath(checkpoint_path),
            'sha256': digest.hexdigest(), 'source_epoch': checkpoint.get('epoch'),
            'optimizer_imported': False, 'history_imported': False, 'best_score_imported': False}


def load_checkpoint_if_available(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    latest_path: str,
    device: torch.device,
    allow_resume: bool,
) -> tuple[int, float, List[dict], dict]:
    start_epoch = 0
    best_val_ate = float("inf")
    history: List[dict] = []

    if not allow_resume or not os.path.exists(latest_path):
        return start_epoch, best_val_ate, history, {}

    print(f"🔄 发现续训存档: {latest_path}")
    ckpt = torch.load(latest_path, map_location=device, weights_only=False)
    if ckpt.get('execution_contract') != EXECUTION_CONTRACT:
        raise ValueError('Checkpoint uses a different execution path; start a fresh run with --init-checkpoint and --no-resume.')
    load_flexible_state_dict(model, ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    start_epoch = int(ckpt["epoch"])
    best_val_ate = float(ckpt.get("best_val_ate", float("inf")))
    history = list(ckpt.get("history", []))
    print(f"✅ 已恢复到 epoch {start_epoch + 1}")
    return start_epoch, best_val_ate, history, dict(ckpt.get('initialization', {}))


def save_history(history: List[dict], run_dir: str) -> str:
    history_path = os.path.join(run_dir, "history.json")
    with open(history_path, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    return history_path


def main() -> None:
    args = parse_args()
    if not args.data_root.strip():
        raise ValueError("--data-root is required; provide the dataset directory described in README.md")
    run_path = Path(args.run_dir).expanduser().resolve()
    if args.init_checkpoint and not args.no_resume:
        raise ValueError('--init-checkpoint requires --no-resume; old optimizer and scores must not be inherited.')
    if args.no_resume and run_path.exists() and any(run_path.iterdir()):
        raise ValueError('A fresh run requires an empty output directory.')
    if args.device.startswith('cuda'):
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
        if len(visible) != 1 or not visible[0].strip() or args.devices != '0' or args.device != 'cuda:0':
            raise ValueError('Pin one physical GPU after preflight and use --device cuda:0 --devices 0.')
        if torch.cuda.device_count() != 1:
            raise ValueError('Exactly one GPU must be visible.')
    if 'test' in parse_eval_splits(args.window_eval_splits) or 'test' in parse_eval_splits(args.fullseq_eval_splits):
        raise ValueError('Use validation for training-time selection; evaluate the test split separately after selection.')
    set_seed(args.seed)

    device = resolve_device(args.device)
    use_amp = device.type == "cuda" and not args.disable_bf16
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    device_ids = parse_device_ids(args.device, args.devices)
    window_eval_splits = parse_eval_splits(args.window_eval_splits)
    fullseq_eval_splits = parse_eval_splits(args.fullseq_eval_splits)

    run_dir = ensure_dir(os.path.abspath(args.run_dir))
    ckpt_dir = ensure_dir(os.path.join(run_dir, "checkpoints"))
    history_path = os.path.join(run_dir, "history.json")
    split_path = os.path.join(run_dir, "subject_split.json")

    subject_split = split_subjects(
        list_subjects(args.data_root),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    print(f"📚 Subject split | {format_split(subject_split)}")
    print(f"📐 Euler convention | {DEFAULT_EULER_ORDER.upper()} + degrees")

    with open(split_path, "w", encoding="utf-8") as handle:
        json.dump(subject_split, handle, indent=2)

    scale_path = (
        os.path.abspath(args.scale_path)
        if args.scale_path
        else os.path.join(run_dir, f"y_scale_{int(args.scale_percentile)}.pt")
    )
    if not os.path.exists(scale_path):
        print(f"📏 未发现 y_scale，开始生成: {scale_path}")
        _, y_scale_np = generate_scale_file(
            data_root=args.data_root,
            subject_ids=subject_split["train"],
            save_path=scale_path,
            percentile=args.scale_percentile,
        )
        print(f"✅ y_scale 完成: {np.round(y_scale_np, 4)}")
    else:
        print(f"📏 使用已有 y_scale: {scale_path}")
    y_scale = load_scale_tensor(scale_path, device=device)

    train_dataset = CalibratedPoseDataset(
        data_root=args.data_root,
        subjects=subject_split["train"],
        seq_len=args.seq_len,
        image_size=args.image_size,
        alpha=args.train_alpha,
        sampling="random",
    )
    val_dataset = CalibratedPoseDataset(
        data_root=args.data_root,
        subjects=subject_split["val"],
        seq_len=args.seq_len,
        image_size=args.image_size,
        sampling="fixed",
        windows_per_scan=args.val_windows_per_scan,
    )
    test_dataset = CalibratedPoseDataset(
        data_root=args.data_root,
        subjects=subject_split["test"],
        seq_len=args.seq_len,
        image_size=args.image_size,
        sampling="fixed",
        windows_per_scan=args.test_windows_per_scan,
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )

    model = MyNet(
        seq_len=args.seq_len,
        input_size=args.image_size,
        mask_ratio=args.mask_ratio,
        mask_depth=args.mask_depth,
        mask_num_heads=args.mask_num_heads,
        enable_local_sacd=args.enable_local_sacd,
        enable_mask_cross=args.enable_mask_cross,
        enable_convlstm=args.enable_convlstm,
        enable_bidirectional_consistency=args.enable_bidirectional_consistency,
    ).to(device)
    initialization = load_initial_weights(model, args.init_checkpoint) if args.init_checkpoint else {'mode': 'fresh_or_same_contract_resume'}
    if device.type == "cuda" and len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])
        print(f"🚀 DataParallel 已启用 | device_ids={device_ids}")
    elif device.type == "cuda":
        print(f"🚀 单卡训练 | device_id={device_ids[0] if device_ids else 0}")
    criterion = MoGLoLoss_Wrapper(
        rot_weight=args.rot_weight,
        mask_weight=0.0,
        triplet_samples=args.triplet_samples,
    ).to(device)

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
    )
    scaler = GradScaler(device.type, enabled=use_amp)

    latest_path = os.path.join(ckpt_dir, "latest_checkpoint.pth")
    best_ckpt_path = os.path.join(ckpt_dir, "best_checkpoint.pth")
    best_model_path = os.path.join(ckpt_dir, "best_model.pth")
    start_epoch, best_val_ate, history, resumed_initialization = load_checkpoint_if_available(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        latest_path=latest_path,
        device=device,
        allow_resume=not args.no_resume,
    )
    if resumed_initialization:
        initialization = resumed_initialization

    print(
        f"🔥 开始训练 | device={device} | train_batches={len(train_loader)} | "
        f"val_batches={len(val_loader)} | test_batches={len(test_loader)}"
    )
    print(
        f"🧪 Eval plan | window_splits={window_eval_splits or ['none']} | "
        f"fullseq_splits={fullseq_eval_splits or ['none']} | "
        f"eval_every={args.eval_every} | scan_eval_every={args.scan_eval_every}"
    )
    use_fullseq_for_best = ("val" in fullseq_eval_splits) and (args.scan_eval_every > 0)
    if use_fullseq_for_best:
        history_fullseq_vals = [
            float(item["val_full_ATE_mm"])
            for item in history
            if "val_full_ATE_mm" in item
        ]
        best_val_ate = min(history_fullseq_vals) if history_fullseq_vals else float("inf")
        print(
            "🏁 Best checkpoint policy | single-best on periodic VAL full-seq ATE only "
            f"(history_fullseq={len(history_fullseq_vals)})"
        )
    else:
        print("🏁 Best checkpoint policy | single-best on VAL proxy ATE")
    monitor_scan_path = ""
    if args.monitor_h5_path.strip():
        monitor_scan_path = resolve_scan_path(args.monitor_h5_path, data_root=args.data_root)
        print(f"🎯 Monitor scan | {monitor_scan_path}")

    stage2_announced = False
    for epoch in range(start_epoch, args.epochs):
        enable_mask = args.enable_mask_cross and (epoch >= args.stage2_start_epoch)
        enable_consistency = (
            args.enable_bidirectional_consistency
            and epoch >= args.consistency_start_epoch
        )

        stage2_lr_active = (enable_mask or args.force_stage2_lr_schedule) and (
            epoch >= args.stage2_start_epoch
        )

        if stage2_lr_active and not stage2_announced:
            if args.stage2_lr > 0:
                for group in optimizer.param_groups:
                    group["lr"] = args.stage2_lr
            print(
                f"🎯 进入 stage2 LR 日程: epoch={epoch + 1}, "
                f"mask_on={enable_mask}, bidir_on={enable_consistency}, "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )
            stage2_announced = True

        loss_weights = {
            "mmae": 1.0,
            "corr": args.corr_weight,
            "tri": args.triplet_weight,
            "mask": 0.0,
            "consist": args.consistency_weight if enable_consistency else 0.0,
        }
        if enable_mask and enable_consistency:
            stage_name = "stage2_mask_bidir"
        elif enable_mask:
            stage_name = "stage2_mask"
        elif enable_consistency:
            stage_name = "stage1_base_bidir"
        else:
            stage_name = "stage1_base"

        model.train()
        train_bar = tqdm(train_loader, desc=f"Ep {epoch + 1:03d} [Train]")
        train_stats: List[Dict[str, float]] = []
        for batch_idx, (frames, pose_gt) in enumerate(train_bar, start=1):
            frames = frames.to(device, non_blocking=True)
            pose_gt = pose_gt.to(device, non_blocking=True)
            pose_scaled = pose_gt / y_scale

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                preds, feats, aux = forward_model(
                    model=model,
                    frames=frames,
                    enable_mask=enable_mask,
                    enable_backward_consistency=enable_consistency,
                )
                loss, loss_mmae, loss_corr, loss_tri, loss_mask, loss_consist = criterion(
                    preds,
                    pose_scaled,
                    feats,
                    aux=aux,
                    loss_weights=loss_weights,
                    y_scale=y_scale,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_stats.append(
                {
                    "total": float(loss.item()),
                    "mmae": float(loss_mmae.item()),
                    "corr": float(loss_corr.item()),
                    "tri": float(loss_tri.item()),
                    "mask": float(loss_mask.item()),
                    "consist": float(loss_consist.item()),
                }
            )
            train_bar.set_postfix(
                total=f"{loss.item():.3f}",
                mmae=f"{loss_mmae.item():.3f}",
                corr=f"{loss_corr.item():.3f}",
                tri=f"{loss_tri.item():.3f}",
                mask=f"{loss_mask.item():.3f}",
                consist=f"{loss_consist.item():.3f}",
            )

            if args.max_train_batches > 0 and batch_idx >= args.max_train_batches:
                break

        if not (stage2_lr_active and args.stage2_lr > 0):
            scheduler.step()

        def eval_loader(
            loader: DataLoader,
            split_name: str,
            max_batches: int,
        ) -> Dict[str, float]:
            model.eval()
            metrics_list: List[Dict[str, float]] = []
            with torch.no_grad():
                for batch_idx, (frames, pose_gt) in enumerate(loader, start=1):
                    frames = frames.to(device, non_blocking=True)
                    pose_gt = pose_gt.to(device, non_blocking=True)

                    with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                        preds, _, _ = forward_model(
                            model=model,
                            frames=frames,
                            enable_mask=enable_mask,
                            enable_backward_consistency=False,
                        )

                    preds_real = preds.float() * y_scale
                    metrics_list.append(
                        compute_window_metrics(
                            preds_real,
                            pose_gt,
                            convention=DEFAULT_EULER_ORDER,
                        )
                    )
                    if max_batches > 0 and batch_idx >= max_batches:
                        break

            summary: Dict[str, float] = {}
            if metrics_list:
                for key in metrics_list[0]:
                    summary[key] = float(np.mean([item[key] for item in metrics_list]))
            print(
                f"Ep {epoch + 1:03d} [{split_name}] | "
                f"rAE={summary.get('rAE', 0.0):.4f} mm | "
                f"aAE={summary.get('aAE', 0.0):.4f} deg | "
                f"rFE={summary.get('rFE', 0.0):.4f} mm | "
                f"aFE={summary.get('aFE', 0.0):.4f} deg | "
                f"Corr={summary.get('Corr', 0.0):.4f} | "
                f"FDR={summary.get('FDR', 0.0):.2f}% | "
                f"ATE={summary.get('ATE_mm', 0.0):.4f} mm | "
                f"RTE={summary.get('RTE_mm', 0.0):.4f} mm | "
                f"RRE={summary.get('RRE_deg', 0.0):.4f} deg"
            )
            return summary

        should_eval = ((epoch + 1) % max(args.eval_every, 1) == 0) or ((epoch + 1) == args.epochs)
        window_summaries: Dict[str, Dict[str, float]] = {}
        eval_loader_map = {
            "val": (val_loader, args.max_val_batches),
            "test": (test_loader, args.max_test_batches),
        }
        if should_eval:
            for split_name in window_eval_splits:
                loader, max_batches = eval_loader_map[split_name]
                window_summaries[split_name] = eval_loader(
                    loader,
                    split_name.upper(),
                    max_batches,
                )
        else:
            print(
                f"Ep {epoch + 1:03d} [Eval] | 跳过窗口级 val/test，"
                f"等待第 {args.eval_every} 的倍数 epoch 再评估"
            )

        epoch_record = {
            "epoch": epoch + 1,
            "stage": stage_name,
            "mask_on": enable_mask,
            "consistency_on": enable_consistency,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_total": float(np.mean([item["total"] for item in train_stats])),
            "train_mmae": float(np.mean([item["mmae"] for item in train_stats])),
            "train_corr": float(np.mean([item["corr"] for item in train_stats])),
            "train_tri": float(np.mean([item["tri"] for item in train_stats])),
            "train_mask": float(np.mean([item["mask"] for item in train_stats])),
            "train_consist": float(np.mean([item["consist"] for item in train_stats])),
        }
        for split_name, summary in window_summaries.items():
            for key, value in summary.items():
                epoch_record[f"{split_name}_{key}"] = value

        run_scan_eval = (
            should_eval
            and args.scan_eval_every > 0
            and (len(fullseq_eval_splits) > 0 or bool(monitor_scan_path))
            and (
                (epoch + 1) % args.scan_eval_every == 0 or (epoch + 1) == args.epochs
            )
        )
        if run_scan_eval:
            for split_name in fullseq_eval_splits:
                full_dir = ensure_dir(
                    os.path.join(run_dir, "fullseq_eval", f"epoch_{epoch + 1:03d}", split_name)
                )
                full_summary = evaluate_split_fullseq(
                    model=model,
                    data_root=args.data_root,
                    subjects=subject_split[split_name],
                    y_scale=y_scale,
                    seq_len=args.seq_len,
                    image_size=args.image_size,
                    device=device,
                    enable_mask=enable_mask,
                    output_dir=full_dir,
                    max_scans=args.scan_eval_max_scans,
                    batch_size=args.scan_eval_batch_size,
                    save_plots=args.save_fullseq_plots,
                )
                if not full_summary:
                    continue
                print(
                    f"Ep {epoch + 1:03d} [{split_name.upper()} FullSeq] | "
                    f"rAE={full_summary.get('rAE', 0.0):.4f} mm | "
                    f"aAE={full_summary.get('aAE', 0.0):.4f} deg | "
                    f"rFE={full_summary.get('rFE', 0.0):.4f} mm | "
                    f"aFE={full_summary.get('aFE', 0.0):.4f} deg | "
                    f"Corr={full_summary.get('Corr', 0.0):.4f} | "
                    f"ATE={full_summary.get('ATE_mm', 0.0):.4f} mm | "
                    f"FDR={full_summary.get('FDR', full_summary.get('FDR_percent', 0.0)):.2f}% | "
                    f"ShapeCorr={full_summary.get('PathShapeCorr', 0.0):.4f} | "
                    f"TurnMAE={full_summary.get('TurnAngleMAE_deg', 0.0):.4f} deg"
                )
                for key, value in full_summary.items():
                    epoch_record[f"{split_name}_full_{key}"] = value

            if monitor_scan_path:
                manual_dir = ensure_dir(
                    os.path.join(run_dir, "fullseq_eval", f"epoch_{epoch + 1:03d}", "manual")
                )
                manual_summary = evaluate_split_fullseq(
                    model=model,
                    data_root=args.data_root,
                    subjects=[],
                    y_scale=y_scale,
                    seq_len=args.seq_len,
                    image_size=args.image_size,
                    device=device,
                    enable_mask=enable_mask,
                    output_dir=manual_dir,
                    max_scans=1,
                    batch_size=args.scan_eval_batch_size,
                    save_plots=1,
                    scan_paths_override=[monitor_scan_path],
                )
                if manual_summary:
                    print(
                        f"Ep {epoch + 1:03d} [MANUAL FullSeq] | "
                        f"scan={os.path.basename(monitor_scan_path)} | "
                        f"rAE={manual_summary.get('rAE', 0.0):.4f} mm | "
                        f"aAE={manual_summary.get('aAE', 0.0):.4f} deg | "
                        f"rFE={manual_summary.get('rFE', 0.0):.4f} mm | "
                        f"aFE={manual_summary.get('aFE', 0.0):.4f} deg | "
                        f"Corr={manual_summary.get('Corr', 0.0):.4f} | "
                        f"ATE={manual_summary.get('ATE_mm', 0.0):.4f} mm | "
                        f"FDR={manual_summary.get('FDR', manual_summary.get('FDR_percent', 0.0)):.2f}% | "
                        f"ShapeCorr={manual_summary.get('PathShapeCorr', 0.0):.4f} | "
                        f"TurnMAE={manual_summary.get('TurnAngleMAE_deg', 0.0):.4f} deg"
                    )
                    epoch_record["manual_scan_path"] = monitor_scan_path
                    for key, value in manual_summary.items():
                        epoch_record[f"manual_full_{key}"] = value

        history.append(epoch_record)
        save_history(history, run_dir)

        if use_fullseq_for_best:
            select_metric = epoch_record.get("val_full_ATE_mm")
        else:
            select_metric = epoch_record.get("val_full_ATE_mm", epoch_record.get("val_ATE_mm"))
            if select_metric is None:
                select_metric = epoch_record.get("test_full_ATE_mm", epoch_record.get("test_ATE_mm"))
        latest_payload = {
            'execution_contract': EXECUTION_CONTRACT,
            'initialization': initialization,
            'network_source_sha256': hashlib.sha256((Path(__file__).with_name('model.py')).read_bytes()).hexdigest(),
            "epoch": epoch + 1,
            "model_state": unwrap_model(model).state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_ate": best_val_ate,
            "scale_path": scale_path,
            "euler_order": DEFAULT_EULER_ORDER,
            "subject_split": subject_split,
            "history": history,
            "args": vars(args),
        }
        torch.save(latest_payload, latest_path)

        if select_metric is not None and select_metric < best_val_ate:
            best_val_ate = float(select_metric)
            latest_payload["best_val_ate"] = best_val_ate
            torch.save(unwrap_model(model).state_dict(), best_model_path)
            torch.save(
                {
                    **latest_payload,
                    "metrics": epoch_record,
                },
                best_ckpt_path,
            )
            print(f"🌟 刷新最佳验证 ATE: {best_val_ate:.4f} mm")
        elif use_fullseq_for_best and "val_full_ATE_mm" not in epoch_record:
            print(
                f"Ep {epoch + 1:03d} [Best] | 本轮未执行 VAL full-seq，保持当前 best={best_val_ate:.4f} mm"
            )

        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            snapshot_path = os.path.join(ckpt_dir, f"snapshot_ep{epoch + 1:03d}.pth")
            torch.save(unwrap_model(model).state_dict(), snapshot_path)

    print(f"💾 训练完成 | run_dir={run_dir}")
    print(f"📜 历史记录: {history_path}")


if __name__ == "__main__":
    main()
