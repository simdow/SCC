import argparse
import json
import os
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from data import (
    DEFAULT_DATA_ROOT,
    DEFAULT_EULER_ORDER,
    DEFAULT_RUN_ROOT,
    format_split,
    list_subjects,
    load_scan_arrays,
    load_scale_tensor,
    relative_poses_from_tforms,
    resolve_scan_path,
    split_subjects,
)
from model import MyNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full-sequence inference for the SCC-Net model."
    )
    parser.add_argument("--data-root", default="")
    parser.add_argument(
        "--ckpt-path",
        default="",
    )
    parser.add_argument(
        "--scale-path",
        default="",
    )
    parser.add_argument(
        "--h5-path",
        default="",
        help="Optional scan to test. If empty, the first test scan is used.",
    )
    parser.add_argument(
        "--scan-path",
        default="",
        help="Preferred single-scan inference entry. Supports absolute path, "
        "path relative to data-root, or a unique .h5 file name.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--mask-ratio", type=float, default=None)
    parser.add_argument("--mask-depth", type=int, default=None)
    parser.add_argument("--mask-num-heads", type=int, default=None)
    parser.add_argument(
        "--stride",
        type=int,
        default=0,
        help="Window step. 0 means seq_len - 1, which avoids duplicate relative poses.",
    )
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--enable-local-sacd", dest="enable_local_sacd", action="store_true")
    parser.add_argument("--disable-local-sacd", dest="enable_local_sacd", action="store_false")
    parser.add_argument("--enable-mask-cross", dest="enable_mask_cross", action="store_true")
    parser.add_argument("--disable-mask-cross", dest="enable_mask_cross", action="store_false")
    parser.add_argument("--enable-convlstm", dest="enable_convlstm", action="store_true")
    parser.add_argument("--disable-convlstm", dest="enable_convlstm", action="store_false")
    parser.add_argument(
        "--enable-bidirectional-consistency",
        dest="enable_bidirectional_consistency",
        action="store_true",
    )
    parser.add_argument(
        "--disable-bidirectional-consistency",
        dest="enable_bidirectional_consistency",
        action="store_false",
    )
    parser.set_defaults(
        enable_local_sacd=None,
        enable_mask_cross=None,
        enable_convlstm=None,
        enable_bidirectional_consistency=None,
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        print("⚠️ CUDA 不可用，自动切换到 CPU。")
        return torch.device("cpu")
    return torch.device(device_arg)


def pick_default_scan(data_root: str, train_ratio: float, val_ratio: float) -> str:
    split_map = split_subjects(
        list_subjects(data_root),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    print(f"📚 Subject split | {format_split(split_map)}")

    for subject in split_map["test"]:
        subject_dir = os.path.join(data_root, subject)
        if not os.path.isdir(subject_dir):
            continue
        for file_name in sorted(os.listdir(subject_dir)):
            if file_name.endswith(".h5"):
                return os.path.join(subject_dir, file_name)

    raise FileNotFoundError("Could not locate a default test scan under the test split.")


def resolve_requested_scan(
    data_root: str,
    scan_path: str,
    h5_path: str,
    train_ratio: float,
    val_ratio: float,
) -> str:
    requested = (scan_path or "").strip() or (h5_path or "").strip()
    if requested:
        return resolve_scan_path(requested, data_root=data_root)
    return pick_default_scan(
        data_root,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )


def load_checkpoint_payload(ckpt_path: str, device: torch.device) -> object:
    return torch.load(ckpt_path, map_location=device, weights_only=False)


def maybe_load_checkpoint_metadata(ckpt_path: str, device: torch.device) -> object:
    payload = load_checkpoint_payload(ckpt_path, device)
    if isinstance(payload, dict) and "model_state" in payload:
        return payload

    ckpt_dir = os.path.dirname(ckpt_path)
    ckpt_name = os.path.basename(ckpt_path)
    sibling_name = ""
    if ckpt_name == "best_model.pth":
        sibling_name = "best_checkpoint.pth"
    elif ckpt_name == "latest_model.pth":
        sibling_name = "latest_checkpoint.pth"

    if sibling_name:
        sibling_path = os.path.join(ckpt_dir, sibling_name)
        if os.path.isfile(sibling_path):
            sibling_payload = load_checkpoint_payload(sibling_path, device)
            if isinstance(sibling_payload, dict) and "model_state" in sibling_payload:
                return sibling_payload
    return payload


def extract_checkpoint_args(payload: object) -> dict:
    if isinstance(payload, dict):
        ckpt_args = payload.get("args", {})
        if isinstance(ckpt_args, dict):
            return ckpt_args
    return {}


def extract_checkpoint_epoch(payload: object) -> Optional[int]:
    if isinstance(payload, dict):
        epoch = payload.get("epoch")
        if isinstance(epoch, int):
            return epoch
    return None


def extract_model_state(payload: object) -> dict:
    if isinstance(payload, dict) and "model_state" in payload:
        return payload["model_state"]
    return payload


def load_model_weights(model: torch.nn.Module, payload: object) -> None:
    model.load_state_dict(extract_model_state(payload))


def resolve_model_value(
    cli_value,
    ckpt_args: dict,
    key: str,
    default,
):
    if cli_value is not None:
        return cli_value
    if key in ckpt_args and ckpt_args[key] is not None:
        return ckpt_args[key]
    return default


def resolve_model_config(args: argparse.Namespace, ckpt_args: dict) -> dict:
    return {
        "seq_len": resolve_model_value(args.seq_len, ckpt_args, "seq_len", 5),
        "input_size": resolve_model_value(args.image_size, ckpt_args, "image_size", 256),
        "mask_ratio": resolve_model_value(args.mask_ratio, ckpt_args, "mask_ratio", 0.1),
        "mask_depth": resolve_model_value(args.mask_depth, ckpt_args, "mask_depth", 2),
        "mask_num_heads": resolve_model_value(args.mask_num_heads, ckpt_args, "mask_num_heads", 8),
        "enable_local_sacd": resolve_model_value(
            args.enable_local_sacd,
            ckpt_args,
            "enable_local_sacd",
            True,
        ),
        "enable_mask_cross": resolve_model_value(
            args.enable_mask_cross,
            ckpt_args,
            "enable_mask_cross",
            False,
        ),
        "enable_convlstm": resolve_model_value(
            args.enable_convlstm,
            ckpt_args,
            "enable_convlstm",
            False,
        ),
        "enable_bidirectional_consistency": resolve_model_value(
            args.enable_bidirectional_consistency,
            ckpt_args,
            "enable_bidirectional_consistency",
            False,
        ),
    }


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


def compute_src_style_metrics(y_pred: torch.Tensor, y_gt: torch.Tensor) -> dict[str, float]:
    t_pred = y_pred[..., :3]
    t_gt = y_gt[..., :3]
    r_pred = y_pred[..., 3:]
    r_gt = y_gt[..., 3:]

    rAE = torch.norm(t_pred - t_gt, dim=-1).mean()
    aAE = torch.norm(r_pred - r_gt, dim=-1).mean()

    path_pred = torch.sum(t_pred, dim=1)
    path_gt = torch.sum(t_gt, dim=1)
    angle_pred = torch.sum(r_pred, dim=1)
    angle_gt = torch.sum(r_gt, dim=1)

    rFE = torch.norm(path_pred - path_gt, dim=-1).mean()
    aFE = torch.norm(angle_pred - angle_gt, dim=-1).mean()

    flat_tp = t_pred.reshape(-1, 3)
    flat_tg = t_gt.reshape(-1, 3)
    corr = F.cosine_similarity(flat_tp, flat_tg, dim=-1).mean()

    path_len = torch.sum(torch.norm(t_gt, dim=-1), dim=-1)
    drift = torch.norm(path_pred - path_gt, dim=-1)
    fdr = (drift / (path_len + 1e-6)).mean() * 100.0

    return {
        "rAE": float(rAE.item()),
        "aAE": float(aAE.item()),
        "rFE": float(rFE.item()),
        "aFE": float(aFE.item()),
        "Corr": float(corr.item()),
        "FDR": float(fdr.item()),
    }


def resolve_runtime_mask_flag(
    args: argparse.Namespace,
    ckpt_args: dict,
    ckpt_epoch: Optional[int],
    model_cfg: dict,
) -> tuple[bool, str]:
    if not model_cfg["enable_mask_cross"]:
        return False, "arch_disabled"

    if args.enable_mask_cross is not None:
        return bool(args.enable_mask_cross), "cli_override"

    stage2_start_epoch = ckpt_args.get("stage2_start_epoch")
    if isinstance(stage2_start_epoch, int) and ckpt_epoch is not None:
        mask_active = (ckpt_epoch - 1) >= stage2_start_epoch
        reason = (
            f"stage2_active@epoch{ckpt_epoch}"
            if mask_active
            else f"stage1_ckpt_epoch{ckpt_epoch}<stage2_{stage2_start_epoch + 1}"
        )
        return mask_active, reason

    return False, "unknown_stage_default_off"


def run_full_inference(
    model: torch.nn.Module,
    frames: np.ndarray,
    gt_rel_all: np.ndarray,
    y_scale: torch.Tensor,
    device: torch.device,
    seq_len: int,
    stride: int,
    image_size: int,
    enable_mask: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(frames) - seq_len
    if max_start < 0:
        raise ValueError(f"Scan only has {len(frames)} frames, but seq_len={seq_len}")

    starts = list(range(0, max_start + 1, stride))
    if starts[-1] != max_start:
        starts.append(max_start)

    pred_chunks = []
    gt_chunks = []
    prev_end_step = -1

    model.eval()
    with torch.no_grad():
        for start_idx in tqdm(starts, desc="Inference"):
            stop_idx = start_idx + seq_len
            imgs = TF.resize(
                torch.from_numpy(frames[start_idx:stop_idx]).float(),
                [image_size, image_size],
                antialias=True,
            ).unsqueeze(0)
            imgs = (imgs - imgs.mean()) / (imgs.std() + 1e-5)

            outputs, _, _ = model(
                imgs.to(device),
                enable_mask=enable_mask,
                enable_backward_consistency=False,
            )
            preds_real = (outputs.float() * y_scale).cpu().squeeze(0)

            step_start = start_idx
            step_end = start_idx + seq_len - 2
            overlap = max(0, prev_end_step - step_start + 1)

            gt_chunk = torch.from_numpy(gt_rel_all[step_start : step_end + 1]).float()
            if overlap > 0:
                preds_real = preds_real[overlap:]
                gt_chunk = gt_chunk[overlap:]

            pred_chunks.append(preds_real)
            gt_chunks.append(gt_chunk)
            prev_end_step = step_end

    y_pred_all = torch.cat(pred_chunks, dim=0).unsqueeze(0).to(device)
    y_gt_all = torch.cat(gt_chunks, dim=0).unsqueeze(0).to(device)
    return y_pred_all, y_gt_all


def evaluate_and_save(
    y_pred: torch.Tensor,
    y_gt: torch.Tensor,
    output_dir: str,
    scan_name: str,
) -> dict:
    src_metrics = compute_src_style_metrics(y_pred, y_gt)
    t_pred, t_gt = y_pred[..., :3], y_gt[..., :3]
    rte_steps = torch.norm(t_pred - t_gt, dim=-1).squeeze(0)

    r_pred = transform_v2m_convention(y_pred)[..., :3, :3]
    r_gt = transform_v2m_convention(y_gt)[..., :3, :3]
    trace = torch.matmul(r_gt.transpose(-1, -2), r_pred).diagonal(dim1=-2, dim2=-1).sum(-1)
    rre_steps = torch.acos(torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)) * 180.0 / np.pi

    t_pred_abs = transform_acum_convention(y_pred)
    t_gt_abs = transform_acum_convention(y_gt)
    pos_pred = t_pred_abs[0, :, :3, 3].cpu().numpy()
    pos_gt = t_gt_abs[0, :, :3, 3].cpu().numpy()

    fd = float(np.linalg.norm(pos_pred[-1] - pos_gt[-1]))
    path_len = float(np.sum(np.linalg.norm(np.diff(pos_gt, axis=0), axis=1)))
    fdr = (fd / (path_len + 1e-6)) * 100.0
    ate = float(np.sqrt(np.mean(np.sum((pos_pred - pos_gt) ** 2, axis=-1))))
    rte = float(rte_steps.mean().item())
    rre = float(rre_steps.mean().item())

    step_trans_cos = F.cosine_similarity(t_pred.squeeze(0), t_gt.squeeze(0), dim=-1).mean().item()
    step_rot_abs_mae = torch.mean(torch.abs(y_pred[..., 3:] - y_gt[..., 3:])).item()

    pos_pred_centered = pos_pred - pos_pred.mean(axis=0, keepdims=True)
    pos_gt_centered = pos_gt - pos_gt.mean(axis=0, keepdims=True)
    path_shape_corr = float(
        np.dot(pos_pred_centered.reshape(-1), pos_gt_centered.reshape(-1))
        / ((np.linalg.norm(pos_pred_centered.reshape(-1)) * np.linalg.norm(pos_gt_centered.reshape(-1))) + 1e-6)
    )

    def turning_angles(points: np.ndarray) -> np.ndarray:
        if len(points) < 3:
            return np.zeros((0,), dtype=np.float32)
        vec_prev = np.diff(points[:-1], axis=0)
        vec_next = np.diff(points[1:], axis=0)
        denom = (np.linalg.norm(vec_prev, axis=1) * np.linalg.norm(vec_next, axis=1)) + 1e-6
        cosine = np.sum(vec_prev * vec_next, axis=1) / denom
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))).astype(np.float32)

    turn_pred = turning_angles(pos_pred)
    turn_gt = turning_angles(pos_gt)
    if len(turn_gt) > 0:
        turn_angle_mae = float(np.mean(np.abs(turn_pred - turn_gt)))
    else:
        turn_angle_mae = 0.0

    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(scan_name)[0]
    fig_path = os.path.join(output_dir, f"{stem}_trajectory.png")
    npz_path = os.path.join(output_dir, f"{stem}_trajectory.npz")
    metrics_path = os.path.join(output_dir, f"{stem}_metrics.json")

    fig = plt.figure(figsize=(18, 8))
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot(pos_gt[:, 0], pos_gt[:, 1], pos_gt[:, 2], "b-", label="Ground Truth", linewidth=2)
    ax1.plot(pos_pred[:, 0], pos_pred[:, 1], pos_pred[:, 2], "r--", label="Prediction", linewidth=2)
    ax1.scatter(pos_gt[0, 0], pos_gt[0, 1], pos_gt[0, 2], c="g", s=90, label="Start")
    ax1.scatter(pos_gt[-1, 0], pos_gt[-1, 1], pos_gt[-1, 2], c="b", s=50)
    ax1.scatter(pos_pred[-1, 0], pos_pred[-1, 1], pos_pred[-1, 2], c="r", s=90, marker="X", label="End")
    ax1.set_title(f"3D Trajectory\nATE={ate:.3f} mm | FDR={fdr:.2f}%")
    ax1.legend()

    ax2 = fig.add_subplot(122)
    steps = np.arange(len(rte_steps.cpu()))
    ax2.plot(steps, rte_steps.cpu().numpy(), "g-", label="RTE (mm)", alpha=0.8)
    ax2_right = ax2.twinx()
    ax2_right.plot(steps, rre_steps.squeeze(0).cpu().numpy(), "r-", label="RRE (deg)", alpha=0.8)
    ax2.set_xlabel("Relative Step")
    ax2.set_ylabel("Translation Error (mm)", color="g")
    ax2_right.set_ylabel("Rotation Error (deg)", color="r")
    ax2.set_title("Step-wise Error")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)

    np.savez(
        npz_path,
        pred_rel=y_pred.squeeze(0).cpu().numpy(),
        gt_rel=y_gt.squeeze(0).cpu().numpy(),
        pred_abs=pos_pred,
        gt_abs=pos_gt,
    )

    metrics = {
        **src_metrics,
        "ATE_mm": ate,
        "FDR_percent": float(fdr),
        "FD_mm": fd,
        "PathLength_mm": path_len,
        "RTE_mm": rte,
        "RRE_deg": rre,
        "StepTranslationCosine": float(step_trans_cos),
        "StepRotationAbsMAE_deg": float(step_rot_abs_mae),
        "PathShapeCorr": float(path_shape_corr),
        "TurnAngleMAE_deg": float(turn_angle_mae),
        "figure_path": fig_path,
        "trajectory_path": npz_path,
        "metrics_path": metrics_path,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return metrics


def main() -> None:
    args = parse_args()
    missing = [name for name, value in {"data-root": args.data_root, "ckpt-path": args.ckpt_path, "scale-path": args.scale_path, "output-dir": args.output_dir}.items() if not value.strip()]
    if missing:
        raise ValueError("Missing required paths: " + ", ".join(missing) + ". See README.md")
    device = resolve_device(args.device)

    h5_path = resolve_requested_scan(
        data_root=args.data_root,
        scan_path=args.scan_path,
        h5_path=args.h5_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    output_dir = os.path.abspath(args.output_dir)
    ckpt_path = os.path.abspath(args.ckpt_path)
    payload = load_checkpoint_payload(ckpt_path, device=device)
    meta_payload = maybe_load_checkpoint_metadata(ckpt_path, device=device)
    ckpt_args = extract_checkpoint_args(meta_payload)
    ckpt_epoch = extract_checkpoint_epoch(meta_payload)
    model_cfg = resolve_model_config(args, ckpt_args)
    runtime_enable_mask, runtime_mask_reason = resolve_runtime_mask_flag(
        args=args,
        ckpt_args=ckpt_args,
        ckpt_epoch=ckpt_epoch,
        model_cfg=model_cfg,
    )
    stride = args.stride if args.stride > 0 else model_cfg["seq_len"] - 1

    print(f"🎯 测试文件: {h5_path}")
    print(f"📐 Euler convention | {DEFAULT_EULER_ORDER.upper()} + degrees")
    print(
        "🧩 Inference model | "
        f"seq_len={model_cfg['seq_len']} | "
        f"SACD={model_cfg['enable_local_sacd']} | "
        f"MaskCrossArch={model_cfg['enable_mask_cross']} | "
        f"MaskCrossRuntime={runtime_enable_mask} ({runtime_mask_reason}) | "
        f"ConvLSTM={model_cfg['enable_convlstm']} | "
        f"BiDir={model_cfg['enable_bidirectional_consistency']}"
    )
    if ckpt_epoch is not None:
        print(f"🗂️ Checkpoint epoch={ckpt_epoch}")

    y_scale = load_scale_tensor(os.path.abspath(args.scale_path), device=device)
    model = MyNet(
        seq_len=model_cfg["seq_len"],
        input_size=model_cfg["input_size"],
        mask_ratio=model_cfg["mask_ratio"],
        mask_depth=model_cfg["mask_depth"],
        mask_num_heads=model_cfg["mask_num_heads"],
        enable_local_sacd=model_cfg["enable_local_sacd"],
        enable_mask_cross=model_cfg["enable_mask_cross"],
        enable_convlstm=model_cfg["enable_convlstm"],
        enable_bidirectional_consistency=model_cfg["enable_bidirectional_consistency"],
    ).to(device)
    load_model_weights(model, payload)

    frames, tforms, t_calib = load_scan_arrays(h5_path, data_root=args.data_root)
    gt_rel_all = relative_poses_from_tforms(
        tforms,
        t_calib=t_calib,
        degrees=True,
        euler_order=DEFAULT_EULER_ORDER,
    )

    y_pred, y_gt = run_full_inference(
        model=model,
        frames=frames,
        gt_rel_all=gt_rel_all,
        y_scale=y_scale,
        device=device,
        seq_len=model_cfg["seq_len"],
        stride=stride,
        image_size=model_cfg["input_size"],
        enable_mask=runtime_enable_mask,
    )
    metrics = evaluate_and_save(
        y_pred=y_pred,
        y_gt=y_gt,
        output_dir=output_dir,
        scan_name=os.path.basename(h5_path),
    )

    print(
        f"📊 SRC风格 | "
        f"rAE={metrics['rAE']:.4f} mm | "
        f"aAE={metrics['aAE']:.4f} deg | "
        f"rFE={metrics['rFE']:.4f} mm | "
        f"aFE={metrics['aFE']:.4f} deg | "
        f"Corr={metrics['Corr']:.4f} | "
        f"FDR={metrics['FDR']:.2f}%"
    )
    print(
        f"📊 FullSeq | "
        f"ATE={metrics['ATE_mm']:.4f} mm | "
        f"FD={metrics['FD_mm']:.4f} mm | "
        f"FDR={metrics['FDR_percent']:.2f}% | "
        f"RTE={metrics['RTE_mm']:.4f} mm | "
        f"RRE={metrics['RRE_deg']:.4f} deg | "
        f"ShapeCorr={metrics['PathShapeCorr']:.4f} | "
        f"TurnMAE={metrics['TurnAngleMAE_deg']:.4f} deg"
    )
    print(f"🖼️ 轨迹图: {metrics['figure_path']}")
    print(f"💾 轨迹数据: {metrics['trajectory_path']}")
    print(f"📝 指标文件: {metrics['metrics_path']}")


if __name__ == "__main__":
    main()
