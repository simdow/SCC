import csv
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
import torchvision.transforms.functional as TF
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset


DEFAULT_DATA_ROOT = ""
DEFAULT_RUN_ROOT = ""
DEFAULT_EULER_ORDER = "yzx"


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def list_subjects(data_root: str) -> List[str]:
    subjects = [
        name
        for name in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, name)) and name.isdigit()
    ]
    return sorted(subjects)


def split_subjects(
    subjects: Sequence[str],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> Dict[str, List[str]]:
    subjects = list(subjects)
    total = len(subjects)
    if total == 0:
        raise ValueError(f"No numeric subject folders were found.")

    if total == 1:
        return {"train": subjects[:], "val": subjects[:], "test": subjects[:]}

    train_count = int(total * train_ratio)
    train_count = min(max(train_count, 1), total - 1)

    remaining = total - train_count
    if remaining == 1:
        val_count = 1
    else:
        val_count = int(total * val_ratio)
        val_count = min(max(val_count, 1), remaining - 1)

    train_subjects = subjects[:train_count]
    val_subjects = subjects[train_count : train_count + val_count]
    test_subjects = subjects[train_count + val_count :]

    if not test_subjects:
        test_subjects = val_subjects[-1:] if val_subjects else train_subjects[-1:]

    return {
        "train": train_subjects,
        "val": val_subjects if val_subjects else train_subjects[-1:],
        "test": test_subjects,
    }


def format_split(split_map: Dict[str, Sequence[str]]) -> str:
    parts = []
    for key in ("train", "val", "test"):
        values = list(split_map.get(key, []))
        preview = ", ".join(values[:4])
        if len(values) > 4:
            preview += ", ..."
        parts.append(f"{key}={len(values)} [{preview}]")
    return " | ".join(parts)


def _parse_multiline_calibration(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(csv_path, "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle.readlines() if line.strip()]

    scale_idx = lines.index("scaling_from_pixel_to_mm")
    spatial_idx = lines.index(
        "spatial_calibration_from_image_coordinate_system_to_tracking_tool_coordinate_system"
    )

    m_scale = np.array(
        [list(map(float, line.split(","))) for line in lines[scale_idx + 1 : scale_idx + 5]],
        dtype=np.float64,
    )
    t_calib = np.array(
        [list(map(float, line.split(","))) for line in lines[spatial_idx + 1 : spatial_idx + 5]],
        dtype=np.float64,
    )
    return m_scale, t_calib


def _parse_singleline_calibration(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        row = next(reader)

    scale_value = float(row["scaling_from_pixel_to_mm"])
    m_scale = np.eye(4, dtype=np.float64)
    m_scale[0, 0] = scale_value
    m_scale[1, 1] = scale_value

    calib_key = None
    for key in row:
        if "spatial_calibration" in key or "T_calib" in key:
            calib_key = key
            break
    if calib_key is None:
        raise KeyError(f"Could not find spatial calibration column in {csv_path}")

    t_calib = np.array(json.loads(row[calib_key]), dtype=np.float64)
    return m_scale, t_calib


def load_global_calibration(data_root: str) -> Tuple[np.ndarray, np.ndarray, str]:
    candidates = [
        os.path.join(data_root, "calib_matrix.csv"),
        os.path.join(os.path.dirname(data_root.rstrip("/")), "calib_matrix.csv"),
    ]

    for csv_path in candidates:
        if not os.path.exists(csv_path):
            continue

        with open(csv_path, "r", encoding="utf-8") as handle:
            first_line = handle.readline().strip()

        if first_line == "scaling_from_pixel_to_mm":
            m_scale, t_calib = _parse_multiline_calibration(csv_path)
        else:
            m_scale, t_calib = _parse_singleline_calibration(csv_path)
        return m_scale, t_calib, csv_path

    raise FileNotFoundError(
        f"Could not find calib_matrix.csv under {data_root} or its parent directory."
    )


def infer_data_root_from_scan(h5_path: str) -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(h5_path)))


def resolve_scan_path(scan_query: str, data_root: str) -> str:
    scan_query = (scan_query or "").strip()
    if not scan_query:
        raise ValueError("scan_query must be a non-empty .h5 path or file name.")

    if os.path.isabs(scan_query):
        if os.path.isfile(scan_query):
            return os.path.abspath(scan_query)
        raise FileNotFoundError(f"Scan file does not exist: {scan_query}")

    direct_path = os.path.join(data_root, scan_query)
    if os.path.isfile(direct_path):
        return os.path.abspath(direct_path)

    basename = os.path.basename(scan_query)
    candidates: List[str] = []
    for root, _, file_names in os.walk(data_root):
        for file_name in file_names:
            if not file_name.endswith(".h5"):
                continue
            abs_path = os.path.join(root, file_name)
            rel_path = os.path.relpath(abs_path, data_root)
            if file_name == basename or rel_path == scan_query:
                candidates.append(os.path.abspath(abs_path))

    unique_candidates = sorted(set(candidates))
    if len(unique_candidates) == 1:
        return unique_candidates[0]
    if len(unique_candidates) > 1:
        preview = ", ".join(unique_candidates[:5])
        raise ValueError(
            f"Ambiguous scan query '{scan_query}'. "
            f"Matched {len(unique_candidates)} files: {preview}"
        )

    raise FileNotFoundError(
        f"Could not resolve scan '{scan_query}' under data_root={data_root}."
    )


def load_scan_arrays(
    h5_path: str,
    data_root: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as handle:
        frames = handle["frames"][:]
        tforms = handle["tforms"][:]
        if "T_calib" in handle:
            t_calib = np.asarray(handle["T_calib"][()], dtype=np.float64)
        else:
            data_root = data_root or infer_data_root_from_scan(h5_path)
            _, t_calib, _ = load_global_calibration(data_root)

    return frames, tforms, t_calib


def relative_poses_from_tforms(
    tforms: np.ndarray,
    t_calib: Optional[np.ndarray] = None,
    degrees: bool = True,
    euler_order: str = DEFAULT_EULER_ORDER,
) -> np.ndarray:
    tforms = np.asarray(tforms, dtype=np.float64)
    if t_calib is not None:
        tforms = tforms @ np.asarray(t_calib, dtype=np.float64)

    t_rel = np.linalg.inv(tforms[:-1]) @ tforms[1:]
    translations = t_rel[:, :3, 3]
    rotations = R.from_matrix(t_rel[:, :3, :3]).as_euler(euler_order, degrees=degrees)
    return np.concatenate([translations, rotations], axis=1).astype(np.float32)


def save_scale_file(
    save_path: str,
    y_scale: np.ndarray,
    percentile: float,
    subject_ids: Sequence[str],
) -> str:
    ensure_dir(os.path.dirname(os.path.abspath(save_path)))
    y_scale = np.asarray(y_scale, dtype=np.float32)
    payload = {
        "y_scale": torch.tensor(y_scale, dtype=torch.float32),
        "My": y_scale.tolist(),
        "percentile": float(percentile),
        "subjects": list(subject_ids),
    }
    torch.save(payload, save_path)
    return save_path


def generate_scale_file(
    data_root: str,
    subject_ids: Sequence[str],
    save_path: str,
    percentile: float = 98.0,
) -> Tuple[str, np.ndarray]:
    all_rel_poses = []
    for subject in subject_ids:
        subject_dir = os.path.join(data_root, subject)
        if not os.path.isdir(subject_dir):
            continue

        for file_name in sorted(os.listdir(subject_dir)):
            if not file_name.endswith(".h5"):
                continue
            h5_path = os.path.join(subject_dir, file_name)
            try:
                _, tforms, t_calib = load_scan_arrays(h5_path, data_root=data_root)
                if len(tforms) < 2:
                    continue
                rel_pose = relative_poses_from_tforms(
                    tforms,
                    t_calib=t_calib,
                    degrees=True,
                    euler_order=DEFAULT_EULER_ORDER,
                )
                all_rel_poses.append(rel_pose)
            except Exception:
                continue

    if not all_rel_poses:
        raise RuntimeError("Failed to compute any valid relative poses for y_scale.")

    merged = np.concatenate(all_rel_poses, axis=0)
    y_scale = np.percentile(np.abs(merged), percentile, axis=0)
    y_scale = np.clip(y_scale, a_min=1e-5, a_max=None).astype(np.float32)
    save_scale_file(save_path, y_scale, percentile, subject_ids)
    return save_path, y_scale


def load_scale_tensor(scale_path: str, device: torch.device) -> torch.Tensor:
    payload = torch.load(scale_path, map_location=device, weights_only=False)
    if isinstance(payload, dict):
        for key in ("y_scale", "My", "M_emp_dy"):
            if key not in payload:
                continue
            value = payload[key]
            tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value)
            return tensor.to(device=device, dtype=torch.float32).view(1, 1, 6)

    if isinstance(payload, torch.Tensor):
        return payload.to(device=device, dtype=torch.float32).view(1, 1, 6)

    raise KeyError(f"Could not find a valid scale key in {scale_path}")


class CalibratedPoseDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        subjects: Sequence[str],
        seq_len: int = 4,
        image_size: int = 256,
        alpha: int = 1,
        sampling: str = "random",
        windows_per_scan: int = 2,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.subjects = list(subjects)
        self.seq_len = seq_len
        self.image_size = image_size
        self.alpha = max(int(alpha), 1)
        self.sampling = sampling
        self.windows_per_scan = max(int(windows_per_scan), 1)

        self.scan_infos: List[Tuple[str, int]] = []
        self.samples: List[Tuple[str, int]] = []

        for subject in self.subjects:
            subject_dir = os.path.join(self.data_root, subject)
            if not os.path.isdir(subject_dir):
                continue

            for file_name in sorted(os.listdir(subject_dir)):
                if not file_name.endswith(".h5"):
                    continue
                h5_path = os.path.join(subject_dir, file_name)
                try:
                    with h5py.File(h5_path, "r") as handle:
                        num_frames = int(handle["frames"].shape[0])
                except Exception:
                    continue

                if num_frames < self.seq_len:
                    continue

                self.scan_infos.append((h5_path, num_frames))
                if self.sampling == "fixed":
                    max_start = num_frames - self.seq_len
                    num_windows = min(self.windows_per_scan, max_start + 1)
                    starts = np.linspace(0, max_start, num=num_windows, dtype=int)
                    for start_idx in np.unique(starts):
                        self.samples.append((h5_path, int(start_idx)))

        if not self.scan_infos:
            raise ValueError(
                f"No valid H5 scans were found for subjects {self.subjects} under {self.data_root}."
            )

        if self.sampling not in {"random", "fixed"}:
            raise ValueError("sampling must be either 'random' or 'fixed'")

    def __len__(self) -> int:
        if self.sampling == "fixed":
            return len(self.samples)
        return len(self.scan_infos) * self.alpha

    def _load_item(self, h5_path: str, start_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        with h5py.File(h5_path, "r") as handle:
            frames = handle["frames"][start_idx : start_idx + self.seq_len]
            tforms = handle["tforms"][start_idx : start_idx + self.seq_len]
            if "T_calib" in handle:
                t_calib = np.asarray(handle["T_calib"][()], dtype=np.float64)
            else:
                _, t_calib, _ = load_global_calibration(self.data_root)

        imgs = TF.resize(
            torch.from_numpy(frames).float(),
            [self.image_size, self.image_size],
            antialias=True,
        )
        imgs = (imgs - imgs.mean()) / (imgs.std() + 1e-5)
        rel_pose = relative_poses_from_tforms(
            tforms,
            t_calib=t_calib,
            degrees=True,
            euler_order=DEFAULT_EULER_ORDER,
        )
        return imgs, torch.from_numpy(rel_pose)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            if self.sampling == "fixed":
                h5_path, start_idx = self.samples[index]
            else:
                h5_path, num_frames = self.scan_infos[index % len(self.scan_infos)]
                start_idx = np.random.randint(0, num_frames - self.seq_len + 1)
            return self._load_item(h5_path, start_idx)
        except Exception:
            retry_index = np.random.randint(0, len(self))
            return self.__getitem__(retry_index)
