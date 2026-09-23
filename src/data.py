from pathlib import Path
import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from torchvision import transforms as T


def relative_labels(absolute):
    absolute = np.asarray(absolute, dtype=np.float32)
    relative = np.linalg.inv(absolute[:-1]) @ absolute[1:]
    return np.concatenate([relative[:, :3, 3], Rotation.from_matrix(relative[:, :3, :3]).as_euler('yzx', degrees=True)], 1).astype(np.float32)


def preprocess(frames):
    images = torch.from_numpy(np.asarray(frames, dtype=np.float32) / 255.)
    if images.ndim == 3:
        images = images.unsqueeze(1)
    assert images.shape[1] == 1
    return T.Resize((256, 256), T.InterpolationMode.BICUBIC, antialias=True)(images)


def reverse_labels(labels):
    labels = labels.detach().cpu().numpy()
    matrices = np.broadcast_to(np.eye(4), (*labels.shape[:2], 4, 4)).copy()
    matrices[..., :3, :3] = Rotation.from_euler('yzx', labels[..., 3:].reshape(-1, 3), degrees=True).as_matrix().reshape(*labels.shape[:2], 3, 3)
    matrices[..., :3, 3] = labels[..., :3]
    inverse = np.linalg.inv(matrices[:, ::-1])
    vectors = np.concatenate([inverse[..., :3, 3], Rotation.from_matrix(inverse[..., :3, :3].reshape(-1, 3, 3)).as_euler('yzx', degrees=True).reshape(*labels.shape[:2], 3)], -1)
    return torch.from_numpy(vectors.astype(np.float32))


def load_scale(path):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(payload, dict):
        for key in ('scale', 'M_emp_dy', 'My', 'y_scale'):
            if key in payload:
                payload = payload[key]
                break
    value = torch.as_tensor(payload, dtype=torch.float32).reshape(1, 1, 6)
    if not torch.isfinite(value).all() or not (value > 0).all():
        raise ValueError('Scale must contain six finite positive numbers')
    return value


class SequenceDataset(Dataset):
    def __init__(self, root, subjects, seed=42, epoch=0, windows=10, fixed=False):
        self.scans = [p for subject in subjects for p in sorted((Path(root) / f'{subject:03d}').glob('*.h5'))]
        if not self.scans:
            raise ValueError('No scans found in the requested split')
        self.lengths = []
        for path in self.scans:
            with h5py.File(path) as handle:
                self.lengths.append(len(handle['frames']))
        if min(self.lengths) < 5:
            raise ValueError('Each scan must contain at least five frames')
        if fixed:
            self.entries = [(i, int(start)) for i, n in enumerate(self.lengths) for start in np.unique(np.linspace(0, n-5, windows, dtype=int))]
        else:
            rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 20260915]))
            indices = np.repeat(np.arange(len(self.scans)), windows)
            rng.shuffle(indices)
            self.entries = [(int(i), int(rng.integers(0, self.lengths[i]-4))) for i in indices]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        scan, begin = self.entries[index]
        with h5py.File(self.scans[scan]) as handle:
            x = preprocess(handle['frames'][begin:begin+5])
            y = torch.from_numpy(relative_labels(handle['tforms'][begin:begin+5]))
        return x, y
