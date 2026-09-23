import random
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rotation_x(angle: torch.Tensor) -> torch.Tensor:
    cos_v = torch.cos(angle)
    sin_v = torch.sin(angle)
    ones = torch.ones_like(angle)
    zeros = torch.zeros_like(angle)
    return torch.stack(
        [
            ones,
            zeros,
            zeros,
            zeros,
            cos_v,
            -sin_v,
            zeros,
            sin_v,
            cos_v,
        ],
        dim=-1,
    ).reshape(angle.shape + (3, 3))


def _rotation_y(angle: torch.Tensor) -> torch.Tensor:
    cos_v = torch.cos(angle)
    sin_v = torch.sin(angle)
    ones = torch.ones_like(angle)
    zeros = torch.zeros_like(angle)
    return torch.stack(
        [
            cos_v,
            zeros,
            sin_v,
            zeros,
            ones,
            zeros,
            -sin_v,
            zeros,
            cos_v,
        ],
        dim=-1,
    ).reshape(angle.shape + (3, 3))


def _rotation_z(angle: torch.Tensor) -> torch.Tensor:
    cos_v = torch.cos(angle)
    sin_v = torch.sin(angle)
    ones = torch.ones_like(angle)
    zeros = torch.zeros_like(angle)
    return torch.stack(
        [
            cos_v,
            -sin_v,
            zeros,
            sin_v,
            cos_v,
            zeros,
            zeros,
            zeros,
            ones,
        ],
        dim=-1,
    ).reshape(angle.shape + (3, 3))


def poses_to_transform_yzx(pose: torch.Tensor) -> torch.Tensor:
    angles = torch.deg2rad(pose[..., 3:])
    rot_y = _rotation_y(angles[..., 0])
    rot_z = _rotation_z(angles[..., 1])
    rot_x = _rotation_x(angles[..., 2])
    rot = rot_y @ rot_z @ rot_x

    eye = torch.eye(4, device=pose.device, dtype=pose.dtype)
    transform = eye.view(*([1] * (pose.dim() - 1)), 4, 4).repeat(*pose.shape[:-1], 1, 1)
    transform[..., :3, :3] = rot
    transform[..., :3, 3] = pose[..., :3]
    return transform

class MMAE(nn.Module):
    def __init__(self, alpha=1, beta=1.0, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, p, y, w=None, per=False, mse=False):
        res = torch.abs(p - y)
        if mse: res = res ** 2
        if per: res = res / (torch.abs(y) + 1e-5)
        if w is not None:
            if w.shape[1] != y.shape[1]: 
                w = torch.concat([torch.zeros([y.shape[0], 1, 6], device=y.device), w], axis=1)
            res = res * ((torch.abs(w) + self.smooth) ** self.beta)
        return res.mean() * self.alpha

class Corr_loss(nn.Module):
    def __init__(self, alpha=1):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, p, y):
        corr = F.cosine_similarity(p.flatten(), y.flatten(), dim=0)
        return (1 - corr) * self.alpha


class Masked_Corr_Loss(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, aux):
        if aux is None:
            return None

        mask = aux.get("mask")
        pred_tokens = aux.get("pred_tokens")
        target_tokens = aux.get("target_tokens")

        if mask is None or pred_tokens is None or target_tokens is None:
            return pred_tokens.new_zeros(()) if isinstance(pred_tokens, torch.Tensor) else None

        if mask.sum() == 0:
            return pred_tokens.new_zeros(())

        pred_masked = F.normalize(pred_tokens[mask], p=2, dim=-1)
        target_masked = F.normalize(target_tokens[mask].detach(), p=2, dim=-1)
        cosine = torch.sum(pred_masked * target_masked, dim=-1)
        return (1.0 - cosine.mean()) * self.alpha


class BidirectionalConsistencyLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, aux, y_scale=None):
        if aux is None or y_scale is None:
            return None

        forward_pose = aux.get("forward_pose_raw")
        backward_pose = aux.get("backward_pose_raw")
        if forward_pose is None or backward_pose is None:
            return None

        scale = y_scale.to(device=forward_pose.device, dtype=forward_pose.dtype)
        forward_pose = (forward_pose * scale).float()
        backward_pose = (backward_pose * scale).float()

        transform_forward = poses_to_transform_yzx(forward_pose)
        transform_backward = poses_to_transform_yzx(backward_pose)
        transform_backward = torch.linalg.inv(transform_backward.reshape(-1, 4, 4)).reshape_as(transform_backward)
        transform_backward = torch.flip(transform_backward, dims=[1])

        trans_loss = torch.mean(torch.abs(transform_forward[..., :3, 3] - transform_backward[..., :3, 3]))
        rot_loss = torch.mean(torch.abs(transform_forward[..., :3, :3] - transform_backward[..., :3, :3]))
        return (trans_loss + rot_loss) * self.alpha

class Triplet_Loss(nn.Module):
    def __init__(self, alpha=1, margin=0.10, n_sample=256, dist='cos'):
        super().__init__()
        self.alpha = alpha
        self.margin = margin
        self.n_sample = n_sample
        if dist == 'cos':
            self.dist = self._cossim
        elif dist == 'norm':
            self.dist = self._norm
    
    def _norm(self, a, b):
        return torch.linalg.norm(a - b, axis=-1)
    
    def _cossim(self, a, b):
        return 1 - F.cosine_similarity(a, b, axis=-1)
    
    def forward(self, v, y):
        B, S = y.shape[0], y.shape[1]
        v = v.reshape(B, S, -1)
        v = v[:, -1, :]
        y = y[:, -1, :]

        combs = list(combinations([i for i in range(v.shape[0])], 3))
        if len(combs) == 0:
            return torch.tensor(0.0, dtype=v.dtype, device=v.device)

        if self.n_sample is not None and len(combs) > self.n_sample:
            combs = random.sample(combs, self.n_sample)

        Y = torch.stack([y[list(combs[i])] for i in range(len(combs))])
        V = torch.stack([v[list(combs[i])] for i in range(len(combs))])

        D1 = self.dist(Y[:, 0], Y[:, 1])
        D2 = self.dist(Y[:, 0], Y[:, 2])
        PN = torch.sign(D2 - D1)
        res = PN * (self._norm(V[:, 0], V[:, 1]) - self._norm(V[:, 0], V[:, 2])) + self.margin
        res = torch.mean(F.relu(res))

        return res * self.alpha

class MoGLoLoss_Wrapper(nn.Module):
    def __init__(self, rot_weight=5.0, mask_weight=0.0, triplet_samples=2):
        super().__init__()
        self.mmae = MMAE(alpha=1.0)
        self.corr = Corr_loss(alpha=1.0)
        self.triplet = Triplet_Loss(alpha=1.0, margin=0.10, n_sample=triplet_samples, dist='cos')
        self.mask_corr = Masked_Corr_Loss(alpha=1.0)
        self.consistency = BidirectionalConsistencyLoss(alpha=1.0)
        self.default_mask_weight = float(mask_weight)
        self.register_buffer('w_mask', torch.tensor([1.0, 1.0, 1.0, rot_weight, rot_weight, rot_weight]))

    def forward(self, preds, y_gt, features, aux=None, loss_weights=None, y_scale=None):
        w = self.w_mask.view(1, 1, 6).to(preds.device)

        loss_mmae = self.mmae(preds * w, y_gt * w)
        loss_corr = self.corr(preds, y_gt)
        loss_tri = self.triplet(features, y_gt)
        loss_mask = preds.new_zeros(())
        loss_consist = self.consistency(aux, y_scale=y_scale)
        if loss_mask is None:
            loss_mask = preds.new_zeros(())
        if loss_consist is None:
            loss_consist = preds.new_zeros(())

        if loss_weights is None:
            loss_weights = {}

        mmae_w = float(loss_weights.get("mmae", 1.0))
        corr_w = float(loss_weights.get("corr", 1.0))
        tri_w = float(loss_weights.get("tri", 0.005))
        mask_w = 0.0
        consist_w = float(loss_weights.get("consist", 0.1))

        total_loss = (
            mmae_w * loss_mmae
            + corr_w * loss_corr
            + tri_w * loss_tri
            + mask_w * loss_mask
            + consist_w * loss_consist
        )
        return total_loss, loss_mmae, loss_corr, loss_tri, loss_mask, loss_consist
