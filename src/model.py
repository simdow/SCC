import contextlib
import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def parallel_run(layer: nn.Module, tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    batch = tensors[0].shape[0]
    stacked = torch.cat(tensors, dim=0)
    outputs = layer(stacked)
    num_chunks = outputs.shape[0] // batch
    return [outputs[i * batch : (i + 1) * batch] for i in range(num_chunks)]


class StructureAwareChannelDecoupling(nn.Module):
    """
    SACD block with optional speckle-decorrelation guidance.

    The original implementation was an SE-style channel gate:
    global avg/max descriptors -> shared MLP -> sigmoid gating.

    When the two shallow frame features are provided, SACD estimates a
    high-frequency decorrelation cue from the pair. The cue is used as a
    reliability mask for the channel descriptors and as a weak channel prior,
    so the local branch is biased toward channels activated on stable speckle
    / structure rather than channels dominated by decorrelated texture.

    This keeps the same learnable parameters as the original block; old
    checkpoints remain load-compatible. Set SACD_DECORRELATION_GUIDE=0 to
    recover the pure SE-style behavior.
    """

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
        self.use_decorrelation_guidance = os.environ.get("SACD_DECORRELATION_GUIDE", "1").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        self.decorrelation_strength = float(os.environ.get("SACD_DECORRELATION_STRENGTH", "0.5"))
        self.corr_ambiguity_weight = float(os.environ.get("SACD_CORR_AMBIGUITY_WEIGHT", "0.25"))
        self.corr_temperature = float(os.environ.get("SACD_CORR_TEMPERATURE", "0.10"))
        self.eps = 1e-6
        self.last_decorrelation_map: Optional[torch.Tensor] = None
        self.last_stability_map: Optional[torch.Tensor] = None
        self.last_channel_stability: Optional[torch.Tensor] = None

    def _normalize_unit(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        x_min = x_float.amin(dim=(-2, -1), keepdim=True)
        x_max = x_float.amax(dim=(-2, -1), keepdim=True)
        x_norm = (x_float - x_min) / (x_max - x_min + self.eps)
        return x_norm.to(dtype=x.dtype)

    def _high_pass(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        low = F.avg_pool2d(x_float, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        return x_float - low

    def _standardize_spatial(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        mean = x_float.mean(dim=(-2, -1), keepdim=True)
        var = (x_float - mean).square().mean(dim=(-2, -1), keepdim=True)
        return (x_float - mean) / torch.sqrt(var + self.eps)

    def _speckle_decorrelation_map(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        target_size: tuple[int, int],
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        h1 = self._high_pass(x1)
        h2 = self._high_pass(x2)

        h1_vec = F.normalize(h1, p=2, dim=1, eps=self.eps)
        h2_vec = F.normalize(h2, p=2, dim=1, eps=self.eps)
        local_corr = torch.sum(h1_vec * h2_vec, dim=1, keepdim=True).clamp(-1.0, 1.0)
        phase_decor = 0.5 * (1.0 - local_corr)

        z1 = self._standardize_spatial(h1)
        z2 = self._standardize_spatial(h2)
        hf_diff = self._normalize_unit(torch.mean(torch.abs(z2 - z1), dim=1, keepdim=True))
        hf_energy = torch.sqrt(
            0.5 * (h1.square().mean(dim=1, keepdim=True) + h2.square().mean(dim=1, keepdim=True)) + self.eps
        )
        hf_energy = self._normalize_unit(hf_energy)

        decor = (0.7 * phase_decor + 0.3 * hf_diff) * hf_energy
        decor = self._normalize_unit(decor)
        if decor.shape[-2:] != target_size:
            decor = F.interpolate(decor, size=target_size, mode="bilinear", align_corners=False)
        return decor.to(dtype=target_dtype)

    def _correlation_ambiguity_map(
        self,
        corr_volume: torch.Tensor,
        target_size: tuple[int, int],
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        corr = corr_volume.float()
        num_offsets = max(int(corr.shape[1]), 2)
        prob = F.softmax(corr / max(self.corr_temperature, self.eps), dim=1)
        peak = prob.amax(dim=1, keepdim=True)
        entropy = -(prob * torch.log(prob + self.eps)).sum(dim=1, keepdim=True) / math.log(num_offsets)
        certainty = 1.0 - entropy
        reliability = torch.sqrt(torch.clamp(peak * certainty, min=0.0))
        ambiguity = 1.0 - self._normalize_unit(reliability)
        if ambiguity.shape[-2:] != target_size:
            ambiguity = F.interpolate(ambiguity, size=target_size, mode="bilinear", align_corners=False)
        return ambiguity.to(dtype=target_dtype)

    def _decorrelation_map(
        self,
        feat: torch.Tensor,
        x1: Optional[torch.Tensor],
        x2: Optional[torch.Tensor],
        corr_volume: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if not self.use_decorrelation_guidance:
            return None

        target_size = feat.shape[-2:]
        decor: Optional[torch.Tensor] = None
        if x1 is not None and x2 is not None:
            decor = self._speckle_decorrelation_map(x1, x2, target_size, feat.dtype)

        if corr_volume is not None:
            ambiguity = self._correlation_ambiguity_map(corr_volume, target_size, feat.dtype)
            if decor is None:
                decor = ambiguity
            else:
                corr_weight = max(0.0, min(self.corr_ambiguity_weight, 1.0))
                decor = (1.0 - corr_weight) * decor + corr_weight * ambiguity

        if decor is None:
            return None
        return decor.clamp(0.0, 1.0).detach()

    def _weighted_avg_pool(self, feat: torch.Tensor, stability: torch.Tensor) -> torch.Tensor:
        denom = stability.sum(dim=(-2, -1), keepdim=True).clamp_min(self.eps)
        return (feat * stability).sum(dim=(-2, -1), keepdim=True) / denom

    def _weighted_max_pool(self, feat: torch.Tensor, stability: torch.Tensor) -> torch.Tensor:
        return torch.amax(feat * stability, dim=(-2, -1), keepdim=True)

    def _channel_stability(self, feat: torch.Tensor, decor: torch.Tensor) -> torch.Tensor:
        energy = feat.abs()
        decor_energy = (energy * decor).sum(dim=(-2, -1), keepdim=True)
        total_energy = energy.sum(dim=(-2, -1), keepdim=True).clamp_min(self.eps)
        channel_decor = (decor_energy / total_energy).clamp(0.0, 1.0)
        return 1.0 - channel_decor

    def forward(
        self,
        feat: torch.Tensor,
        x1: Optional[torch.Tensor] = None,
        x2: Optional[torch.Tensor] = None,
        corr_volume: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        decor = self._decorrelation_map(feat, x1=x1, x2=x2, corr_volume=corr_volume)
        if decor is None:
            avg_feat = self.avg_pool(feat)
            max_feat = self.max_pool(feat)
            channel_stability = None
            stability = None
        else:
            stability = 1.0 - decor
            avg_feat = self._weighted_avg_pool(feat, stability)
            max_feat = self._weighted_max_pool(feat, stability)
            channel_stability = self._channel_stability(feat, decor)

        avg_desc = self.shared_mlp(avg_feat)
        max_desc = self.shared_mlp(max_feat)
        weight = self.sigmoid(avg_desc + max_desc)
        if channel_stability is not None and self.decorrelation_strength > 0:
            centered = channel_stability - channel_stability.mean(dim=1, keepdim=True)
            prior = 1.0 + self.decorrelation_strength * centered
            weight = (weight * prior.clamp(0.25, 1.75)).clamp(0.02, 1.5)

        self.last_decorrelation_map = decor
        self.last_stability_map = stability.detach() if stability is not None else None
        self.last_channel_stability = channel_stability.detach() if channel_stability is not None else None
        return feat * weight, weight


class FeedForwardBlock(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = FeedForwardBlock(dim=dim, mlp_ratio=mlp_ratio)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_in = self.norm_attn(x)
        attn_out, _ = self.attn(
            attn_in,
            attn_in,
            attn_in,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn_out
        x = x + self.ffn(self.norm_ffn(x))
        return x


class CrossAttentionDecoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm_self = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_cross_q = nn.LayerNorm(dim)
        self.norm_cross_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = FeedForwardBlock(dim=dim, mlp_ratio=mlp_ratio)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        self_in = self.norm_self(x)
        self_out, _ = self.self_attn(self_in, self_in, self_in, need_weights=False)
        x = x + self_out

        cross_q = self.norm_cross_q(x)
        cross_kv = self.norm_cross_kv(memory)
        cross_out, _ = self.cross_attn(cross_q, cross_kv, cross_kv, need_weights=False)
        x = x + cross_out
        x = x + self.ffn(self.norm_ffn(x))
        return x


class MaskedLatentCrossBlock(nn.Module):
    """
    CroCo-inspired low-ratio masked cross-view completion.
    The MoGLo backbone stays unchanged; we only refine the local branch with:
    1) visible-token encoding on the current frame,
    2) full-token encoding on the reference frame,
    3) decoder self/cross attention to reconstruct masked current tokens.
    """

    def __init__(
        self,
        dim: int = 64,
        num_heads: int = 8,
        mask_ratio: float = 0.1,
        depth: int = 2,
        pooled_size: int = 16,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio
        self.pooled_size = pooled_size
        self.pool = nn.AdaptiveAvgPool2d((pooled_size, pooled_size))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, pooled_size * pooled_size, dim))
        self.gamma = nn.Parameter(torch.ones(1) * 0.5)
        self.input_norm = nn.LayerNorm(dim)
        self.encoder_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
                for _ in range(max(1, depth // 2))
            ]
        )
        self.decoder_blocks = nn.ModuleList(
            [
                CrossAttentionDecoderBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
                for _ in range(max(1, depth))
            ]
        )
        self.out_norm = nn.LayerNorm(dim)
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

    def _sample_mask(self, batch: int, num_tokens: int, device: torch.device) -> torch.Tensor:
        if num_tokens <= 0 or self.mask_ratio <= 0:
            return torch.zeros((batch, num_tokens), dtype=torch.bool, device=device)
        num_mask = max(1, int(round(num_tokens * self.mask_ratio)))
        noise = torch.rand(batch, num_tokens, device=device)
        rank = torch.argsort(noise, dim=1)
        return rank < num_mask

    def _pack_visible_tokens(
        self,
        tokens: torch.Tensor,
        keep_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        batch, _, dim = tokens.shape
        counts = keep_mask.sum(dim=1)
        max_visible = max(int(counts.max().item()), 1)

        packed = tokens.new_zeros(batch, max_visible, dim)
        padding_mask = torch.ones(batch, max_visible, dtype=torch.bool, device=tokens.device)
        visible_indices: list[torch.Tensor] = []

        for batch_idx in range(batch):
            idx = torch.nonzero(keep_mask[batch_idx], as_tuple=False).squeeze(1)
            visible_indices.append(idx)
            if idx.numel() == 0:
                continue
            packed[batch_idx, : idx.numel()] = tokens[batch_idx, idx]
            padding_mask[batch_idx, : idx.numel()] = False

        return packed, padding_mask, visible_indices

    def _encode_tokens(
        self,
        tokens: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = self.input_norm(tokens)
        for block in self.encoder_blocks:
            out = block(out, key_padding_mask=key_padding_mask)
        return out

    def forward(self, current_feat: torch.Tensor, reference_feat: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        current_small = self.pool(current_feat)
        reference_small = self.pool(reference_feat)

        batch, channels, height, width = current_small.shape
        num_tokens = height * width
        pos_embed = self.pos_embed[:, :num_tokens].to(device=current_feat.device, dtype=current_feat.dtype)

        current_tokens = current_small.flatten(2).transpose(1, 2)
        reference_tokens = reference_small.flatten(2).transpose(1, 2)
        target_tokens = current_tokens.detach()

        if self.training:
            mask = self._sample_mask(batch, num_tokens, current_feat.device)
        else:
            mask = torch.zeros((batch, num_tokens), dtype=torch.bool, device=current_feat.device)

        keep_mask = ~mask
        current_visible, current_pad_mask, visible_indices = self._pack_visible_tokens(
            current_tokens + pos_embed,
            keep_mask=keep_mask,
        )
        current_encoded = self._encode_tokens(current_visible, key_padding_mask=current_pad_mask)
        reference_encoded = self._encode_tokens(reference_tokens + pos_embed)
        current_encoded = current_encoded.to(dtype=current_feat.dtype)
        reference_encoded = reference_encoded.to(dtype=current_feat.dtype)

        decoded = self.mask_token.to(device=current_feat.device, dtype=current_feat.dtype).expand(batch, num_tokens, -1)
        decoded = decoded + pos_embed
        for batch_idx, idx in enumerate(visible_indices):
            if idx.numel() == 0:
                continue
            decoded[batch_idx, idx] = current_encoded[batch_idx, : idx.numel()]

        out = decoded
        for block in self.decoder_blocks:
            out = block(out, memory=reference_encoded)

        pred_tokens = self.out_norm(out).to(dtype=current_feat.dtype)
        delta_tokens = pred_tokens - current_tokens
                                                                                         

        refined_small = delta_tokens.transpose(1, 2).reshape(batch, channels, height, width)
        refined = F.interpolate(
            refined_small,
            size=current_feat.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        fused = current_feat + self.gamma * refined

        aux = {
            "mask": mask,
            "pred_tokens": pred_tokens,
            "target_tokens": target_tokens,
            "mask_cross_feature": F.adaptive_avg_pool2d(refined, 8).detach(),
        }
        return fused, aux


class DistanceModule(nn.Module):
    def __init__(self, mode: str = "cos") -> None:
        super().__init__()
        self.mode = mode

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = x1.reshape(x1.shape[0], -1)
        x2 = x2.reshape(x2.shape[0], -1)
        if self.mode == "cos":
            return (F.cosine_similarity(x1, x2, dim=1) + 1.0) / 2.0
        if self.mode == "l2":
            return torch.mean((x1 - x2) ** 2, dim=1)
        raise ValueError(f"Unsupported distance mode: {self.mode}")


class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16, return_score: bool = False) -> None:
        super().__init__()
        self.return_score = return_score
        hidden = max(in_channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.SiLU(),
            nn.Linear(hidden, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_score = self.excitation(self.avg_pool(x).view(x.shape[0], -1))
        max_score = self.excitation(self.max_pool(x).view(x.shape[0], -1))
        score = (avg_score + max_score).reshape(x.shape[0], x.shape[1], 1, 1)
        if self.return_score:
            return score
        return x * score


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 3, return_score: bool = False) -> None:
        super().__init__()
        self.return_score = return_score
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.max(x, dim=1, keepdim=True)[0]
        score = self.conv(torch.cat([avg_map, max_map], dim=1))
        if self.return_score:
            return score
        return x * score


class ResBlock(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        kernel_size: int = 3,
        bottleneck_ratio: int = 1,
        down: float = 1,
        act: nn.Module = nn.SiLU(),
    ) -> None:
        super().__init__()
        self.down = down
        self.act = act
        hidden = dim_out // bottleneck_ratio
        stride = 2 if down == 2 else 1

        if bottleneck_ratio > 1:
            self.encoder = nn.Sequential(
                nn.Conv2d(dim_in, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                act,
                nn.Conv2d(hidden, hidden, kernel_size, stride=stride, padding=kernel_size // 2, bias=False),
                nn.BatchNorm2d(hidden),
                act,
                nn.Conv2d(hidden, dim_out, 1, bias=False),
                nn.BatchNorm2d(dim_out),
            )
        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size, stride=stride, padding=kernel_size // 2, bias=False),
                nn.BatchNorm2d(dim_out),
                act,
                nn.Conv2d(dim_out, dim_out, kernel_size, padding=kernel_size // 2, bias=False),
                nn.BatchNorm2d(dim_out),
            )

        self.eq_channel = nn.Conv2d(dim_in, dim_out, 1, stride=1)
        self.eq_size_up = nn.Upsample(scale_factor=1 / down, mode="bilinear", align_corners=True)
        self.eq_size_down = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.encoder(x)
        if y.shape[1] != x.shape[1]:
            x = self.eq_channel(x)
        if self.down == 2:
            x = self.eq_size_down(x)
        if self.down == 0.5:
            x = self.eq_size_up(x)
            y = self.eq_size_up(y)
        return self.act(y + x)


class GLModule(nn.Module):
    def __init__(
        self,
        dim_in_local: int,
        dim_in_global: int,
        margin: tuple[int, int, int, int] = (0, 0, 0, 0),
        projection_ratio: float = 1.0,
        sizes: tuple[int, int] = (64, 4),
        simple_att_local: bool = True,
        simple_att_global: bool = True,
        gl_att: bool = True,
        mode: str = "cos",
    ) -> None:
        super().__init__()
        self.simple_att_local = simple_att_local
        self.simple_att_global = simple_att_global
        self.gl_att = gl_att
        self.mode = mode

        self.dim_down = int(dim_in_local * projection_ratio)
        self.size_full = sizes[0]
        self.size_patch = sizes[1]
        self.margin = list(margin)
        self.num_cells = int(self.size_full / self.size_patch)
        self.region_count = (self.num_cells - (self.margin[0] + self.margin[1])) * (
            self.num_cells - (self.margin[2] + self.margin[3])
        )

        self.ca_global = ChannelAttention(dim_in_global, return_score=True)
        self.sa_global = SpatialAttention(kernel_size=3, return_score=True)
        self.ca_local = ChannelAttention(dim_in_local, return_score=True)
        self.distance = DistanceModule(mode=mode)

        self.proj_global = nn.Sequential(
            nn.Conv2d(dim_in_global, self.dim_down, 1, bias=False),
            nn.BatchNorm2d(self.dim_down),
            nn.SiLU(),
        )
        if dim_in_local != self.dim_down:
            self.proj_local = nn.Sequential(
                nn.Conv2d(dim_in_local, self.dim_down, 1, bias=False),
                nn.BatchNorm2d(self.dim_down),
                nn.SiLU(),
            )
        else:
            self.proj_local = nn.Identity()

        local_out_channels = max(int(dim_in_global / dim_in_local), 1)
        self.local_final = nn.Sequential(
            nn.Conv3d(self.region_count, local_out_channels, 1, bias=False),
            nn.BatchNorm3d(local_out_channels),
            nn.SiLU(),
        )

    def forward(
        self,
        local_feat: torch.Tensor,
        global_feat: torch.Tensor,
    ) -> tuple[list[torch.Tensor], dict[str, object], torch.Tensor]:
        batch = local_feat.shape[0]
        patches = torch.stack(
            [
                local_feat[
                    :,
                    :,
                    i * self.size_patch : (i + 1) * self.size_patch,
                    j * self.size_patch : (j + 1) * self.size_patch,
                ]
                for i in range(self.margin[0], self.num_cells - self.margin[1])
                for j in range(self.margin[2], self.num_cells - self.margin[3])
            ],
            dim=1,
        )
        global_branch = global_feat

        if self.simple_att_global:
            global_branch = self.ca_global(global_branch) * global_branch
            global_branch = self.sa_global(global_branch) * global_branch

        if self.simple_att_local:
            local_score = torch.mean(patches, dim=1)
            local_score = self.ca_local(local_score).unsqueeze(1)
            patches = patches * local_score

        att_score = torch.ones(
            (batch, self.region_count, 1, 1, 1),
            device=local_feat.device,
            dtype=local_feat.dtype,
        )
        if self.gl_att:
            patch_proj = patches.reshape(batch * self.region_count, patches.shape[2], patches.shape[3], patches.shape[4])
            patch_proj = self.proj_local(patch_proj)
            global_proj = self.proj_global(global_branch)
            global_proj = global_proj.unsqueeze(1).repeat(1, self.region_count, 1, 1, 1)
            global_proj = global_proj.reshape(batch * self.region_count, global_proj.shape[2], global_proj.shape[3], global_proj.shape[4])

            att_score = self.distance(patch_proj, global_proj)
            att_score = att_score.reshape(batch, self.region_count).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            if self.mode == "l2":
                att_score = F.softmax(att_score, dim=1)
            patches = patches * att_score

        local_branch = self.local_final(patches)
        local_branch = local_branch.reshape(
            batch,
            local_branch.shape[1] * local_branch.shape[2],
            local_branch.shape[3],
            local_branch.shape[4],
        )

        meta = {
            "margin": self.margin,
            "Size": self.size_full,
            "size": self.size_patch,
            "rn": self.region_count,
        }
        return [local_branch, global_branch], meta, att_score


def corr_operation(
    fmap1: torch.Tensor,
    fmap2: torch.Tensor,
    window: int = 1,
    stride: int = 4,
    radius: int = 8,
    use_fp16: bool = True,
) -> torch.Tensor:
    dtype = fmap1.dtype
    kernel = window * 2 + 1
    context = torch.no_grad()

    with context:
        batch, channels, size_x, size_y = fmap1.shape
        padding = (radius, radius, radius, radius)
        fmap1 = F.pad(fmap1, mode="constant", pad=padding, value=0)
        fmap2 = F.pad(fmap2, mode="constant", pad=padding, value=0)

        if use_fp16:
            fmap1 = fmap1.to(torch.float16)
            fmap2 = fmap2.to(torch.float16)

        fmap1 = fmap1[
            :,
            :,
            radius - 1 : radius - 1 + size_x,
            radius - 1 : radius - 1 + size_y,
        ].unfold(2, kernel, stride).unfold(3, kernel, stride)
        fmap1 = fmap1.reshape(batch, channels, -1, kernel, kernel)
        fmap1 = fmap1.permute(2, 0, 1, 3, 4).transpose(0, 1)

        mask1 = torch.norm(fmap1, dim=[3, 4])
        mask1 = torch.norm(mask1, dim=2).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        fmap1 = fmap1 / (mask1 + 1e-6)
        fmap1 = fmap1.unsqueeze(2)

        fmap2 = fmap2.unfold(2, radius * 2 + 1, stride).unfold(3, radius * 2 + 1, stride)
        fmap2 = fmap2.reshape(batch, channels, -1, radius * 2 + 1, radius * 2 + 1)
        fmap2 = fmap2.permute(2, 0, 1, 3, 4).transpose(0, 1)
        fmap2 = fmap2.unfold(3, kernel, 1).unfold(4, kernel, 1)
        fmap2 = fmap2.reshape(batch, fmap2.shape[1], channels, -1, kernel, kernel)
        fmap2 = fmap2.permute(3, 0, 1, 2, 4, 5).transpose(0, 1).transpose(1, 2)

        mask2 = torch.norm(fmap2, dim=[4, 5])
        mask2 = torch.norm(mask2, dim=3).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        fmap2 = fmap2 / (mask2 + 1e-6)

        corr = fmap1 * fmap2
        corr = torch.sum(corr, dim=[3, 4, 5])
        corr_size = int(corr.shape[2] ** 0.5)
        corr = corr.reshape(batch, corr.shape[1], corr_size, corr_size)
        corr = corr.transpose(-1, -2)
        corr = corr.to(dtype)

    return corr


class CorrBlock(nn.Module):
    def __init__(
        self,
        window: int = 1,
        stride: int = 4,
        radius: int = 8,
        use_fp16: bool = True,
        scale_factor: float = 16 / 15,
    ) -> None:
        super().__init__()
        self.window = window
        self.stride = stride
        self.radius = radius
        self.use_fp16 = use_fp16
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True)

    def forward(self, fmap1: torch.Tensor, fmap2: torch.Tensor) -> torch.Tensor:
        corr = corr_operation(
            fmap1,
            fmap2,
            window=self.window,
            stride=self.stride,
            radius=self.radius,
            use_fp16=self.use_fp16,
        )
        return self.upsample(corr)


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.gates = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(self, x: torch.Tensor, state=None) -> tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            h_prev = torch.zeros(
                x.shape[0],
                self.hidden_dim,
                x.shape[-2],
                x.shape[-1],
                device=x.device,
                dtype=x.dtype,
            )
            c_prev = torch.zeros_like(h_prev)
        else:
            h_prev, c_prev = state

        gates = self.gates(torch.cat([x, h_prev], dim=1))
        i_gate, f_gate, o_gate, g_gate = torch.chunk(gates, 4, dim=1)
        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        o_gate = torch.sigmoid(o_gate)
        g_gate = torch.tanh(g_gate)

        c_next = f_gate * c_prev + i_gate * g_gate
        h_next = o_gate * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTMSequence(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.cell = ConvLSTMCell(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        state = None
        for step_idx in range(x.shape[1]):
            h_state, c_state = self.cell(x[:, step_idx], state)
            outputs.append(h_state)
            state = (h_state, c_state)
        return torch.stack(outputs, dim=1)


class MyNet(nn.Module):
    """
    MoGLo-style dense correlation backbone with the requested modifications:
    encoder_1 -> corr volume -> encoder_2(base/local) -> local SACD
    -> optional stage-2 masked CrossBlock refinement -> encoder_3/4(global with cv)
    -> GL module -> ConvLSTM heads -> 6-DoF regression.
    The same shared topology is also applied to the flipped sequence so we can
    add a bidirectional consistency prior during training.
    """

    def __init__(
        self,
        seq_len: int = 5,
        dim_in: int = 1,
        dim_base: int = 64,
        input_size: int = 256,
        mask_ratio: float = 0.1,
        mask_depth: int = 2,
        mask_num_heads: int = 8,
        enable_local_sacd: bool = True,
        enable_mask_cross: bool = False,
        enable_convlstm: bool = False,
        enable_bidirectional_consistency: bool = False,
    ) -> None:
        super().__init__()
        _ = input_size
        self.seq_len = seq_len
        self.enable_local_sacd = enable_local_sacd
        self.enable_mask_cross = enable_mask_cross
        self.enable_convlstm = enable_convlstm
        self.enable_bidirectional_consistency = enable_bidirectional_consistency

        self.encoder_1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_base, 7, stride=2, padding=3),
            nn.BatchNorm2d(dim_base),
            nn.ReLU(),
            ResBlock(dim_base, dim_base),
            ResBlock(dim_base, dim_base),
            ResBlock(dim_base, dim_base, down=2),
        )
        self.encoder_2 = nn.Sequential(
            ResBlock(dim_base * 2, dim_base * 2),
            ResBlock(dim_base * 2, dim_base * 2),
            ResBlock(dim_base * 2, dim_base * 2),
            ResBlock(dim_base * 2, dim_base * 2, down=1),
        )
        self.encoder_3 = nn.Sequential(
            ResBlock(dim_base * 2, dim_base * 4, down=2),
            ResBlock(dim_base * 4, dim_base * 4),
            ResBlock(dim_base * 4, dim_base * 4),
            ResBlock(dim_base * 4, dim_base * 4),
            ResBlock(dim_base * 4, dim_base * 4),
            ResBlock(dim_base * 4, dim_base * 4, down=2),
        )
        self.encoder_4 = nn.Sequential(
            ResBlock(dim_base * 4 + 256, dim_base * 8),
            ResBlock(dim_base * 8, dim_base * 8),
            ResBlock(dim_base * 8, dim_base * 8, down=2),
        )

        self.sacd_local = StructureAwareChannelDecoupling(dim_base * 2)

        self.mask_cross = MaskedLatentCrossBlock(
            dim=dim_base,
            num_heads=mask_num_heads,
            mask_ratio=mask_ratio,
            depth=mask_depth,
            pooled_size=16,
        )
        self.mask_to_local = nn.Sequential(
            nn.Conv2d(dim_base, dim_base * 2, 1, bias=False),
            nn.BatchNorm2d(dim_base * 2),
            nn.SiLU(),
        )
        self.mask_local_gain = nn.Parameter(torch.tensor(0.5))

        self.corr = CorrBlock(window=1, stride=4, radius=8, scale_factor=16 / 15)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool_adap = nn.AdaptiveAvgPool2d((1, 1))
        self.att = GLModule(
            dim_in_local=dim_base * 2,
            dim_in_global=dim_base * 8,
            margin=(0, 0, 0, 0),
            sizes=(64, 4),
            simple_att_local=True,
            simple_att_global=True,
            gl_att=True,
            mode="cos",
        )

        self.convlstm_local = ConvLSTMSequence(dim_base * 8, dim_base * 8, kernel_size=3)
        self.convlstm_global = ConvLSTMSequence(dim_base * 8, dim_base * 8, kernel_size=3)
        self.lstm_local = nn.LSTM(dim_base * 8, dim_base * 8, num_layers=1, batch_first=True)
        self.lstm_global = nn.LSTM(dim_base * 8, dim_base * 8, num_layers=1, batch_first=True)
        self.fc_local = nn.Linear(dim_base * 8, 6)
        self.fc_global = nn.Linear(dim_base * 8, 6)

    def _forward_once(self, x: torch.Tensor, enable_mask: bool) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
        batch = x.shape[0]
        steps = x.shape[1] - 1

        x1 = x[:, :-1].reshape(batch * steps, x.shape[2], x.shape[3], x.shape[4])
        x2 = x[:, 1:].reshape(batch * steps, x.shape[2], x.shape[3], x.shape[4])
        x1, x2 = parallel_run(self.encoder_1, [x1, x2])

        high_freq = torch.cat([x1, x2], dim=1)
        cv = self.corr(x1, x2)
        x_base = self.encoder_2(torch.cat([x1, x2], dim=1))

        local_before_sacd = x_base
        if self.enable_local_sacd:
            x_local, local_score = self.sacd_local(x_base, x1=x1, x2=x2, corr_volume=cv)
        else:
            x_local, local_score = x_base, None

        mask_aux: dict[str, torch.Tensor] = {}
        if self.enable_mask_cross and enable_mask:
            mask_context, mask_aux = self.mask_cross(x2, x1)
            x_local = x_local + self.mask_local_gain * self.mask_to_local(mask_context)
        local_after_refine = x_local

        x_global = self.encoder_3(x_base)
        x_global = torch.cat([x_global, cv], dim=1)
        x_global = self.encoder_4(x_global)
        x_global = self.pool(x_global)

        x_emb, _, gl_score = self.att(local_after_refine, x_global)
        local_branch, global_branch = x_emb
        feat = torch.cat(x_emb, dim=1)

        if self.enable_convlstm:
            local_branch = local_branch.reshape(batch, steps, local_branch.shape[1], local_branch.shape[2], local_branch.shape[3])
            global_branch = global_branch.reshape(batch, steps, global_branch.shape[1], global_branch.shape[2], global_branch.shape[3])

            local_branch = self.convlstm_local(local_branch)
            global_branch = self.convlstm_global(global_branch)

            local_branch = self.pool_adap(
                local_branch.reshape(batch * steps, local_branch.shape[2], local_branch.shape[3], local_branch.shape[4])
            ).reshape(batch, steps, -1)
            global_branch = self.pool_adap(
                global_branch.reshape(batch * steps, global_branch.shape[2], global_branch.shape[3], global_branch.shape[4])
            ).reshape(batch, steps, -1)

        else:
            local_branch = self.pool_adap(local_branch).reshape(batch, steps, -1)
            global_branch = self.pool_adap(global_branch).reshape(batch, steps, -1)

                                                                                     
        local_pose = self.fc_local(self.lstm_local(local_branch)[0])
        global_pose = self.fc_global(self.lstm_global(global_branch)[0])
        pose = 0.5 * (local_pose + global_pose)

        sacd_decorrelation = getattr(self.sacd_local, "last_decorrelation_map", None)
        sacd_stability = getattr(self.sacd_local, "last_stability_map", None)
        sacd_channel_stability = getattr(self.sacd_local, "last_channel_stability", None)
        aux = {
            "mask": mask_aux.get("mask"),
            "pred_tokens": mask_aux.get("pred_tokens"),
            "target_tokens": mask_aux.get("target_tokens"),
            "record_high_freq": F.adaptive_avg_pool2d(high_freq, 8).detach(),
            "record_corr_volume": F.adaptive_avg_pool2d(cv, 4).detach(),
            "record_mask_cross": mask_aux.get("mask_cross_feature"),
            "record_local_pre_sacd": F.adaptive_avg_pool2d(local_before_sacd, 8).detach(),
            "record_local_post_sacd": F.adaptive_avg_pool2d(local_after_refine, 8).detach(),
            "record_sacd_local": local_score.detach() if local_score is not None else None,
            "record_sacd_decorrelation": (
                F.adaptive_avg_pool2d(sacd_decorrelation, 8).detach()
                if sacd_decorrelation is not None
                else None
            ),
            "record_sacd_stability": (
                F.adaptive_avg_pool2d(sacd_stability, 8).detach() if sacd_stability is not None else None
            ),
            "record_sacd_channel_stability": (
                sacd_channel_stability.detach() if sacd_channel_stability is not None else None
            ),
            "record_gl_score": gl_score.detach(),
            "record_pose_local": local_pose.detach(),
            "record_pose_global": global_pose.detach(),
            "forward_pose_raw": pose,
        }
        return pose, feat, aux

    def forward(
        self,
        x: torch.Tensor,
        enable_mask: bool = True,
        enable_backward_consistency: Optional[bool] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
        if x.dim() == 4:
            x = x.unsqueeze(2)
        elif x.dim() == 5 and x.shape[2] != 1:
            raise ValueError(f"Expected a singleton channel dimension, got {x.shape}")

        pose_fw, feat_fw, aux_fw = self._forward_once(x, enable_mask=enable_mask)
        if enable_backward_consistency is None:
            enable_backward_consistency = self.enable_bidirectional_consistency
        if self.enable_bidirectional_consistency and enable_backward_consistency:
            pose_bw, _, aux_bw = self._forward_once(torch.flip(x, dims=[1]), enable_mask=enable_mask)
            aux_fw["backward_pose_raw"] = pose_bw
            aux_fw["record_backward_mask_cross"] = aux_bw.get("record_mask_cross")
            aux_fw["record_backward_sacd_local"] = aux_bw.get("record_sacd_local")
            aux_fw["record_backward_sacd_decorrelation"] = aux_bw.get("record_sacd_decorrelation")
            aux_fw["record_backward_sacd_stability"] = aux_bw.get("record_sacd_stability")
            aux_fw["record_backward_sacd_channel_stability"] = aux_bw.get("record_sacd_channel_stability")
            aux_fw["record_pose_backward_local"] = aux_bw.get("record_pose_local")
            aux_fw["record_pose_backward_global"] = aux_bw.get("record_pose_global")
        return pose_fw, feat_fw, aux_fw
