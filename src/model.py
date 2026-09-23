import math
import torch
from torch import nn
import torch.nn.functional as F
from . import backbone
from .correlation import corr_operation
from .spatial_attention import install_vectorized_patches
from .frequency import FrequencyFeatureInjection

def sinusoid(values, width):
    rates = torch.exp(torch.arange(0, width, 2, device=values.device, dtype=values.dtype) * (-math.log(10000.0) / width))
    angle = values[..., None] * rates
    return torch.stack((angle.sin(), angle.cos()), -1).flatten(-2)

def position_code(steps, side, device, dtype, reverse=False, time_offset=0.5):
    times = torch.arange(steps, device=device, dtype=dtype) + time_offset
    if reverse:
        times = times.flip(0)
    (yy, xx) = torch.meshgrid(torch.arange(side, device=device, dtype=dtype), torch.arange(side, device=device, dtype=dtype), indexing='ij')
    return torch.cat([sinusoid(times, 256)[:, None].expand(-1, side * side, -1), sinusoid(yy.flatten(), 128)[None].expand(steps, -1, -1), sinusoid(xx.flatten(), 128)[None].expand(steps, -1, -1)], -1)

def candidate_channels(c0):
    (n, anchors, dy, dx) = c0.shape
    side = math.isqrt(anchors)
    assert side * side == anchors
    return c0.flatten(2).transpose(1, 2).reshape(n, dy * dx, side, side)

def align_anchors(value, height, width, stride=4):
    (sy, sx) = value.shape[-2:]
    (yy, xx) = torch.meshgrid(torch.arange(height, device=value.device, dtype=value.dtype), torch.arange(width, device=value.device, dtype=value.dtype), indexing='ij')
    grid = torch.stack((2 * xx / (stride * (sx - 1)) - 1, 2 * yy / (stride * (sy - 1)) - 1), -1)
    return F.grid_sample(value, grid[None].expand(value.shape[0], -1, -1, -1), mode='bilinear', padding_mode='border', align_corners=True)

def patchify_images(images, patch=32):
    (b, t, c, h, w) = images.shape
    assert c == 1 and h % patch == 0 and (w % patch == 0)
    return images.reshape(b, t, c, h // patch, patch, w // patch, patch).permute(0, 1, 3, 5, 4, 6, 2).reshape(b, t, h // patch * (w // patch), patch * patch * c)

def image_patch_mask(batch, frames, side=8, ratio=0.25, device='cpu', generator=None, mode='random'):
    count = frames * side * side
    if mode == 'tube':
        order = torch.rand(batch, side * side, device=device, generator=generator).argsort(1)
        frame = torch.zeros(batch, side * side, dtype=torch.bool, device=device)
        frame.scatter_(1, order[:, :int(side * side * ratio)], True)
        return frame[:, None].expand(-1, frames, -1).clone()
    assert mode == 'random'
    for _ in range(100):
        order = torch.rand(batch, count, device=device, generator=generator).argsort(1)
        mask = torch.zeros(batch, count, dtype=torch.bool, device=device)
        mask.scatter_(1, order[:, :int(count * ratio)], True)
        mask = mask.reshape(batch, frames, side * side)
        if bool((mask.sum(-1) < side * side).all()):
            return mask
    raise RuntimeError('Unable to retain visible patches in every frame')

def apply_image_mask(images, mask, patch=32):
    (b, t, c, h, w) = images.shape
    assert tuple(mask.shape) == (b, t, h // patch * (w // patch))
    pixel_mask = mask.reshape(b, t, h // patch, w // patch).repeat_interleave(patch, -2).repeat_interleave(patch, -1).unsqueeze(2)
    return (images.masked_fill(pixel_mask, 0.0), pixel_mask)

class SequenceCross(nn.Module):

    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(512)
        self.attention = nn.MultiheadAttention(512, 8, batch_first=True, dropout=0.0)
        self.ffn_norm = nn.LayerNorm(512)
        self.ffn = nn.Sequential(nn.Linear(512, 1024), nn.GELU(), nn.Linear(1024, 512))
        self.direction = nn.Parameter(torch.zeros(2, 512))
        self.stream = nn.Parameter(torch.zeros(2, 512))
        nn.init.normal_(self.direction, std=0.02)
        nn.init.normal_(self.stream, std=0.02)
        self.residual_scale = 0.1

    def attend(self, query, reference, stream, query_reverse, reference_reverse, temporal=True):
        (b, s, p, d) = query.shape
        posq = position_code(s, math.isqrt(p), query.device, query.dtype, query_reverse)
        posr = position_code(s, math.isqrt(p), query.device, query.dtype, reference_reverse)
        (qn, rn) = (self.norm(query), self.norm(reference))
        q = qn + posq + self.direction[int(query_reverse)] + self.stream[stream]
        k = rn + posr + self.direction[int(reference_reverse)] + self.stream[1 - stream]
        shape = (b, s * p, d) if temporal else (b * s, p, d)
        update = self.attention(q.reshape(shape), k.reshape(shape), rn.reshape(shape), need_weights=False)[0].reshape(b, s, p, d)
        return query + self.residual_scale * (update + self.ffn(self.ffn_norm(query + update)))

    def forward(self, query, reference, query_reverse=False, reference_reverse=False, temporal=True):
        return (self.attend(query[0], reference[1], 0, query_reverse, reference_reverse, temporal), self.attend(query[1], reference[0], 1, query_reverse, reference_reverse, temporal))

class TemporalC0Workflow(nn.Module):

    def __init__(self):
        super().__init__()
        self.legacy = backbone
        self.legacy.corr_operation = corr_operation
        self.backbone = self.legacy.MoGLo_Net(dim_in=1)
        install_vectorized_patches(self)
        self.backbone.att.vectorized_patches = True
        self.frequency = FrequencyFeatureInjection()
        self.local_correlation = nn.Sequential(nn.Conv2d(225, 16, 1), nn.GELU())
        self.local_fuse = nn.Conv2d(144, 128, 1)
        nn.init.zeros_(self.local_fuse.bias)
        with torch.no_grad():
            self.local_fuse.weight.zero_()
            self.local_fuse.weight[:, :128, 0, 0].copy_(torch.eye(128))
            nn.init.xavier_uniform_(self.local_fuse.weight[:, 128:])
            self.local_fuse.weight[:, 128:].mul_(0.01)
        self.cross = SequenceCross()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, 512))
        self.reconstruction = nn.Linear(512, 32 * 32)
        self.trace = {}
        self.record_trace = False

    def capture(self, name, value):
        if self.record_trace:
            self.trace[name] = value.detach()

    def encode(self, x, local_corr=True, prefix='forward'):
        (b, t) = x.shape[:2]
        s = t - 1
        assert tuple(x.shape[2:]) == (1, 256, 256)
        (first, second) = self.legacy.parallel_run(self.backbone.encoder_1, [x[:, :-1].reshape(b * s, 1, 256, 256), x[:, 1:].reshape(b * s, 1, 256, 256)])
        c0 = self.legacy.corr_operation(first, second, window=1, stride=4, radius=8, f16=True)
        assert not c0.requires_grad
        (first, second) = self.frequency(torch.cat([first, second], 0)).chunk(2, 0)
        base = self.backbone.encoder_2(torch.cat([first, second], 1))
        local = base
        if local_corr:
            descriptor = self.local_correlation(candidate_channels(c0))
            aligned = align_anchors(descriptor, *base.shape[-2:])
            local = self.local_fuse(torch.cat([base, aligned], 1))
            self.capture(prefix + '/local_c0', aligned)
        global_ = self.backbone.encoder_3(base)
        global_ = self.backbone.encoder_4(torch.cat([global_, self.backbone.corr.upsample(c0)], 1))
        global_ = self.backbone.pool(global_)
        (pair, _, score) = self.backbone.att(local, global_)
        tokens = tuple((v.flatten(2).transpose(1, 2).reshape(b, s, 16, 512) for v in pair))
        self.capture(prefix + '/C0', c0)
        for (name, value) in zip(['local_tokens', 'global_tokens'], tokens):
            self.capture(prefix + '/' + name, value)
        return (tokens, score)

    def decode(self, tokens):
        (local, global_) = tokens
        features = torch.cat([v.reshape(-1, 16, 512).transpose(1, 2).reshape(-1, 512, 4, 4) for v in tokens], 1)
        pl = self.backbone.fc_L(self.backbone.lstm_L(local.mean(2))[0])
        pg = self.backbone.fc_G(self.backbone.lstm_G(global_.mean(2))[0])
        return ([pl, pg], features)

    @staticmethod
    def mask_weight(epoch):
        return 0.05 * max(0.0, min(1.0, (epoch - 100) / 100.0))

    def decode_patches(self, reference, frames, reverse_query):
        (b, s, p, d) = reference[0].shape
        pos = position_code(frames, 8, reference[0].device, reference[0].dtype, reverse_query, time_offset=0.0)
        token = self.mask_token.reshape(1, 1, d).expand(b, frames * 64, d)
        query = self.cross.norm(token) + pos.reshape(1, frames * 64, d) + self.cross.direction[int(reverse_query)]
        refpos = position_code(s, 4, reference[0].device, reference[0].dtype, not reverse_query).reshape(1, s * p, d)
        values = torch.cat([self.cross.norm(v.reshape(b, s * p, d)) for v in reference], 1)
        keys = torch.cat([self.cross.norm(v.reshape(b, s * p, d)) + refpos + self.cross.stream[i] + self.cross.direction[int(not reverse_query)] for (i, v) in enumerate(reference)], 1)
        update = self.cross.attention(query, keys, values, need_weights=False)[0]
        decoded = token + update + self.cross.ffn(self.cross.ffn_norm(token + update))
        return self.reconstruction(decoded).reshape(b, frames, 64, 32 * 32)

    def complete_features(self, forward, reverse, mask):
        assert tuple(mask.shape) == tuple(forward[0].shape[:3])
        query = tuple((v.detach().masked_fill(mask[..., None], 0.0) for v in forward))
        reference = tuple((v.detach() for v in reverse))
        refined = self.cross(query, reference, False, True)
        (heads, features) = self.decode(refined)
        return dict(heads=heads, features=features, mask=mask)

    def forward(self, x, local_corr=True, temporal=True, bidirectional=True, epoch=0, patch_mask=None):
        self.trace = {}
        (forward, score) = self.encode(x, local_corr, 'forward')
        reverse = None
        if bidirectional:
            (reverse, _) = self.encode(x.flip(1), local_corr, 'reverse')
        refined = self.cross(forward, reverse if bidirectional else forward, False, bidirectional, temporal)
        (heads, features) = self.decode(refined)
        result = dict(heads=heads, features=features, score=score, mask_weight=self.mask_weight(epoch))
        for (name, value) in zip(['local', 'global'], refined):
            self.capture('cross/' + name, value)
        if bidirectional:
            rrefined = self.cross(reverse, forward, True, False, temporal)
            (result['reverse_heads'], result['reverse_features']) = self.decode(rrefined)
        if self.training and result['mask_weight'] > 0:
            assert bidirectional and patch_mask is not None
            result['mask'] = self.complete_features(forward, reverse, patch_mask)
        return result
SCCNet = TemporalC0Workflow
