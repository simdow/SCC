import torch
import torch.nn.functional as F

def descriptors(first, second):
    (n, c, h, w) = first.shape
    (p1, p2) = (F.pad(first, (8, 8, 8, 8)), F.pad(second, (8, 8, 8, 8)))
    source = p1[:, :, 7:7 + h, 7:7 + w].unfold(2, 3, 4).unfold(3, 3, 4)
    source = source.permute(0, 2, 3, 1, 4, 5).reshape(n, -1, c, 3, 3)
    source_norm = torch.norm(torch.norm(source, dim=(-2, -1)), dim=2)
    source = (source / (source_norm[:, :, None, None, None] + 1e-06)).flatten(2)
    target = p2.unfold(2, 3, 1).unfold(3, 3, 1)
    target = target.permute(0, 2, 3, 1, 4, 5).reshape(n, -1, c, 3, 3)
    target_norm = torch.norm(torch.norm(target, dim=(-2, -1)), dim=2)
    target = (target / (target_norm[:, :, None, None, None] + 1e-06)).flatten(2)
    return (source, target)

@torch.no_grad()
def corr_operation(first, second, window=1, stride=4, radius=8, f16=True, chunk_size=16):
    assert window == 1 and stride == 4 and (radius == 8)
    assert first.shape == second.shape and first.shape[-1] == first.shape[-2]
    original_dtype = first.dtype
    (n, c, h, w) = first.shape
    if f16:
        (first, second) = (first.to(torch.float16), second.to(torch.float16))
    (source, target) = descriptors(first, second)
    dr = torch.arange(15, device=first.device)
    offsets = (dr[:, None] * (w + 14) + dr[None, :]).flatten()
    pieces = []
    for start in range(0, source.shape[1], chunk_size):
        end = min(start + chunk_size, source.shape[1])
        index = torch.arange(start, end, device=first.device)
        origin = index // (h // 4) * 4 * (w + 14) + index % (h // 4) * 4
        candidates = target[:, origin[:, None] + offsets[None, :]]
        product = (source[:, start:end, None, :] * candidates).reshape(n, end - start, 225, c, 3, 3)
        score = product.sum(dim=(-3, -2, -1))
        pieces.append(score.reshape(n, end - start, 15, 15).transpose(-1, -2))
    return torch.cat(pieces, dim=1).to(original_dtype)
