import random
from itertools import combinations
import torch
from torch import nn
import torch.nn.functional as F

class MMAE(nn.Module):

    def __init__(self, alpha=1, beta=1.0, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, p, y, w=None, per=False, mse=False):
        res = torch.abs(p - y)
        if mse:
            res = res ** 2
        if per:
            res = res / (torch.abs(y) + 1e-05)
        if w is not None:
            if w.shape[1] != y.shape[1]:
                w = torch.concat([torch.zeros([y.shape[0], 1, 6]).to(y.device), w], axis=1)
            res = res * (torch.abs(w) + self.smooth) ** self.beta
        res = res.mean()
        return res * self.alpha

class Corr_loss(nn.Module):

    def __init__(self, alpha=1):
        super().__init__()
        self.alpha = alpha

    def forward(self, p, y):
        corr = F.cosine_similarity(p.flatten(), y.flatten(), dim=0)
        res = 1 - corr
        return res * self.alpha

class Triplet_Loss(nn.Module):

    def __init__(self, alpha=1, margin=0.1, n_sample=2, dist='cos'):
        super().__init__()
        '\n        p: Sampling %\n        bs: Batch*Seq\n        p_bs: sampling number\n        n_sample: [None, num]\n        '
        self.alpha = alpha
        self.margin = margin
        self.n_sample = n_sample
        if dist == 'cos':
            self.dist = self._cossim
        if dist == 'norm':
            self.dist = self._norm

    def _norm(self, a, b):
        return torch.linalg.norm(a - b, axis=-1)

    def _cossim(self, a, b):
        return 1 - F.cosine_similarity(a, b, axis=-1)

    def forward(self, v, y):
        (B, S) = (y.shape[0], y.shape[1])
        v = v.reshape(B, S, -1)
        v = v[:, -1, :]
        y = y[:, -1, :]
        combs = list(combinations([i for i in range(v.shape[0])], 3))
        if len(combs) == 0:
            return torch.tensor(0.0).to(v.dtype)
        if self.n_sample != None:
            if len(combs) < self.n_sample:
                return torch.tensor(0.0).to(v.dtype)
            else:
                combs = random.sample(combs, self.n_sample)
        Y = torch.stack([y[[combs[i]]] for i in range(len(combs))])
        V = torch.stack([v[[combs[i]]] for i in range(len(combs))])
        D1 = self.dist(Y[:, 0], Y[:, 1])
        D2 = self.dist(Y[:, 0], Y[:, 2])
        PN = torch.sign(D2 - D1)
        res = PN * (self._norm(V[:, 0], V[:, 1]) - self._norm(V[:, 0], V[:, 2])) + self.margin
        res = torch.mean(F.relu(res))
        return res * self.alpha

def pose_loss(heads, feature, y, criteria):
    prediction = 0.5 * (heads[0] + heads[1])
    mae = criteria[0](prediction, y, w=y)
    corr = criteria[1](prediction, y)
    triplet = criteria[2](feature, y)
    loss = mae + corr + 0.005 * triplet
    return (loss, {'mae': float(mae.detach()), 'corr': float(corr.detach()), 'triplet': float(triplet.detach())})

def make_criteria():
    return [MMAE(alpha=1.0, beta=1.0, smooth=1.0), Corr_loss(alpha=1.0), Triplet_Loss(alpha=1.0, margin=0.1, n_sample=2)]
