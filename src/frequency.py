import torch
from torch import nn
import torch.nn.functional as F

def initialize_linear(module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

class BandAttention(nn.Module):

    def __init__(self):
        super().__init__()
        self.channel = nn.Sequential(nn.Linear(64, 8), nn.SiLU(), nn.Linear(8, 64))
        self.spatial = nn.Conv2d(2, 1, 3, padding=1)
        self.apply(initialize_linear)
        nn.init.zeros_(self.channel[-1].weight)
        nn.init.zeros_(self.spatial.weight)

    def forward(self, x):
        gate = 2 * torch.sigmoid(self.channel(x.mean((-2, -1))) + self.channel(x.amax((-2, -1))))
        value = x * gate[:, :, None, None]
        spatial = 2 * torch.sigmoid(self.spatial(torch.cat([value.mean(1, keepdim=True), value.amax(1, keepdim=True)], 1)))
        return value * spatial

class FrequencyCorrRefiner(nn.Module):

    @staticmethod
    def split(x):
        low = F.avg_pool2d(x, 3, 1, 1, count_include_pad=False)
        return (low, x - low)

class FrequencyFeatureInjection(nn.Module):

    def __init__(self):
        super().__init__()
        self.low_attention = BandAttention()
        self.high_attention = BandAttention()
        self.mlp = nn.Sequential(nn.Conv2d(128, 64, 1), nn.GELU(), nn.Conv2d(64, 64, 1))
        for layer in [self.mlp[0], self.mlp[2]]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        with torch.no_grad():
            self.mlp[2].weight.mul_(0.01)
        self.alpha = 0.1
        self.last_diagnostics = {}
        self.record_diagnostics = False

    def forward(self, features):
        (low, high) = FrequencyCorrRefiner.split(features)
        attended = torch.cat([self.low_attention(low), self.high_attention(high)], 1)
        delta = self.alpha * torch.tanh(self.mlp(attended))
        self.last_diagnostics = {}
        if self.record_diagnostics:
            self.last_diagnostics = dict(input=features.detach(), delta=delta.detach())
        return features + delta
