import torch
from torch import nn


class ConvLSTMCell(nn.Module):
    def __init__(self, channels=512):
        super().__init__()
        self.channels = channels
        self.gates = nn.Conv2d(2*channels, 4*channels, 3, padding=1)

    def forward(self, x, state=None):
        if state is None:
            h, c = torch.zeros_like(x), torch.zeros_like(x)
        else:
            h, c = state
        if h.shape != x.shape or c.shape != x.shape:
            raise ValueError('State and input shapes must match')
        i, f, o, g = self.gates(torch.cat((x,h),dim=1)).chunk(4,dim=1)
        c = f.sigmoid()*c + i.sigmoid()*g.tanh()
        h = o.sigmoid()*c.tanh()
        return h,c


class ConvLSTMSequence(nn.Module):
    def __init__(self, channels=512):
        super().__init__()
        self.cell = ConvLSTMCell(channels)

    def forward(self, x, state=None):
        if x.ndim != 5 or x.shape[1] < 1 or x.shape[2] != self.cell.channels:
            raise ValueError('Expected nonempty B,S,C,H,W input')
        outputs = []
        for frame in x.unbind(1):
            state = self.cell(frame,state)
            outputs.append(state[0])
        return torch.stack(outputs,dim=1), state


def spatial_sequence(tokens):
    b,s,p,c = tokens.shape
    if p != 16 or c != 512:
        raise ValueError('Expected B,S,16,512 tokens')
    return tokens.transpose(-1,-2).reshape(b,s,c,4,4)
