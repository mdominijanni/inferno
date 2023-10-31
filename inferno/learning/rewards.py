import torch
import torch.nn as nn


class FlorianReward(nn.Module):

    def __init__(self, inner, baseline=0, sign=True):
        nn.Module.__init__(self)
        self.inner = inner
        self.baseline = baseline
        self.sign = sign
        self.register_buffer('prior', torch.tensor(0.0))

    def forward(self, x, target):
        diff = self.inner(x, target)
        delta = diff - self.prior
        if self.sign:
            delta = torch.sign(delta)
        self.prior.fill_(diff)
        return self.baseline + delta
