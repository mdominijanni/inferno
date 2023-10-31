import torch
import torch.nn as nn


class DCClassifier(nn.Module):

    def __init__(
        self,
        dynamics_shape,
        num_classes,
        alpha
    ):
        nn.Module.__init__(self)
        self.register_buffer('rates', torch.zeros([num_classes] + list(dynamics_shape), requires_grad=False))
        self.register_buffer('alpha', torch.tensor(alpha), requires_grad=False)

    def forward(
        self,
        inputs: torch.Tensor,  # (batch_size, dynamics_shape..., timesteps)
        labels: torch.Tensor   # (batch_size,)
    ):
        if self.rates.abs().sum() == 0.0:
            self.rates.scatter_add_(
                0,
                labels.view([-1] + [1 for _ in self.rates.shape[1:]]).expand([-1] + [sn for sn in self.rates.shape[1:]]),
                inputs
            )
        else:
            self.rates.mul_(self.alpha)
            self.rates.scatter_add_(
                0,
                labels.view([-1] + [1 for _ in self.rates.shape[1:]]).expand([-1] + [sn for sn in self.rates.shape[1:]]),
                inputs * (1 - self.alpha)
            )
