import einops as ein
from inferno.typing import OneToOne
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal
from .. import Connection, SynapseConstructor
from .._mixins import WeightBiasDelayMixin


class Conv2D(WeightBiasDelayMixin, Connection):
    def __init__(
        self,
        in_channels: int,
        in_height: int,
        in_width: int,
        out_channels: int,
        step_time: float,
        kernel_size: int | tuple[int, int],
        *,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 1,
        dilation: int | tuple[int, int] = 0,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
        synapse: SynapseConstructor,
        batch_size: int = 1,
        bias: bool = False,
        delay: float | None = None,
        weight_init: OneToOne | None = None,
        bias_init: OneToOne | None = None,
        delay_init: OneToOne | None = None,
    ):
        pass
