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
        height: int,
        width: int,
        channels: int,
        filters: int,
        step_time: float,
        kernel_size: int | tuple[int, int],
        *,
        stride: int | tuple[int, int] = 1,
        dilation: int | tuple[int, int] = 0,
        synapse: SynapseConstructor,
        batch_size: int = 1,
        bias: bool = False,
        delay: float | None = None,
        weight_init: OneToOne | None = None,
        bias_init: OneToOne | None = None,
        delay_init: OneToOne | None = None,
    ):
        # todo
        # fix this, needs to have im2col/col2im shaped synapses, etc.
        # check variables and cast accordingly
        # shapve variables
        height = int(height)
        if height < 1:
            raise ValueError(f"height must be positive, received {height}.")

        width = int(width)
        if width < 1:
            raise ValueError(f"width must be positive, received {width}.")

        channels = int(channels)
        if channels < 1:
            raise ValueError(
                f"number of channels must be positive, received {channels}."
            )

        filters = int(filters)
        if filters < 1:
            raise ValueError(f"number of filters must be positive, received {filters}.")

        # kernel variables
        try:
            kernel_size = (int(kernel_size), int(kernel_size))
            if kernel_size[0] < 1:
                raise ValueError(
                    f"kernel size must be positive, received {kernel_size[0]}."
                )
        except TypeError:
            if len(kernel_size) != 2:
                raise ValueError(
                    f"non-scalar kernel size must be a 2-tuple, received a {len(kernel_size)}-tuple."
                )
            kernel_size = tuple(int(v) for v in kernel_size)
            if kernel_size[0] < 1 or kernel_size[1] < 1:
                raise ValueError(
                    f"kernel size must be positive, received {kernel_size}."
                )

        try:
            stride = (int(stride), int(stride))
            if stride[0] < 1:
                raise ValueError(f"stride must be positive, received {stride[0]}.")
        except TypeError:
            if len(stride) != 2:
                raise ValueError(
                    f"non-scalar stride must be a 2-tuple, received a {len(stride)}-tuple."
                )
            stride = tuple(int(v) for v in stride)
            if stride[0] < 1 or stride[1] < 1:
                raise ValueError(f"stride must be positive, received {stride}.")

        try:
            dilation = (int(dilation), int(dilation))
            if dilation[0] < 0:
                raise ValueError(
                    f"dilation must be non-negative, received {dilation[0]}."
                )
        except TypeError:
            if len(dilation) != 2:
                raise ValueError(
                    f"non-scalar dilation must be a 2-tuple, received a {len(dilation)}-tuple."
                )
            dilation = tuple(int(v) for v in dilation)
            if dilation[0] < 0 or dilation[1] < 0:
                raise ValueError(f"dilation must be non-negative, received {dilation}.")

        # other variables
        step_time = float(step_time)
        if step_time <= 0:
            raise ValueError(f"step time must be positive, received {step_time}.")

        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError(f"batch size must be positive, received {batch_size}.")

        delay = None if delay is None else float(delay)
        if delay is not None and delay <= 0:
            raise ValueError(f"delay, if not none, must be positive, received {delay}.")

        out_height, out_width = (
            math.floor(
                (size - dilation[hw] * (kernel_size[hw] - 1) - 1) / stride[hw] + 1
            )
            for hw, size in enumerate((height, width))
        )

        # call superclass constructor
        Connection.__init__(
            self,
            synapse=synapse(
                (channels * math.prod(kernel_size), out_height * out_width),
                float(step_time),
                int(batch_size),
                None if not delay else delay,
            ),
        )

        # call mixin constructor
        WeightBiasDelayMixin.__init__(
            weight=torch.rand(filters, channels, *kernel_size),
            bias=(None if not bias else torch.rand(filters)),
            delay=(None if not bias else torch.zeros(filters, channels, *kernel_size)),
            requires_grad=False,
        )

        # register extras
        self.register_extra("stride", stride)
        self.register_extra("dilation", dilation)
        self.register_extra("in_shape", (height, width))
        self.register_extra("out_shape", (out_height, out_width))

        # initialize parameters
        if weight_init:
            self.weight = weight_init(self.weight)

        if bias_init and bias:
            self.bias = bias_init(self.bias)

        if delay_init and delay:
            self.delay = delay_init(self.delay)

    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""Generates connection output from inputs, after passing through the synapse.

        Outputs are determined as the learned two-dimensional convolution applied to synaptic
        currents, after new input is applied to the synapse.

        Args:
            inputs (torch.Tensor): inputs to the connection.

        Returns:
            torch.Tensor: outputs from the connection.

        Note:
            Keyword arguments are passed to :py:class:`~inferno.neural.Synapse`
            :py:meth:`~inferno.neural.Synapse.forward` call.
        """
        if self.delayed:
            self.synapse(inputs, **kwargs)

            data = ein.rearrange(
                self.synapse.current_at(
                    ein.rearrange(self.delay, "f c h w -> 1 (c h w) 1 f").expand(
                        self.bsize, -1, self.synapse.shape[-1], -1
                    )
                ),
                "b n l f -> b f n l",
            )
            kernel = ein.rearrange(self.weight, "f c h w -> f 1 (c h w)")

            res = ein.rearrange(
                torch.matmul(kernel, data),
                "b f 1 (oh ow) -> b f oh ow",
                oh=self.outshape[0],
                ow=self.outshape[1],
            )
        else:
            self.synapse(inputs, **kwargs)

            data = self.synapse.current()  # B N L
            kernel = ein.rearrange(self.weight, "f c h w -> f (c h w)")

            res = ein.rearrange(
                torch.matmul(kernel, data),
                "b f (oh ow) -> b f oh ow",
                oh=self.outshape[0],
                ow=self.outshape[1],
            )

        if self.biased:
            pass  # add bias component and return
        else:
            return res

    @property
    def inshape(self) -> tuple[int, int, int]:
        return (self.weight.shape[1], *self.in_shape)

    @property
    def outshape(self) -> tuple[int, int, int]:
        return (self.weight.shape[0], *self.out_shape)
