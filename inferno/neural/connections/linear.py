import einops as ein
from inferno.typing import OneToOne
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import Connection, SynapseConstructor


class DenseLinear(Connection):
    def __init__(
        self,
        input_shape: tuple[int, ...] | int,
        output_shape: tuple[int, ...] | int,
        step_time: float,
        *,
        synapse: SynapseConstructor,
        batch_size: int = 1,
        bias: bool = False,
        delay: int | float | None = None,
        weight_init: OneToOne | None = None,
        bias_init: OneToOne | None = None,
        delay_init: OneToOne | None = None,
    ):
        # convert shapes
        try:
            input_shape = (int(input_shape),)
        except TypeError:
            input_shape = tuple(int(s) for s in input_shape)
        try:
            output_shape = (int(output_shape),)
        except TypeError:
            output_shape = tuple(int(s) for s in output_shape)

        input_size = math.prod(input_shape)
        output_size = math.prod(output_shape)

        # check that the step time is valid
        if float(step_time) <= 0:
            raise ValueError(
                f"step time must be greater than zero, received {float(step_time)}"
            )

        # check that the delay is valid
        if delay is not None:
            if delay == 0:
                raise ValueError(
                    f"delay, if not none, it must be greater than zero, received {delay}"
                )

        # build parameters
        weights = nn.Parameter(torch.rand(output_size, input_size), requires_grad=False)

        if bias:
            biases = nn.Parameter(torch.rand(output_size, 1), requires_grad=False)
        else:
            biases = None

        if delay:
            if isinstance(delay, int):
                synapse_delay = delay
                delays = nn.Parameter(
                    torch.zeros(output_size, input_size, dtype=torch.int64),
                    requires_grad=False,
                )
            if isinstance(delay, float):
                synapse_delay = int(delay // step_time)
                delays = nn.Parameter(
                    torch.zeros(output_size, input_size),
                    requires_grad=False,
                )
        else:
            synapse_delay = None
            delays = None

        # call superclass constructor
        Connection.__init__(
            self,
            synapse=synapse(
                input_size, float(step_time), int(batch_size), synapse_delay
            ),
            weight=weights,
            bias=biases,
            delay=delays,
        )

        # register extras
        self.register_extra("input_shape", input_shape)
        self.register_extra("output_shape", output_shape)

        # initialize parameters
        if weight_init:
            self.weight = weight_init(self.weight)

        if bias_init and bias:
            self.bias = bias_init(self.bias)

        if delay_init and delay:
            self.delay = delay_init(self.delay)

    @property
    def inshape(self) -> tuple[int]:
        return self.input_shape

    @property
    def outshape(self) -> tuple[int]:
        return self.output_shape

    def forward(self, inputs: torch.Tensor):
        if self.delayed:
            _ = self.synapse(ein.rearrange(inputs, "b ... -> b (...)"))
            res = self.synapse.dcurrent(self.delay)

            if self.bias is not None:
                res = torch.sum(res * self.weight + self.bias, dim=-1)
            else:
                res = torch.sum(res * self.weight, dim=-1)

        else:
            res = self.synapse(ein.rearrange(inputs, "b ... -> b (...)"))
            res = F.linear(res, self.weight, self.bias)

        return res.view(-1, *self.outshape)


class DirectLinear(Connection):
    def __init__(
        self,
        shape: tuple[int, ...] | int,
        step_time: float,
        *,
        synapse: SynapseConstructor,
        batch_size: int = 1,
        bias: bool = False,
        delay: int | float | None = None,
        weight_init: OneToOne | None = None,
        bias_init: OneToOne | None = None,
        delay_init: OneToOne | None = None,
    ):
        # convert shapes
        try:
            shape = (int(shape),)
        except TypeError:
            shape = tuple(int(s) for s in shape)

        size = math.prod(shape)

        # check that the step time is valid
        if float(step_time) <= 0:
            raise ValueError(
                f"step time must be greater than zero, received {float(step_time)}"
            )

        # check that the delay is valid
        if delay is not None:
            if delay == 0:
                raise ValueError(
                    f"delay, if not none, it must be greater than zero, received {delay}"
                )

        # build parameters
        weights = nn.Parameter(torch.rand(size), requires_grad=False)

        if bias:
            biases = nn.Parameter(torch.rand(size), requires_grad=False)
        else:
            biases = None

        if delay:
            if isinstance(delay, int):
                synapse_delay = delay
                delays = nn.Parameter(
                    torch.zeros(size, dtype=torch.int64),
                    requires_grad=False,
                )
            if isinstance(delay, float):
                synapse_delay = int(delay // step_time)
                delays = nn.Parameter(
                    torch.zeros(size),
                    requires_grad=False,
                )
        else:
            synapse_delay = None
            delays = None

        # call superclass constructor
        Connection.__init__(
            self,
            synapse=synapse(size, float(step_time), int(batch_size), synapse_delay),
            weight=weights,
            bias=biases,
            delay=delays,
        )

        # register extras
        self.register_extra("shape", shape)

        # initialize parameters
        if weight_init:
            self.weight = weight_init(self.weight)

        if bias_init and bias:
            self.bias = bias_init(self.bias)

        if delay_init and delay:
            self.delay = delay_init(self.delay)

    @property
    def inshape(self) -> tuple[int]:
        return self.shape

    @property
    def outshape(self) -> tuple[int]:
        return self.shape

    def forward(self, inputs: torch.Tensor):
        if self.delayed:
            _ = self.synapse(ein.rearrange(inputs, "b ... -> b (...)"))
            res = ein.rearrange(
                self.synapse.dcurrent(self.delay.view(-1, 1)), "b 1 n -> b n"
            )

            if self.bias is not None:
                res = res * self.weight + self.bias
            else:
                res = res * self.weight

        else:
            res = self.synapse(ein.rearrange(inputs, "b ... -> b (...)"))

            if self.bias is not None:
                res = res * self.weight + self.bias
            else:
                res = res * self.weight

        return res.view(-1, *self.outshape)
