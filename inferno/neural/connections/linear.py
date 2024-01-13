import einops as ein
from inferno.typing import OneToOne
import math
import torch
import torch.nn.functional as F
from .. import Connection, SynapseConstructor
from ._mixins import WeightBiasDelayMixin


class DenseLinear(WeightBiasDelayMixin, Connection):
    r"""Linear all-to-all connection.

    .. math::
        y = x W^\intercal + b

    Args:
        in_shape (tuple[int, ...] | int): expected shape of input tensor, excluding batch.
        out_shape (tuple[int, ...] | int): expected shape of output tensor, excluding batch.
        step_time (float): length of a simulation time step, in :math:`\mathrm{ms}`.
        synapse (SynapseConstructor): partial constructor for inner :py:class:`~inferno.neural.Synapse`.
        batch_size (int, optional): size of input batches for simualtion. Defaults to 1.
        bias (bool, optional): if the connection should include learnable additive bias. Defaults to False.
        delay (float | None, optional): length of time the connection should support delays for. Defaults to None.
        weight_init (OneToOne | None, optional): initializer for weights. Defaults to None.
        bias_init (OneToOne | None, optional): initializer for biases. Defaults to None.
        delay_init (OneToOne | None, optional): initializer for delays. Defaults to None.

    Raises:
        ValueError: step time must be a positive real.
        ValueError: delay, if not none, must be a positive real.
    """

    def __init__(
        self,
        in_shape: tuple[int, ...] | int,
        out_shape: tuple[int, ...] | int,
        step_time: float,
        *,
        synapse: SynapseConstructor,
        batch_size: int = 1,
        bias: bool = False,
        delay: float | None = None,
        weight_init: OneToOne | None = None,
        bias_init: OneToOne | None = None,
        delay_init: OneToOne | None = None,
    ):
        # convert shapes
        try:
            in_shape = (int(in_shape),)
        except TypeError:
            in_shape = tuple(int(s) for s in in_shape)
        try:
            out_shape = (int(out_shape),)
        except TypeError:
            out_shape = tuple(int(s) for s in out_shape)

        input_size = math.prod(in_shape)
        output_size = math.prod(out_shape)

        # check that the step time is valid
        if float(step_time) <= 0:
            raise ValueError(f"step time must be positive, received {float(step_time)}")

        # check that the delay is valid
        if delay is not None and float(delay) <= 0:
            raise ValueError(
                f"delay, if not none, must be positive, received {float(delay)}"
            )

        # call superclass constructor
        Connection.__init__(
            self,
            synapse=synapse(
                input_size,
                float(step_time),
                int(batch_size),
                None if not delay else float(delay),
            ),
        )

        # call mixin constructor
        WeightBiasDelayMixin.__init__(
            weight=torch.rand(output_size, input_size),
            bias=(None if not bias else torch.rand(output_size, 1)),
            delay=(None if not bias else torch.zeros(output_size, input_size)),
            requires_grad=False,
        )

        # register extras
        self.register_extra("in_shape", in_shape)
        self.register_extra("out_shape", out_shape)

        # initialize parameters
        if weight_init:
            self.weight = weight_init(self.weight)

        if bias_init and bias:
            self.bias = bias_init(self.bias)

        if delay_init and delay:
            self.delay = delay_init(self.delay)

    def presyn_receptive(self, data: torch.Tensor) -> torch.Tensor:
        r"""Reshapes data like the synapse data for pre/post learning methods.

        Args:
            data (torch.Tensor): data to reshape.

        Returns:
            torch.Tensor: reshaped data.

         Shape:
            Input:
            :math:`B \times N_\mathrm{in} \times [N_\mathrm{out}]`

            Output:
            :math:`B \times 1 \times N_\mathrm{in} \times 1` or
            :math:`B \times N_\mathrm{out} \times N_\mathrm{in} \times 1`

            Where :math:`N_\mathrm{in}` is the number of connection inputs and :math:`N_\mathrm{out}` is the number
            of connection outputs.
        """
        match data.ndim:
            case 3:
                return ein.rearrange(data, "b i -> b 1 i 1")
            case 2:
                return ein.rearrange(data, "b i o -> b o i 1")
            case _:
                raise RuntimeError(
                    f"data with invalid number of dimensions {data.ndim} received."
                )

    def postsyn_receptive(self, data: torch.Tensor) -> torch.Tensor:
        r"""Reshapes data like the output for pre/post learning methods.

        Args:
            data (torch.Tensor): data to reshape.

        Returns:
            torch.Tensor: reshaped data.

         Shape:
            Input:
            :math:`B \times N_\mathrm{out}`

            Output:
            :math:`B \times 1 \times N_\mathrm{out} \times 1`

            Where :math:`N_\mathrm{out}` is the number of connection outputs.
        """
        return ein.rearrange(data, "b o -> b o 1 1")

    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""Generates connection output from inputs, after passing through the synapse.

        Outputs are determined as the learned linear transformation applied to synaptic
        currents, after new input is applied to the synapse. These are reshaped according
        to the specified output shape.

        Args:
            inputs (torch.Tensor): inputs to the connection.

        Returns:
            torch.Tensor: outputs from the connection.

        Note:
            Keyword arguments are passed to :py:class:`~inferno.neural.Synapse`
            :py:meth:`~inferno.neural.Synapse.forward` call.
        """
        if self.delayed:
            _ = self.synapse(ein.rearrange(inputs, "b ... -> b (...)"), **kwargs)
            res = self.syncurrent

            if self.bias is not None:
                res = torch.sum(res * self.weight.t() + self.bias, dim=1)
            else:
                res = torch.sum(res * self.weight.t(), dim=1)

        else:
            res = self.synapse(ein.rearrange(inputs, "b ... -> b (...)"), **kwargs)
            res = F.linear(res, self.weight, self.bias)

        return res.view(-1, *self.outshape)

    @property
    def syncurrent(self) -> torch.Tensor:
        r"""Currents from the synapse at the time last used by the connection.

        Returns:
             torch.Tensor: delay-offset synaptic currents.
        """
        if self.delayed:
            return self.synapse.current_at(
                ein.rearrange(self.delay, "o i -> 1 i o").expand(self.bsize, -1, -1)
            )
        else:
            return self.synapse.current

    @property
    def synspike(self) -> torch.Tensor:
        r"""Spikes to the synapse at the time last used by the connection.

        Returns:
             torch.Tensor: delay-offset synaptic spikes.
        """
        if self.delayed:
            return self.synapse.spike_at(
                ein.rearrange(self.delay, "o i -> 1 i o").expand(self.bsize, -1, -1)
            )
        else:
            return self.synapse.spike

    @property
    def inshape(self) -> tuple[int]:
        r"""Shape of inputs to the connection, excluding the batch dimension.

        Returns:
            tuple[int]: shape of inputs to the connection.
        """
        return self.in_shape

    @property
    def outshape(self) -> tuple[int]:
        r"""Shape of outputs from the connection, excluding the batch dimension.

        Returns:
            tuple[int]: shape of outputs from the connection.
        """
        return self.out_shape


class DirectLinear(WeightBiasDelayMixin, Connection):
    r"""Linear one-to-one connection.

    .. math::
        y = x \left(W^\intercal \odot I\right) + b

    Args:
        shape (tuple[int, ...] | int): expected shape of input/output tensor, excluding batch.
        step_time (float): length of a simulation time step, in :math:`\mathrm{ms}`.
        synapse (SynapseConstructor): partial constructor for inner :py:class:`~inferno.neural.Synapse`.
        batch_size (int, optional): size of input batches for simualtion. Defaults to 1.
        bias (bool, optional): if the connection should include learnable additive bias. Defaults to False.
        delay (float | None, optional): length of time the connection should support delays for. Defaults to None.
        weight_init (OneToOne | None, optional): initializer for weights. Defaults to None.
        bias_init (OneToOne | None, optional): initializer for biases. Defaults to None.
        delay_init (OneToOne | None, optional): initializer for delays. Defaults to None.

    Raises:
        ValueError: step time must be a positive real.
        ValueError: delay, if not none, must be a positive real.
    """

    def __init__(
        self,
        shape: tuple[int, ...] | int,
        step_time: float,
        *,
        synapse: SynapseConstructor,
        batch_size: int = 1,
        bias: bool = False,
        delay: float | None = None,
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
        if delay is not None and float(delay) <= 0:
            raise ValueError(
                f"delay, if not none, must be positive, received {float(delay)}"
            )

        # call superclass constructor
        Connection.__init__(
            self,
            synapse=synapse(
                size,
                float(step_time),
                int(batch_size),
                None if not delay else float(delay),
            ),
        )

        # call mixin constructor
        WeightBiasDelayMixin.__init__(
            weight=torch.rand(size),
            bias=(None if not bias else torch.rand(size)),
            delay=(None if not bias else torch.zeros(size)),
            requires_grad=False,
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

    def presyn_receptive(self, data: torch.Tensor) -> torch.Tensor:
        r"""Reshapes data like the synapse data for pre/post learning methods.

        Args:
            data (torch.Tensor): data to reshape.

        Returns:
            torch.Tensor: reshaped data.

         Shape:
            Input:
            :math:`B \times N`

            Output:
            :math:`B \times N \times 1` or

            Where :math:`N` is the number of connection inputs/outputs.
        """
        return ein.rearrange(data, "b n -> b n 1")

    def postsyn_receptive(self, data: torch.Tensor) -> torch.Tensor:
        r"""Reshapes data like the output for pre/post learning methods.

        Args:
            data (torch.Tensor): data to reshape.

        Returns:
            torch.Tensor: reshaped data.

         Shape:
            Input:
            :math:`B \times N`

            Output:
            :math:`B \times N \times 1` or

            Where :math:`N` is the number of connection inputs/outputs.
        """
        return ein.rearrange(data, "b n -> b n 1")

    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""Generates connection output from inputs, after passing through the synapse.

        Outputs are determined as the learned linear transformation applied to synaptic
        currents, after new input is applied to the synapse. These are reshaped according
        to the specified output shape.

        Args:
            inputs (torch.Tensor): inputs to the connection.

        Returns:
            torch.Tensor: outputs from the connection.

        Note:
            Keyword arguments are passed to :py:class:`~inferno.neural.Synapse`
            :py:meth:`~inferno.neural.Synapse.forward` call.
        """
        if self.delayed:
            _ = self.synapse(ein.rearrange(inputs, "b ... -> b (...)"), **kwargs)
            res = ein.rearrange(self.current, "b n 1 -> b n")
        else:
            res = self.synapse(ein.rearrange(inputs, "b ... -> b (...)"), **kwargs)

        if self.bias is not None:
            res = res * self.weight + self.bias
        else:
            res = res * self.weight

        return res.view(-1, *self.outshape)

    @property
    def syncurrent(self) -> torch.Tensor:
        r"""Currents from the synapse at the time last used by the connection.

        Returns:
             torch.Tensor: delay-offset synaptic currents.
        """
        if self.delayed:
            return self.synapse.current_at(
                ein.rearrange(self.delay, "n -> 1 n 1").expand(self.bsize, -1, -1)
            )
        else:
            return self.synapse.current

    @property
    def synspike(self) -> torch.Tensor:
        r"""Spikes to the synapse at the time last used by the connection.

        Returns:
             torch.Tensor: delay-offset synaptic spikes.
        """
        if self.delayed:
            return self.synapse.spike_at(
                ein.rearrange(self.delay, "n -> 1 n 1").expand(self.bsize, -1, -1)
            )
        else:
            return self.synapse.spike

    @property
    def inshape(self) -> tuple[int]:
        r"""Shape of inputs to the connection, excluding the batch dimension.

        Returns:
            tuple[int]: shape of inputs to the connection.
        """
        return self.shape

    @property
    def outshape(self) -> tuple[int]:
        r"""Shape of outputs from the connection, excluding the batch dimension.

        Returns:
            tuple[int]: shape of outputs from the connection.
        """
        return self.shape


class LateralLinear(WeightBiasDelayMixin, Connection):
    r"""Linear all-to-"all but one" connection.

    .. math::
        y = x \left(W^\intercal \odot (1 - I\right)) + b

    Args:
        shape (tuple[int, ...] | int): expected shape of input/output tensor, excluding batch.
        step_time (float): length of a simulation time step, in :math:`\mathrm{ms}`.
        synapse (SynapseConstructor): partial constructor for inner :py:class:`~inferno.neural.Synapse`.
        batch_size (int, optional): size of input batches for simualtion. Defaults to 1.
        bias (bool, optional): if the connection should include learnable additive bias. Defaults to False.
        delay (float | None, optional): length of time the connection should support delays for. Defaults to None.
        weight_init (OneToOne | None, optional): initializer for weights. Defaults to None.
        bias_init (OneToOne | None, optional): initializer for biases. Defaults to None.
        delay_init (OneToOne | None, optional): initializer for delays. Defaults to None.

    Raises:
        ValueError: step time must be a positive real.
        ValueError: delay, if not none, must be a positive real.
    """

    def __init__(
        self,
        shape: tuple[int, ...] | int,
        step_time: float,
        *,
        synapse: SynapseConstructor,
        batch_size: int = 1,
        bias: bool = False,
        delay: float | None = None,
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
            raise ValueError(f"step time must be positive, received {float(step_time)}")

        # check that the delay is valid
        if delay is not None and float(delay) <= 0:
            raise ValueError(
                f"delay, if not none, must be positive, received {float(delay)}"
            )

        # call superclass constructor
        Connection.__init__(
            self,
            synapse=synapse(
                size,
                float(step_time),
                int(batch_size),
                None if not delay else float(delay),
            ),
        )

        # call mixin constructor
        WeightBiasDelayMixin.__init__(
            weight=torch.rand(size, size) * (1 - torch.eye(size)),
            bias=(None if not bias else torch.rand(size, 1)),
            delay=(None if not bias else torch.zeros(size, size)),
            requires_grad=False,
        )

        # register buffer
        self.register_buffer("mask", 1 - torch.eye(size))

        # register extras
        self.register_extra("shape", shape)

        # initialize parameters
        if weight_init:
            self.weight = weight_init(self.weight)

        if bias_init and bias:
            self.bias = bias_init(self.bias)

        if delay_init and delay:
            self.delay = delay_init(self.delay)

    def presyn_receptive(self, data: torch.Tensor) -> torch.Tensor:
        r"""Reshapes data like the synapse data for pre/post learning methods.

        Args:
            data (torch.Tensor): data to reshape.

        Returns:
            torch.Tensor: reshaped data.

         Shape:
            Input:
            :math:`B \times N \times [N]`

            Output:
            :math:`B \times 1 \times N \times 1` or
            :math:`B \times N \times N \times 1`

            Where :math:`N` is the number of connection inputs/outputs.
        """
        match data.ndim:
            case 3:
                return ein.rearrange(data, "b i -> b 1 i 1")
            case 2:
                return ein.rearrange(data, "b i o -> b o i 1")
            case _:
                raise RuntimeError(
                    f"data with invalid number of dimensions {data.ndim} received."
                )

    def postsyn_receptive(self, data: torch.Tensor) -> torch.Tensor:
        r"""Reshapes data like the output for pre/post learning methods.

        Args:
            data (torch.Tensor): data to reshape.

        Returns:
            torch.Tensor: reshaped data.

         Shape:
            Input:
            :math:`B \times N`

            Output:
            :math:`B \times 1 \times N \times 1`

            Where :math:`N` is the number of connection inputs/outputs.
        """
        return ein.rearrange(data, "b o -> b o 1 1")

    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""Generates connection output from inputs, after passing through the synapse.

        Outputs are determined as the learned linear transformation applied to synaptic
        currents, after new input is applied to the synapse. These are reshaped according
        to the specified output shape.

        Args:
            inputs (torch.Tensor): inputs to the connection.

        Returns:
            torch.Tensor: outputs from the connection.

        Note:
            Keyword arguments are passed to :py:class:`~inferno.neural.Synapse`
            :py:meth:`~inferno.neural.Synapse.forward` call.
        """
        if self.delayed:
            _ = self.synapse(ein.rearrange(inputs, "b ... -> b (...)"), **kwargs)
            res = self.syncurrent

            if self.bias is not None:
                res = torch.sum(res * self.weight.t() + self.bias, dim=1)
            else:
                res = torch.sum(res * self.weight.t(), dim=1)

        else:
            res = self.synapse(ein.rearrange(inputs, "b ... -> b (...)"), **kwargs)
            res = F.linear(res, self.weight, self.bias)

        return res.view(-1, *self.outshape)

    @property
    def syncurrent(self) -> torch.Tensor:
        r"""Currents from the synapse at the time last used by the connection.

        Returns:
             torch.Tensor: delay-offset synaptic currents.
        """
        if self.delayed:
            return self.synapse.current_at(
                ein.rearrange(self.delay, "o i -> 1 i o").expand(self.bsize, -1, -1)
            )
        else:
            return self.synapse.current

    @property
    def synspike(self) -> torch.Tensor:
        r"""Spikes to the synapse at the time last used by the connection.

        Returns:
             torch.Tensor: delay-offset synaptic spikes.
        """
        if self.delayed:
            return self.synapse.spike_at(
                ein.rearrange(self.delay, "o i -> 1 i o").expand(self.bsize, -1, -1)
            )
        else:
            return self.synapse.spike

    @property
    def inshape(self) -> tuple[int]:
        r"""Shape of inputs to the connection, excluding the batch dimension.

        Returns:
            tuple[int]: shape of inputs to the connection.
        """
        return self.shape

    @property
    def outshape(self) -> tuple[int]:
        r"""Shape of outputs from the connection, excluding the batch dimension.

        Returns:
            tuple[int]: shape of outputs from the connection.
        """
        return self.shape

    @property
    def weight(self) -> torch.Tensor:
        r"""Learnable weights of the connection.

        Args:
            value (torch.Tensor): new weights.

        Returns:
            torch.Tensor: current weights.
        """
        return WeightBiasDelayMixin.weight.fget(self)

    @weight.setter
    def weight(self, value: torch.Tensor):
        WeightBiasDelayMixin.weight.fset(self, value * self.mask)

    @property
    def delay(self) -> torch.Tensor | None:
        r"""Learnable delays of the connection.

        Args:
            value (torch.Tensor): new delays.

        Returns:
            torch.Tensor | None: current delays, if the connection has any.

        Raises:
            RuntimeError: ``delay`` cannot be set on a connection without learnable delays.
        """
        WeightBiasDelayMixin.delay.fget(self)

    @delay.setter
    def delay(self, value: torch.Tensor):
        if self.delay_ is not None:
            WeightBiasDelayMixin.weight.fset(self, value * self.mask)
        else:
            WeightBiasDelayMixin.weight.fset(self, value)
