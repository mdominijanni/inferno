from .mixins import WeightBiasDelayMixin
from ..base import Connection, SynapseConstructor
from ..._internal import argtest
from ...types import OneToOne
import einops as ein
import math
import torch
import torch.nn.functional as F


class LinearDense(WeightBiasDelayMixin, Connection):
    r"""Linear all-to-all connection.

    .. math::
        y = x W^\intercal + b

    Args:
        in_shape (tuple[int, ...] | int): expected shape of input tensor,
            excluding batch dimension.
        out_shape (tuple[int, ...] | int): expected shape of output tensor,
            excluding batch dimension.
        step_time (float): length of a simulation time step, in :math:`\text{ms}`.
        synapse (SynapseConstructor): partial constructor for inner :py:class:`Synapse`.
        bias (bool, optional): if the connection should support
            learnable additive bias. Defaults to ``False``.
        delay (float | None, optional): maximum supported delay length, in
            :math:`\text{ms}`, excludes delays when ``None``. Defaults to ``None``.
        batch_size (int, optional): size of input batches for simulation.
            Defaults to ``1``.
        weight_init (OneToOne[torch.Tensor] | None, optional): initializer for weights.
            Defaults to ``None``.
        bias_init (OneToOne[torch.Tensor] | None, optional): initializer for biases.
            Defaults to ``None``.
        delay_init (OneToOne[torch.Tensor] | None, optional): initializer for delays.
            Defaults to ``None``.

    .. admonition:: Shape
        :class: tensorshape

        ``LinearDense.weight``, ``LinearDense.delay``:

        :math:`\prod(N_0, \ldots) \times \prod(M_0, \ldots)`

        ``LinearDense.bias``:

        :math:`\prod(N_0 \cdot \cdots)`

        Where:
            * :math:`N_0, \ldots` are the unbatched output dimensions.
            * :math:`M_0, \ldots` are the unbatched input dimensions.

    Note:
        When ``delay`` is None, no ``delay_`` parameter is created and altering the
        maximum delay of :py:attr:`~Connection.synapse` will have no effect. Setting to 0 will
        create and register a ``delay_`` parameter but not use delays unless it is
        later changed.

    Note:
        If ``weight_init`` or ``bias_init`` are None, ``weight`` and ``bias`` are,
        respectively, initialized as uniform random values over the interval
        :math:`[0, 1)` using :py:func:`torch.rand`.

        If ``delay_init`` is None, ``delay`` is initialized as zeros using
        :py:func:`torch.rand`.
    """

    def __init__(
        self,
        in_shape: tuple[int, ...] | int,
        out_shape: tuple[int, ...] | int,
        step_time: float,
        *,
        synapse: SynapseConstructor,
        bias: bool = False,
        delay: float | None = None,
        batch_size: int = 1,
        weight_init: OneToOne[torch.Tensor] | None = None,
        bias_init: OneToOne[torch.Tensor] | None = None,
        delay_init: OneToOne[torch.Tensor] | None = None,
    ):
        # connection attributes
        try:
            self.in_shape = argtest.ofsequence("in_shape", in_shape, argtest.gt, 0, int)
        except TypeError:
            self.in_shape = (argtest.gt("in_shape", in_shape, 0, int),)

        try:
            self.out_shape = argtest.ofsequence(
                "out_shape", out_shape, argtest.gt, 0, int
            )
        except TypeError:
            self.out_shape = (argtest.gt("out_shape", out_shape, 0, int),)

        # intermediate values
        in_size, out_size = math.prod(self.in_shape), math.prod(self.out_shape)

        # call superclass constructor
        Connection.__init__(
            self,
            synapse=synapse(
                in_size,
                step_time,
                0.0 if delay is None else delay,
                batch_size,
            ),
        )

        # call mixin constructor
        WeightBiasDelayMixin.__init__(
            self,
            weight=torch.rand(out_size, in_size),
            bias=(None if not bias else torch.rand(out_size)),
            delay=(None if delay is None else torch.zeros(out_size, in_size)),
            requires_grad=False,
        )

        # initialize parameters
        if weight_init:
            self.weight = weight_init(self.weight)

        if bias_init and self.biased:
            self.bias = bias_init(self.bias)

        if delay_init and self.delayedby is not None:
            self.delay = delay_init(self.delay)

    @property
    def inshape(self) -> tuple[int, ...]:
        r"""Shape of inputs to the connection, excluding the batch dimension.

        Returns:
            tuple[int]: shape of inputs to the connection.
        """
        return self.in_shape

    @property
    def outshape(self) -> tuple[int, ...]:
        r"""Shape of outputs from the connection, excluding the batch dimension.

        Returns:
            tuple[int]: shape of outputs from the connection.
        """
        return self.out_shape

    @property
    def selector(self) -> torch.Tensor | None:
        r"""Learned delays as a selector for synaptic currents and delays.

        Returns:
            torch.Tensor | None: delay selector if the connection has learnable delays.

        .. admonition:: Shape
            :class: tensorshape

            :math:`1 \times M \times N`

            Where:
                * :math:`M` is the number of elements across input dimensions.
                * :math:`N` is the number of elements across output dimensions.
        """
        if self.delayedby is not None:
            delays = self.delay
        else:
            delays = torch.zeros_like(self.weight)
        return ein.rearrange(delays, "o i -> 1 i o").expand(self.batchsz, -1, -1)

    def like_bias(self, data: torch.Tensor) -> torch.Tensor:
        r"""Reshapes data like reduced postsynaptic receptive spikes to connection bias.

        Args:
            data (torch.Tensor): data shaped like reduced postsynaptic receptive spikes.

        Returns:
            torch.Tensor: reshaped data.

        .. admonition:: Shape
            :class: tensorshape

            ``data``:

            :math:`M \times 1`

            ``return``:

            :math:`M`

            Where:
                * :math:`M` is the number of elements across output dimensions.
        """
        return ein.rearrange(data, "m 1 -> m")

    def like_input(self, data: torch.Tensor) -> torch.Tensor:
        r"""Reshapes data like synapse input to connection input.

        Args:
            data (torch.Tensor): data shaped like synapse input.

        Returns:
            torch.Tensor: reshaped data.

        .. admonition:: Shape
            :class: tensorshape

            ``data``:

            :math:`B \times M`

            ``return``:

            :math:`B \times M_0 \times \cdots`

            Where:
                * :math:`B` is the batch size.
                * :math:`M` is the number of elements across input dimensions.
                * :math:`M_0, \ldots` are the unbatched input dimensions.
        """
        return data.view(-1, *self.inshape)

    def like_synaptic(self, data: torch.Tensor) -> torch.Tensor:
        r"""Reshapes data like connection input to synapse input.

        Args:
            data (torch.Tensor): data shaped like connection input.

        Returns:
            torch.Tensor: reshaped data.

        .. admonition:: Shape
            :class: tensorshape

            ``data``:

            :math:`B \times M_0 \times \cdots`

            ``return``:

            :math:`B \times M`

            Where:
                * :math:`B` is the batch size.
                * :math:`M_0, \ldots` are the unbatched input dimensions.
                * :math:`M` is the number of elements across input dimensions.
        """
        return ein.rearrange(data, "b ... -> b (...)")

    def presyn_receptive(self, data: torch.Tensor) -> torch.Tensor:
        r"""Reshapes data like the synapse state for pre-post learning methods.

        Args:
            data (torch.Tensor): data shaped like output of :py:meth:`like_synaptic`.

        Returns:
            torch.Tensor: reshaped data.

        .. admonition:: Shape
            :class: tensorshape

            ``data``:

            :math:`B \times M \times [N]`

            ``return``:

            :math:`B \times N \times M \times 1`

            or

            :math:`B \times 1 \times M \times 1`

            Where:
                * :math:`B` is the batch size.
                * :math:`M` is the number of elements across input dimensions.
                * :math:`N` is the number of elements across output dimensions.
        """
        return ein.rearrange(data, "b i ... -> b (...) i 1")

    def postsyn_receptive(self, data: torch.Tensor) -> torch.Tensor:
        r"""Reshapes data like connection output for pre-post learning methods.

        Args:
            data (torch.Tensor): data shaped like output of :py:meth:`forward`.

        Returns:
            torch.Tensor: reshaped data.

        .. admonition:: Shape
            :class: tensorshape

            ``data``:

            :math:`B \times N_0 \times \cdots`

            ``return``:

            :math:`B \times N \times 1 \times 1`

            Where:
                * :math:`B` is the batch size.
                * :math:`N_0, \ldots` are the unbatched output dimensions.
                * :math:`N` is the number of elements across output dimensions.
        """
        return ein.rearrange(data, "b ... -> b (...) 1 1")

    def forward(self, *inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""Generates connection output from inputs, after passing through the synapse.

        Outputs are determined as the learned linear transformation applied to synaptic
        currents, after new input is applied to the synapse, then reshaped to match
        :py:attr:`~Connection.batched_outshape`.

        Args:
            *inputs (torch.Tensor): inputs to the connection.

        Returns:
            torch.Tensor: outputs from the connection.

        .. admonition:: Shape
            :class: tensorshape

            ``*inputs``:

            :math:`B \times M_0 \times \cdots`

            ``return``:

            :math:`B \times N_0 \times \cdots`

            Where:
                * :math:`B` is the batch size.
                * :math:`M_0, \ldots` are the unbatched input dimensions.
                * :math:`N_0, \ldots` are the unbatched output dimensions.

        Note:
            ``*inputs`` are reshaped using :py:meth:`like_synaptic` then passed to
            py:meth:`~Synapse.forward` of :py:attr:`~Connection.synapse`. Keyword arguments are
            also passed through.
        """
        # reshape inputs and perform synapse simulation
        res = self.synapse(
            *(self.like_synaptic(inp) for inp in inputs), **kwargs
        )  # B I

        if self.delayedby:
            res = self.syncurrent  # B I O

            if self.biased:
                res = ein.einsum(res, self.weight, "b i o, o i -> b o") + self.bias
            else:
                res = ein.einsum(res, self.weight, "b i o, o i -> b o")

        else:
            res = F.linear(res, self.weight, self.bias)

        return res.view(-1, *self.outshape)


class LinearDirect(WeightBiasDelayMixin, Connection):
    r"""Linear one-to-one connection.

    .. math::
        y = x \odot W + b

    Args:
        shape (tuple[int, ...] | int): expected shape of input and output tensors,
            excluding batch dimension.
        step_time (float): length of a simulation time step, in :math:`\text{ms}`.
        synapse (SynapseConstructor): partial constructor for inner :py:class:`Synapse`.
        bias (bool, optional): if the connection should support
            learnable additive bias. Defaults to ``False``.
        delay (float | None, optional): maximum supported delay length, in
            :math:`\text{ms}`, excludes delays when ``None``. Defaults to ``None``.
        batch_size (int, optional): size of input batches for simulation.
            Defaults to ``1``.
        weight_init (OneToOne[torch.Tensor] | None, optional): initializer for weights.
            Defaults to ``None``.
        bias_init (OneToOne[torch.Tensor] | None, optional): initializer for biases.
            Defaults to ``None``.
        delay_init (OneToOne[torch.Tensor] | None, optional): initializer for delays.
            Defaults to ``None``.

    .. admonition:: Shape
        :class: tensorshape

        ``LinearDirect.weight``, ``LinearDirect.delay``, and ``LinearDirect.bias``:

        :math:`\prod(N_0, \ldots)`

        Where:
            * :math:`N_0, \ldots` are the unbatched input/output dimensions.

    Note:
        When ``delay`` is None, no ``delay_`` parameter is created and altering the
        maximum delay of :py:attr:`~Connection.synapse` will have no effect. Setting to 0 will
        create and register a ``delay_`` parameter but not use delays unless it is
        later changed.

    Note:
        If ``weight_init`` or ``bias_init`` are None, ``weight`` and ``bias`` are,
        respectively, initialized as uniform random values over the interval
        :math:`[0, 1)` using :py:func:`torch.rand`.

        If ``delay_init`` is None, ``delay`` is initialized as zeros using
        :py:func:`torch.rand`.
    """

    def __init__(
        self,
        shape: tuple[int, ...] | int,
        step_time: float,
        *,
        synapse: SynapseConstructor,
        bias: bool = False,
        delay: float | None = None,
        batch_size: int = 1,
        weight_init: OneToOne[torch.Tensor] | None = None,
        bias_init: OneToOne[torch.Tensor] | None = None,
        delay_init: OneToOne[torch.Tensor] | None = None,
    ):
        # connection attribute
        try:
            self.shape = argtest.ofsequence("shape", shape, argtest.gt, 0, int)
        except TypeError:
            self.shape = (argtest.gt("shape", shape, 0, int),)

        # intermediate value
        size = math.prod(self.shape)

        # call superclass constructor
        Connection.__init__(
            self,
            synapse=synapse(
                size,
                step_time,
                0.0 if delay is None else delay,
                batch_size,
            ),
        )

        # call mixin constructor
        WeightBiasDelayMixin.__init__(
            self,
            weight=torch.rand(size),
            bias=(None if not bias else torch.rand(size)),
            delay=(None if delay is None else torch.zeros(size)),
            requires_grad=False,
        )

        # initialize parameters
        if weight_init:
            self.weight = weight_init(self.weight)

        if bias_init and self.biased:
            self.bias = bias_init(self.bias)

        if delay_init and self.delayedby is not None:
            self.delay = delay_init(self.delay)

    @property
    def inshape(self) -> tuple[int, ...]:
        r"""Shape of inputs to the connection, excluding the batch dimension.

        Returns:
            tuple[int]: shape of inputs to the connection.
        """
        return self.shape

    @property
    def outshape(self) -> tuple[int, ...]:
        r"""Shape of outputs from the connection, excluding the batch dimension.

        Returns:
            tuple[int]: shape of outputs from the connection.
        """
        return self.shape

    @property
    def selector(self) -> torch.Tensor | None:
        r"""Learned delays as a selector for synaptic currents and delays.

        Returns:
            torch.Tensor | None: delay selector if the connection has learnable delays.

        .. admonition:: Shape
            :class: tensorshape

            :math:`1 \times N \times 1`

            Where:
                * :math:`N` is the number of elements across input/output dimensions.
        """
        if self.delayedby is not None:
            delays = self.delay
        else:
            delays = torch.zeros_like(self.weight)

        return ein.rearrange(delays, "n -> 1 n 1").expand(self.batchsz, -1, -1)

    def like_input(self, data: torch.Tensor) -> torch.Tensor:
        r"""Reshapes data like synapse input to connection input.

        Args:
            data (torch.Tensor): data shaped like synapse input.

        Returns:
            torch.Tensor: reshaped data.

        .. admonition:: Shape
            :class: tensorshape

            ``data``:

            :math:`B \times N`

            ``return``:

            :math:`B \times N_0 \times \cdots`

            Where:
                * :math:`B` is the batch size.
                * :math:`N` is the number of elements across input/output dimensions.
                * :math:`N_0, \ldots` are the unbatched input/output dimensions.
        """
        return data.view(-1, *self.inshape)

    def like_bias(self, data: torch.Tensor) -> torch.Tensor:
        r"""Reshapes data like reduced postsynaptic receptive spikes to connection bias.

        Args:
            data (torch.Tensor): data shaped like reduced postsynaptic receptive spikes.

        Returns:
            torch.Tensor: reshaped data.

        .. admonition:: Shape
            :class: tensorshape

            ``data``:

            :math:`N`

            ``return``:

            :math:`N`

            Where:
                * :math:`N` is the number of elements across input/output dimensions.
        """
        return ein.rearrange(data, "n -> n")

    def like_synaptic(self, data: torch.Tensor) -> torch.Tensor:
        r"""Reshapes data like connection input to synapse input.

        Args:
            data (torch.Tensor): data shaped like connection input.

        Returns:
            torch.Tensor: reshaped data.

        .. admonition:: Shape
            :class: tensorshape

            ``data``:

            :math:`B \times N_0 \times \cdots`

            ``return``:

            :math:`B \times N`

            Where:
                * :math:`B` is the batch size.
                * :math:`N_0, \ldots` are the unbatched input/output dimensions.
                * :math:`N` is the number of elements across input/output dimensions.
        """
        return ein.rearrange(data, "b ... -> b (...)")

    def presyn_receptive(self, data: torch.Tensor) -> torch.Tensor:
        r"""Reshapes data like the synapse state for pre-post learning methods.

        Args:
            data (torch.Tensor): data shaped like output of :py:meth:`like_synaptic`.

        Returns:
            torch.Tensor: reshaped data.

        .. admonition:: Shape
            :class: tensorshape

            ``data``:

            :math:`B \times N \times [1]`

            ``return``:

            :math:`B \times N \times 1`

            Where:
                * :math:`B` is the batch size.
                * :math:`N` is the number of elements across output dimensions.
        """
        return ein.rearrange(data, "b n ... -> b n (...)")

    def postsyn_receptive(self, data: torch.Tensor) -> torch.Tensor:
        r"""Reshapes data like connection output for pre-post learning methods.

        Args:
            data (torch.Tensor): data shaped like output of :py:meth:`forward`.

        Returns:
            torch.Tensor: reshaped data.

        .. admonition:: Shape
            :class: tensorshape

            ``data``:

            :math:`B \times N_0 \times \cdots`

            ``return``:

            :math:`B \times N \times 1`

            Where:
                * :math:`B` is the batch size.
                * :math:`N_0, \ldots` are the unbatched input/output dimensions.
                * :math:`N` is the number of elements across input/output dimensions.
        """
        return ein.rearrange(data, "b ... -> b (...) 1")

    def forward(self, *inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""Generates connection output from inputs, after passing through the synapse.

        Outputs are determined as the learned linear transformation applied to synaptic
        currents, after new input is applied to the synapse, then reshaped to match
        :py:attr:`~Connection.batched_outshape`.

        Args:
            inputs (torch.Tensor): inputs to the connection.

        Returns:
            torch.Tensor: outputs from the connection.

        .. admonition:: Shape
            :class: tensorshape

            ``*inputs``:

            :math:`B \times N_0 \times \cdots`

            ``return``:

            :math:`B \times N_0 \times \cdots`

            Where:
                * :math:`B` is the batch size.
                * :math:`N_0, \ldots` are the unbatched input/output dimensions.

        Note:
            ``*inputs`` are reshaped using :py:meth:`like_synaptic` then passed to
            py:meth:`~Synapse.forward` of :py:attr:`~Connection.synapse`. Keyword arguments are
            also passed through.
        """
        # reshape inputs and perform synapse simulation
        res = self.synapse(
            *(self.like_synaptic(inp) for inp in inputs), **kwargs
        )  # B N

        if self.delayedby:
            res = ein.rearrange(self.syncurrent, "b n 1 -> b n")

        if self.biased:
            res = res * self.weight + self.bias
        else:
            res = res * self.weight

        return res.view(-1, *self.outshape)


class LinearLateral(WeightBiasDelayMixin, Connection):
    r"""Linear all-to-"all but one" connection.

    .. math::
        y = x \left(W^\intercal \odot (1 - I_N\right)) + b

    Args:
        shape (tuple[int, ...] | int): expected shape of input and output tensors,
            excluding batch dimension.
        step_time (float): length of a simulation time step, in :math:`\text{ms}`.
        synapse (SynapseConstructor): partial constructor for inner :py:class:`Synapse`.
        bias (bool, optional): if the connection should support
            learnable additive bias. Defaults to ``False``.
        delay (float | None, optional): maximum supported delay length, in
            :math:`\text{ms}`, excludes delays when ``None``. Defaults to ``None``.
        batch_size (int, optional): size of input batches for simulation. Defaults to ``1``.
        weight_init (OneToOne[torch.Tensor] | None, optional): initializer for weights.
            Defaults to ``None``.
        bias_init (OneToOne[torch.Tensor] | None, optional): initializer for biases.
            Defaults to ``None``.
        delay_init (OneToOne[torch.Tensor] | None, optional): initializer for delays.
            Defaults to ``None``.

    .. admonition:: Shape
        :class: tensorshape

        ``LinearDense.weight``, ``LinearDense.delay``:

        :math:`\prod(N_0, \ldots) \times \prod(N_0, \ldots)`

        ``LinearDense.bias``:

        :math:`(N_0 \cdot \cdots)`

    Note:
        If ``weight_init`` or ``bias_init`` are None, ``weight`` and ``bias`` are,
        respectively, initialized as uniform random values over the interval
        :math:`[0, 1)` using :py:func:`torch.rand`.

        If ``delay_init`` is None, ``delay`` is initialized as zeros using
        :py:func:`torch.rand`.

    Note:
        Weights and delays are stored internally like in :py:class:`LinearDense`, but on
        assignment by :py:attr:`weight` and creation are masked by a tensor
        :math:`1 - I_N`, where :math:`N = (N_0 \cdot \cdots)`.
    """

    def __init__(
        self,
        shape: tuple[int, ...] | int,
        step_time: float,
        *,
        synapse: SynapseConstructor,
        bias: bool = False,
        delay: float | None = None,
        batch_size: int = 1,
        weight_init: OneToOne | None = None,
        bias_init: OneToOne | None = None,
        delay_init: OneToOne | None = None,
    ):
        # connection attribute
        try:
            self.shape = argtest.ofsequence("shape", shape, argtest.gt, 0, int)
        except TypeError:
            self.shape = (argtest.gt("shape", shape, 0, int),)

        # intermediate value
        size = math.prod(self.shape)

        # call superclass constructor
        Connection.__init__(
            self,
            synapse=synapse(
                size,
                step_time,
                0.0 if delay is None else delay,
                batch_size,
            ),
        )

        # register buffer
        self.register_buffer("mask", 1 - torch.eye(size), persistent=False)

        # call mixin constructor
        WeightBiasDelayMixin.__init__(
            self,
            weight=(torch.rand(size, size) * self.mask),
            bias=(None if not bias else torch.rand(size)),
            delay=(None if delay is None else torch.zeros(size, size) * self.mask),
            requires_grad=False,
        )

        # initialize parameters
        if weight_init:
            self.weight = weight_init(self.weight)

        if bias_init and self.biased:
            self.bias = bias_init(self.bias)

        if delay_init and self.delayedby is not None:
            self.delay = delay_init(self.delay)

    @property
    def weight(self) -> torch.Tensor:
        r"""Learnable connection weights.

        Args:
            value (torch.Tensor): new weights.

        Returns:
            torch.Tensor: present weights.

        Note:
            Setter masks weights before assignment.
        """
        return WeightBiasDelayMixin.weight.fget(self)

    @weight.setter
    def weight(self, value: torch.Tensor) -> None:
        WeightBiasDelayMixin.weight.fset(self, value * self.mask)

    @property
    def delay(self) -> torch.Tensor | None:
        r"""Learnable delays of the connection.

        Args:
            value (torch.Tensor): new delays.

        Returns:
            torch.Tensor | None: current delays, if the connection has any.

        Note:
            Setter masks delays before assignment.
        """
        return WeightBiasDelayMixin.delay.fget(self)

    @delay.setter
    def delay(self, value: torch.Tensor) -> None:
        WeightBiasDelayMixin.delay.fset(self, value * self.mask)

    @property
    def inshape(self) -> tuple[int, ...]:
        r"""Shape of inputs to the connection, excluding the batch dimension.

        Returns:
            tuple[int]: shape of inputs to the connection.
        """
        return self.shape

    @property
    def outshape(self) -> tuple[int, ...]:
        r"""Shape of outputs from the connection, excluding the batch dimension.

        Returns:
            tuple[int]: shape of outputs from the connection.
        """
        return self.shape

    @property
    def selector(self) -> torch.Tensor | None:
        r"""Learned delays as a selector for synaptic currents and delays.

        Returns:
            torch.Tensor | None: delay selector if the connection has learnable delays.

        .. admonition:: Shape
            :class: tensorshape

            :math:`B \times N \times N`

            Where:
                * :math:`B` is the batch size.
                * :math:`N` is the number of elements across input/output dimensions.
        """
        return LinearDense.selector.fget(self)

    def like_input(self, data: torch.Tensor) -> torch.Tensor:
        r"""Reshapes data like synapse input to connection input.

        Args:
            data (torch.Tensor): data shaped like synapse input.

        Returns:
            torch.Tensor: reshaped data.

        .. admonition:: Shape
            :class: tensorshape

            ``data``:

            :math:`B \times N`

            ``return``:

            :math:`B \times N_0 \times \cdots`

            Where:
                * :math:`B` is the batch size.
                * :math:`N` is the number of elements across input/output dimensions.
                * :math:`N_0, \ldots` are the unbatched input/output dimensions.
        """
        return LinearDense.like_input(self, data)

    def like_bias(self, data: torch.Tensor) -> torch.Tensor:
        r"""Reshapes data like reduced postsynaptic receptive spikes to connection bias.

        Args:
            data (torch.Tensor): data shaped like reduced postsynaptic receptive spikes.

        Returns:
            torch.Tensor: reshaped data.

        .. admonition:: Shape
            :class: tensorshape

            ``data``:

            :math:`N \times 1`

            ``return``:

            :math:`N`

            Where:
                * :math:`N` is the number of elements across input/output dimensions.
        """
        return ein.rearrange(data, "n 1 -> n")

    def like_synaptic(self, data: torch.Tensor) -> torch.Tensor:
        r"""Reshapes data like connection input to synapse input.

        Args:
            data (torch.Tensor): data shaped like connection input.

        Returns:
            torch.Tensor: reshaped data.

        .. admonition:: Shape
            :class: tensorshape

            ``data``:

            :math:`B \times N_0 \times \cdots`

            ``return``:

            :math:`B \times N`

            Where:
                * :math:`B` is the batch size.
                * :math:`N_0, \ldots` are the unbatched input/output dimensions.
                * :math:`N` is the number of elements across input/output dimensions.
        """
        return LinearDense.like_synaptic(self, data)

    def presyn_receptive(self, data: torch.Tensor) -> torch.Tensor:
        r"""Reshapes data like the synapse state for pre-post learning methods.

        Args:
            data (torch.Tensor): data shaped like output of :py:meth:`like_synaptic`.

        Returns:
            torch.Tensor: reshaped data.

        .. admonition:: Shape
            :class: tensorshape

            ``data``:

            :math:`B \times N \times [N]`

            ``return``:

            :math:`B \times N \times N \times 1`

            or

            :math:`B \times 1 \times N \times 1`

            Where:
                * :math:`B` is the batch size.
                * :math:`N` is the number of elements across input/output dimensions.
        """
        return LinearDense.presyn_receptive(self, data)

    def postsyn_receptive(self, data: torch.Tensor) -> torch.Tensor:
        r"""Reshapes data like connection output for pre-post learning methods.

        Args:
            data (torch.Tensor): data shaped like output of :py:meth:`forward`.

        Returns:
            torch.Tensor: reshaped data.

        .. admonition:: Shape
            :class: tensorshape

            ``data``:

            :math:`B \times N_0 \times \cdots`

            ``return``:

            :math:`B \times N \times 1 \times 1`

            Where:
                * :math:`B` is the batch size.
                * :math:`N_0, \ldots` are the unbatched input/output dimensions.
                * :math:`N` is the number of elements across input/output dimensions.
        """
        return LinearDense.postsyn_receptive(self, data)

    def forward(self, *inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""Generates connection output from inputs, after passing through the synapse.

        Outputs are determined as the learned linear transformation applied to synaptic
        currents, after new input is applied to the synapse, then reshaped to match
        :py:attr:`~Connection.batched_outshape`.

        Args:
            *inputs (torch.Tensor): inputs to the connection.

        Returns:
            torch.Tensor: outputs from the connection.

        .. admonition:: Shape
            :class: tensorshape

            ``*inputs``:

            :math:`B \times N_0 \times \cdots`

            ``return``:

            :math:`B \times N_0 \times \cdots`

            Where:
                * :math:`B` is the batch size.
                * :math:`N_0, \ldots` are the unbatched input/output dimensions.

        Note:
            ``*inputs`` are reshaped using :py:meth:`like_synaptic` then passed to
            :py:meth:`~Synapse.forward` of :py:attr:`~Connection.synapse`.
            Keyword arguments are also passed through.
        """
        return LinearDense.forward(self, *inputs, **kwargs)
