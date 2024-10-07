from .mixins import WeightBiasDelayMixin
from ..base import Connection, SynapseConstructor
from ...core import ones
from ..._internal import argtest
from ...types import OneToOne
import einops as ein
import math
import torch
import torch.nn.functional as F


class Conv2D(WeightBiasDelayMixin, Connection):
    r"""Convolutional connection along two spatial dimensions with separate input planes.

    Args:
        height (int): height of the expected inputs.
        width (int): width of the expected inputs.
        channels (int): number of channels in the input tensor.
        filters (int): number of convolutional filters (channels of the output tensor).
        step_time (float): length of a simulation time step, in :math:`\text{ms}`.
        kernel (int | tuple[int, int]): size of the convolution kernel.
        stride (int | tuple[int, int], optional): stride of the convolution.
            Defaults to ``1``.
        padding (int | tuple[int, int], optional): amount of zero padding added to
            height and width. Defaults to ``0``.
        dilation (int | tuple[int, int], optional): dilation of the convolution.
            Defaults to ``1``.
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

        ``Conv2D.weight``, ``Conv2D.delay``:

        :math:`F \times C \times H \times W`

        ``Conv2D.bias``:

        :math:`F`

        Where:
            * :math:`F` is the number of filters (output channels).
            * :math:`C` is the number of input channels.
            * :math:`kH` is the kernel height.
            * :math:`kW` is the kernel width.

    Note:
        When ``delay`` is ``None``, no ``delay_`` parameter is created and altering the
        maximum delay of :py:attr:`~Connection.synapse` will have no effect. Setting to 0 will
        create and register a ``delay_`` parameter but not use delays unless it is
        later changed.

    Note:
        If ``weight_init`` or ``bias_init`` are ``None``, ``weight`` and ``bias`` are,
        respectively, initialized as uniform random values over the interval
        :math:`[0, 1)` using :py:func:`torch.rand`.

        If ``delay_init`` is ``None``, ``delay`` is initialized as zeros using
        :py:func:`torch.rand`.

    Tip:
        The added padding is applied after the synapse. Inputs must still be of uniform
        size. Only zero padding is supported, if another type of padding is required,
        it should be performed before being inputted to the connection.
    """

    def __init__(
        self,
        height: int,
        width: int,
        channels: int,
        filters: int,
        step_time: float,
        kernel: int | tuple[int, int],
        *,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        synapse: SynapseConstructor,
        bias: bool = False,
        delay: float | None = None,
        batch_size: int = 1,
        weight_init: OneToOne[torch.Tensor] | None = None,
        bias_init: OneToOne[torch.Tensor] | None = None,
        delay_init: OneToOne[torch.Tensor] | None = None,
    ):
        # connection attributes
        self.height = argtest.gt("height", height, 0, int)
        self.width = argtest.gt("width", width, 0, int)
        self.channels = argtest.gt("channels", channels, 0, int)
        self.filters = argtest.gt("filters", filters, 0, int)

        try:
            self.kernel = argtest.ofsequence("kernel", kernel, argtest.gt, 0, int)
        except TypeError:
            k = argtest.gt("kernel", kernel, 0, int)
            self.kernel = (k, k)
        except IndexError:
            raise ValueError(
                "nonscalar 'kernel' must be of length 2, is of " f"length {len(kernel)}"
            )

        try:
            self.stride = argtest.ofsequence("stride", stride, argtest.gt, 0, int)
        except TypeError:
            s = argtest.gt("stride", stride, 0, int)
            self.stride = (s, s)
        except IndexError:
            raise ValueError(
                "nonscalar 'stride' must be of length 2, is of " f"length {len(stride)}"
            )

        try:
            self.padding = argtest.ofsequence("padding", padding, argtest.gte, 0, int)
        except TypeError:
            p = argtest.gte("padding", padding, 0, int)
            self.padding = (p, p)
        except IndexError:
            raise ValueError(
                "nonscalar 'padding' must be of length 2, is of "
                f"length {len(padding)}"
            )

        try:
            self.dilation = argtest.ofsequence("dilation", dilation, argtest.gt, 0, int)
        except TypeError:
            d = argtest.gt("dilation", dilation, 0, int)
            self.dilation = (d, d)
        except IndexError:
            raise ValueError(
                "nonscalar 'dilation' must be of length 2, is of "
                f"length {len(dilation)}"
            )

        self.outheight, self.outwidth = (
            math.floor(
                (
                    size
                    + 2 * self.padding[d]
                    - self.dilation[d] * (self.kernel[d] - 1)
                    - 1
                )
                / self.stride[d]
                + 1
            )
            for d, size in enumerate((self.height, self.width))
        )

        # call superclass constructor
        Connection.__init__(
            self,
            synapse=synapse(
                (
                    self.channels * math.prod(self.kernel),
                    self.outheight * self.outwidth,
                ),
                step_time,
                0.0 if delay is None else delay,
                batch_size,
            ),
        )

        # call mixin constructor
        WeightBiasDelayMixin.__init__(
            self,
            weight=torch.rand(self.filters, self.channels, *self.kernel),
            bias=(None if not bias else torch.rand(self.filters)),
            delay=(
                None
                if delay is None
                else torch.zeros(self.filters, self.channels, *self.kernel)
            ),
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
    def inshape(self) -> tuple[int, int, int]:
        r"""Shape of inputs to the connection, excluding the batch dimension.

        Returns:
            tuple[int]: shape of inputs to the connection.

        Note:
            Resulting tuple will be:

            :math:`(C, H, W)`

            Where:
                * :math:`C` is the number of input channels.
                * :math:`H` is the input height.
                * :math:`W` is the input width.
        """
        return (self.channels, self.height, self.width)

    @property
    def outshape(self) -> tuple[int, int, int]:
        r"""Shape of outputs from the connection, excluding the batch dimension.

        Returns:
            tuple[int]: shape of outputs from the connection.

        Note:
            Resulting tuple will be:

            :math:`(F, H_\text{out}, W_\text{out})`

            Where:

            .. math::

                \begin{align*}
                    H_\text{out} &= \left\lfloor \frac{H + 2 \times p_H - d_H
                    \times (k_H - 1) - 1}{s_H} + 1 \right\rfloor \\
                    W_\text{out} &= \left\lfloor \frac{W + 2 \times p_W - d_W
                    \times (k_W - 1) - 1}{s_W} + 1 \right\rfloor
                \end{align*}

            And:
                * :math:`F` is the number of filters (output channels).
                * :math:`(H, W)` are the input height and width.
                * :math:`(pH, pW)` are the per-side padding height and width.
                * :math:`(dH, dW)` are the dilation height and width.
                * :math:`(kH, kW)` are the kernel height and width.
                * :math:`(sH, sW)` are the stride height and width.
        """
        return (self.filters, self.outheight, self.outwidth)

    @property
    def selector(self) -> torch.Tensor | None:
        r"""Learned delays as a selector for synaptic currents and delays.

        Returns:
            torch.Tensor | None: delay selector if the connection has learnable delays.

        .. admonition:: Shape
            :class: tensorshape

            :math:`B \times (C \cdot kH \cdot kW) \times (H_\text{out}
            \cdot W_\text{out}) \times F`

            Where:
                * :math:`B` is the batch size.
                * :math:`C` is the number of input channels.
                * :math:`kH` is the kernel height.
                * :math:`kW` is the kernel width.
                * :math:`H_\text{out}` is the output height.
                * :math:`W_\text{out}` is the output width.
                * :math:`F` is the number of filters (output channels).

        Caution:
            This operation relies upon :py:meth:`torch.Tensor.expand`, and
            consequentially multiple elements may reference the same underlying
            memory.
        """
        if self.delayedby is not None:
            delays = self.delay
        else:
            delays = torch.zeros_like(self.weight)

        return ein.rearrange(delays, "f c h w -> 1 (c h w) 1 f").expand(
            self.batchsz, -1, self.synapse.shape[-1], -1
        )

    def like_input(self, data: torch.Tensor) -> torch.Tensor:
        r"""Reshapes data like synapse input to connection input.

        Args:
            data (torch.Tensor): data shaped like synapse input.

        Returns:
            torch.Tensor: reshaped data.

        .. admonition:: Shape
            :class: tensorshape

            ``data``:

            :math:`B \times (C \cdot kH \cdot kW) \times (H_\text{out}
            \cdot W_\text{out})`

            ``return``:

            :math:`B \times C \times H \times W`

            Where:
                * :math:`B` is the batch size.
                * :math:`C` is the number of input channels.
                * :math:`kH` is the kernel height.
                * :math:`kW` is the kernel width.
                * :math:`H_\text{out}` is the output height.
                * :math:`W_\text{out}` is the output width.
                * :math:`H` is the input height.
                * :math:`W` is the input width.

        Note:
            PyTorch's :py:func:`~torch.nn.functional.fold` and
            :py:func:`~torch.nn.functional.unfold` are only implemented for floating
            point values. Intermediate casting to the same datatype as connection
            weights will be performed if required.
        """
        if torch.is_floating_point(data):
            return F.fold(
                data,
                (self.height, self.width),
                self.kernel,
                dilation=self.dilation,
                padding=self.padding,
                stride=self.stride,
            ) / F.fold(
                torch.ones_like(data),
                (self.height, self.width),
                self.kernel,
                dilation=self.dilation,
                padding=self.padding,
                stride=self.stride,
            )
        else:
            return (
                F.fold(
                    data.to(dtype=self.weight.dtype),
                    (self.height, self.width),
                    self.kernel,
                    dilation=self.dilation,
                    padding=self.padding,
                    stride=self.stride,
                )
                / F.fold(
                    ones(data, dtype=self.weight.dtype),
                    (self.height, self.width),
                    self.kernel,
                    dilation=self.dilation,
                    padding=self.padding,
                    stride=self.stride,
                )
            ).to(dtype=data.dtype)

    def like_bias(self, data: torch.Tensor) -> torch.Tensor:
        r"""Reshapes data like reduced postsynaptic receptive spikes to connection bias.

        Args:
            data (torch.Tensor): data shaped like reduced postsynaptic receptive spikes.

        Returns:
            torch.Tensor: reshaped data.

        .. admonition:: Shape
            :class: tensorshape

            ``data``:

            :math:`F \times 1 \times 1 \times 1`

            ``return``:

            :math:`F`

            Where:
                * :math:`F` is the number of filters (output channels).
        """
        return ein.rearrange(data, "f 1 1 1 -> f")

    def like_synaptic(self, data: torch.Tensor) -> torch.Tensor:
        r"""Reshapes data like connection input to synapse input.

        Args:
            data (torch.Tensor): data shaped like connection input.

        Returns:
            torch.Tensor: reshaped data.

        .. admonition:: Shape
            :class: tensorshape

            ``data``:

            :math:`B \times C \times H \times W`

            ``return``:

            :math:`B \times (C \cdot kH \cdot kW) \times (H_\text{out}
            \cdot W_\text{out})`

            Where:
                * :math:`B` is the batch size.
                * :math:`C` is the number of input channels.
                * :math:`H` is the input height.
                * :math:`W` is the input width.
                * :math:`kH` is the kernel height.
                * :math:`kW` is the kernel width.
                * :math:`H_\text{out}` is the output height.
                * :math:`W_\text{out}` is the output width.

        Note:
            PyTorch's :py:func:`~torch.nn.functional.fold` and
            :py:func:`~torch.nn.functional.unfold` are only implemented for floating
            point values. Intermediate casting to the same datatype as connection
            weights will be performed if required.
        """
        if torch.is_floating_point(data):
            return F.unfold(
                data,
                self.kernel,
                dilation=self.dilation,
                padding=self.padding,
                stride=self.stride,
            )
        else:
            return F.unfold(
                data.to(dtype=self.weight.dtype),
                self.kernel,
                dilation=self.dilation,
                padding=self.padding,
                stride=self.stride,
            ).to(dtype=data.dtype)

    def presyn_receptive(self, data: torch.Tensor) -> torch.Tensor:
        r"""Reshapes data like the synapse state for pre-post learning methods.

        Args:
            data (torch.Tensor): data shaped like output of :py:meth:`like_synaptic`.

        Returns:
            torch.Tensor: reshaped data.

        .. admonition:: Shape
            :class: tensorshape

            ``data``:

            :math:`B \times (C \cdot kH \cdot kW) \times (H_\text{out}
            \cdot W_\text{out}) \times [F]`

            ``return``:

            :math:`B \times F \times C \times kH \times kW \times
            (H_\text{out} \cdot W_\text{out})`

            or

            :math:`B \times 1 \times C \times kH \times kW \times
            (H_\text{out} \cdot W_\text{out})`

            Where:
                * :math:`B` is the batch size.
                * :math:`C` is the number of input channels.
                * :math:`kH` is the kernel height.
                * :math:`kW` is the kernel width.
                * :math:`H_\text{out}` is the output height.
                * :math:`W_\text{out}` is the output width.
                * :math:`F` is the number of filters (output channels).
        """
        return ein.rearrange(
            data,
            "b n l ... -> b (...) c kh kw l",
            c=self.channels,
            kh=self.kernel[0],
            kw=self.kernel[1],
        )

    def postsyn_receptive(self, data: torch.Tensor) -> torch.Tensor:
        r"""Reshapes data like connection output for pre-post learning methods.

        Args:
            data (torch.Tensor): data shaped like output of :py:meth:`forward`.

        Returns:
            torch.Tensor: reshaped data.

        .. admonition:: Shape
            :class: tensorshape

            ``data``:

            :math:`B \times F \times H_\text{out} \times W_\text{out}`

            ``return``:

            :math:`B \times F \times 1 \times 1 \times 1 \times
            (H_\text{out} \cdot W_\text{out})`

            Where:
                * :math:`B` is the batch size.
                * :math:`F` is the number of filters (output channels).
                * :math:`H_\text{out}` is the output height.
                * :math:`W_\text{out}` is the output width.
        """
        return ein.rearrange(data, "b f oh ow -> b f 1 1 1 (oh ow)")

    def forward(self, *inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""Generates connection output from inputs, after passing through the synapse.

        Outputs are determined as the learned two-dimensional convolution applied to
        synaptic currents, after new input is applied to the synapse.

        Args:
            *inputs (torch.Tensor): inputs to the connection.

        Returns:
            torch.Tensor: outputs from the connection.

        .. admonition:: Shape
            :class: tensorshape

            ``*inputs``:

            :math:`B \times C \times H \times W`

            ``return``:

            :math:`B \times F \times H_\text{out} \times W_\text{out}`

            Where:
                * :math:`B` is the batch size.
                * :math:`C` is the number of input channels.
                * :math:`H` is the input height.
                * :math:`W` is the input width.
                * :math:`F` is the number of filters (output channels).
                * :math:`H_\text{out}` is the output height.
                * :math:`W_\text{out}` is the output width.

        Note:
            ``*inputs`` are reshaped using :py:meth:`like_synaptic` then passed to
            py:meth:`Synapse.forward` of :py:attr:`~Connection.synapse`. Keyword arguments are
            also passed through.

        See Also:
            The formulae for output height and width are detailed in the documentation
            for :py:attr:`outshape`.
        """
        # reshape inputs and perform synapse simulation
        res = self.synapse(
            *(self.like_synaptic(inp) for inp in inputs), **kwargs
        )  # B N L

        kernel = ein.rearrange(self.weight, "f c h w -> f (c h w)")  # F N

        if self.delayedby:
            res = ein.rearrange(self.syncurrent, "b n l f -> b f n l")

            res = ein.rearrange(
                ein.einsum(kernel, res, "f n, b f n l -> b f l"),
                "b f (oh ow) -> b f oh ow",
                oh=self.outheight,
                ow=self.outwidth,
            )
        else:
            res = ein.rearrange(
                torch.matmul(kernel, res),
                "b f (oh ow) -> b f oh ow",
                oh=self.outheight,
                ow=self.outwidth,
            )

        if self.biased:
            return res + ein.rearrange(self.bias, "f -> 1 f 1 1")
        else:
            return res
