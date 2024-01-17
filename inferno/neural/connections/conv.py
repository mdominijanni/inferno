import einops as ein
from ..base import Connection, SynapseConstructor
from ..base import Synapse  # noqa:F401
from inferno._internal import numeric_limit
from inferno.typing import OneToOne
import math
import torch
import torch.nn.functional as F
from .mixins import WeightBiasDelayMixin


class Conv2D(WeightBiasDelayMixin, Connection):
    r"""Convolutional connection along two spatial dimensions with separate input planes.

    Args:
        height (int): height of the expected inputs.
        width (int): width of the expected inputs.
        channels (int): number of channels in the input tensor.
        filters (int): number of convolutional filters (channels of the output tensor).
        step_time (float): length of a simulation time step, in :math:`\mathrm{ms}`.
        kernel (int | tuple[int, int]): size of the convolution kernel.
        stride (int | tuple[int, int], optional): stride of the convolution.
            Defaults to 1.
        padding (int | tuple[int, int], optional): amount of zero padding added to
            height and width. Defaults to 0.
        dilation (int | tuple[int, int], optional): dilation of the convolution.
            Defaults to 1.
        synapse (SynapseConstructor): partial constructor for inner :py:class:`Synapse`.
        bias (bool, optional): if the connection should support
            learnable additive bias. Defaults to False.
        delay (float | None, optional): maximum supported delay length, in
            :math:`\mathrm{ms}`, excludes delays when None. Defaults to None.
        batch_size (int, optional): size of input batches for simualtion. Defaults to 1.
        weight_init (OneToOne[torch.Tensor] | None, optional): initializer for weights.
            Defaults to None.
        bias_init (OneToOne[torch.Tensor] | None, optional): initializer for biases.
            Defaults to None.
        delay_init (OneToOne[torch.Tensor] | None, optional): initializer for delays.
            Defaults to None.

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
        When ``delay`` is None, no ``delay_`` parameter is created and altering the
        maximum delay of :py:attr:`synapse` will have no effect. Setting to 0 will
        create and register a ``delay_`` parameter but not use delays unless it is
        later changed.

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
        self.height = numeric_limit("`height`", height, 0, "gt", int)
        self.width = numeric_limit("`width`", width, 0, "gt", int)
        self.channels = numeric_limit("`channels`", channels, 0, "gt", int)
        self.filters = numeric_limit("`filters`", filters, 0, "gt", int)

        try:
            self.kernel = (
                numeric_limit("`kernel[0]`", kernel[0], 0, "gt", int),
                numeric_limit("`kernel[1]`", kernel[1], 0, "gt", int),
            )
        except TypeError:
            self.kernel = (
                numeric_limit("`kernel`", kernel, 0, "gt", int),
                numeric_limit(int(kernel)),
            )
        except IndexError:
            raise ValueError(
                "nonscalar `kernel` must be of length 2, is of "
                f"length {len(kernel)}."
            )

        try:
            self.stride = (
                numeric_limit("`stride[0]`", stride[0], 0, "gt", int),
                numeric_limit("`stride[1]`", stride[1], 0, "gt", int),
            )
        except TypeError:
            self.stride = (
                numeric_limit("`stride`", stride, 0, "gt", int),
                numeric_limit(int(stride)),
            )
        except IndexError:
            raise ValueError(
                "nonscalar `stride` must be of length 2, is of "
                f"length {len(stride)}."
            )

        try:
            self.padding = (
                numeric_limit("`padding[0]`", padding[0], 0, "gte", int),
                numeric_limit("`padding[1]`", padding[1], 0, "gte", int),
            )
        except TypeError:
            self.padding = (
                numeric_limit("`padding`", padding, 0, "gte", int),
                numeric_limit(int(padding)),
            )
        except IndexError:
            raise ValueError(
                "nonscalar `padding` must be of length 2, is of "
                f"length {len(padding)}."
            )

        try:
            self.dilation = (
                numeric_limit("`dilation[0]`", dilation[0], 0, "gt", int),
                numeric_limit("`dilation[1]`", dilation[1], 0, "gt", int),
            )
        except TypeError:
            self.dilation = (
                numeric_limit("`dilation`", dilation, 0, "gt", int),
                numeric_limit(int(dilation)),
            )
        except IndexError:
            raise ValueError(
                "nonscalar `dilation` must be of length 2, is of "
                f"length {len(dilation)}."
            )

        self.outheight, self.outwidth = (
            math.floor(
                (size + 2 * padding[d] - dilation[d] * (kernel[d] - 1) - 1) / stride[d]
                + 1
            )
            for d, size in enumerate((self.height, self.width))
        )

        # call superclass constructor
        Connection.__init__(
            self,
            synapse=synapse(
                (channels * math.prod(kernel), self.outheight * self.outwidth),
                step_time,
                0.0 if delay is None else delay,
                batch_size,
            ),
        )

        # call mixin constructor
        WeightBiasDelayMixin.__init__(
            weight=torch.rand(filters, channels, *kernel),
            bias=(None if not bias else torch.rand(filters)),
            delay=(None if delay is None else torch.zeros(filters, channels, *kernel)),
            requires_grad=False,
        )

        # initialize parameters
        if weight_init:
            self.weight = weight_init(self.weight)

        if bias_init and bias:
            self.bias = bias_init(self.bias)

        if delay_init and delay:
            self.delay = delay_init(self.delay)

    @property
    def inshape(self) -> tuple[int, int, int]:
        r"""Shape of inputs to the connection, excluding the batch dimension.

        Returns:
            tuple[int]: shape of inputs to the connection.

        Raises:
            NotImplementedError: ``inshape`` must be implemented by the subclass.
        """
        return (self.channels, self.height, self.width)

    @property
    def outshape(self) -> tuple[int, int, int]:
        return (self.filters, self.outheight, self.outwidth)

    @property
    def selector(self) -> torch.Tensor | None:
        r"""Learned delays as a selector for synaptic currents and delays.

        Returns:
            torch.Tensor | None: delay selector if the connection has learnable delays.

        .. admonition:: Shape
            :class: tensorshape

            :math:`B \times (C \cdot kH \cdot kW) \times (H_\mathrm{out}
            \cdot W_\mathrm{out}) \times F`

            Where:
                * :math:`B` is the batch size.
                * :math:`C` is the number of input channels.
                * :math:`kH` is the kernel height.
                * :math:`kW` is the kernel width.
                * :math:`H_\mathrm{out}` is the output height.
                * :math:`W_\mathrm{out}` is the output width.
                * :math:`F` is the number of filters (output channels).
        """
        return ein.rearrange(self.delay, "f c h w -> 1 (c h w) 1 f").expand(
            self.bsize, -1, self.synapse.shape[-1], -1
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

            :math:`B \times (C \cdot kH \cdot kW) \times (H_\mathrm{out}
            \cdot W_\mathrm{out})`

            ``return``:

            :math:`B \times C \times H \times W`

            Where:
                * :math:`B` is the batch size.
                * :math:`C` is the number of input channels.
                * :math:`kH` is the kernel height.
                * :math:`kW` is the kernel width.
                * :math:`H_\mathrm{out}` is the output height.
                * :math:`W_\mathrm{out}` is the output width.
                * :math:`H` is the input height.
                * :math:`W` is the input width.
        """
        return F.fold(
            data,
            (self.outheight, self.outwidth),
            self.kernel,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        ) / F.fold(
            torch.ones_like(data),
            (self.outheight, self.outwidth),
            self.kernel,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        )

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

            :math:`B \times (C \cdot kH \cdot kW) \times (H_\mathrm{out}
            \cdot W_\mathrm{out})`

            Where:
                * :math:`B` is the batch size.
                * :math:`C` is the number of input channels.
                * :math:`H` is the input height.
                * :math:`W` is the input width.
                * :math:`kH` is the kernel height.
                * :math:`kW` is the kernel width.
                * :math:`H_\mathrm{out}` is the output height.
                * :math:`W_\mathrm{out}` is the output width.
        """
        return F.unfold(
            data,
            self.kernel,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        )

    def presyn_receptive(self, data: torch.Tensor) -> torch.Tensor:
        r"""Reshapes data like the synapse state for pre-post learning methods.

        Args:
            data (torch.Tensor): data shaped like :py:attr:`syncurrent`
                or :py:attr:`synspike`.

        Raises:
            NotImplementedError: ``presyn_receptive`` must be
            implemented by the subclass.

        Returns:
            torch.Tensor: reshaped data.

        .. admonition:: Shape
            :class: tensorshape

            ``data``:

            :math:`B \times (C \cdot kH \cdot kW) \times (H_\mathrm{out}
            \cdot W_\mathrm{out}) \times [F]`

            ``return``:

            :math:`B \times F \times C \times kH \times kW \times
            (H_\mathrm{out} \cdot W_\mathrm{out})`

            or

            :math:`B \times 1 \times C \times kH \times kW \times
            (H_\mathrm{out} \cdot W_\mathrm{out})`

            Where:
                * :math:`B` is the batch size.
                * :math:`C` is the number of input channels.
                * :math:`kH` is the kernel height.
                * :math:`kW` is the kernel width.
                * :math:`H_\mathrm{out}` is the output height.
                * :math:`W_\mathrm{out}` is the output width.
                * :math:`F` is the number of filters (output channels).
        """
        match data.ndim:
            case 4:
                return ein.rearrange(
                    data,
                    "b n l f -> b f c kh kw l",
                    c=self.channels,
                    kh=self.kernel[0],
                    kw=self.kernel[1],
                )
            case 3:
                return ein.rearrange(
                    data,
                    "b n l -> b 1 c kh kw l",
                    c=self.channels,
                    kh=self.kernel[0],
                    kw=self.kernel[1],
                )
            case _:
                raise RuntimeError(
                    f"`data` must have 3 or 4 dimensions, has {data.ndim} dimensions."
                )

    def postsyn_receptive(self, data: torch.Tensor) -> torch.Tensor:
        r"""Reshapes data like the output for pre-post learning methods.

        Args:
            data (torch.Tensor): data shaped like connection output.

        Raises:
            NotImplementedError: ``postsyn_receptive`` must be
            implemented by the subclass.

        Returns:
            torch.Tensor: reshaped data.

        .. admonition:: Shape
            :class: tensorshape

            ``data``:

            :math:`B \times F \times H_\mathrm{out} \times W_\mathrm{out}`

            ``return``:

            :math:`B \times F \times 1 \times 1 \times 1 \times
            (H_\mathrm{out} \cdot W_\mathrm{out})`

            Where:
                * :math:`B` is the batch size.
                * :math:`F` is the number of filters (output channels).
                * :math:`H_\mathrm{out}` is the output height.
                * :math:`W_\mathrm{out}` is the output width.
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

        Note:
            ``*inputs`` can either be a single tensor representing the input current,
            or it can be two tensors where the first is the spikes and the second is
            an override for the currents. See documentation for the corresponding
            :py:meth:`Synapse.forward` method for details.

        Note:
            Keyword arguments are passed to :py:meth:`Synapse.forward`.
        """
        # reshape inputs and perform synapse simulation
        data = self.synapse(
            *(self.like_synaptic(inp) for inp in inputs), **kwargs
        )  # B N L

        if self.delayed:
            data = ein.rearrange(self.syncurrent, "b n l f -> b f n l")
            kernel = ein.rearrange(self.weight, "f c h w -> f 1 (c h w)")  # F 1 L

            res = ein.rearrange(
                torch.matmul(kernel, data),
                "b f 1 (oh ow) -> b f oh ow",
                oh=self.outshape[0],
                ow=self.outshape[1],
            )
        else:
            kernel = ein.rearrange(self.weight, "f c h w -> f (c h w)")  # F L

            res = ein.rearrange(
                torch.matmul(kernel, data),
                "b f (oh ow) -> b f oh ow",
                oh=self.outshape[0],
                ow=self.outshape[1],
            )

        if self.biased:
            return res + ein.rearrange(self.bias, "f -> 1 f 1 1")
        else:
            return res
