import einops as ein
from inferno.typing import OneToOne
import math
import torch
import torch.nn.functional as F
from .. import Connection, SynapseConstructor
from ._mixins import WeightBiasDelayMixin


class Conv2D(WeightBiasDelayMixin, Connection):
    r"""Convolutional connection along two spatial dimensions.

    Args:
        height (int): height of the input tensor.
        width (int): width of the input tensor.
        channels (int): number of channels in the input tensor.
        filters (int): number of convolutional filters, channels of the output tensor.
        step_time (float): length of a simulation time step, in :math:`\mathrm{ms}`.
        kernel (int | tuple[int, int]): size of the convolution kernel.
        stride (int | tuple[int, int], optional): stride of the convolution. Defaults to 1.
        padding (int | tuple[int, int], optional): amount of zero padding added to height and width. Defaults to 0.
        dilation (int | tuple[int, int], optional): dilation of the convolution. Defaults to 1.
        synapse (SynapseConstructor): partial constructor for inner :py:class:`~inferno.neural.Synapse`.
        batch_size (int, optional): size of input batches for simualtion. Defaults to 1.
        bias (bool, optional): if the connection should include learnable additive bias. Defaults to False.
        delay (float | None, optional): length of time the connection should support delays for. Defaults to None.
        weight_init (OneToOne | None, optional): initializer for weights. Defaults to None.
        bias_init (OneToOne | None, optional): initializer for biases. Defaults to None.
        delay_init (OneToOne | None, optional): initializer for delays. Defaults to None.

    Raises:
        ValueError: ``height`` must be positive.
        ValueError: ``width`` must be positive.
        ValueError: ``channels`` must be positive.
        ValueError: ``filters`` must be positive.
        ValueError: ``kernel`` must be a scalar or 2-tuple with positive values.
        ValueError: ``stride`` must be a scalar or 2-tuple with positive values.
        ValueError: ``padding`` must be a scalar or 2-tuple with non-negative values.
        ValueError: ``dilation`` must be a scalar or 2-tuple with positive values.
        ValueError: ``step_time`` must be positive.
        ValueError: ``batch_size`` must be positive.
        ValueError: ``delay`` must be non-negative if not None.
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
        batch_size: int = 1,
        bias: bool = False,
        delay: float | None = None,
        weight_init: OneToOne | None = None,
        bias_init: OneToOne | None = None,
        delay_init: OneToOne | None = None,
    ):
        # check variables and cast accordingly
        # shape variables
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
            kernel = (int(kernel), int(kernel))
            if kernel[0] < 1:
                raise ValueError(f"kernel size must be positive, received {kernel[0]}.")
        except TypeError:
            if len(kernel) != 2:
                raise ValueError(
                    f"non-scalar kernel size must be a 2-tuple, received a {len(kernel)}-tuple."
                )
            kernel = tuple(int(v) for v in kernel)
            if kernel[0] < 1 or kernel[1] < 1:
                raise ValueError(f"kernel size must be positive, received {kernel}.")

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
            padding = (int(padding), int(padding))
            if padding[0] < 0:
                raise ValueError(f"stride must be non-negative, received {padding[0]}.")
        except TypeError:
            if len(padding) != 2:
                raise ValueError(
                    f"non-scalar padding must be a 2-tuple, received a {len(padding)}-tuple."
                )
            padding = tuple(int(v) for v in padding)
            if padding[0] < 0 or padding[1] < 0:
                raise ValueError(f"padding must be non-negative, received {padding}.")

        try:
            dilation = (int(dilation), int(dilation))
            if dilation[0] < 1:
                raise ValueError(f"dilation must be positive, received {dilation[0]}.")
        except TypeError:
            if len(dilation) != 2:
                raise ValueError(
                    f"non-scalar dilation must be a 2-tuple, received a {len(dilation)}-tuple."
                )
            dilation = tuple(int(v) for v in dilation)
            if dilation[0] < 1 or dilation[1] < 1:
                raise ValueError(f"dilation must be positive, received {dilation}.")

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
                (size + 2 * padding[hw] - dilation[hw] * (kernel[hw] - 1) - 1)
                / stride[hw]
                + 1
            )
            for hw, size in enumerate((height, width))
        )

        # call superclass constructor
        Connection.__init__(
            self,
            synapse=synapse(
                (channels * math.prod(kernel), out_height * out_width),
                float(step_time),
                int(batch_size),
                None if not delay else delay,
            ),
        )

        # call mixin constructor
        WeightBiasDelayMixin.__init__(
            weight=torch.rand(filters, channels, *kernel),
            bias=(None if not bias else torch.rand(filters)),
            delay=(None if not bias else torch.zeros(filters, channels, *kernel)),
            requires_grad=False,
        )

        # convolution properties
        # kernel
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # input
        self.channels = channels
        self.in_height = height
        self.in_width = width

        # output
        self.filters = filters
        self.out_height = out_height
        self.out_width = out_width

        # initialize parameters
        if weight_init:
            self.weight = weight_init(self.weight)

        if bias_init and bias:
            self.bias = bias_init(self.bias)

        if delay_init and delay:
            self.delay = delay_init(self.delay)

    def _unfold_input(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.unfold(
            inputs,
            self.kernel,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        )

    def presyn_receptive(self, data: torch.Tensor) -> torch.Tensor:
        r"""Reshapes data like the synapse data for pre/post learning methods.

        Args:
            data (torch.Tensor): data to reshape.

        Returns:
            torch.Tensor: reshaped data.

         Shape:
            Input: :math:`B \times N \times L \times [F]`

            Output:
            :math:`B \times 1 \times N \times L` or :math:`B \times F \times N \times L`

            Where :math:`N = C \cdot kH \ cdot kW` and
            :math:`L = H_\mathrm{out} \cdot W_\mathrm{out}`.
            Here, math:`F` is the number of filters (output channels),
            :math:`kH` is the height of the kernel, :math:`kW` is the width of the kernel,
            :math:`H_\mathrm{out}` is the height of the output,
            and :math:`W_\mathrm{out}` is the width of the output.

        Note:
            The inputs to this method are shaped like
            the `unfolded <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html>`_
            inputs to the connection.
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
            :math:`B \times F \times H_\mathrm{out} \times W_\mathrm{out}`

            Output:
            :math:`B \times F \times 1 \times L`

            Where :math:`L = H_\mathrm{out} \cdot W_\mathrm{out}`.
            Here, :math:`F` is the number of filters (output channels), :math:`H_\mathrm{out}`
            is the height of the output, and :math:`W_\mathrm{out}` is the width of the output.
        """
        return ein.rearrange(data, "b f oh ow -> b f 1 1 1 (oh ow)")

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
            _ = self.synapse(self._unfold_input(inputs), **kwargs)

            data = ein.rearrange(self.syncurrent, "b n l f -> b f n l")
            kernel = ein.rearrange(self.weight, "f c h w -> f 1 (c h w)")

            res = ein.rearrange(
                torch.matmul(kernel, data),
                "b f 1 (oh ow) -> b f oh ow",
                oh=self.outshape[0],
                ow=self.outshape[1],
            )
        else:
            data = self.synapse(inputs, **kwargs)  # B N L
            kernel = ein.rearrange(self.weight, "f c h w -> f (c h w)")

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

    @property
    def syncurrent(self) -> torch.Tensor:
        r"""Currents from the synapse at the time last used by the connection.

        Returns:
             torch.Tensor: delay-offset synaptic currents.
        """
        if self.delayed:
            return self.synapse.current_at(self.selector)
        else:
            return self.synapse.current

    @property
    def synspike(self) -> torch.Tensor:
        r"""Spikes to the synapse at the time last used by the connection.

        Returns:
             torch.Tensor: delay-offset synaptic spikes.
        """
        if self.delayed:
            return self.synapse.spike_at(self.selector)
        else:
            return self.synapse.spike

    @property
    def selector(self) -> torch.Tensor | None:
        r"""Learned delays as a selector for history.

        Returns:
             torch.Tensor | None: delay selector if one exists.
        """
        return ein.rearrange(self.delay, "f c h w -> 1 (c h w) 1 f").expand(
            self.bsize, -1, self.synapse.shape[-1], -1
        )

    @property
    def inshape(self) -> tuple[int, int, int]:
        return (self.channels, self.in_height, self.in_width)

    @property
    def outshape(self) -> tuple[int, int, int]:
        return (self.filters, self.out_height, self.out_width)
