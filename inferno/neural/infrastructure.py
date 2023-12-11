from __future__ import annotations
from inferno import Module, Batched, Temporal
import math
from typing import Callable


class Group(Batched, Module):
    r"""Base class for representing modules with common size and batch-size dependent state.

    Args:
        shape (tuple[int, ...] | int): shape of the group being represented, excluding batch size.
        batch_size (int): initial batch size.
        batched_parameters (tuple[tuple[str, ~torch.nn.parameter.Parameter] | str, ...] | None): batch size sensitive
            module parameters.
        batched_buffers (tuple[tuple[str, ~torch.Tensor] | str, ...] | None): batch size sensitive module buffers.
        on_batch_resize (Callable[[], None]): function to call when after the batch size is altered.

    Raises:
        ValueError: ``batch_size`` must be a positive integer.
        RuntimeError: all :py:class:`~torch.nn.parameter.Parameter` objects of ``batched_parameters`` must have
            ``batch_size`` sized :math:`0^\text{th}` dimension.
        RuntimeError: all :py:class:`~torch.Tensor` objects of ``batched_buffers`` must have ``batch_size`` sized
            :math:`0^\text{th}` dimension.

    Note:
        If a pre-defined :py:class:`~torch.Tensor` or :py:class:`~torch.nn.parameter.Parameter` is specified, the
        size of its :math:`0^\text{th}` dimension must equal ``batch_size``. If unspecified, an empty tensor of shape
        (batch_size, \*shape) will be constructed for each, and for parameter construction ``requires_grad``
        will be set to False.
    """

    def __init__(
        self,
        shape: tuple[int, ...] | int,
        batch_size: int,
        *,
        batched_parameters: tuple[str, ...] | None,
        batched_buffers: tuple[str, ...] | None,
        on_batch_resize: Callable[[], None],
    ):
        # superclass constructors
        Module.__init__(self)
        Batched.__init__(self, batch_size, on_batch_resize)

        # register buffers and parameters
        for param in batched_parameters if batched_parameters else []:
            self.register_parameter(param, None)
            self.register_batched(param)

        for buffer in batched_buffers if batched_buffers else []:
            self.register_buffer(buffer, None)
            self.register_batched(buffer)

        # register extras
        try:
            self.register_extra("_shape", (int(shape),))
        except TypeError:
            self.register_extra("_shape", tuple(int(s) for s in shape))

        self.register_extra("_count", math.prod(self._shape))

        # register buffers and parameters
        if not batched_parameters:
            batched_parameters = ()
        if not batched_buffers:
            batched_buffers = ()

    @property
    def shape(self) -> tuple[int, ...]:
        r"""Shape of the group.

        Returns:
            tuple[int, ...]: Shape of the group.
        """
        return self._shape

    @property
    def bshape(self) -> tuple[int, ...]:
        r"""Batch shape of the group

        Returns:
            tuple[int, ...]: Shape of the group, including the batch dimension.
        """
        return (self.bsize,) + self._shape

    @property
    def count(self) -> int:
        r"""Number of elements in the group, excluding batch.

        Returns:
            int: number of elements in the group.
        """
        return self._count


class TemporalGroup(Temporal, Batched, Module):
    r"""Base class for representing modules with common size and batch-size dependent state recorded over time.

    Args:
        shape (tuple[int, ...] | int): shape of the group being represented, excluding batch size.
        batch_size (int): initial batch size.
        batched_parameters (tuple[tuple[str, ~torch.nn.parameter.Parameter] | str, ...] | None): batch size sensitive
            module parameters.
        batched_buffers (tuple[tuple[str, ~torch.Tensor] | str, ...] | None): batch size sensitive module buffers.
        on_batch_resize (Callable[[], None]): function to call when after the batch size is altered.

    Raises:
        ValueError: ``batch_size`` must be a positive integer.
        RuntimeError: all :py:class:`~torch.nn.parameter.Parameter` objects of ``batched_parameters`` must have
            ``batch_size`` sized :math:`0^\text{th}` dimension.
        RuntimeError: all :py:class:`~torch.Tensor` objects of ``batched_buffers`` must have ``batch_size`` sized
            :math:`0^\text{th}` dimension.

    Note:
        If a pre-defined :py:class:`~torch.Tensor` or :py:class:`~torch.nn.parameter.Parameter` is specified, the
        size of its :math:`0^\text{th}` dimension must equal ``batch_size``. If unspecified, an empty tensor of shape
        (batch_size, \*shape) will be constructed for each, and for parameter construction ``requires_grad``
        will be set to False.
    """

    def __init__(
        self,
        shape: tuple[int, ...] | int,
        batch_size: int,
        *,
        batched_parameters: tuple[str, ...] | None,
        batched_buffers: tuple[str, ...] | None,
        on_batch_resize: Callable[[], None],
    ):
        # superclass constructors
        Module.__init__(self)
        Batched.__init__(self, batch_size, on_batch_resize)

        # register buffers and parameters
        for param in batched_parameters if batched_parameters else []:
            self.register_parameter(param, None)
            self.register_batched(param)

        for buffer in batched_buffers if batched_buffers else []:
            self.register_buffer(buffer, None)
            self.register_batched(buffer)

        # register extras
        try:
            self.register_extra("_shape", (int(shape),))
        except TypeError:
            self.register_extra("_shape", tuple(int(s) for s in shape))

        self.register_extra("_count", math.prod(self._shape))

        # register buffers and parameters
        if not batched_parameters:
            batched_parameters = ()
        if not batched_buffers:
            batched_buffers = ()

    @property
    def shape(self) -> tuple[int, ...]:
        r"""Shape of the group.

        Returns:
            tuple[int, ...]: Shape of the group.
        """
        return self._shape

    @property
    def bshape(self) -> tuple[int, ...]:
        r"""Batch shape of the group

        Returns:
            tuple[int, ...]: Shape of the group, including the batch dimension.
        """
        return (self.bsize,) + self._shape

    @property
    def count(self) -> int:
        r"""Number of elements in the group, excluding batch.

        Returns:
            int: number of elements in the group.
        """
        return self._count