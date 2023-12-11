from inferno._internal import rgetattr, rsetattr
from numbers import Number
import math
import torch
import torch.nn as nn
from typing import Callable


class Batched:
    """Mixin for modules with batch-size dependent parameters or buffers.

    Args:
        size (int): initial batch size.
        on_batch_resize (Callable[[], None]): function to call when after the batch size is altered.

    Raises:
        RuntimeError: batch size must be positive.

    Note:
        This must be added to a class which inherits from :py:class:`Module`.

    Note:
        The mixin constructor must be called after the constructor for the class
        which calls the :py:class:`Module` constructor is.
    """

    def __init__(
        self,
        batch_size: int,
        on_batch_resize: Callable[[], None],
    ):
        if int(batch_size) < 1:
            raise RuntimeError(f"size must be positive, received {int(batch_size)}")

        # register non-persistent
        self._on_batch_resize = on_batch_resize

        # register extras
        self.register_extra("_batched_bsize", int(batch_size))
        self.register_extra("_batched_buffers", set())
        self.register_extra("_batched_parameters", set())

    def cshape(self, shape: tuple[int]):
        return (self.bsize,) + shape

    def register_batched(self, attr: str, fill: Number = 0):
        try:
            buffer = self.get_buffer(attr)
        except AttributeError:
            pass
        else:
            if (
                buffer is not None
                and buffer.numel > 0
                and buffer.shape[0] != self._batched_bsize
            ):
                raise RuntimeError(
                    f"attribute {attr} has zeroth dimension of size "
                    f"{buffer.shape[0]} which does not match "
                    f"batch size {self._batched_bsize}"
                )
            else:
                self._batched_buffers.add(attr)

        try:
            param = self.get_parameter(attr)
        except AttributeError:
            pass
        else:
            if (
                param is not None
                and param.numel > 0
                and param.shape[0] != self._batched_bsize
            ):
                raise RuntimeError(
                    f"attribute {attr} has zeroth dimension of size "
                    f"{param.shape[0]} which does not match "
                    f"batch size {self._batched_bsize}"
                )
            else:
                self._batched_parameters.add(attr)

        raise AttributeError(
            f"attribute {attr} is not a registered buffer or parameter"
        )

    def deregister_batched(self, attr: str):
        if attr in self._batched_buffers:
            self._batched_buffers.remove(attr)

        if attr in self._batched_parameters:
            self._batched_parameters.remove(attr)

    @property
    def bsize(self) -> int:
        r"""Batch size of the group.

        Args:
            value (int): new batch size.

        Returns:
            int: current batch size.

        Raises:
            ValueError: ``value`` must be a positive integer.

        Note:
            When reallocating tensors for batched parameters and buffers, memory is not pinned
            and the default ``torch.contiguous_format``
            `memory format <https://pytorch.org/docs/stable/tensor_attributes.html#torch.memory_format>_` is used.

        Note:
           Batched tensors are reallocated using :py:func:`torch.empty`. After reallocation, ``on_batch_resize``
           from initialization is called. Any additional processing should be done there.
        """
        return self._batched_bsize

    @bsize.setter
    def bsize(self, value: int):
        if int(value) == self._batched_bsize:
            return
        if int(value) < 1:
            raise ValueError(f"batch size must be positive, received {int(value)}")

        self._batched_bsize = int(value)

        for attr in self._batched_parameters:
            param = rgetattr(self, attr)

            if param is None:
                continue
            if param.numel() == 0:
                continue

            rsetattr(
                self,
                attr,
                nn.Parameter(
                    torch.empty(
                        (self._batched_bsize,) + param.shape[1:],
                        dtype=param.data.dtype,
                        layout=param.data.layout,
                        device=param.data.device,
                        requires_grad=param.data.requires_grad,
                    ),
                    requires_grad=param.requires_grad,
                ),
            )

        for attr in self._batched_buffers:
            buffer = rgetattr(self, attr)

            if buffer is None:
                continue
            if buffer.numel() == 0:
                continue

            rsetattr(
                self,
                buffer,
                torch.empty(
                    (self._batched_bsize,) + buffer.shape[1:],
                    dtype=buffer.dtype,
                    layout=buffer.layout,
                    device=buffer.device,
                    requires_grad=buffer.requires_grad,
                ),
            )

        self._on_batch_resize()


class Temporal:
    r"""Mixin for modules with parameters or buffers stored over time.

    Args:
        step_time (float): length of a simulation time step, in :math:`\mathrm{ms}`.
        history_len (float | None): maximum length of time to record for before present, in :math:`\mathrm{ms}`.
        on_history_resize (Callable[[], None]): function to call when after the history count is altered.

    Raises:
        RuntimeError: batch size must be positive.

    Note:
        This must be added to a class which inherits from :py:class:`Module`.

    Note:
        The mixin constructor must be called after the constructor for the class
        which calls the :py:class:`Module` constructor is.

    Note:
        When `history_len` is zero, only the most recent result is recorded.

    """

    def __init__(
        self,
        step_time: float,
        history_len: float,
        on_history_resize: Callable[[], None],
    ):
        if float(step_time) <= 0:
            raise RuntimeError(
                f"step time must be positive, received {float(step_time)}"
            )

        if float(history_len) < 0:
            raise RuntimeError(
                f"history length must be non-negative, received {float(history_len)}"
            )

        # register non-persistent
        self._on_history_resize = on_history_resize

        # register extras
        self.register_extra("_temporal_dt", float(step_time))
        self.register_extra("_temporal_hlen", float(history_len))
        self.register_extra("_temporal_buffers", set())
        self.register_extra("_temporal_parameters", set())

    def cshape(self, shape: tuple[int]):
        return tuple() + shape + (math.ceil(self._temporal_hlen / self._temporal_dt) + 1,)

    def register_temporal(self, attr: str):
        count = math.ceil(self._temporal_hlen / self._temporal_dt) + 1

        try:
            buffer = self.get_buffer(attr)
        except AttributeError:
            pass
        else:
            if buffer is not None and buffer.numel > 0 and buffer.shape[-1] != count:
                raise RuntimeError(
                    f"attribute {attr} has final dimension of size "
                    f"{buffer.shape[-1]} which does not match "
                    f"required number of elements {count}"
                )
            else:
                self._history_buffers.add(attr)

        try:
            param = self.get_parameter(attr)
        except AttributeError:
            pass
        else:
            if param is not None and param.numel > 0 and param.shape[0] != count:
                raise RuntimeError(
                    f"attribute {attr} has final dimension of size "
                    f"{param.shape[-1]} which does not match "
                    f"required number of elements {count}"
                )
            else:
                self._history_parameters.add(attr)

        raise AttributeError(
            f"attribute {attr} is not a registered buffer or parameter"
        )

    def deregister_temporal(self, attr: str):
        if attr in self._batched_buffers:
            self._batched_buffers.remove(attr)

        if attr in self._batched_parameters:
            self._batched_parameters.remove(attr)

    @property
    def dt(self) -> float:
        r"""Length of the simulation time step, in milliseconds.

        Args:
            value (float): new step length, in :math:`\mathrm{ms}`.

        Returns:
            float: length of the simulation time step.

        Raises:
            ValueError: ``value`` must be a positive integer.

        Note:
            When reallocating tensors for batched parameters and buffers, memory is not pinned
            and the default ``torch.contiguous_format``
            `memory format <https://pytorch.org/docs/stable/tensor_attributes.html#torch.memory_format>_` is used.

        Note:
           Batched tensors are reallocated using :py:func:`torch.empty`. After reallocation, ``on_batch_resize``
           from initialization is called. Any additional processing should be done there.
        """
        return self._temporal_dt

    @dt.setter
    def dt(self, value: float):
        if float(value) == self._temporal_dt:
            return

        if float(value) <= 0:
            raise RuntimeError(f"step time must be positive, received {float(value)}")

        self._temporal_dt = float(value)

        count = math.ceil(self._temporal_hlen / self._temporal_dt) + 1

        for attr in self._batched_parameters:
            param = rgetattr(self, attr)

            if param is None:
                continue
            if param.numel() == 0:
                continue

            rsetattr(
                self,
                attr,
                nn.Parameter(
                    torch.empty(
                        tuple() + param.shape[:-1] + (count,),
                        dtype=param.data.dtype,
                        layout=param.data.layout,
                        device=param.data.device,
                        requires_grad=param.data.requires_grad,
                    ),
                    requires_grad=param.requires_grad,
                ),
            )

        for attr in self._batched_buffers:
            buffer = rgetattr(self, attr)

            if buffer is None:
                continue
            if buffer.numel() == 0:
                continue

            rsetattr(
                self,
                buffer,
                torch.empty(
                    tuple() + buffer.shape[:-1] + (count,),
                    dtype=buffer.dtype,
                    layout=buffer.layout,
                    device=buffer.device,
                    requires_grad=buffer.requires_grad,
                ),
            )

        self._on_history_resize()

    @property
    def hlen(self) -> float:
        r"""Length of history past the current moment, in milliseconds.

        Args:
            value (float): new step length, in :math:`\mathrm{ms}`.

        Returns:
            float: length of the simulation time step.

        Raises:
            ValueError: ``value`` must be a non-negative integer.

        Note:
            When reallocating tensors for batched parameters and buffers, memory is not pinned
            and the default ``torch.contiguous_format``
            `memory format <https://pytorch.org/docs/stable/tensor_attributes.html#torch.memory_format>_` is used.

        Note:
           Batched tensors are reallocated using :py:func:`torch.empty`. After reallocation, ``on_batch_resize``
           from initialization is called. Any additional processing should be done there.
        """
        return self._temporal_hlen

    @hlen.setter
    def hlen(self, value: float):
        if float(value) == self._temporal_hlen:
            return

        if float(value) < 0:
            raise RuntimeError(
                f"history length must be non-negative, received {float(value)}"
            )

        self._temporal_hlen = float(value)

        count = math.ceil(self._temporal_hlen / self._temporal_dt) + 1

        for attr in self._batched_parameters:
            param = rgetattr(self, attr)

            if param is None:
                continue
            if param.numel() == 0:
                continue

            rsetattr(
                self,
                attr,
                nn.Parameter(
                    torch.empty(
                        tuple() + param.shape[:-1] + (count,),
                        dtype=param.data.dtype,
                        layout=param.data.layout,
                        device=param.data.device,
                        requires_grad=param.data.requires_grad,
                    ),
                    requires_grad=param.requires_grad,
                ),
            )

        for attr in self._batched_buffers:
            buffer = rgetattr(self, attr)

            if buffer is None:
                continue
            if buffer.numel() == 0:
                continue

            rsetattr(
                self,
                buffer,
                torch.empty(
                    tuple() + buffer.shape[:-1] + (count,),
                    dtype=buffer.dtype,
                    layout=buffer.layout,
                    device=buffer.device,
                    requires_grad=buffer.requires_grad,
                ),
            )

        self._on_history_resize()
