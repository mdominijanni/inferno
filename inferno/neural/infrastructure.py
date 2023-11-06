from __future__ import annotations
from abc import ABC, abstractmethod, abstractproperty
from inferno import Module
import math
import torch
import torch.nn as nn
from typing import Callable, Protocol


class Group(Module):
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
        batched_parameters: tuple[tuple[str, nn.Parameter], ...] | None,
        batched_buffers: tuple[tuple[str, torch.Tensor], ...] | None,
        on_batch_resize: Callable[[], None],
    ):
        # superclass constructor
        Module.__init__(self)

        # non-persistant function
        self._on_batch_resize = on_batch_resize

        # register extras
        try:
            self.register_extra("_shape", (int(shape),))
        except TypeError:
            self.register_extra("_shape", tuple(int(s) for s in shape))

        self.register_extra("_count", math.prod(self._shape))

        if int(batch_size) < 1:
            raise ValueError(
                f"batch size must be at least one, received {int(batch_size)}"
            )
        self.register_extra("_batch_size", int(batch_size))

        self.register_extra("_batched_parameters", [])
        self.register_extra("_batched_buffers", [])

        # register buffers and parameters
        if not batched_parameters:
            batched_parameters = ()
        if not batched_buffers:
            batched_buffers = ()

        for param in batched_parameters:
            if isinstance(param, str):
                self.register_parameter(
                    param,
                    nn.Parameter(torch.empty(self._batch_size, *self._shape), False),
                )
                self._batched_parameters.append(param)
            else:
                name, data = param[0], param[1]
                if data.shape[0] != self._batch_size:
                    raise RuntimeError(
                        "Parameter object in `batched_parameters` has zeroth dimension "
                        f"of size {data.shape[0]} not equal to batch size {self._batch_size}"
                    )
                else:
                    self.register_parameter(name, data)
                    self._batched_parameters.append(name)

        for buffer in batched_buffers:
            if isinstance(buffer, str):
                self.register_buffer(
                    buffer, torch.empty(self._batch_size, *self._shape)
                )
                self._batched_buffers.append(buffer)
            else:
                name, data = buffer[0], buffer[1]
                if data.shape[0] != self._batch_size:
                    raise RuntimeError(
                        "Tensor object in `batched_buffers` has zeroth dimension "
                        f"of size {data.shape[0]} not equal to batch size {self._batch_size}"
                    )
                else:
                    self.register_buffer(name, data)
                    self._batched_buffers.append(name)

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
        return (self._batch_size,) + self._shape

    @property
    def count(self) -> int:
        r"""Number of elements in the group, excluding batch.

        Returns:
            int: number of elements in the group.
        """
        return self._count

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
        return self._batch_size

    @bsize.setter
    def bsize(self, value):
        if int(value) == self._batch_size:
            return
        if int(value) < 1:
            raise ValueError(f"batch size must be at least one, received {int(value)}")

        self._batch_size = int(value)

        for param in self._batched_parameters:
            newparam = nn.Parameter(
                torch.empty(
                    (self._batch_size,) + self._shape,
                    dtype=getattr(self, param).data.dtype,
                    layout=getattr(self, param).data.layout,
                    device=getattr(self, param).data.device,
                    requires_grad=getattr(self, param).data.requires_grad,
                ),
                requires_grad=getattr(self, param).requires_grad,
            )
            delattr(self, param)
            self.register_parameter(param, newparam)

        for buffer in self._batched_buffers:
            newbuffer = torch.empty(
                (self._batch_size,) + self._shape,
                dtype=getattr(self, buffer).dtype,
                layout=getattr(self, buffer).layout,
                device=getattr(self, buffer).device,
                requires_grad=getattr(self, buffer).requires_grad,
            )
            delattr(self, buffer)
            self.register_buffer(buffer, newbuffer)

        self.on_batch_resize()


class Neuron(Group, ABC):
    r"""Base class for representing a group of neurons with a common mode of dynamics.

    Args:
        shape (tuple[int, ...] | int): shape of the group being represented, excluding batch size.
        batch_size (int): initial batch size.
        batched_parameters (tuple[tuple[str, ~torch.nn.parameter.Parameter] | str, ...] | None, optional): batch size
            sensitive module parameters. Defaults to None.
        batched_buffers (tuple[tuple[str, ~torch.Tensor] | str, ...] | None, optional): batch size sensitive module
            buffers. Defaults to None.
        on_batch_resize (Callable[[], None] | None, optional): function to call when after the batch size is altered.
            Defaults to None.

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

    Note:
        If no ``on_batch_resize`` is specified, then :py:meth:`clear` will be called without arguments.
    """

    def __init__(
        self,
        shape: tuple[int, ...] | int,
        batch_size: int,
        batched_parameters: tuple[tuple[str, nn.Parameter] | str, ...] | None = None,
        batched_buffers: tuple[tuple[str, torch.Tensor] | str, ...] | None = None,
        on_batch_resize: Callable[[], None] | None = None,
    ):
        # superclass constructor
        Group.__init__(
            self,
            shape,
            batch_size,
            batched_parameters=batched_parameters,
            batched_buffers=batched_buffers,
            on_batch_resize=self.clear if on_batch_resize is None else on_batch_resize,
        )

    @property
    @abstractmethod
    def dt(self) -> float:
        r"""Length of the simulation time step, in milliseconds.

        Args:
            value (float): new simulation time step length.

        Returns:
            float: length of the simulation time step.

        Raises:
            NotImplementedError: ``dt`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"Neuron `{type(self).__name__}` must implement the getter for property `dt`"
        )

    @dt.setter
    @abstractmethod
    def dt(self, value: float):
        raise NotImplementedError(
            f"Neuron `{type(self).__name__}` must implement the setter for property `dt`"
        )

    @property
    @abstractmethod
    def spike(self) -> torch.Tensor:
        r"""Which neurons generated an action potential on the last simulation step.

        Returns:
            torch.Tensor: if the correspond neuron generated an action potential last step.

        Raises:
            NotImplementedError: ``spike`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"Neuron `{type(self).__name__}` must implement the getter for property `spike`"
        )

    @property
    @abstractmethod
    def voltage(self) -> torch.Tensor:
        r"""Membrane voltages of the neurons, in millivolts.

        Args:
            value (torch.Tensor): new membrane voltages.

        Returns:
            torch.Tensor: current membrane voltages.

        Raises:
            NotImplementedError: ``voltage`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"Neuron `{type(self).__name__}` must implement the getter for property `voltage`"
        )

    @voltage.setter
    def voltage(self, value: torch.Tensor):
        raise NotImplementedError(
            f"Neuron `{type(self).__name__}` must implement the setter for property `voltage`"
        )

    @property
    @abstractmethod
    def refrac(self) -> torch.Tensor:
        r"""Remaining refractory periods of the neurons, in milliseconds.

        Args:
            value (torch.Tensor): new remaining refractory periods.

        Returns:
            torch.Tensor: current remaining refractory periods.

        Raises:
            NotImplementedError: ``refrac`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"Neuron `{type(self).__name__}` must implement the getter for property `refrac`"
        )

    @abstractmethod
    def clear(self, *args, **kwargs):
        r"""Resets the state of the neurons.

        Raises:
            NotImplementedError: ``clear`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"Neuron `{type(self).__name__}` must implement the method `clear`"
        )


class SynapseConstructor(Protocol):
    """Common constructor for synapses, used by :py:class:`Connection` objects.

    Args:
        shape (tuple[int, ...] | int): expected shape of the inputs, excluding batch dimension.
        step_time (float): step time of the simulation, in :math:`ms`.
        batch_size (int): size of the batch dimension.
        delay (int | None): maximum delay, as an integer multiple of step time, or None for an undelayed variant.

    Returns:
        Synapse: newly constructed synapse.
    """

    def __call__(
        self,
        shape: tuple[int, ...] | int,
        step_time: float,
        batch_size: int,
        delay: int | None,
    ) -> Synapse:
        r"""Callback protocol function."""
        ...


class Synapse(Group, ABC):
    r"""Base class for representing the input synapses for a connection.

    Args:
        shape (tuple[int, ...] | int): shape of the group being represented, excluding batch size.
        batch_size (int): initial batch size.
        batched_parameters (tuple[tuple[str, ~torch.nn.parameter.Parameter] | str, ...] | None, optional): batch size
            sensitive module parameters. Defaults to None.
        batched_buffers (tuple[tuple[str, ~torch.Tensor] | str, ...] | None, optional): batch size sensitive module
            buffers. Defaults to None.
        on_batch_resize (Callable[[], None] | None, optional): function to call when after the batch size is altered.
            Defaults to None.

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

    Note:
        If no ``on_batch_resize`` is specified, then :py:meth:`clear` will be called without arguments.
    """

    def __init__(
        self,
        shape: tuple[int, ...] | int,
        batch_size: int,
        batched_parameters: tuple[tuple[str, nn.Parameter] | str, ...] | None = None,
        batched_buffers: tuple[tuple[str, torch.Tensor] | str, ...] | None = None,
        on_batch_resize: Callable[[], None] | None = None,
    ):
        # superclass constructor
        Group.__init__(
            self,
            shape,
            batch_size,
            batched_parameters=batched_parameters,
            batched_buffers=batched_buffers,
            on_batch_resize=self.clear if on_batch_resize is None else on_batch_resize,
        )

    @classmethod
    @abstractmethod
    def partialconstructor(self, *args, **kwargs) -> SynapseConstructor:
        r"""Returns a function with a common signature for synapse construction.

        Raises:
            NotImplementedError: ``partialconstructor`` must be implemented by the subclass.

        Returns:
           SynapseConstructor: partial constructor for synapse.
        """
        raise NotImplementedError(
            f"Synapse `{type(self).__name__}` must implement the method `partialconstructor`"
        )

    @property
    @abstractmethod
    def dt(self) -> float:
        r"""Length of the simulation time step, in milliseconds.

        Args:
            value (float): new simulation time step length.

        Returns:
            float: length of the simulation time step.

        Raises:
            NotImplementedError: ``dt`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"Synapse `{type(self).__name__}` must implement the getter for property `dt`"
        )

    @dt.setter
    @abstractmethod
    def dt(self, value: float):
        raise NotImplementedError(
            f"Synapse `{type(self).__name__}` must implement the setter for property `dt`"
        )

    @property
    @abstractmethod
    def delay(self) -> float | None:
        r"""Maximum supported delay, in milliseconds.

        Returns:
            float | None: maximum delay, in milliseconds.

        Raises:
            NotImplementedError: ``delay`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"Synapse `{type(self).__name__}` must implement the getter for property `delay`"
        )

    @property
    @abstractmethod
    def current(self) -> torch.Tensor:
        r"""Currents of the synapses.

        Args:
            value (torch.Tensor): new synapse currents.

        Returns:
            torch.Tensor: currents of the synapses.

        Raises:
            NotImplementedError: ``current`` must be implemented by the subclass.

        Note:
            This will return the currents over the entire delay history.
        """
        raise NotImplementedError(
            f"Synapse `{type(self).__name__}` must implement the getter for property `current`"
        )

    @current.setter
    @abstractmethod
    def current(self, value: torch.Tensor):
        raise NotImplementedError(
            f"Synapse `{type(self).__name__}` must implement the setter for property `current`"
        )

    @property
    @abstractmethod
    def spike(self) -> torch.Tensor:
        r"""Spikes to the synapses.

        Returns:
            torch.Tensor: spikes to the synapses.

        Raises:
            NotImplementedError: ``spike`` must be implemented by the subclass.

        Note:
            This will return the spikes over the entire delay history.
        """
        raise NotImplementedError(
            f"Synapse `{type(self).__name__}` must implement the getter for property `spike`"
        )

    @abstractmethod
    def current_at(
        self, selector: torch.Tensor, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        r"""Returns currents at times specified by delays.

        Args:
            selector (torch.Tensor): delays for selection of currents.
            out (torch.Tensor | None, optional): output tensor, if required. Defaults to None.

        Returns:
            torch.Tensor: currents selected at the given times.

        Raises:
            NotImplementedError: ``currents`` must be implemented by the subclass.

        Note:
            It is expected that if ``selector`` is of a floating point datatype, as assessed
            by :py:func:`torch.Tensor.is_floating_point`, then it will be assumed to be in
            :math:`ms`. If it is not, it will be assumed to be an integer multiple of step time.
        """
        raise NotImplementedError(
            f"Synapse `{type(self).__name__}` must implement the method `dcurrent`"
        )

    @abstractmethod
    def spike_at(
        self, selector: torch.Tensor, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        r"""Returns spikes at times specified by delays.

        Args:
            selector (torch.Tensor): delays for selection of spikes.
            out (torch.Tensor | None, optional): output tensor, if required. Defaults to None.

        Returns:
            torch.Tensor: spikes selected at the given times.

        Raises:
            NotImplementedError: ``spikes`` must be implemented by the subclass.

        Note:
            It is expected that if ``selector`` is of a floating point datatype, as assessed
            by :py:func:`torch.Tensor.is_floating_point`, then it will be assumed to be in
            :math:`ms`. If it is not, it will be assumed to be an integer multiple of step time.
        """
        raise NotImplementedError(
            f"Synapse `{type(self).__name__}` must implement the method `dspike`"
        )

    @abstractmethod
    def clear(self, *args, **kwargs):
        r"""Resets the state of the synapses.

        Raises:
            NotImplementedError: ``clear`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"Synapse `{type(self).__name__}` must implement the method `clear`"
        )


class Connection(Module, ABC):
    r"""Base class for representing a weighted connection between two groups of neurons."""

    def __init__(
        self,
        synapse: Synapse,
        weight: nn.Parameter | None,
        bias: nn.Parameter | None,
        delay: nn.Parameter | None,
    ):
        # superclass constructor
        Module.__init__(self)

        # register submodule
        self.register_module("synapses", synapse)

        # register parameters
        self.register_parameter("weights", weight)
        self.register_parameter("biases", bias)
        self.register_parameter("delays", delay)

    @property
    def synapse(self) -> Synapse:
        r"""Synapse registered with this connection.

        Args:
            value (Synapse): replacement synapse for this connection.

        Returns:
           Synapse: registered synapse.
        """
        return self.synapses

    @synapse.setter
    def synapse(self, value: Synapse):
        self.synapses = value

    @property
    def bsize(self) -> int:
        r"""Batch size of the connection.

        Args:
            value (int): new batch size.

        Returns:
            int: current batch size.

        Raises:
            ValueError: ``value`` must be a positive integer.

        Note:
            This calls the property :py:attr:`Synapse.bsize`, assuming the connection
            itself does not use any batch size dependant tensors.
        """
        return self.synapse.bsize

    @bsize.setter
    def bsize(self, value: int):
        self.synapse.bsize = value

    @property
    def dt(self) -> float:
        r"""Length of the simulation time step, in milliseconds.

        Returns:
            float: length of the simulation time step.

        Raises:
            This calls the property :py:attr:`Synapse.dt`.
        """
        return self.synapse.dt

    @property
    def insize(self) -> int:
        return math.prod(self.inshape)

    @property
    def outsize(self) -> tuple[int]:
        return math.prod(self.outshape)

    @property
    def delayed(self) -> int | float | None:
        if self.delay is None:
            return None
        elif self.delay.is_floating_point():
            return float(self.synapse.delay * self.dt)
        else:
            return int(self.synapse.delay)

    @property
    def weight(self) -> torch.Tensor:
        return self.weights.data

    @weight.setter
    def weight(self, value: torch.Tensor):
        self.weights.data = value

    @property
    def bias(self) -> torch.Tensor | None:
        if self.biases is not None:
            return self.biases.data

    @bias.setter
    def bias(self, value: torch.Tensor):
        if self.biases is not None:
            self.biases.data = value
        else:
            raise RuntimeError(
                f"cannot set `bias` on a {type(self).__name__} without trainable biases"
            )

    @property
    def delay(self) -> torch.Tensor | None:
        if self.delays is not None:
            return self.delays.data

    @delay.setter
    def delay(self, value: torch.Tensor):
        if self.delays is not None:
            self.delays.data = value
        else:
            raise RuntimeError(
                f"cannot set `delay` on a {type(self).__name__} without trainable delays"
            )

    def clear(self, *args, **kwargs):
        self.synapse.clear(*args, **kwargs)

    @abstractproperty
    def inshape(self) -> tuple[int]:
        raise NotImplementedError(
            f"Connection `{type(self).__name__}` must implement the getter for property `inshape`"
        )

    @abstractproperty
    def outshape(self) -> tuple[int]:
        raise NotImplementedError(
            f"Connection `{type(self).__name__}` must implement the getter for property `outshape`"
        )
