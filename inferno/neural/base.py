from __future__ import annotations
from abc import ABC, abstractmethod
from inferno import DimensionalModule, HistoryModule, Module
import math
from functools import cached_property
import torch
from typing import Protocol
from .infrastructure import ShapeMixin


class Neuron(ShapeMixin, DimensionalModule, ABC):
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
    ):
        # superclass constructors
        DimensionalModule.__init__(self)
        ShapeMixin.__init__(self, shape, batch_size)

    @property
    def bsize(self) -> int:
        r"""Batch size of the neuron group.

        Args:
            value (int): new batch size.

        Returns:
            int: current batch size.
        """
        return ShapeMixin.bsize.fget(self)

    @bsize.setter
    def bsize(self, value: int) -> None:
        ShapeMixin.bsize.fset(self, value)
        self.clear()

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

    @refrac.setter
    def refrac(self, value: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            f"Neuron `{type(self).__name__}` must implement the setter for property `refrac`"
        )

    @abstractmethod
    def clear(self, **kwargs):
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


class Synapse(ShapeMixin, HistoryModule, ABC):
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
    """

    def __init__(
        self,
        shape: tuple[int, ...] | int,
        step_time: float,
        batch_size: int,
        delay: float | None,
    ):
        # superclass constructors
        HistoryModule.__init__(self, step_time, delay if delay is not None else 0.0)
        ShapeMixin.__init__(self, shape, batch_size)

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
    def dt(self) -> float:
        r"""Length of the simulation time step, in milliseconds.

        Args:
            value (float): new simulation time step length.

        Returns:
            float: length of the simulation time step.
        """
        return HistoryModule.dt.fget(self)

    @dt.setter
    def dt(self, value: float):
        HistoryModule.dt.fset(self, value)
        self.clear()

    @property
    def delay(self) -> float | None:
        r"""Maximum supported delay, in milliseconds.

        Returns:
            float | None: maximum delay, in milliseconds.

        Raises:
            NotImplementedError: ``delay`` must be implemented by the subclass.
        """
        if self.hsize == 0:
            return None
        else:
            return self.hlen

    @delay.setter
    def delay(self, value: float) -> None:
        hsize = self.hsize
        self.hlen = value
        if hsize != self.hsize:
            self.clear()

    @property
    @abstractmethod
    def current(self) -> torch.Tensor:
        r"""Currents of the synapses at the present time.

        Args:
            value (torch.Tensor): new synapse currents.

        Returns:
            torch.Tensor: currents of the synapses.

        Raises:
            NotImplementedError: ``current`` must be implemented by the subclass.
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
        r"""Spikes to the synapses at the present time.

        Returns:
            torch.Tensor: spikes to the synapses.

        Raises:
            NotImplementedError: ``spike`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"Synapse `{type(self).__name__}` must implement the getter for property `spike`"
        )

    @abstractmethod
    def current_at(self, selector: torch.Tensor) -> torch.Tensor:
        r"""Returns currents at times specified by delays.

        Args:
            selector (torch.Tensor): delays for selection of currents.

        Returns:
            torch.Tensor: currents selected at the given times.

        Raises:
            NotImplementedError: ``current_at`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"Synapse `{type(self).__name__}` must implement the method `dcurrent`"
        )

    @abstractmethod
    def spike_at(self, selector: torch.Tensor) -> torch.Tensor:
        r"""Returns spikes at times specified by delays.

        Args:
            selector (torch.Tensor): delays for selection of spikes.

        Returns:
            torch.Tensor: spikes selected at the given times.

        Raises:
            NotImplementedError: ``spike_at`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"Synapse `{type(self).__name__}` must implement the method `dspike`"
        )

    @abstractmethod
    def clear(self, **kwargs):
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
    ):
        # superclass constructor
        Module.__init__(self)

        # register submodule
        self.register_module("synapse_", synapse)

    @property
    def synapse(self) -> Synapse:
        r"""Synapse registered with this connection.

        Args:
            value (Synapse): replacement synapse for this connection.

        Returns:
           Synapse: registered synapse.
        """
        return self.synapse_

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

        Note:
            This calls the property :py:attr:`Synapse.bsize`, assuming the connection
            itself does not use any batch size dependant tensors.
        """
        return self.synapse.bsize

    @bsize.setter
    def bsize(self, value: int):
        self.synapse.bsize = value

    @property
    @abstractmethod
    def inshape(self) -> tuple[int]:
        r"""Shape of inputs to the connection, excluding the batch dimension.

        Returns:
            tuple[int]: shape of inputs to the connection.

        Raises:
            NotImplementedError: ``inshape`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"Connection `{type(self).__name__}` must implement the getter for property `inshape`"
        )

    @property
    @abstractmethod
    def outshape(self) -> tuple[int]:
        r"""Shape of outputs from the connection, excluding the batch dimension.

        Returns:
            tuple[int]: shape of outputs from the connection.

        Raises:
            NotImplementedError: ``outshape`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"Connection `{type(self).__name__}` must implement the getter for property `outshape`"
        )

    @cached_property
    def insize(self) -> int:
        r"""Number of inputs to the connection, excluding the batch dimension.

        Returns:
            int: number of inputs to the connection.
        """
        return math.prod(self.inshape)

    @cached_property
    def outsize(self) -> tuple[int]:
        r"""Number of outputs from the connection, excluding the batch dimension.

        Returns:
            int: number of outputs from the connection.
        """
        return math.prod(self.outshape)

    @property
    def dt(self) -> float:
        r"""Length of the simulation time step, in milliseconds.

        Args:
            value (float): new length of the simulation time step.

        Returns:
            float: current length of the simulation time step.

        Note:
            This calls the property :py:attr:`Synapse.dt`, assuming the
            connection itself is not dependent on step time.
        """
        return self.synapse.dt

    @dt.setter
    def dt(self, value: float):
        self.synapse.dt = value

    @property
    def biased(self) -> bool:
        r"""If the connection has learnable biases.

        Returns:
            bool: if the connection has learnable biases.
        """
        return self.bias is None

    @property
    def delayed(self) -> float | None:
        r"""Maxmimum length of the learned delays, in milliseconds.

        Returns:
            float | None: maxmimum length of the learned delays, none if no delays.

        Note:
            This calls the property :py:attr:`Synapse.delay`.
        """
        return self.synapse.delay

    @property
    @abstractmethod
    def weight(self) -> torch.Tensor:
        r"""Learnable weights of the connection.

        Args:
            value (torch.Tensor): new weights.

        Returns:
            torch.Tensor: current weights.

        Raises:
            NotImplementedError: ``weight`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"Connection `{type(self).__name__}` must implement the getter for property `weight`"
        )

    @weight.setter
    @abstractmethod
    def weight(self, value: torch.Tensor):
        raise NotImplementedError(
            f"Connection `{type(self).__name__}` must implement the setter for property `weight`"
        )

    @property
    @abstractmethod
    def bias(self) -> torch.Tensor | None:
        r"""Learnable biases of the connection.

        Args:
            value (torch.Tensor): new biases.

        Returns:
            torch.Tensor | None: current biases, if the connection has any.

        Raises:
            NotImplementedError: ``bias`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"Connection `{type(self).__name__}` must implement the getter for property `bias`"
        )

    @bias.setter
    @abstractmethod
    def bias(self, value: torch.Tensor):
        raise NotImplementedError(
            f"Connection `{type(self).__name__}` must implement the setter for property `bias`"
        )

    @property
    @abstractmethod
    def delay(self) -> torch.Tensor | None:
        r"""Learnable delays of the connection.

        Args:
            value (torch.Tensor): new delays.

        Returns:
            torch.Tensor | None: current delays, if the connection has any.

        Raises:
            NotImplementedError: ``delay`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"Connection `{type(self).__name__}` must implement the getter for property `delay`"
        )

    @delay.setter
    @abstractmethod
    def delay(self, value: torch.Tensor):
        raise NotImplementedError(
            f"Connection `{type(self).__name__}` must implement the setter for property `delay`"
        )

    def clear(self, **kwargs):
        r"""Resets the state of the connection.

        Note:
            This calls the method :py:meth:`Synapse.clear`, assuming the connection
            itself maintains no clearable state.
        """
        self.synapse.clear(**kwargs)
