from __future__ import annotations
from .mixins import BatchShapeMixin, DelayedMixin
from .modeling import Updatable, Updater
from .. import Module
from abc import ABC, abstractmethod
import math
import torch
import torch.nn as nn
from typing import Protocol


class Neuron(Module, ABC):
    r"""Base class for representing a group of neurons with a common mode of dynamics."""

    def __init__(self):
        # superclass constructors
        Module.__init__(self)

    def extra_repr(self) -> str:
        r"""Returns extra information on this module."""
        return f"shape={self.shape}, batchsz={self.batchsz}, dt={self.dt}"

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        r"""Shape of the group of neurons.

        Returns:
            tuple[int, ...]: shape of the group of neurons.

        Raises:
            NotImplementedError: ``shape`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__}(Neuron) must implement "
            "the getter for property `shape`."
        )

    @property
    @abstractmethod
    def count(self) -> int:
        r"""Number of neurons in the group.

        Returns:
            int: number of neurons in the group.

        Raises:
            NotImplementedError: ``count`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__}(Neuron) must implement "
            "the getter for property 'count'"
        )

    @property
    @abstractmethod
    def batchsz(self) -> int:
        r"""Batch size of the neuron group.

        Args:
            value (int): new batch size.

        Returns:
            int: present batch size.

        Raises:
            NotImplementedError: ``batchsz`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__}(Neuron) must implement "
            "the getter for property 'batchsz'"
        )

    @batchsz.setter
    @abstractmethod
    def batchsz(self, value: int) -> None:
        raise NotImplementedError(
            f"{type(self).__name__}(Neuron) must implement "
            "the setter for property 'batchsz'"
        )

    @property
    @abstractmethod
    def batchedshape(self) -> tuple[int, ...]:
        r"""Batch shape of the module

        Returns:
            tuple[int, ...]: shape of the group of neurons, including batch size.

        Raises:
            NotImplementedError: ``batchedshape`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__}(Neuron) must implement "
            "the getter for property 'batchedshape'"
        )

    @property
    @abstractmethod
    def dt(self) -> float:
        r"""Length of the simulation time step, in milliseconds.

        Args:
            value (float): new simulation time step length.

        Returns:
            float: present simulation time step length.

        Raises:
            NotImplementedError: ``dt`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__}(Neuron) must implement "
            "the getter for property 'dt'"
        )

    @dt.setter
    @abstractmethod
    def dt(self, value: float) -> None:
        raise NotImplementedError(
            f"{type(self).__name__}(Neuron) must implement "
            "the setter for property 'dt'"
        )

    @property
    @abstractmethod
    def voltage(self) -> torch.Tensor:
        r"""Membrane voltages in millivolts.

        Args:
            value (torch.Tensor): new membrane voltages.

        Returns:
            torch.Tensor: present membrane voltages.

        Raises:
            NotImplementedError: ``voltage`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__}(Neuron) must implement "
            "the getter for property 'voltage'"
        )

    @voltage.setter
    @abstractmethod
    def voltage(self, value: torch.Tensor) -> None:
        raise NotImplementedError(
            f"{type(self).__name__}(Neuron) must implement "
            "the setter for property 'voltage'"
        )

    @property
    @abstractmethod
    def refrac(self) -> torch.Tensor:
        r"""Remaining refractory periods, in milliseconds.

        Args:
            value (torch.Tensor): new remaining refractory periods.

        Returns:
            torch.Tensor: present remaining refractory periods.

        Raises:
            NotImplementedError: ``refrac`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__}(Neuron) must implement "
            "the getter for property 'refrac'"
        )

    @refrac.setter
    @abstractmethod
    def refrac(self, value: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            f"{type(self).__name__}(Neuron) must implement "
            "the setter for property 'refrac'"
        )

    @property
    @abstractmethod
    def spike(self) -> torch.Tensor:
        r"""Action potentials last generated.

        Returns:
            torch.Tensor: if the corresponding neuron generated an action potential
                during the prior step.

        Raises:
            NotImplementedError: ``spike`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__}(Neuron) must implement "
            "the getter for property 'spike'"
        )

    @abstractmethod
    def clear(self, **kwargs) -> None:
        r"""Resets neurons to their resting state.

        Raises:
            NotImplementedError: ``clear`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__}(Neuron) must implement the method 'clear'"
        )

    @abstractmethod
    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""Runs a simulation step of the neuronal dynamics.

        Args:
            inputs (torch.Tensor): input currents to the neurons.

        Returns:
            torch.Tensor: postsynaptic spikes from integration of inputs.

        Raises:
            NotImplementedError: ``forward`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__}(Neuron) must implement the method 'forward'"
        )


class InfernoNeuron(BatchShapeMixin, Neuron):
    r"""Base class for neurons included in the Inferno library.

    Unlike :py:class:`Neuron` which only defines an interface, this uses
    :py:class:`~inferno.neural.mixins.BatchShapeMixin` to work with the included mixins
    to automatically reshape batch-size dependent buffers and parameters.

    It also assumes :py:meth:`clear` will be implemented so it can be called without
    arguments and will, by default, not reset state which should persist, such as
    adaptations.
    """

    def __init__(self, shape: tuple[int, ...] | int, batch_size: int):
        # superclass constructors
        Neuron.__init__(self)

        # call mixin constructors
        BatchShapeMixin.__init__(self, shape, batch_size)

    @property
    def batchsz(self) -> int:
        r"""Batch size of the module.

        Args:
            value (int): new batch size.

        Returns:
            int: present batch size.
        """
        return BatchShapeMixin.batchsz.fget(self)

    @batchsz.setter
    def batchsz(self, value: int) -> None:
        BatchShapeMixin.batchsz.fset(self, value)
        self.clear()

    @property
    def dt(self) -> float:
        r"""Length of the simulation time step, in milliseconds.

        Args:
            value (float): new simulation time step length.

        Returns:
            float: present simulation time step length.

        Raises:
            NotImplementedError: ``dt`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__}(InfernoNeuron) must implement "
            "the getter for property 'dt'"
        )

    @dt.setter
    def dt(self, value: float) -> None:
        raise NotImplementedError(
            f"{type(self).__name__}(InfernoNeuron) must implement "
            "the setter for property 'dt'"
        )

    @property
    def voltage(self) -> torch.Tensor:
        r"""Membrane voltages in millivolts.

        Args:
            value (torch.Tensor): new membrane voltages.

        Returns:
            torch.Tensor: present membrane voltages.

        Raises:
            NotImplementedError: ``voltage`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__}(InfernoNeuron) must implement "
            "the getter for property 'voltage'"
        )

    @voltage.setter
    def voltage(self, value: torch.Tensor) -> None:
        raise NotImplementedError(
            f"{type(self).__name__}(InfernoNeuron) must implement "
            "the setter for property 'voltage'"
        )

    @property
    def refrac(self) -> torch.Tensor:
        r"""Remaining refractory periods, in milliseconds.

        Args:
            value (torch.Tensor): new remaining refractory periods.

        Returns:
            torch.Tensor: present remaining refractory periods.

        Raises:
            NotImplementedError: ``refrac`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__}(InfernoNeuron) must implement "
            "the getter for property 'refrac'"
        )

    @refrac.setter
    def refrac(self, value: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            f"{type(self).__name__}(InfernoNeuron) must implement "
            "the setter for property 'refrac'"
        )

    @property
    def spike(self) -> torch.Tensor:
        r"""Action potentials last generated.

        Returns:
            torch.Tensor: if the corresponding neuron generated an action potential
                during the prior step.

        Raises:
            NotImplementedError: ``spike`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__}(InfernoNeuron) must implement "
            "the getter for property 'spike'"
        )

    def clear(self, **kwargs) -> None:
        r"""Resets neurons to their resting state.

        Raises:
            NotImplementedError: ``clear`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__}(InfernoNeuron) must implement the method 'clear'"
        )

    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""Runs a simulation step of the neuronal dynamics.

        Args:
            inputs (torch.Tensor): input currents to the neurons.

        Returns:
            torch.Tensor: postsynaptic spikes from integration of inputs.

        Raises:
            NotImplementedError: ``forward`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__}(InfernoNeuron) must implement the method 'forward'"
        )


class SynapseConstructor(Protocol):
    r"""Common constructor for synapses, used by :py:class:`Connection` objects.

    Args:
        shape (tuple[int, ...] | int): shape of the group of synapses,
            excluding the batch dim.
        step_time (float): length of a simulation time step, in :math:`ms`.
        delay (float): maximum supported delay, in :math:`ms`.
        batch_size (int): size of the batch dimension.

    Returns:
        Synapse: newly constructed synapse.
    """

    def __call__(
        self,
        shape: tuple[int, ...] | int,
        step_time: float,
        delay: float,
        batch_size: int,
    ) -> Synapse:
        r"""Callback protocol function."""
        ...


class Synapse(Module, ABC):
    r"""Base class for representing a group of input synapses for a connection."""

    def __init__(self):
        # superclass constructors
        Module.__init__(self)

    def extra_repr(self) -> str:
        r"""Returns extra information on this module."""
        return f"shape={self.shape}, batchsz={self.batchsz}, dt={self.dt}, delay={self.delay}"

    @classmethod
    @abstractmethod
    def partialconstructor(cls, *args, **kwargs) -> SynapseConstructor:
        r"""Returns a function with a common signature for synapse construction.

        Raises:
            NotImplementedError: ``partialconstructor`` must be implemented
                by the subclass.

        Returns:
            SynapseConstructor: partial constructor for synapses of a given class.
        """
        raise NotImplementedError(
            f"{cls.__name__}(Synapse) must implement " "the method 'partialconstructor'"
        )

    @property
    @abstractmethod
    def dt(self) -> float:
        r"""Length of the simulation time step, in milliseconds.

        Args:
            value (float): new simulation time step length.

        Returns:
            float: present simulation time step length.
        """
        raise NotImplementedError(
            f"{self.__name__}(Synapse) must implement " "the getter for property 'dt'"
        )

    @dt.setter
    def dt(self, value: float) -> None:
        raise NotImplementedError(
            f"{self.__name__}(Synapse) must implement " "the setter for property 'dt'"
        )

    @property
    @abstractmethod
    def delay(self) -> float:
        r"""Maximum supported delay, in milliseconds.

        Returns:
            float: maximum supported delay.
        """
        raise NotImplementedError(
            f"{self.__name__}(Synapse) must implement "
            "the getter for property 'delay'"
        )

    @delay.setter
    @abstractmethod
    def delay(self, value: float) -> None:
        raise NotImplementedError(
            f"{self.__name__}(Synapse) must implement "
            "the setter for property 'delay'"
        )

    @property
    @abstractmethod
    def current(self) -> torch.Tensor:
        r"""Currents of the synapses at present, in nanoamperes.

        Args:
            value (torch.Tensor): new synapse currents.

        Returns:
            torch.Tensor: present synaptic currents.

        Raises:
            NotImplementedError: ``current`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__}(Synapse) must implement "
            "the getter for property 'current'"
        )

    @current.setter
    @abstractmethod
    def current(self, value: torch.Tensor) -> None:
        raise NotImplementedError(
            f"{type(self).__name__}(Synapse) must implement "
            "the setter for property 'current'"
        )

    @property
    @abstractmethod
    def spike(self) -> torch.Tensor:
        r"""Spike input to the synapses at present.

        Args:
            value (torch.Tensor): new spike input.

        Returns:
            torch.Tensor: present spike input.

        Raises:
            NotImplementedError: ``spike`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__}(Synapse) must implement "
            "the getter for property 'spike'"
        )

    @spike.setter
    @abstractmethod
    def spike(self, value: torch.Tensor) -> None:
        raise NotImplementedError(
            f"{type(self).__name__}(Synapse) must implement "
            "the setter for property 'spike'"
        )

    @abstractmethod
    def current_at(self, selector: torch.Tensor) -> torch.Tensor:
        r"""Currents, in nanoamperes, at times specified by delays, in milliseconds.

        Args:
            selector (torch.Tensor): delays for selection of currents.

        Returns:
            torch.Tensor: synaptic currents at the specified times.

        Raises:
            NotImplementedError: ``current_at`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__}(Synapse) must implement the method 'current_at'"
        )

    @abstractmethod
    def spike_at(self, selector: torch.Tensor) -> torch.Tensor:
        r"""Spikes, as booleans, at times specified by delays, in milliseconds.

        Args:
            selector (torch.Tensor): delays for selection of spikes.

        Returns:
            torch.Tensor: spike input at the given times.

        Raises:
            NotImplementedError: ``spike_at`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__}(Synapse) must implement the method 'spike_at'"
        )

    @abstractmethod
    def clear(self, **kwargs) -> None:
        r"""Resets synapses to their resting state.

        Raises:
            NotImplementedError: ``clear`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__}(Synapse) must implement the method 'clear'"
        )

    @abstractmethod
    def forward(self, *inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""Runs a simulation step of the synaptic dynamics.

        Args:
            *inputs (torch.Tensor): tensors shaped like the synapse.

        Returns:
            torch.Tensor: synaptic currents after integration of the inputs.

        Raises:
            NotImplementedError: ``forward`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__}(Synapse) must implement the method 'forward'"
        )


class InfernoSynapse(DelayedMixin, BatchShapeMixin, Synapse):
    r"""Base class for representing synapses included in the Inferno library.

    Unlike :py:class:`Neuron` which only defines an interface, this uses
    :py:class:`~inferno.neural.mixins.BatchShapeMixin` and
    :py:class:`~inferno.neural.mixins.DelayedMixin` to work with the included
    mixins to automatically reshape batch-size dependent buffers and parameters, and to
    manage recorded history for delay-stored buffers and parameters.

    Args:
        shape (tuple[int, ...] | int): shape of the group of synapses,
            excluding the batch dim.
        step_time (float): length of a simulation time step, in :math:`ms`.
        delay (float): maximum supported delay, in :math:`ms`.
        batch_size (int): size of the batch dimension.
        inplace (bool): if write operations on :py:class:`~inferno.RecordTensor`
            attributes should be performed with in-place operations.
    """

    def __init__(
        self,
        shape: tuple[int, ...] | int,
        step_time: float,
        delay: float,
        batch_size: int,
        inplace: bool,
    ):
        # superclass constructors
        Synapse.__init__(self)

        # mixin constructors
        BatchShapeMixin.__init__(self, shape, batch_size)
        DelayedMixin.__init__(self, step_time, delay)

        # set attributes
        self.__inplace = bool(inplace)

    @classmethod
    def partialconstructor(cls, *args, **kwargs) -> SynapseConstructor:
        r"""Returns a function with a common signature for synapse construction.

        Raises:
            NotImplementedError: ``partialconstructor`` must be implemented
                by the subclass.

        Returns:
            SynapseConstructor: partial constructor for synapses of a given class.
        """
        raise NotImplementedError(
            f"{cls.__name__}(Synapse) must implement " "the method 'partialconstructor'"
        )

    @property
    def dt(self) -> float:
        r"""Length of the simulation time step, in milliseconds.

        Args:
            value (float): new simulation time step length.

        Returns:
            float: present simulation time step length.
        """
        return DelayedMixin.dt.fget(self)

    @dt.setter
    def dt(self, value: float) -> None:
        DelayedMixin.dt.fset(self, value)
        self.clear()

    @property
    def delay(self) -> float:
        r"""Maximum supported delay, in milliseconds.

        Returns:
            float: maximum supported delay.
        """
        return DelayedMixin.delay.fget(self)

    @delay.setter
    def delay(self, value: float) -> None:
        DelayedMixin.delay.fset(self, value)
        self.clear()

    @property
    def inplace(self) -> bool:
        r"""If write operations should be performed in-place.

        Args:
            value (bool): if write operations should be performed in-place.

        Returns:
            bool: if write operations should be performed in-place.

        Note:
            Generally if gradient computation is required, this should be set to
            ``False``.
        """
        return self.__inplace

    @inplace.setter
    def inplace(self, value: bool) -> None:
        self.__inplace = bool(value)

    @property
    def current(self) -> torch.Tensor:
        r"""Currents of the synapses at present, in nanoamperes.

        Args:
            value (torch.Tensor): new synapse currents.

        Returns:
            torch.Tensor: present synaptic currents.

        Raises:
            NotImplementedError: ``current`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__}(Synapse) must implement "
            "the getter for property 'current'"
        )

    @current.setter
    def current(self, value: torch.Tensor) -> None:
        raise NotImplementedError(
            f"{type(self).__name__}(Synapse) must implement "
            "the setter for property 'current'"
        )

    @property
    def spike(self) -> torch.Tensor:
        r"""Spike input to the synapses at present.

        Args:
            value (torch.Tensor): new spike input.

        Returns:
            torch.Tensor: present spike input.

        Raises:
            NotImplementedError: ``spike`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__}(Synapse) must implement "
            "the getter for property 'spike'"
        )

    @spike.setter
    def spike(self, value: torch.Tensor) -> None:
        raise NotImplementedError(
            f"{type(self).__name__}(Synapse) must implement "
            "the setter for property 'spike'"
        )

    def current_at(self, selector: torch.Tensor) -> torch.Tensor:
        r"""Currents, in nanoamperes, at times specified by delays, in milliseconds.

        Args:
            selector (torch.Tensor): delays for selection of currents.

        Returns:
            torch.Tensor: synaptic currents at the specified times.

        Raises:
            NotImplementedError: ``current_at`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__}(Synapse) must implement the method 'current_at'"
        )

    def spike_at(self, selector: torch.Tensor) -> torch.Tensor:
        r"""Spikes, as booleans, at times specified by delays, in milliseconds.

        Args:
            selector (torch.Tensor): delays for selection of spikes.

        Returns:
            torch.Tensor: spike input at the given times.

        Raises:
            NotImplementedError: ``spike_at`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__}(Synapse) must implement the method 'spike_at'"
        )

    def clear(self, **kwargs) -> None:
        r"""Resets synapses to their resting state.

        Raises:
            NotImplementedError: ``clear`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__}(Synapse) must implement the method 'clear'"
        )

    def forward(self, *inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""Runs a simulation step of the synaptic dynamics.

        Args:
            *inputs (torch.Tensor): tensors shaped like the synapse.

        Returns:
            torch.Tensor: synaptic currents after integration of the inputs.

        Raises:
            NotImplementedError: ``forward`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__}(Synapse) must implement the method 'forward'"
        )


class Connection(Updatable, Module, ABC):
    r"""Base class for representing a weighted connection between two groups of neurons.

    Args:
        synapse (Synapse): synapse used to generate currents from inputs.
    """

    def __init__(
        self,
        synapse: Synapse,
    ):
        # superclass constructors
        Module.__init__(self)
        Updatable.__init__(self)

        # register submodule
        self.register_module("synapse_", synapse)

    def extra_repr(self) -> str:
        r"""Returns extra information on this module."""
        return (
            f"inshape={self.inshape}, outshape={self.outshape}, delay={self.delayedby}"
        )

    @property
    def synapse(self) -> Synapse:
        r"""Synapse registered with this connection.

        Args:
            value (Synapse): new synapse for this connection.

        Returns:
            Synapse: registered synapse.
        """
        return self.synapse_

    @synapse.setter
    def synapse(self, value: Synapse) -> None:
        self.synapses = value

    @property
    def batchsz(self) -> int:
        r"""Batch size of the connection.

        Args:
            value (int): new batch size.

        Returns:
            int: current batch size.

        Note:
            This calls the property :py:attr:`~inferno.neural.Synapse.batchsz` on
            :py:attr:`synapse`, assuming the connection has no batch size dependent
            state.
        """
        return self.synapse.batchsz

    @batchsz.setter
    def batchsz(self, value: int) -> None:
        self.synapse.batchsz = value

    @property
    def dt(self) -> float:
        r"""Length of the simulation time step, in milliseconds.

        Args:
            value (float): new length of the simulation time step.

        Returns:
            float: current length of the simulation time step.

        Note:
            This calls the property :py:attr:`~inferno.neural.Synapse.dt` on
            :py:attr:`synapse`, assuming the connection has no step time dependent
            state.
        """
        return self.synapse.dt

    @dt.setter
    def dt(self, value: float) -> None:
        self.synapse.dt = value

    @property
    @abstractmethod
    def inshape(self) -> tuple[int, ...]:
        r"""Shape of inputs to the connection, excluding the batch dimension.

        Returns:
            tuple[int]: shape of inputs to the connection.

        Raises:
            NotImplementedError: ``inshape`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__}(Connection) must implement "
            "the getter for property `inshape`."
        )

    @property
    @abstractmethod
    def outshape(self) -> tuple[int, ...]:
        r"""Shape of outputs from the connection, excluding the batch dimension.

        Returns:
            tuple[int]: shape of outputs from the connection.

        Raises:
            NotImplementedError: ``outshape`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__}(Connection) must implement "
            "the getter for property `outshape`"
        )

    @property
    def batched_inshape(self) -> tuple[int, ...]:
        r"""Shape of inputs to the connection, including the batch dimension.

        Returns:
            tuple[int]: shape of inputs to the connection.
        """
        return (self.batchsz,) + self.inshape

    @property
    def batched_outshape(self) -> tuple[int, ...]:
        r"""Shape of outputs from the connection, including the batch dimension.

        Returns:
            tuple[int]: shape of outputs from the connection.
        """
        return (self.batchsz,) + self.outshape

    def insize(self) -> int:
        r"""Number of inputs to the connection, excluding the batch dimension.

        Returns:
            int: number of inputs to the connection.

        Caution:
            This is a cached property based on :py:attr:`inshape`. When subclassing,
            if the computed value for ``inshape`` is changed, ``insize`` must be
            deleted to refresh the cache.
        """
        return math.prod(self.inshape)

    def outsize(self) -> int:
        r"""Number of outputs from the connection, excluding the batch dimension.

        Returns:
            int: number of outputs from the connection.

        Caution:
            This is a cached property based on :py:attr:`outshape`. When subclassing,
            if the computed value for ``outshape`` is changed, ``outsize`` must be
            deleted to refresh the cache.
        """
        return math.prod(self.outshape)

    @property
    @abstractmethod
    def weight(self) -> nn.Parameter:
        r"""Learnable connection weights.

        Args:
            value (torch.Tensor | nn.Parameter): new weights.

        Returns:
            nn.Parameter: present weights.

        Raises:
            NotImplementedError: ``weight`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__}(Connection) must implement "
            "the getter for property `weight`."
        )

    @weight.setter
    @abstractmethod
    def weight(self, value: torch.Tensor | nn.Parameter) -> None:
        raise NotImplementedError(
            f"{type(self).__name__}(Connection) must implement "
            "the setter for property `weight`."
        )

    @property
    @abstractmethod
    def bias(self) -> nn.Parameter | None:
        r"""Learnable connection biases.

        Args:
            value (torch.Tensor | nn.Parameter): new biases.

        Returns:
            nn.Parameter | None: present biases, if any.

        Raises:
            NotImplementedError: ``bias`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__}(Connection) must implement "
            "the getter for property `bias`."
        )

    @bias.setter
    @abstractmethod
    def bias(self, value: torch.Tensor | nn.Parameter) -> None:
        raise NotImplementedError(
            f"{type(self).__name__}(Connection) must implement "
            "the setter for property `bias`."
        )

    @property
    @abstractmethod
    def delay(self) -> nn.Parameter | None:
        r"""Learnable delays of the connection.

        Args:
            value (torch.Tensor | nn.Parameter): new delays.

        Returns:
            nn.Parameter | None: current delays, if the connection has any.

        Raises:
            NotImplementedError: ``delay`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__}(Connection) must implement "
            "the getter for property `delay`."
        )

    @delay.setter
    @abstractmethod
    def delay(self, value: torch.Tensor | nn.Parameter) -> None:
        raise NotImplementedError(
            f"{type(self).__name__}(Connection) must implement "
            "the setter for property `delay`."
        )

    @property
    @abstractmethod
    def selector(self) -> torch.Tensor | None:
        r"""Learned delays as a selector for synaptic currents and delays.

        Returns:
            torch.Tensor | None: delay selector if the connection has learnable delays.

        Raises:
            NotImplementedError: ``selector`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__}(Connection) must implement "
            "the getter for property `selector`."
        )

    @property
    def biased(self) -> bool:
        r"""If the connection has learnable biases.

        Returns:
            bool: if the connection has learnable biases.
        """
        return self.bias is not None

    @property
    def delayedby(self) -> float | None:
        r"""Maximum valid learned delay, in milliseconds.

        Returns:
            float: maximum valid learned delays.

        Note:
            This calls the property :py:attr:`Synapse.delay` on :py:attr:`synapse`
            if the connections has delays, otherwise returns None.
        """
        if self.delay is not None:
            return self.synapse.delay

    @property
    def syncurrent(self) -> torch.Tensor:
        r"""Currents from the synapse at the time last used by the connection.

        Returns:
            torch.Tensor: delay-offset synaptic currents.

        Note:
            If :py:attr:`delayedby` is None, this should return the most recent synaptic
            currents, otherwise it should return those indexed by :py:attr:`delay`.
        """
        if self.delayedby:
            return self.synapse.current_at(self.selector)
        else:
            return self.synapse.current

    @property
    def synspike(self) -> torch.Tensor:
        r"""Spikes to the synapse at the time last used by the connection.

        Returns:
            torch.Tensor: delay-offset synaptic spikes.
        """
        if self.delayedby:
            return self.synapse.spike_at(self.selector)
        else:
            return self.synapse.spike

    def clear(self, **kwargs) -> None:
        r"""Resets the state of the connection.

        Note:
            This calls the method :py:meth:`Synapse.clear` on :py:attr:`synapse` and
            :py:meth:`Updater.clear` on :py:attr`updater`, assuming the connection
            itself has no clearable state. Keyword arguments are passed through.
        """
        Updatable.clear(self, **kwargs)
        self.synapse.clear(**kwargs)

    @abstractmethod
    def like_input(self, data: torch.Tensor) -> torch.Tensor:
        r"""Reshapes data like synapse input to connection input.

        Args:
            data (torch.Tensor): data shaped like synapse input.

        Raises:
            NotImplementedError: ``like_input`` must be implemented by the subclass.

        Returns:
            torch.Tensor: reshaped data.

        .. admonition:: Shape
            :class: tensorshape

            ``data``:

            like :py:attr:`syncurrent` or :py:attr:`synspike`

            ``return``:

            :py:attr:`batched_inshape`
        """
        raise NotImplementedError(
            f"{type(self).__name__}(Connection) must implement "
            "the method `like_input`."
        )

    @abstractmethod
    def like_bias(self, data: torch.Tensor) -> torch.Tensor:
        r"""Reshapes data like reduced postsynaptic receptive spikes to connection bias.

        Args:
            data (torch.Tensor): data shaped like reduced postsynaptic receptive spikes.

        Raises:
            NotImplementedError: ``like_bias`` must be implemented by the subclass.

        Returns:
            torch.Tensor: reshaped data.

        .. admonition:: Shape
            :class: tensorshape

            ``data``:

            like :py:meth:`postsyn_receptive`, excluding the first (batch) and
            last (receptive field) dimension.

            ``return``:

            like :py:attr:`bias`
        """
        raise NotImplementedError(
            f"{type(self).__name__}(Connection) must implement "
            "the method `like_bias`."
        )

    @abstractmethod
    def like_synaptic(self, data: torch.Tensor) -> torch.Tensor:
        r"""Reshapes data like connection input to synapse input.

        Args:
            data (torch.Tensor): data shaped like connection input.

        Raises:
            NotImplementedError: ``like_synaptic`` must be implemented by the subclass.

        Returns:
            torch.Tensor: reshaped data.

        .. admonition:: Shape
            :class: tensorshape

            ``data``:

            :py:attr:`batched_inshape`

            ``return``:

            like :py:attr:`syncurrent` or :py:attr:`synspike`
        """
        raise NotImplementedError(
            f"{type(self).__name__}(Connection) must implement "
            "the method `like_synaptic`."
        )

    @abstractmethod
    def postsyn_receptive(self, data: torch.Tensor) -> torch.Tensor:
        r"""Reshapes data like connection output for pre-post learning methods.

        Args:
            data (torch.Tensor): data shaped like output of :py:meth:`forward`.

        Raises:
            NotImplementedError: ``postsyn_receptive`` must be
            implemented by the subclass.

        Returns:
            torch.Tensor: reshaped data.

        .. admonition:: Shape
            :class: tensorshape

            ``data``:

            :py:attr:`batched_outshape`

            ``return``:

            :math:`B \times` `broadcastable <https://pytorch.org/docs/stable/notes/broadcasting.html>`_
            with :py:attr:`weight` \times :math:`\times L`

            Where:
                * :math:`B` is the batch size.
                * :math:`L` is a connection-dependent value.
        """
        raise NotImplementedError(
            f"{type(self).__name__}(Connection) must implement "
            "the method `postsyn_receptive`."
        )

    @abstractmethod
    def presyn_receptive(self, data: torch.Tensor) -> torch.Tensor:
        r"""Reshapes data like the synapse state for pre-post learning methods.

        Args:
            data (torch.Tensor): data shaped like output of :py:meth:`like_synaptic`.

        Raises:
            NotImplementedError: ``presyn_receptive`` must be
            implemented by the subclass.

        Returns:
            torch.Tensor: reshaped data.

        .. admonition:: Shape
            :class: tensorshape

            ``data``:

            like :py:attr:`syncurrent` or :py:attr:`synspike`

            ``return``:

            :math:`B \times` `broadcastable <https://pytorch.org/docs/stable/notes/broadcasting.html>`_
            with :py:attr:`weight` :math:`\times L`

            Where:
                * :math:`B` is the batch size.
                * :math:`L` is a connection-dependent value.
        """
        raise NotImplementedError(
            f"{type(self).__name__}(Connection) must implement "
            "the method `presyn_receptive`."
        )

    def defaultupdater(
        self,
        *includes: str,
        exclude_weight: bool = False,
        exclude_bias: bool = False,
        exclude_delay: bool = False,
    ) -> Updater:
        r"""Default updater for this connection.

        Args:
            *includes (str): additional instance-specific parameters to include.
            exclude_weight (bool, optional): if weight should not be an updatable
                parameter. Defaults to ``False``.
            exclude_bias (bool, optional): if bias should not be an updatable
                parameter. Defaults to ``False``.
            exclude_delay (bool, optional): if delay should not be an updatable
                parameter. Defaults to ``False``.

        This will set and return an :py:class:`Updater` with the following trainable
        parameters:

        * ``weight``
        * ``bias``, if :py:attr:`biased` is ``True``
        * ``delay``, if :py:attr:`delayedby` is not ``None``

        Returns:
            Updater: newly set updater.
        """
        # determine updatable parameters
        params = []
        if not exclude_weight:
            params.append("weight")
        if self.biased and not exclude_bias:
            params.append("bias")
        if self.delayedby is not None and not exclude_delay:
            params.append("delay")

        # return the updater
        return Updater(self, *(*params, *includes))

    @abstractmethod
    def forward(self, *inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""Runs a simulation step of the connection.

        Args:
            *inputs (torch.Tensor): inputs which will be reshaped like the composed
                synapse and passed to its :py:meth:`Synapse.forward` call.

        Returns:
            torch.Tensor: resulting postsynaptic currents.

        Raises:
            NotImplementedError: ``forward`` must be implemented by the subclass.

        Note:
            When subclassing, keyword arguments should also be passed to the composed
            :py:class:`Synapse`.
        """
        raise NotImplementedError(
            f"{type(self).__name__}(Connection) must implement the method `forward`."
        )
