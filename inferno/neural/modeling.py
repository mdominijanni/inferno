from . import Connection, Neuron, Synapse
import functools
from inferno import Module
from inferno.observe import ManagedMonitor, MonitorConstructor
import torch
from typing import Any
import warnings

class Trainable(Module):

    def __init__(self, connection: Connection, neuron: Neuron):
        r"""A trainable connection-neuron pair.

        This is a construct used to associate a single connection and neuron object
        for the purposes of training. The contained connection may produce output for
        multiple neurons and the neuron may take input from multiple connections.

        When implementing a new updater, the properties here should be used when
        accessing or alering the model parameters.

        Args:
            connection (Connection): connection which produces output for the neuron.
            neuron (Neuron): neuron which takes output from the connection.
        """
        # call superclass constructor
        Module.__init__(self)

        # component elements
        self.connection_ = connection
        self.neuron_ = neuron

    @property
    def connection(self) -> Connection:
        r"""Connection submodule.

        Args:
            value (Connection): replacement connection.

        Returns:
            Connection: existing connection.
        """
        return self.connection_

    @connection.setter
    def connection(self, value: Connection) -> None:
        self.connection_ = value

    @property
    def neuron(self) -> Neuron:
        r"""Neuron submodule.

        Args:
            value (Neuron): replacement neuron.

        Returns:
            Neuron: existing neuron.
        """
        return self.neuron_

    @neuron.setter
    def neuron(self, value: Neuron) -> None:
        self.neuron_ = value

    @property
    def synapse(self) -> Synapse:
        r"""Synapse submodule.

        Args:
            value (Synapse): replacement synapse.

        Returns:
            Synapse: existing synapse.
        """
        return self.connection_.synapse

    @synapse.setter
    def synapse(self, value: Synapse) -> None:
        self.connection_.synapse = value

    @property
    def precurrent(self) -> torch.Tensor:
        r"""Currents from the synapse at the time last used by the connection.

        Alias for ``connection.syncurrent``.

        Returns:
            torch.Tensor: delay-offset synaptic currents.
        """
        return self.connection_.syncurrent

    @property
    def prespike(self) -> torch.Tensor:
        r"""Spikes to the synapse at the time last used by the connection.

        Alias for ``connection.synspike``.

        Returns:
            torch.Tensor: delay-offset synaptic spikes.
        """
        return self.connection_.synspike

    @property
    def postvoltage(self) -> torch.Tensor:
        r"""Membrane voltages in millivolts.

        Returns:
            torch.Tensor: membrane voltages.
        """
        return self.neuron_.voltage

    @property
    def postspike(self) -> torch.Tensor:
        r"""Action potentials last generated.

        Returns:
            torch.Tensor: membrane voltages.
        """
        return self.neuron_.spike

class MonitorPool(Module):

    def __init__(self):
        self.


class Layer(Module):
    r"""Container for sequential Connection and Neuron objects.

    This is used as the base building block of spiking neural networks in Inferno,
    and is used for training models.

    Args:
        connection (Connection): connection between the layer inputs and the neurons.
        neuron (Neuron): neurons which take their input from the connection and their
            output is returned.
        connection_kwargs (dict[str, str] | None, optional): keyword argument
                mapping for connection methods. Defaults to None.
        neuron_kwargs (dict[str, str] | None, optional): keyword argument
                mapping for neuron methods. Defaults to None.

    Note:
        The keyword argument mappings are a dictionary, where the key is a
        kwarg in a :py:class:`Layer` method, and the corresponding value is
        the name for that kwarg which will be passed to the dependent method in
        in the :py:class:`Connection` or :py:meth:`Neuron`.

        When None, *all kwargs* are passed in. Included classes in Inferno are
        written to avoid conflicts, but that is not always guaranteed.

    Tip:
        The composed :py:class:`Neuron` does not need to be unique to this layer, and
        some architectures explicitly have multiple connections going to the same
        group of neurons. The uniqueness of the composed :py:class:`Connection` is
        not enforced, but unexpected behavior may occur if it is not unique.
    """

    def __init__(
        self,
        connection: Connection,
        neuron: Neuron,
        connection_kwargs: dict[str, str] | None = None,
        neuron_kwargs: dict[str, str] | None = None,
    ):
        Module.__init__(self)
        # warn if connection and neuron are inconsistent
        if connection.dt != neuron.dt:
            warnings.warn(
                f"inconsistent step times, {connection.dt} "
                f"for connection and {neuron.dt} for neuron."
            )

        # error if incompatible
        if connection.bsize != neuron.bsize:
            raise RuntimeError(
                f"incompatible batch sizes, {connection.bsize} "
                f"for connection and {neuron.bsize} for neuron."
            )

        if connection.outshape != neuron.shape:
            raise RuntimeError(
                f"incompatible shapes, {connection.outshape} "
                f"for connection output and {neuron.shape} for neuron."
            )

        # register submodules
        self.register_module("connection_", connection)
        self.register_module("neuron_", neuron)

        # keyword argument mapping functions
        def filterkwargs(kwargs: dict[str, Any], kwamap: dict[str, str | str]):
            return {
                kwamap.get(arg): val for arg, val in kwargs.values() if arg in kwamap
            }

        # kwarg mapping for connection
        if connection_kwargs is None:
            self.kwargmap_c = lambda x: x
        else:
            self.kwargmap_c = functools.partial(filterkwargs, connection_kwargs)

        # kwarg mapping for neuron
        if neuron_kwargs is None:
            self.kwargmap_n = lambda x: x
        else:
            self.kwargmap_n = functools.partial(filterkwargs, neuron_kwargs)

    @property
    def connection(self) -> Connection:
        r"""Connection submodule.

        Args:
            value (Connection): replacement connection.

        Returns:
            Connection: existing connection.
        """
        return self.connection_

    @connection.setter
    def connection(self, value: Neuron):
        self.connection_ = value

    @property
    def neuron(self) -> Neuron:
        r"""Neuron submodule.

        Args:
            value (Neuron): replacement neuron.

        Returns:
            Neuron: existing neuron.
        """
        return self.neuron_

    @neuron.setter
    def neuron(self, value: Connection):
        self.neuron_ = value

    @property
    def synapse(self) -> Synapse:
        r"""Synapse submodule.

        Args:
            value (Synapse): replacement synapse.

        Returns:
            Synapse: existing synapse.
        """
        return self.connection_.synapse

    @synapse.setter
    def synapse(self, value: Synapse):
        self.connection_.synapse = value

    def clear(self, **kwargs):
        r"""Resets connections and neurons to their resting state.

        Keyword arguments are filtered and then passed to :py:meth:`Connection.clear`
        and :py:meth:`Neuron.clear`.
        """
        self.connection.clear(**self.kwargmap_c(kwargs))
        self.neuron.clear(**self.kwargmap_n(kwargs))

    def forward(self, *inputs, **kwargs):
        r"""Runs a simulation step of the connection and then the neurons.

        Keyword arguments are filtered and then passed to :py:meth:`Connection.forward`
        and :py:meth:`Neuron.forward`. It is expected that :py:meth:`Connection.forward`
        outputs a single tensor and :py:meth:`Neuron.forward` and takes a single
        positional argument. The output of the former is used for the input of the
        latter.
        """
        return self.neuron(
            self.connection(*inputs, **self.kwargmap_c(kwargs)),
            **self.kwargmap_n(kwargs),
        )


class CompositeLayer(Module):
    r"""Container for parallel Connections in sequence parallel Neuron objects.

    A CompositeLayer will apply the same input to multiple connections, these inputs
    can then be combined and sent to the contained neurons. Each connection-neuron
    pair can be accessed as its own layer and monitored as such, including for the
    purpose of updating.

    Args:
        named_connections (tuple[tuple[Connection, str], ...]): connection objects to
            which input will be passed, and unique valid identifiers for each.
        named_neurons (tuple[tuple[Connection, str], ...]): connection objects to
            which input will be passed, and unique valid identifiers for each.
        neuron (Neuron): neurons which take their input from the connection and their
            output is returned.
        connection_kwargs (dict[str, str] | None, optional): keyword argument
                mapping for connection methods. Defaults to None.
        neuron_kwargs (dict[str, str] | None, optional): keyword argument
                mapping for neuron methods. Defaults to None.

    Note:
        The keyword argument mappings are a dictionary, where the key is a
        kwarg in a :py:class:`Layer` method, and the corresponding value is
        the name for that kwarg which will be passed to the dependent method in
        in the :py:class:`Connection` or :py:meth:`Neuron`.

        When None, *all kwargs* are passed in. Included classes in Inferno are
        written to avoid conflicts, but that is not always guaranteed.

    Tip:
        The composed :py:class:`Neuron` does not need to be unique to this layer, and
        some architectures explicitly have multiple connections going to the same
        group of neurons. The uniqueness of the composed :py:class:`Connection` is
        not enforced, but unexpected behavior may occur if it is not unique.
    """

    def __init__(
        self,
        connection: Connection,
        neuron: Neuron,
        connection_kwargs: dict[str, str] | None = None,
        neuron_kwargs: dict[str, str] | None = None,
    ):
        Module.__init__(self)
        # warn if connection and neuron are inconsistent
        if connection.dt != neuron.dt:
            warnings.warn(
                f"inconsistent step times, {connection.dt} "
                f"for connection and {neuron.dt} for neuron."
            )

        # error if incompatible
        if connection.bsize != neuron.bsize:
            raise RuntimeError(
                f"incompatible batch sizes, {connection.bsize} "
                f"for connection and {neuron.bsize} for neuron."
            )

        if connection.outshape != neuron.shape:
            raise RuntimeError(
                f"incompatible shapes, {connection.outshape} "
                f"for connection output and {neuron.shape} for neuron."
            )

        # register submodules
        self.register_module("connection_", connection)
        self.register_module("neuron_", neuron)

        # keyword argument mapping functions
        def filterkwargs(kwargs: dict[str, Any], kwamap: dict[str, str | str]):
            return {
                kwamap.get(arg): val for arg, val in kwargs.values() if arg in kwamap
            }

        # kwarg mapping for connection
        if connection_kwargs is None:
            self.kwargmap_c = lambda x: x
        else:
            self.kwargmap_c = functools.partial(filterkwargs, connection_kwargs)

        # kwarg mapping for neuron
        if neuron_kwargs is None:
            self.kwargmap_n = lambda x: x
        else:
            self.kwargmap_n = functools.partial(filterkwargs, neuron_kwargs)

    @property
    def connection(self) -> Connection:
        r"""Connection submodule.

        Args:
            value (Connection): replacement connection.

        Returns:
            Connection: existing connection.
        """
        return self.connection_

    @connection.setter
    def connection(self, value: Neuron):
        self.connection_ = value

    @property
    def neuron(self) -> Neuron:
        r"""Neuron submodule.

        Args:
            value (Neuron): replacement neuron.

        Returns:
            Neuron: existing neuron.
        """
        return self.neuron_

    @neuron.setter
    def neuron(self, value: Connection):
        self.neuron_ = value

    @property
    def synapse(self) -> Synapse:
        r"""Synapse submodule.

        Args:
            value (Synapse): replacement synapse.

        Returns:
            Synapse: existing synapse.
        """
        return self.connection_.synapse

    @synapse.setter
    def synapse(self, value: Synapse):
        self.connection_.synapse = value

    def clear(self, **kwargs):
        r"""Resets connections and neurons to their resting state.

        Keyword arguments are filtered and then passed to :py:meth:`Connection.clear`
        and :py:meth:`Neuron.clear`.
        """
        self.connection.clear(**self.kwargmap_c(kwargs))
        self.neuron.clear(**self.kwargmap_n(kwargs))

    def forward(self, *inputs, **kwargs):
        r"""Runs a simulation step of the connection and then the neurons.

        Keyword arguments are filtered and then passed to :py:meth:`Connection.forward`
        and :py:meth:`Neuron.forward`. It is expected that :py:meth:`Connection.forward`
        outputs a single tensor and :py:meth:`Neuron.forward` and takes a single
        positional argument. The output of the former is used for the input of the
        latter.
        """
        return self.neuron(
            self.connection(*inputs, **self.kwargmap_c(kwargs)),
            **self.kwargmap_n(kwargs),
        )
