from __future__ import annotations
from . import Connection, Neuron, Synapse
from .modeling import Updater
from .. import Module
from .._internal import Proxy, argtest, rgetitem
from ..types import OneToOne, OneToMany
from ..observe import Observable
from ..extra import identity, tuplewrap
from abc import ABC, abstractmethod
from collections.abc import Iterator, Iterable, Sequence
import einops as ein
from itertools import chain
import torch
import torch.nn as nn
from typing import Any, Callable, Literal


class Cell(Module, Observable):
    r"""Pair of a Connection and Neuron produced used for training.

    Args:
        layer (Layer): layer which owns this cell.
        connection (Connection): connection for the cell.
        neuron (Neuron): neuron for the cell.
        names (tuple[str, str]): names used by the layer to uniquely identify this cell.
    """

    def __init__(
        self,
        layer: Layer,
        connection: Connection,
        neuron: Neuron,
        names: tuple[str, str],
    ):
        # call superclass constructors
        Module.__init__(self)
        Observable.__init__(self, layer, "_realign_attribute", names, None)

        # test for shape compatibility
        if connection.outshape != neuron.shape:
            raise RuntimeError(
                f"connection output shape {connection.outshape} is incompatible "
                f"with neuron shape {neuron.shape}"
            )

        # component elements
        self.connection_ = connection
        self.neuron_ = neuron

    def local_remap(self, attr: str) -> tuple[tuple[Any, ...], dict[str, Any]]:
        r"""Locally remaps an attribute for pooled monitors.

        This method should alias any local attributes being referenced
        as required. The callback ``realign`` given on initialization will
        accept the output of this as positional and keyword arguments.

        Args:
            attr (str): dot-separated attribute relative to self, to realign.

        Returns:
            tuple[tuple[Any, ...], dict[str, Any]]: tuple of positional arguments and
            keyword arguments for ``realign`` method specified on initialization.
        """
        # check that the attribute is a valid dot-chain identifier if non-empty
        if attr:
            _ = argtest.nestedidentifier("attr", attr)

        # split the identifier and check for ownership
        attrchain = attr.split(".")

        # ensure the top-level attribute is in this cell
        if attr and not hasattr(self, attrchain[0]):
            raise RuntimeError(f"cell does not have an attribute '{attrchain[0]}'")

        # remap the top-level target if pointing to a private attribute
        attrchain[0] = {
            "connection_": "connection",
            "neuron_": "neuron",
        }.get(attrchain[0], attrchain[0])

        # test against Inferno-defined alias attributes
        attrsub = {
            "updater": ["connection", "updater"],
            "synapse": ["connection", "synapse"],
            "precurrent": ["connection", "syncurrent"],
            "prespike": ["connection", "synspike"],
            "postvoltage": ["neuron", "voltage"],
            "postspike": ["neuron", "spike"],
        }.get(attrchain[0], [attrchain[0]])
        attrchain = attrsub + attrchain[1:]

        # split the chain into target and attribute
        match attrchain[0]:
            case "connection":
                return ("connection", ".".join(attrchain[1:])), {}
            case "neuron":
                return ("neuron", ".".join(attrchain[1:])), {}
            case _:
                return ("cell", ".".join(attrchain)), {}

    @property
    def connection(self) -> Connection:
        r"""Connection submodule.

        Returns:
            Connection: composed connection.
        """
        return self.connection_

    @property
    def neuron(self) -> Neuron:
        r"""Neuron submodule.

        Returns:
            Neuron: composed neuron.
        """
        return self.neuron_

    @property
    def synapse(self) -> Synapse:
        r"""Synapse submodule.

        Alias for ``connection.synapse``.

        Returns:
            Synapse: composed synapse.
        """
        return self.connection_.synapse

    @property
    def updater(self) -> Updater | None:
        r"""Updater submodule.

        Alias for ``connection.updater``.

        Returns:
            Updater | None: composed updater, if any.
        """
        return self.connection_.updater

    @property
    def precurrent(self) -> torch.Tensor:
        r"""Currents from the synapse at the time last used by the connection.

        Alias for ``connection.syncurrent``.

        Returns:
            torch.Tensor: delay-offset synaptic currents.
        """
        return self.connection.syncurrent

    @property
    def prespike(self) -> torch.Tensor:
        r"""Spikes to the synapse at the time last used by the connection.

        Alias for ``connection.synspike``.

        Returns:
            torch.Tensor: delay-offset synaptic spikes.
        """
        return self.connection.synspike

    @property
    def postvoltage(self) -> torch.Tensor:
        r"""Membrane voltages in millivolts.

        Alias for ``neuron.voltage``.

        Returns:
            torch.Tensor: membrane voltages.
        """
        return self.neuron.voltage

    @property
    def postspike(self) -> torch.Tensor:
        r"""Action potentials last generated.

        Alias for ``neuron.spike``.

        Returns:
            torch.Tensor: membrane voltages.
        """
        return self.neuron.spike

    def forward(self) -> None:
        r"""Forward call.

        Raises:
            RuntimeError: Cell cannot have its forward method called.
        """
        raise RuntimeError(
            f"'forward' method of {type(self).__name__}(Cell) cannot be called"
        )


class Layer(Module, ABC):
    r"""Representation of simultaneously processed connections and neurons."""

    def __init__(self):
        # call superclass constructor
        Module.__init__(self)

        # inner modules
        self.connections_ = nn.ModuleDict()
        self.neurons_ = nn.ModuleDict()
        self.cells_ = nn.ModuleDict()

    def clear(self, submodules: bool = True, **kwargs) -> None:
        r"""Clears the state of the layer.

        Args:
            submodules (bool, optional): if the state of connections and neurons should
                also be cleared. Defaults to ``True``.
            **kwargs (Any): keyword arguments passed to connection and neuron submodule
                ``clear`` methods, if ``submodules`` is ``True``.
        """
        if submodules:
            for connection in self.connections_:
                connection.clear(**kwargs)
            for neuron in self.neurons_:
                neuron.clear(**kwargs)

    def add_cell(self, connection: str, neuron: str) -> Cell:
        r"""Creates and adds a cell if it doesn't exist.

        If a cell already exists with the given connection and neuron, this will
        return the existing cell rather than create a new one.

        Args:
            connection (str): name of the connection for the cell to add.
            neuron (str): name of the neuron for the cell to add.

        Raises:
            AttributeError: ``connection`` does not specify a connection.
            AttributeError: ``neuron`` does not specify a neuron.

        Returns:
            Cell: cell specified by the connection and neuron.
        """
        if connection not in self.connections_:
            raise AttributeError(
                f"'connection' ('{connection}') is not a registered connection"
            )

        elif neuron not in self.neurons_:
            raise AttributeError(f"'neuron' ('{neuron}') is not a registered neuron")

        else:
            if connection not in self.cells_:
                self.cells_[connection] = nn.ModuleDict()

            if neuron not in self.cells_[connection]:
                self.cells_[connection][neuron] = Cell(
                    self,
                    self.connections_[connection],
                    self.neurons_[neuron],
                    (connection, neuron),
                )

            return self.cells_[connection][neuron]

    def get_cell(self, connection: str, neuron: str) -> Cell:
        r"""Gets a created cell if it exists.

        Args:
            connection (str): name of the connection for the cell to get.
            neuron (str): name of the neuron for the cell to get.

        Raises:
            AttributeError: no cell has been created with the specified connection
                and neuron.

        Returns:
            Cell: cell specified by the connection and neuron.
        """
        try:
            return self.cells_[connection][neuron]
        except KeyError:
            raise AttributeError(
                "no cell with the connection-neuron pair "
                f"('{connection}', '{neuron}') exists"
            )

    def del_cell(self, connection: str, neuron: str) -> None:
        r"""Deletes a created cell if it exists.

        Even if a cell hasn't been created with the given pair, if the pair is valid,
        this will not raise an error.

        Args:
            connection (str): name of the connection for the cell to delete.
            neuron (str): name of the neuron for the cell to delete.

        Raises:
            AttributeError: ``connection`` does not specify a connection.
            AttributeError: ``neuron`` does not specify a neuron.
        """
        if connection not in self.connections_:
            raise AttributeError(
                f"'connection' ('{connection}') is not a registered connection"
            )

        if neuron not in self.neurons_:
            raise AttributeError(f"'neuron' ('{neuron}') is not a registered neuron")

        if connection in self.cells_:
            if neuron in self.cells_[connection]:
                del self.cells_[connection][neuron]
            if not len(self.cells_[connection]):
                del self.cells_[connection]

    def add_connection(self, name: str, connection: Connection) -> Connection:
        r"""Adds a new connection.

        Args:
            name (str): name of the connection to add.
            connection (Connection): connection to add.

        Raises:
            RuntimeError: ``name`` already specifies a connection

        Returns:
            Connection: added connection.
        """
        if name in self.connections_:
            raise RuntimeError(f"'name' ('{name}') is already a registered connection")
        else:
            _ = argtest.identifier("name", name)
            self.connections_[name] = connection
            return self.connections_[name]

    def get_connection(self, name: str) -> Connection:
        r"""Gets an existing connection.

        Args:
            name (str): name of the connection to get.

        Raises:
            AttributeError: ``name`` does not specify a connection.

        Returns:
            Connection: connection with specified name.
        """
        try:
            return self.connections_[name]
        except KeyError:
            raise AttributeError(f"'name' ('{name}') is not a registered connection")

    def del_connection(self, name: str) -> None:
        r"""Deletes an existing connection.

        Args:
            name (str): name of the connection to delete.

        Raises:
            AttributeError: ``name`` does not specify a connection.
        """
        if name not in self.connections_:
            raise AttributeError(f"'name' ('{name}') is not a registered connection")
        else:
            del self.connections_[name]
            if name in self.cells_:
                del self.cells_[name]

    def add_neuron(self, name: str, neuron: Neuron) -> Neuron:
        r"""Adds a new neuron.

        Args:
            name (str): name of the neuron to add.
            neuron (Neuron): neuron to add.

        Raises:
            RuntimeError: ``name`` already specifies a neuron

        Returns:
            Neuron: added neuron.
        """
        if name in self.neurons_:
            raise RuntimeError(f"'name' ('{name}') is already a registered neuron")
        else:
            _ = argtest.identifier("name", name)
            self.neurons_[name] = neuron
            return self.neurons_[name]

    def get_neuron(self, name: str) -> Neuron:
        r"""Gets an existing neuron.

        Args:
            name (str): name of the neuron to get.

        Raises:
            AttributeError: ``name`` does not specify a neuron.

        Returns:
            Neuron: neuron with specified name
        """
        try:
            return self.neurons_[name]
        except KeyError:
            raise AttributeError(f"'name' ('{name}') is not a registered neuron")

    def del_neuron(self, name: str) -> None:
        r"""Deletes an existing neuron.

        Args:
            name (str): name of the neuron to delete.

        Raises:
            ValueError: ``name`` does not specify a neuron.
        """
        if name not in self.neurons_:
            raise ValueError(f"'name' ('{name}') is not a registered neuron")
        else:
            del self.neurons_[name]
            for conn in [*self.cells_]:
                if name in self.cells_[conn]:
                    del self.cells_[conn][name]
                if not len(self.cells_[conn]):
                    del self.cells_[conn]

    def _realign_attribute(
        self, connection: str, neuron: str, target: str, attr: str
    ) -> str:
        r"""Gets the attribute path for monitoring relative to the layer.

        Args:
            connection (str): name of the associated connection.
            neuron (str): name of the associated neuron.
            target (str): layer-level top attribute to target.
            attr (str): cell-relative dot-separated attribute to monitor.

        Returns:
            str: dot-separated layer-origin attribute to monitor.
        """
        # con
        match target:
            case "connection":
                if connection not in self.connections_:
                    raise AttributeError(f"'connection' ('{connection}') is not valid")
                else:
                    return f"connections_.{connection}{'.' if attr else ''}{attr}"
            case "neuron":
                if neuron not in self.neurons_:
                    raise AttributeError(f"'neuron' ('{neuron}') is not valid")
                else:
                    return f"neurons_.{neuron}{'.' if attr else ''}{attr}"
            case "cell":
                if not rgetitem(self.cells_, (connection, neuron), None):
                    raise AttributeError(
                        f"cell 'connection', 'neuron' ('{connection}', '{neuron}') is not valid"
                    )
                else:
                    return f"cells_.{connection}.{neuron}{'.' if attr else ''}{attr}"
            case _:
                raise ValueError(
                    f"invalid 'target' ('{target}') specified, expected one of: "
                    "'neuron', 'connection', 'cell'"
                )

    @property
    def connections(self) -> Proxy:
        r"""Registered connections.

        For a given ``name`` of a :py:class:`Connection` set via
        ``layer.add_connection(name)``, it can be accessed as ``layer.connections.name``.

        It can be modified in-place (including setting other attributes, adding
        monitors, etc), but it can neither be deleted nor reassigned.

        This is primarily used when targeting ``Connection`` objects with a monitor.

        Returns:
            Proxy: safe access to registered connections.
        """
        return Proxy(self.connections_, "")

    @property
    def neurons(self) -> Proxy:
        r"""Registered neurons.

        For a given ``name`` of a :py:class:`Neuron` set via
        ``layer.add_neuron(name, neuron)``, it can be accessed as
        ``layer.neurons.name``.

        It can be modified in-place (including setting other attributes, adding
        monitors, etc), but it can neither be deleted nor reassigned.

        This is primarily used when targeting ``Neuron`` objects with a monitor.

        Returns:
            Proxy: safe access to registered neurons.
        """
        return Proxy(self.neurons_, "")

    @property
    def cells(self) -> Proxy:
        r"""Registered cells.

        For a given ``connection_name`` and ``neuron_name``, the :py:class:`Cell`
        constructed via ``layer.add_cell(connection_name, neuron_name)``
        can be accessed as ``layer.cells.connection_name.neuron_name``.

        It can be modified in-place (including setting other attributes, adding
        monitors, etc), but it can neither be deleted nor reassigned.

        This is primarily used when targeting ``Cell`` objects with a monitor.

        Returns:
            Proxy: safe access to registered cells.
        """
        return Proxy(self.cells_, "", "")

    @property
    def named_connections(self) -> Iterator[tuple[str, Connection]]:
        r"""Iterable of registered connections and their names.

        Yields:
            tuple[str, Connection]: tuple of a registered connection and its name.
        """
        return ((k, v) for k, v in self.connections_.items())

    @property
    def named_neurons(self) -> Iterator[tuple[str, Neuron]]:
        r"""Iterable of registered neurons and their names.

        Yields:
            tuple[str, Neuron]: tuple of a registered neuron and its name.
        """
        return ((k, v) for k, v in self.neurons_.items())

    @property
    def named_synapses(self) -> Iterator[tuple[str, Synapse]]:
        r"""Iterable of registered connection's synapses and their names.

        Yields:
            tuple[str, Synapse]: tuple of a registered synapse and its name.
        """
        return ((k, v.synapse) for k, v in self.connections_.items())

    @property
    def named_cells(self) -> Iterator[tuple[tuple[str, str], Cell]]:
        r"""Iterable of registered cells and tuples of the connection and neuron names.

        Yields:
            tuple[tuple[str, str], torch.Tensor]: tuple of a registered cell and a tuple
            of the connection name and neuron name corresponding to it.
        """
        return chain.from_iterable(
            (((n0, n1), c) for n1, c in g.items()) for n0, g in self.cells_.items()
        )

    @abstractmethod
    def wiring(
        self, inputs: dict[str, torch.Tensor], **kwargs
    ) -> dict[str, torch.Tensor]:
        r"""Connection logic between connection outputs and neuron inputs.

        The inputs are given as a dictionary where each key is a registered input name
        and the value is the tensor output from that connection. This is expected to
        return a dictionary where each key is the name of a registered output and the
        value is the tensor to be passed to its ``__call__`` method.

        Args:
            inputs (dict[str, torch.Tensor]): dictionary of input names to tensors.

        Raises:
            NotImplementedError: ``wiring`` must be implemented by the subclass.

        Returns:
            dict[str, torch.Tensor]: dictionary of output names to tensors.
        """
        raise NotImplementedError(
            f"{type(self).__name__}(Layer) must implement " "the method `wiring`."
        )

    def update(self, clear: bool = True, **kwargs) -> None:
        r"""Applies all cumulative updates.

        This calls every updated which applies cumulative updates and any updater
        hooks are automatically called (e.g. parameter clamping).

        Args:
            clear (bool, optional): if accumulators should be cleared after updating.
                Defaults to ``True``.
        """
        for connection in self.connections_.values():
            connection.update(clear=clear, **kwargs)

    def forward(
        self,
        inputs: dict[str, tuple[torch.Tensor, ...]],
        connection_kwargs: dict[str, dict[str, Any]] | None = None,
        neuron_kwargs: dict[str, dict[str, Any]] | None = None,
        capture_intermediate: bool = False,
        **kwargs: Any,
    ) -> (
        dict[str, torch.Tensor]
        | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]
    ):
        r"""Computes a forward pass.

        The keys for ``inputs`` and ``connection_kwargs`` are the names of registered
        :py:class:`Connection` objects.

        The keys for ``neuron_kwargs`` are the names of the registered :py:class:`Neuron`
        objects.

        Underlying :py:class:`Connection` and :py:class:`Neuron` objects are called
        using ``__call__``, which in turn call :py:meth:`Connection.forward` and
        :py:meth:`Neuron.forward` respectively. The keyword argument dictionaries will
        be unpacked for each call automatically, and the inputs will be unpacked as
        positional arguments for each ``Connection`` call.

        Only input modules that have keys in ``inputs`` will be run and added to
        the positional argument of :py:meth:`wiring`.

        Args:
            inputs (dict[str, tuple[torch.Tensor, ...]]): inputs passed to the
                registered connections' forward calls.
            connection_kwargs (dict[str, dict[str, Any]] | None, optional): keyword
                arguments passed to registered connections' forward calls. Defaults to ``None``.
            neuron_kwargs (dict[str, dict[str, Any]] | None, optional): keyword
                arguments passed to registered neurons' forward calls. Defaults to ``None``.
            capture_intermediate (bool, optional): if output from the connections should
                also be returned. Defaults to ``False``.
            **kwargs (Any): keyword arguments passed to :py:meth:`wiring`.

        Returns:
            dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
            tensors from neurons and the associated neuron names, if ``capture_intermediate``,
            this is the first element of a tuple, the second being a tuple of tensors from
            connections and the associated connection names.
        """
        # replace none with empty dictionaries
        ckw = connection_kwargs if connection_kwargs else {}
        nkw = neuron_kwargs if neuron_kwargs else {}

        # get connection outputs
        res = {k: self.connections_[k](*v, **ckw.get(k, {})) for k, v in inputs.items()}

        if capture_intermediate:
            outputs = self.wiring(res, **kwargs)
            outputs = {
                k: self.neurons_[k](v, **nkw.get(k, {})) for k, v in outputs.items()
            }
            return (outputs, res)
        else:
            res = self.wiring(res, **kwargs)
            res = {k: self.neurons_[k](v, **nkw.get(k, {})) for k, v in res.items()}
            return res


class Biclique(Layer):
    r"""Layer structured as a complete bipartite graph.

    Each input is processed by its corresponding connection, with an optional
    transformation applied, before being combined with the results of all other
    connections. These are then, for each group of neurons, optionally transformed
    and then passed in.

    Each element of ``connections`` and ``c`` must be a tuple with at least two
    elements and at most three. The first of these is a string, which must be a
    Python identifier and unique to across the ``connections`` and ``neurons``. The
    second is the module itself (:py:class:`Connection` or :py:class:`Neuron`
    respectively).

    The optional third is a function which is a callable that takes and returns a
    :py:class:`~torch.Tensor`. If present, this will be applied to the output tensor
    of the corresponding ``Connection`` or input tensor of the corresponding ``Neuron``.

    Either a function to combine the tensors from the modules in ``connections`` to be passed
    into ``inputs`` or a string literal may be provided. These may be "sum", "mean",
    "prod", "min", or "max". All use ``einops`` to reduce. When providing
    a function, it must take a tuple of tensors (equal to the number of inputs) and
    produce a single tensor output.

    Args:
        connections (Iterable[tuple[str, Connection] | tuple[str, Connection, OneToOne[torch.Tensor]]]):
            modules which receive inputs given to the layer.
        neurons (Iterable[tuple[str, Neuron] | tuple[str, Neuron, OneToOne[torch.Tensor]]]):
            modules which produce output from the layer.
        combine (Callable[[dict[str, torch.Tensor]], torch.Tensor] | Literal["sum", "mean", "prod", "min", "max"], optional):
            function to combine tensors from inputs into a single tensor for outputs.
            Defaults to ``"sum"``.

    Note:
        When ``combine`` is not a string, keyword arguments passed into ``__call__``,
        other than those captured in :py:meth:`forward` will be passed in.
    """

    def __init__(
        self,
        connections: Iterable[
            tuple[str, Connection] | tuple[str, Connection, OneToOne[torch.Tensor]]
        ],
        neurons: Iterable[
            tuple[str, Neuron] | tuple[str, Neuron, OneToOne[torch.Tensor]]
        ],
        combine: (
            Callable[[dict[str, torch.Tensor]], torch.Tensor]
            | Literal["sum", "mean", "prod", "min", "max"]
        ) = "sum",
    ):
        # superclass constructor
        Layer.__init__(self)

        # callables
        self.post_input = {}
        self.pre_output = {}
        match (combine.lower() if isinstance(combine, str) else combine):

            case "sum" | "mean" | "prod" | "min" | "max":

                def combinefn(tensors, **kwargs):
                    return ein.reduce(
                        list(tensors.values()), "s ... -> ...", combine.lower()
                    )

                self._combine = combinefn

            case _:
                if isinstance(combine, str):
                    raise ValueError(
                        f"'combine' ('{combine}'), when a string, must be one of: "
                        "'sum', 'mean', 'prod', 'min', 'max'"
                    )
                else:
                    self._combine = combine

        # unpack arguments and ensure they are non-empty
        connections = [*connections]
        if not len(connections):
            raise ValueError("'connections' cannot be empty")

        neurons = [*neurons]
        if not len(neurons):
            raise ValueError("'neurons' cannot be empty")

        # add inputs
        for idx, c in enumerate(connections):
            match len(c):
                case 2:
                    Layer.add_connection(self, *c)
                    self.post_input[c[0]] = lambda x: x
                case 3:
                    Layer.add_connection(self, *c[:-1])
                    self.post_input[c[0]] = c[2]
                case _:
                    raise ValueError(
                        f"element at position {idx} in 'connections' has invalid "
                        f"number of elements {len(c)}"
                    )

        # add outputs
        for idx, n in enumerate(neurons):
            match len(n):
                case 2:
                    Layer.add_neuron(self, *n)
                    self.pre_output[n[0]] = lambda x: x
                case 3:
                    Layer.add_neuron(self, *n[:-1])
                    self.pre_output[n[0]] = n[2]
                case _:
                    raise ValueError(
                        f"element at position {idx} in 'neurons' has invalid "
                        f"number of elements {len(n)}"
                    )

        # construct cells
        for c in connections:
            for n in neurons:
                _ = Layer.add_cell(self, c[0], n[0])

    def add_cell(self, *args, **kwargs) -> None:
        raise RuntimeError(
            f"{type(self).__name__}(Biclique) does not support adding cells"
        )

    def del_cell(self, *args, **kwargs) -> None:
        raise RuntimeError(
            f"{type(self).__name__}(Biclique) does not support removing cells"
        )

    def add_connection(self, *args, **kwargs) -> None:
        raise RuntimeError(
            f"{type(self).__name__}(Biclique) does not support adding connections"
        )

    def del_connection(self, *args, **kwargs) -> None:
        raise RuntimeError(
            f"{type(self).__name__}(Biclique) does not support removing connections"
        )

    def add_neuron(self, *args, **kwargs) -> None:
        raise RuntimeError(
            f"{type(self).__name__}(Biclique) does not support adding neurons"
        )

    def del_neuron(self, *args, **kwargs) -> None:
        raise RuntimeError(
            f"{type(self).__name__}(Biclique) does not support removing neurons"
        )

    def wiring(
        self, inputs: dict[str, torch.Tensor], **kwargs
    ) -> dict[str, torch.Tensor]:
        r"""Connection logic between connection outputs and neuron inputs.

        This implements the forward logic of the biclique topology where the tensors
        from the inputs are transformed, combined, and transformed again before
        being passed to the outputs. Transforms which were unspecified are assumed to
        be identity.

        Args:
            inputs (dict[str, torch.Tensor]): dictionary of connection names to tensors.

        Returns:
            dict[str, torch.Tensor]: dictionary of output names to tensors.
        """
        return {
            k: v(
                self._combine(
                    {k: self.post_input[k](v) for k, v in inputs.items()}, **kwargs
                )
            )
            for k, v in self.pre_output.items()
        }


class Serial(Layer):
    r"""Layer with a single connection and single neuron group.

    This wraps :py:class:`Layer` to provide simplified accessors and a simplified
    :py:meth:`forward` method for layers with one connection and one neuron group.

    Args:
        connection (Connection): module which receives input to the layer.
        neuron (Neuron): module which generates output from the layer.
        transform (OneToOne[torch.Tensor] | None, optional): function
            to apply to connection output before passing into neurons. Defaults to ``None``.
        connection_name (str, optional): name for the connection in the layer. Defaults
            to ``"serial"``.
        neuron_name (str, optional): name for the neuron in the layer. Defaults to
            ``"serial"``.

    Note:
        When ``transform`` is not specified, the identity function is used. Keyword
        arguments passed into ``__call__``, other than those captured in
        :py:meth:`forward` will be passed in.

    Note:
        The :py:class:`Layer` object underlying a ``Serial`` object has ``connection``
        and ``neuron`` registered with names ``"serial"``. Convenience properties can be
        used to avoid accessing manually.
    """

    def __init__(
        self,
        connection: Connection,
        neuron: Neuron,
        transform: OneToOne[torch.Tensor] | None = None,
        connection_name: str = "serial",
        neuron_name: str = "serial",
    ):
        # call superclass constructor
        Layer.__init__(self)

        # set names
        self.__connection_name = connection_name
        self.__neuron_name = neuron_name

        # add connection and neuron
        Layer.add_connection(self, self.__connection_name, connection)
        Layer.add_neuron(self, self.__neuron_name, neuron)
        _ = Layer.add_cell(self, self.__connection_name, self.__neuron_name)

        # set transformation used
        if transform:
            self._transform = transform
        else:

            def transfn(tensor, **kwargs):
                return tensor

            self._transform = transfn

    def add_cell(self, *args, **kwargs) -> None:
        raise RuntimeError(
            f"{type(self).__name__}(Serial) does not support adding cells"
        )

    def del_cell(self, *args, **kwargs) -> None:
        raise RuntimeError(
            f"{type(self).__name__}(Serial) does not support removing cells"
        )

    def add_connection(self, *args, **kwargs) -> None:
        raise RuntimeError(
            f"{type(self).__name__}(Serial) does not support adding connections"
        )

    def del_connection(self, *args, **kwargs) -> None:
        raise RuntimeError(
            f"{type(self).__name__}(Serial) does not support removing connections"
        )

    def add_neuron(self, *args, **kwargs) -> None:
        raise RuntimeError(
            f"{type(self).__name__}(Serial) does not support adding neurons"
        )

    def del_neuron(self, *args, **kwargs) -> None:
        raise RuntimeError(
            f"{type(self).__name__}(Serial) does not support removing neurons"
        )

    @property
    def connection(self) -> Connection:
        r"""Registered connection.

        Returns:
            Connection: registered connection.
        """
        return self.get_connection(self.__connection_name)

    @property
    def neuron(self) -> Neuron:
        r"""Registered neuron.

        Returns:
            Neuron: registered neuron.
        """
        return self.get_neuron(self.__neuron_name)

    @property
    def synapse(self) -> Synapse:
        r"""Registered synapse.

        Returns:
            Synapse: registered connection's synapse.
        """
        return self.get_connection(self.__connection_name).synapse

    @property
    def updater(self) -> Updater:
        r"""Registered updater.

        Returns:
            Updater: registered connection's updater.
        """
        return self.get_connection(self.__connection_name).updater

    @property
    def cell(self) -> Cell:
        r"""Registered cell.

        Returns:
            Cell: registered cell.
        """
        return self.get_cell(self.__connection_name, self.__neuron_name)

    def wiring(
        self, inputs: dict[str, torch.Tensor], **kwargs
    ) -> dict[str, torch.Tensor]:
        r"""Connection logic between connection outputs and neuron inputs.

        This implements the forward logic of the serial topology. The ``transform`` is
        applied to the result of the connection before being passed to the neuron. If
        not specified, it is assumed to be identity.

        Args:
            inputs (dict[str, torch.Tensor]): dictionary of input names to tensors.

        Returns:
            dict[str, torch.Tensor]: dictionary of output names to tensors.
        """
        return {
            self.__neuron_name: self._transform(
                inputs[self.__connection_name], **kwargs
            )
        }

    def forward(
        self,
        *inputs: torch.Tensor,
        connection_kwargs: dict[str, Any] | None = None,
        neuron_kwargs: dict[str, Any] | None = None,
        capture_intermediate: bool = False,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        r"""Computes a forward pass.

        Args:
            *inputs (torch.Tensor): values passed to the connection.
            connection_kwargs (dict[str, dict[str, Any]] | None, optional): keyword
                arguments for the connection's forward call. Defaults to ``None``.
            neuron_kwargs (dict[str, dict[str, Any]] | None, optional): keyword
                arguments for the neuron's forward call. Defaults to ``None``.
            capture_intermediate (bool, optional): if output from the connections should
                also be returned. Defaults to ``False``.
            **kwargs (Any): keyword arguments passed to :py:meth:`wiring`.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: output from the neurons,
            if ``capture_intermediate``, this is the first element of a tuple, the second
            being the output from the connection.
        """
        # wrap non-empty dictionaries
        ckw = {self.__connection_name: connection_kwargs} if connection_kwargs else None
        nkw = {self.__neuron_name: neuron_kwargs} if neuron_kwargs else None

        # call parent forward
        res = Layer.forward(
            self,
            {self.__connection_name: inputs},
            connection_kwargs=ckw,
            neuron_kwargs=nkw,
            capture_intermediate=capture_intermediate,
            **kwargs,
        )

        # unpack to sensible output
        if capture_intermediate:
            return (
                res[0][self.__neuron_name],
                res[1][self.__connection_name],
            )
        else:
            return res[self.__neuron_name]


class RecurrentSerial(Layer):
    r"""Layer with a single feedforward connection and neuron group, and two feedback connections with a neuron group in-between.

    .. math::
        \begin{align*}
            \texttt{inputs} &\rightarrow \texttt{feedfwd_connection} \\
            \texttt{feedfwd_connection} + \texttt{feedback_connection} &\rightarrow \texttt{feedfwd_neuron} \\
            \texttt{feedfwd_neuron} &\rightarrow \texttt{lateral_connection} \\
            \texttt{lateral_connection} &\rightarrow \texttt{feedback_neuron} \\
            \texttt{feedback_neuron} &\rightarrow \texttt{feedback_connection}
        \end{align*}

    This wraps :py:class:`Layer` to provide simplified accessors and a simplified
    :py:meth:`forward` method for layers with one feedforward connection, two feedback
    connections, and two neuron groups.

    Args:
        feedfwd_connection (Connection): module which receives input to the layer.
        lateral_connection (Connection): module which receives input from the
            feedforward neurons.
        feedback_connection (Connection): module which receives input from the
            feedback neurons and applies it to the feedforward neurons.
        feedfwd_neuron (Neuron): module which generates output from the layer.
        feedback_neuron (Neuron): module which generates feedback spikes.
        feedfwd_out_transform (OneToOne[torch.Tensor] | None, optional): function
            to apply to feedforward connection output. Defaults to ``None``.
        lateral_out_transform (OneToOne[torch.Tensor] | None, optional): function
            to apply to lateral connection output. Defaults to ``None``.
        feedback_out_transform (OneToOne[torch.Tensor] | None, optional): function
            to apply to feedback connection output. Defaults to ``None``.
        lateral_in_transform (OneToMany[torch.Tensor] | None, optional): function
            to apply to lateral connection input. Defaults to ``None``.
        feedback_in_transform (OneToMany[torch.Tensor] | None, optional): function
            to apply to feedback connection input. Defaults to ``None``.
        feedfwd_connection_name (str, optional): name for the feedforward
            connection in the layer. Defaults to ``"feedfwd"``.
        lateral_connection_name (str, optional): name for the lateral
            connection in the layer. Defaults to ``"lateral"``.
        feedback_connection_name (str, optional): name for the feedback
            connection in the layer. Defaults to ``"feedback"``.
        feedfwd_neuron_name (str, optional): name for the neuron in the layer.
            Defaults to ``"feedfwd"``.
        feedback_neuron_name (str, optional): name for the neuron in the layer.
            Defaults to ``"feedback"``.
        trainable_feedback (bool, optional): if feedback connections should be
            trainable. Defaults to ``False``.

    Note:
        When any of ``feedfwd_out_transform``, ``lateral_out_transform``, ``feedback_out_transform``,
        or ``feedback_in_transform`` is not specified, the identity function is used.
        Keyword arguments passed into ``__call__``, other than those captured in
        :py:meth:`forward` will be passed in.

    Important:
        When ``trainable_feedback`` is set to ``True``, the feedback connection and
        neuron shapes need to be compatible for creating :py:class:`~inferno.neural.Cell`
        objects.

    Note:
        When any of ``feedfwd_out_transform``, ``lateral_out_transform``,
        ``feedback_out_transform``, ``lateral_in_transform``,
        or ``feedback_in_transform`` is not specified, the identity function is used
        (the latter two also wrapping the input in a tuple). Keyword arguments passed
        into ``__call__``, other than those captured in :py:meth:`forward` will be passed
        in. The ``lateral_in_transform`` and ``feedback_in_transform`` functions are
        only applied to the spiking input from the feedforward and feedback neurons
        respectively.
    """

    def __init__(
        self,
        feedfwd_connection: Connection,
        lateral_connection: Connection,
        feedback_connection: Connection,
        feedfwd_neuron: Neuron,
        feedback_neuron: Neuron,
        *,
        feedfwd_out_transform: OneToOne[torch.Tensor] | None = None,
        lateral_out_transform: OneToOne[torch.Tensor] | None = None,
        feedback_out_transform: OneToOne[torch.Tensor] | None = None,
        lateral_in_transform: OneToMany[torch.Tensor] | None = None,
        feedback_in_transform: OneToMany[torch.Tensor] | None = None,
        feedfwd_connection_name: str = "feedfwd",
        lateral_connection_name: str = "lateral",
        feedback_connection_name: str = "feedback",
        feedfwd_neuron_name: str = "feedfwd",
        feedback_neuron_name: str = "feedback",
        trainable_feedback: bool = False,
    ):
        # call superclass constructor
        Layer.__init__(self)

        # register feedback tensor
        self.register_buffer("feedback_spikes", None)

        # set names
        self.__feedfwd_connection_name = feedfwd_connection_name
        self.__lateral_connection_name = lateral_connection_name
        self.__feedback_connection_name = feedback_connection_name
        self.__feedfwd_neuron_name = feedfwd_neuron_name
        self.__feedback_neuron_name = feedback_neuron_name

        # add connections and neurons
        Layer.add_connection(self, self.__feedfwd_connection_name, feedfwd_connection)
        Layer.add_connection(self, self.__lateral_connection_name, lateral_connection)
        Layer.add_connection(self, self.__feedback_connection_name, feedback_connection)

        Layer.add_neuron(self, self.__feedfwd_neuron_name, feedfwd_neuron)
        Layer.add_neuron(self, self.__feedback_neuron_name, feedback_neuron)

        _ = Layer.add_cell(
            self, self.__feedfwd_connection_name, self.__feedfwd_neuron_name
        )
        if trainable_feedback:
            _ = Layer.add_cell(
                self,
                self.__lateral_connection_name,
                self.__feedback_neuron_name,
            )
            _ = Layer.add_cell(
                self, self.__feedback_connection_name, self.__feedfwd_neuron_name
            )

        # set transformations used
        self._feedfwd_out_transform = (
            feedfwd_out_transform if feedfwd_out_transform else identity
        )
        self._lateral_out_transform = (
            lateral_out_transform if lateral_out_transform else identity
        )
        self._feedback_out_transform = (
            feedback_out_transform if feedback_out_transform else identity
        )
        self._lateral_in_transform = (
            lateral_in_transform if lateral_in_transform else tuplewrap
        )
        self._feedback_in_transform = (
            feedback_in_transform if feedback_in_transform else tuplewrap
        )

    def clear(
        self, clear_feedback: bool = True, submodules: bool = True, **kwargs
    ) -> None:
        r"""Clears the state of the layer.

        Args:
            clear_feedback (bool, optional): if the feedback spikes should be cleared.
                Defaults to ``True``.
            submodules (bool, optional): if the state of connections and neurons should
                also be cleared. Defaults to ``True``.
            **kwargs (Any): keyword arguments passed to connection and neuron submodule
                ``clear`` methods, if ``submodules`` is ``True``.
        """
        if clear_feedback:
            self.feedback_spikes = None

        if submodules:
            for connection in self.connections_:
                connection.clear(**kwargs)
            for neuron in self.neurons_:
                neuron.clear(**kwargs)

    def add_cell(self, *args, **kwargs) -> None:
        raise RuntimeError(
            f"{type(self).__name__}(FeedbackSerial) does not support adding cells"
        )

    def del_cell(self, *args, **kwargs) -> None:
        raise RuntimeError(
            f"{type(self).__name__}(FeedbackSerial) does not support removing cells"
        )

    def add_connection(self, *args, **kwargs) -> None:
        raise RuntimeError(
            f"{type(self).__name__}(FeedbackSerial) does not support adding connections"
        )

    def del_connection(self, *args, **kwargs) -> None:
        raise RuntimeError(
            f"{type(self).__name__}(FeedbackSerial) does not support removing connections"
        )

    def add_neuron(self, *args, **kwargs) -> None:
        raise RuntimeError(
            f"{type(self).__name__}(FeedbackSerial) does not support adding neurons"
        )

    def del_neuron(self, *args, **kwargs) -> None:
        raise RuntimeError(
            f"{type(self).__name__}(FeedbackSerial) does not support removing neurons"
        )

    @property
    def feedfwd_connection(self) -> Connection:
        r"""Registered feedforward connection.

        Returns:
            Connection: registered feedforward connection.
        """
        return self.get_connection(self.__feedfwd_connection_name)

    @property
    def lateral_connection(self) -> Connection:
        r"""Registered lateral connection.

        Returns:
            Connection: registered lateral connection.
        """
        return self.get_connection(self.__lateral_connection_name)

    @property
    def feedback_connection(self) -> Connection:
        r"""Registered feedback connection.

        Returns:
            Connection: registered feedback connection.
        """
        return self.get_connection(self.__feedback_connection_name)

    @property
    def feedfwd_neuron(self) -> Neuron:
        r"""Registered feedforward neuron.

        Returns:
            Neuron: registered feedforward neuron.
        """
        return self.get_neuron(self.__feedfwd_neuron_name)

    @property
    def feedback_neuron(self) -> Neuron:
        r"""Registered feedback neuron.

        Returns:
            Neuron: registered feedback neuron.
        """
        return self.get_neuron(self.__feedback_neuron_name)

    @property
    def feedfwd_synapse(self) -> Synapse:
        r"""Registered feedforward synapse.

        Returns:
            Synapse: registered feedforward connection's synapse.
        """
        return self.get_connection(self.__feedfwd_connection_name).synapse

    @property
    def lateral_synapse(self) -> Synapse:
        r"""Registered lateral synapse.

        Returns:
            Synapse: registered lateral connection's synapse.
        """
        return self.get_connection(self.__lateral_connection_name).synapse

    @property
    def feedback_synapse(self) -> Synapse:
        r"""Registered feedback synapse.

        Returns:
            Synapse: registered feedback connection's synapse.
        """
        return self.get_connection(self.__feedback_connection_name).synapse

    @property
    def feedfwd_updater(self) -> Updater:
        r"""Registered feedforward updater.

        Returns:
            Updater: registered feedforward connection's updater.
        """
        return self.get_connection(self.__feedfwd_connection_name).updater

    @property
    def lateral_updater(self) -> Updater | None:
        r"""Registered lateral updater.

        Returns:
            Updater: registered lateral connection's updater.
        """
        return self.get_connection(self.__lateral_connection_name).updater

    @property
    def feedback_updater(self) -> Updater | None:
        r"""Registered feedback updater.

        Returns:
            Updater: registered feedback connection's updater.
        """
        return self.get_connection(self.__feedback_connection_name).updater

    @property
    def feedfwd_cell(self) -> Cell:
        r"""Registered feedforward cell.

        Returns:
            Cell: registered feedforward cell.
        """
        return self.get_cell(self.__feedfwd_connection_name, self.__feedfwd_neuron_name)

    @property
    def lateral_cell(self) -> Cell | None:
        r"""Registered lateral cell.

        Returns:
            Cell: registered lateral cell, if constructed with
            ``trainable_feedback``, otherwise ``None``.
        """
        if self.__lateral_connection_name in self.cells_:
            return self.get_cell(
                self.__lateral_connection_name, self.__feedback_neuron_name
            )
        else:
            return None

    @property
    def feedback_cell(self) -> Cell | None:
        r"""Registered feedback cell.

        Returns:
            Cell: registered feedback cell, if constructed with
            ``trainable_feedback``, otherwise ``None``.
        """
        if self.__feedback_connection_name in self.cells_:
            return self.get_cell(
                self.__feedback_connection_name, self.__feedfwd_neuron_name
            )
        else:
            return None

    def wiring(
        self,
        inputs: dict[str, torch.Tensor],
        forward_pass: bool,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        r"""Connection logic between connection outputs and neuron inputs.

        This implements the forward logic of the feedback serial topology. The
        ``transform`` is applied to the result of the connection before being passed to the neuron. If
        not specified, it is assumed to be identity.

        Args:
            inputs (dict[str, torch.Tensor]): dictionary of input names to tensors.
            forward_pass (bool) if this is a forward-pass step.

        Returns:
            dict[str, torch.Tensor]: dictionary of output names to tensors.
        """
        if forward_pass:
            return {
                self.__feedfwd_neuron_name: self._feedfwd_out_transform(
                    inputs[self.__feedfwd_connection_name]
                )
                + self._feedback_out_transform(inputs[self.__feedback_connection_name])
            }
        else:
            return {
                self.__feedback_neuron_name: self._lateral_out_transform(
                    inputs[self.__lateral_connection_name]
                )
            }

    def forward(
        self,
        *inputs: torch.Tensor,
        lateral_connection_args: Sequence[torch.Tensor] | None = None,
        feedback_connection_args: Sequence[torch.Tensor] | None = None,
        feedfwd_connection_kwargs: dict[str, Any] | None = None,
        lateral_connection_kwargs: dict[str, Any] | None = None,
        feedback_connection_kwargs: dict[str, Any] | None = None,
        feedfwd_neuron_kwargs: dict[str, Any] | None = None,
        feedback_neuron_kwargs: dict[str, Any] | None = None,
        capture_intermediate: bool = False,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        r"""Computes a forward pass.

        Args:
            *inputs (torch.Tensor): values passed to the connection.
            lateral_connection_args (Sequence[torch.Tensor] | None, optional): additional
                positional arguments for lateral connection's forward call.
                Defaults to ``None``.
            feedback_connection_args (Sequence[torch.Tensor] | None, optional): additional
                positional arguments for feedback connection's forward call.
                Defaults to ``None``.
            feedfwd_connection_kwargs (dict[str, dict[str, Any]] | None, optional): keyword
                arguments for the feedforward  connection's forward call.
                Defaults to ``None``.
            lateral_connection_kwargs (dict[str, dict[str, Any]] | None, optional): keyword
                arguments for the lateral connection's forward call.
                Defaults to ``None``.
            feedback_connection_kwargs (dict[str, dict[str, Any]] | None, optional): keyword
                arguments for the feedback connection's forward call.
                Defaults to ``None``.
            feedfwd_neuron_kwargs (dict[str, dict[str, Any]] | None, optional): keyword
                arguments for the feedforward neuron's forward call.
                Defaults to ``None``.
            feedback_neuron_kwargs (dict[str, dict[str, Any]] | None, optional): keyword
                arguments for the feedback neuron's forward call.
                Defaults to ``None``.
            capture_intermediate (bool, optional): if output from the connections should
                also be returned. Defaults to ``False``.
            **kwargs (Any): keyword arguments passed to :py:meth:`wiring`.

        Returns:
            tuple[torch.Tensor, torch.Tensor] | tuple[tuple[torch.Tensor, torch.Tensor], dict[str, torch.Tensor]]: tuple
            of output from the feedforward and feedback neurons, in that order. If
            ``capture_intermediate``, this is the first element of a tuple, the second
            being the outputs from the connections, as a dictionary of connection names
            to their corresponding outputs.

        Note:
            On the first input, zero-valued tensors are given as input for calculating
            feedback.
        """
        # wrap non-empty dictionaries
        ckw = (
            {}
            | (
                {self.__feedfwd_connection_name: feedfwd_connection_kwargs}
                if feedfwd_connection_kwargs
                else {}
            )
            | (
                {self.__lateral_connection_name: lateral_connection_kwargs}
                if lateral_connection_kwargs
                else {}
            )
            | (
                {self.__feedback_connection_name: feedback_connection_kwargs}
                if feedback_connection_kwargs
                else {}
            )
        )
        nkw = (
            {}
            | (
                {self.__feedfwd_neuron_name: feedfwd_neuron_kwargs}
                if feedfwd_neuron_kwargs
                else {}
            )
            | (
                {self.__feedback_neuron_name: feedback_neuron_kwargs}
                if feedback_neuron_kwargs
                else {}
            )
        )

        # set recurrent spikes
        if self.feedback_spikes is None:
            self.feedback_spikes = torch.zeros_like(
                self.get_neuron(self.__feedback_neuron_name).spike
            )

        # call parent forward (forward-pass)
        fres = Layer.forward(
            self,
            {
                self.__feedfwd_connection_name: inputs,
                self.__feedback_connection_name: self._feedback_in_transform(
                    self.feedback_spikes
                )
                + (tuple(feedback_connection_args) if feedback_connection_args else ()),
            },
            connection_kwargs=ckw,
            neuron_kwargs=nkw,
            capture_intermediate=True,
            forward_pass=True,
            **kwargs,
        )

        # call parent forward (feedback-pass)
        bres = Layer.forward(
            self,
            {
                self.__lateral_connection_name: self._lateral_in_transform(
                    self.get_neuron(self.__feedfwd_neuron_name).spike
                )
                + (tuple(lateral_connection_args) if lateral_connection_args else ()),
            },
            connection_kwargs=ckw,
            neuron_kwargs=nkw,
            capture_intermediate=True,
            forward_pass=False,
            **kwargs,
        )

        # update recurrent spikes
        self.feedback_spikes = self.get_neuron(self.__feedback_neuron_name).spike

        # unpack to sensible output
        if capture_intermediate:
            return (
                (
                    fres[0][self.__feedfwd_neuron_name],
                    bres[0][self.__feedback_neuron_name],
                ),
                fres[1] | bres[1],
            )
        else:
            return (
                fres[0][self.__feedfwd_neuron_name],
                bres[0][self.__feedback_neuron_name],
            )
