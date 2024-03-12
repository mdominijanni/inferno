from __future__ import annotations
from . import Connection, Neuron, Synapse
from .modeling import Updater
from .hooks import Normalization, Clamping  # noqa:F401; ignore, used for docs
from .. import Module
from .._internal import Proxy, argtest, rgetattr, rgetitem
from ..types import OneToOne
from ..observe import Monitor, MonitorConstructor
from abc import ABC, abstractmethod
from collections.abc import Iterator, Iterable, Mapping
import einops as ein
from itertools import chain
import torch
import torch.nn as nn
from typing import Any, Callable, Literal
import weakref


class Cell(Module):
    def __init__(
        self,
        layer: Layer,
        connection: Connection,
        neuron: Neuron,
        names: tuple[str, str],
    ):
        # call superclass constructor
        Module.__init__(self)

        # component elements
        self.connection_ = connection
        self.neuron_ = neuron

        # reference to owner and names
        self._layer = weakref.ref(layer)
        self._names = names

    def get_monitor(
        self,
        name: str,
        attr: str,
        monitor: MonitorConstructor,
        pool: Iterable[Cell, Mapping[str, Monitor]] | None = None,
        **tags: Any,
    ) -> Monitor:
        r"""Creates or alias a monitor on the owning layer.

        Args:
            name (str): name of the monitor to add.
            attr (str): dot-separated attribute path, relative to this cell, to monitor.
            monitor (MonitorConstructor): partial constructor for the monitor to add.
            pool (Iterable[Cell, Mapping[str, Monitor]] | None, optional): pool to
                search for compatible monitor, always creates a new one if None.
                Defaults to None.
            **tags (Any): tags to add to the monitor and to test for uniqueness.

        Raises:
            RuntimeError: attribute must be a member of this trainable.
            RuntimeError: 'updater.connection' is the only valid head of the attribute
                chain starting with 'updater'.

        Returns:
            Monitor: added monitor.

        Important:
            All monitors added this way will be hooked to the :py:class:`Layer` which
            owns this ``Cell`, and therefore will be associated with its ``forward``
            call. If a monitor targeting an updater should trigger on updating, it
            should be directly added to the updater. These tags are added via reflection
            and do not persist in the state dictionary. If no pool is specified, the
            new monitor will not have added tags, which prevents future aliasing.
        """
        # check that the attribute is a valid dot-chain identifier
        _ = argtest.nestedidentifier("attr", attr)

        # split the identifier and check for ownership
        attrchain = attr.split(".")

        # ensure the top-level attribute is in this cell
        if not hasattr(self, attrchain[0]):
            raise RuntimeError(f"this cell does not have an attribute '{attrchain[0]}'")

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
                target, attr = "connection", ".".join(attrchain[1:])
            case "neuron":
                target, attr = "neuron", ".".join(attrchain[1:])
            case _:
                target, attr = "cell", ".".join(attrchain)

        # get resolved attribute if the layer exists
        if not self._layer():
            raise RuntimeError("layer is no longer in memory")
        else:
            attr = self._layer()._align_attribute(*self._names, target, attr)

        # return new monitor if pool is undefined (do not tag, guaranteed unique)
        if not pool:
            return monitor(attr, self._layer())

        # get test tags
        tags = {"attr": attr, **tags}
        found = None

        for cell, monitors in pool:
            # skip invalid cells or cells from a different layer
            if not (cell._layer() or id(cell._layer()) == id(self._layer())):
                continue

            # skip if the named monitor doesn't exist
            if name not in monitors:
                continue

            # create the alias if tags match
            if hasattr(monitors[name], "__tags") and monitors[name].__tags == tags:
                found = monitors[name]
                if id(cell) == id(self):  # break if identical cell
                    break

        # return alias or create new monitor
        if found:
            return found
        else:
            monitor = monitor(attr, self._layer())
            monitor.__tags = tags
            return monitor

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
        """Updater submodule.

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
        """Forward call.

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

        # set cells dict so it is not part of PyTorch's state
        object.__setattr__(self, "cells_", nn.ModuleDict())

    def __getitem__(self, name: str | tuple[str, str]) -> Connection | Neuron | Cell:
        r"""Retrieves a previously added connection, neuron, or cell.

        The given ``name`` can either be a string (in which case it checks first for
        a :py:class:`Connection` then a :py:class:`Neuron`) or a 2-tuple of strings,
        in which case it expects the first string to be the name of a ``Connection`` and
        the second to be the name of a ``Neuron``, and the corresponding :py:class`Cell`
        is retrieved. If the ``Cell`` has not yet been accessed in this way, it will
        first be created.

        Args:
            name (str | tuple[str, str]): name of the component module to retrieve.

        Returns:
            Connection | Neuron | Cell: retrieved module.
        """
        try:
            if isinstance(name, tuple):
                if name[0] in self.connections_:
                    # create cell group for connection if it does not exist
                    if name[0] not in self.cells_:
                        self.cells_[name[0]] = nn.ModuleDict()

                    if name[1] in self.neurons_:
                        # create cell for neuron if it does not exist
                        if name[1] not in self.cells_[name[0]]:
                            self.cells_[name[0]][name[1]] = Cell(
                                self,
                                self.connections_[name[0]],
                                self.neurons_[name[1]],
                                name,
                            )
                        return self.cells_[name[0]][name[1]]

                    else:
                        raise AttributeError(
                            f"'name' ('{name}') is not a registered neuron"
                        )

                else:
                    raise AttributeError(
                        f"'name' ('{name}') is not a registered connection"
                    )

            elif name in self.connections_:
                return self.connections_[name]

            elif name in self.neurons_:
                return self.neurons_[name]

            else:
                raise AttributeError(
                    f"'name' ('{name}') is not a registered connection or neuron"
                )

        except IndexError:
            raise ValueError("tuple 'name' must have exactly two elements")

    def __setitem__(self, name: str, module: Connection | Neuron) -> None:
        r"""Registers a new connection or neuron.

        The specified ``name`` must be a valid Python identifier, and the ``Layer``
        must not already have a :py:class:`Connection` or :py:class:`Neuron`
        registered with the same name.

        If a connection is not updatable (i.e. if it does not contain an
        :py:class:`Updater`), the default updater will be added to it.

        Args:
            name (str): attribute name for the connection or neuron.
            module (Connection | Neuron): connection used for inputs or neuron used for
                outputs.
        """
        # ensure the name is not already assigned
        if name in self.connections_ or name in self.neurons_:
            raise ValueError(
                "item assignment cannot be used to reassign registered modules"
            )

        # ensure the name is a valid identifier
        _ = argtest.identifier("name", name)

        if isinstance(module, Connection):
            self.connections_[name] = module

        elif isinstance(module, Neuron):
            self.neurons_[name] = module

        else:
            _ = argtest.instance("module", module, (Connection, Neuron))

    def __delitem__(self, name: str | tuple[str, str]) -> None:
        """Deletes a connection, neuron, or cell.

        Args:
            name (str | tuple[str, str]): name of the component module to delete.

        Raises:
            AttributeError: name does not specify a registered connection, neuron, or cell.

        Note:
            When ``name`` is a tuple, that cell  will be deleted but the associated
            connection and neuron will not be. Deletion of a cell will only fail if the
            connection or neuron do not exist, even if that cell doesn't.
        """
        connections = []
        neurons = []

        if isinstance(name, tuple):
            try:
                if name[0] in self.connections_:
                    if name[1] in self.neurons_:
                        connections.append(name[0])
                        neurons.append(name[1])
                    else:
                        raise AttributeError(f"'name' ('{name}') not an added neuron")
                else:
                    raise AttributeError(f"'name' ('{name}') not an added connection")

            except IndexError:
                raise ValueError("tuple 'name' must have exactly two elements")

        elif name in self.connections_:
            del self.connections_[name]
            connections.append(name)
            neurons.extend(self.neurons_.keys())

        elif name in self.neurons_:
            del self.neurons_[name]
            connections.extend(self.connections_.keys())
            neurons.append(name)

        else:
            raise AttributeError(
                f"'name' ('{name}') is not a registered connection or neuron"
            )

        # clean up cells
        for c in connections:
            if c in self.cells_:
                for n in neurons:
                    if n in self.cells_[c]:
                        del self.cells_[c][n]

                if not len(self.cells_[c]):
                    del self.cells_[c]

        else:
            raise AttributeError(
                f"'name' ('{name}') is not a registered connection or neuron"
            )

    def __contains__(self, name: str | tuple[str, str]) -> bool:
        r"""Checks if a connection, neuron, or cell is in the layer.

        Args:
            name (str | tuple[str, str]): name of the connection or neuron, or a tuple
                of names specifying a cell to test for.

        Returns:
            bool: if the specified connection, neuron, or cell is in the layer.
        """
        if isinstance(name, tuple):
            try:
                return name[0] in self.cells_ and name[1] in self.cells_[name[0]]
            except IndexError:
                raise ValueError("tuple 'name' must have exactly two elements")
        else:
            return name in self.connections_ or name in self.neurons_

    def _align_attribute(
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
                if not rgetitem(self._cells, (connection, neuron), None):
                    raise AttributeError(
                        f"cell 'connection', 'neuron' ('{connection}', '{neuron}') is not valid"
                    )
                else:
                    return f"cell_.{connection}.{neuron}{'.' if attr else ''}{attr}"
            case _:
                raise ValueError(
                    f"invalid 'target' ('{target}') specified, expected one of: "
                    "'neuron', 'connection', 'cell'"
                )

    @property
    def connections(self) -> Proxy:
        r"""Registred connections.

        For a given ``name`` of a :py:class:`Connection` set via
        ``layer[name] = connection``, it can be accessed as ``layer.connections.name``.

        It can be modified in-place (including setting other attributes, adding
        monitors, etc), but it can neither be deleted nor reassigned.

        This is primarily used when targeting ``Connection`` objects with a monitor.

        Returns:
            Proxy: safe access to registered connections.
        """
        return Proxy(self.connections_, "")

    @property
    def neurons(self) -> Proxy:
        r"""Registred neurons.

        For a given ``name`` of a :py:class:`Neuron` set via
        ``layer[name] = neuron``, it can be accessed as ``layer.neurons.name``.

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
        automatically constructed on ``cell = layer[connection_name, neuron_name]``,
        it can be accessed as ``layer.cells.connection_name.neuron_name``.

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

    @property
    def named_updaters(self) -> Iterator[tuple[str, Updater]]:
        r"""Iterable of registered connection's updaters and their names.

        Yields:
            tuple[str, Updater]: tuple of a registered updater and its name.
        """
        return ((k, v.updater) for k, v in self.connections_.items() if v.updatable)

    @abstractmethod
    def wiring(
        self, inputs: dict[str, torch.Tensor], **kwargs
    ) -> dict[str, torch.Tensor]:
        r"""Connection logic between connection outputs and neuron inputs.

        The inputs are given as a dictionary where each key is a registered input name
        and the value is the tensor output from that connection. This is expected to
        return a dictionary where each key is the name of a registered output and the
        value is the tensor to be passed to its :py:meth:`~torch.nn.Module.__call__`.

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
                Defaults to True.
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

        The keys for ``neuron_kwargs`` are the names of the registered :py:class`Neuron`
        objects.

        Underlying :py:class:`Connection` and :py:class:`Neuron` objects are called
        using :py:meth:`~torch.nn.Module.__call__`, which in turn call
        :py:meth:`Connection.forward` and :py:meth:`Neuron.forward` respectively.
        The keyword argument dictionaries will be unpacked for each call automatically,
        and the inputs will be unpacked as positional arguments for each ``Connection``
        call.

        Only input modules which have keys in ``inputs`` will be run and added to
        the positional argument of :py:meth:`wiring`.

        Args:
            inputs (dict[str, tuple[torch.Tensor, ...]]): inputs passed to the
                registered connections' forward calls.
            connection_kwargs (dict[str, dict[str, Any]] | None, optional): keyword
                arguments passed to registered connections' forward calls. Defaults to None.
            neuron_kwargs (dict[str, dict[str, Any]] | None, optional): keyword
                arguments passed to registered neurons' forward calls. Defaults to None.
            capture_intermediate (bool, optional): if output from the connections should
                also be returned. Defaults to False.
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
        res = {
            k: rgetattr(self.connections_, k)(*v, **ckw.get(k, {}))
            for k, v in inputs.items()
        }

        if capture_intermediate:
            outputs = self.wiring(res, **kwargs)
            outputs = {
                k: rgetattr(self.neurons_, k)(v, **nkw.get(k, {}))
                for k, v in outputs.items()
            }
            return (outputs, res)
        else:
            res = self.wiring(res, **kwargs)
            res = {
                k: rgetattr(self.neurons_, k)(v, **nkw.get(k, {}))
                for k, v in res.items()
            }
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
    This may be used, for example, to reshape or pad a tensor.

    Either a function to combine the tensors from the modules in ``connections`` to be passed
    into ``inputs`` or a string literal may be provided. These may be "sum", "mean",
    "prod", "min", "max", or "stack". All except for "stack" use ``einops`` to reduce
    them, "stack" will stack the tensors along a new final dimension. When providing
    a function, it must take a tuple of tensors (equal to the number of inputs) and
    produce a single tensor output.

    Args:
        connections (Iterable[tuple[str, Connection] | tuple[str, Connection, OneToOne[torch.Tensor]]]):
            modules which receive inputs given to the layer.
        neurons (Iterable[tuple[str, Neuron] | tuple[str, Neuron, OneToOne[torch.Tensor]]]):
            modules which produce output from the layer.
        combine (Callable[[dict[str, torch.Tensor]], torch.Tensor] | Literal["stack", "sum", "mean", "prod", "min", "max"], optional):
            function to combine tensors from inputs into a single tensor for ouputs.
            Defaults to "stack".

    Caution:
        When a string literal is used as an argument for ``combine``, especially
        important when using ``stack``, the tensors are used in "insertion order" based
        on the dictionary passed into ``inputs`` in :py:meth:`Layer.forward`.

        When a custom function is given, keyword arguments passed into :py:meth:`call`,
        other than those captured in :py:meth`forward` will be passed in.
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
            | Literal["stack", "sum", "mean", "prod", "min", "max"]
        ) = "stack",
    ):
        # superclass constructor
        Layer.__init__(self)

        # callables
        self.post_input = {}
        self.pre_output = {}
        match (combine.lower() if isinstance(combine, str) else combine):
            case "stack":

                def combinefn(tensors, **kwargs):
                    return torch.stack(list(tensors.values()), dim=-1)

                self._combine = combinefn

            case "sum" | "mean" | "prod" | "min" | "max":

                def combinefn(tensors, **kwargs):
                    return ein.reduce(
                        list(tensors.values()), "s ... -> () ...", combine.lower()
                    )

                self._combine = combinefn

            case _:
                if isinstance(combine, str):
                    raise ValueError(
                        f"'combine' ('{combine}'), when a string, must be one of: "
                        "'stack', 'sum', 'mean', 'prod', 'min', 'max'"
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
                    Layer.__setitem__(self, *c)
                    self.post_input[c[0]] = lambda x: x
                case 3:
                    Layer.__setitem__(self, *c[:-1])
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
                    Layer.__setitem__(self, *n)
                    self.pre_output[n[0]] = lambda x: x
                case 3:
                    Layer.__setitem__(self, *n[:-1])
                    self.pre_output[n[0]] = n[2]
                case _:
                    raise ValueError(
                        f"element at position {idx} in 'neurons' has invalid "
                        f"number of elements {len(n)}"
                    )

        # construct cells
        for c in connections:
            for n in neurons:
                _ = self[c[0], n[0]]

    def __setitem__(self, *args, **kwargs) -> None:
        raise RuntimeError(
            f"{type(self).__name__}(Biclique) does not support item assignment"
        )

    def __delitem__(self, *args, **kwargs) -> None:
        raise RuntimeError(
            f"{type(self).__name__}(Biclique) does not support item deletion"
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

    This wraps :py:class:`Layer` to provid

    Args:
        connection (Connection): module which receives input to the layer.
        neuron (Neuron): module which generates output from the layer.
        transform (OneToOne[torch.Tensor] | None, optional): function
            to apply to connection output before passing into neurons. Defaults to None.
        connection_name (str, optional): name for the connection in the layer. Defaults
            to "serial_c".
        neuron_name (str, optional): name for the neuron in the layer. Defaults to
            "serial_n".

    Note:
        When ``transform`` is not specified, the identity function is used. Keyword
        arguments passed into :py:meth:`call`, other than those captured in
        :py:meth`forward` will be passed in.

    Note:
        The :py:class:`Layer` object underlying a ``Serial`` object has ``connection``
        and ``neuron`` registered with names ``"serial_c"`` and ``"serial_n"``
        respectively. Convenience properties can be used to avoid accessing manually.
    """

    def __init__(
        self,
        connection: Connection,
        neuron: Neuron,
        transform: OneToOne[torch.Tensor] | None = None,
        connection_name: str = "serial_c",
        neuron_name: str = "serial_n",
    ):
        # call superclass constructor
        Layer.__init__(self)

        # set names
        self._serial_connection_name = connection_name
        self._serial_neuron_name = neuron_name

        # add connection and neuron
        Layer.__setitem__(self, self._serial_connection_name, connection)
        Layer.__setitem__(self, self._serial_neuron_name, neuron)

        # set transformation used
        if transform:
            self._transform = transform
        else:

            def transfn(tensor, **kwargs):
                return tensor

            self._transform = transfn

    def __setitem__(self, *args, **kwargs) -> None:
        raise RuntimeError(
            f"{type(self).__name__}(Biclique) does not support item assignment"
        )

    def __delitem__(self, *args, **kwargs) -> None:
        raise RuntimeError(
            f"{type(self).__name__}(Biclique) does not support item deletion"
        )

    @property
    def connection(self) -> Connection:
        r"""Registered connection.

        Returns:
            Connection: registered connection.
        """
        return self[self._serial_connection_name]

    @property
    def neuron(self) -> Neuron:
        r"""Registered neuron.

        Returns:
            Neuron: registered neuron.
        """
        return self[self._serial_neuron_name]

    @property
    def synapse(self) -> Synapse:
        r"""Registered synapse.

        Returns:
            Synapse: registered connection's synapse.
        """
        return self[self._serial_connection_name].synapse

    @property
    def updater(self) -> Updater:
        r"""Registered updater.

        Returns:
            Updater: registered connection's updater.
        """
        return self[self._serial_connection_name].updater

    @property
    def cell(self) -> Cell:
        r"""Registered cell.

        Returns:
            Cell: registered cell.
        """
        return self[self._serial_connection_name, self._serial_neuron_name]

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
            self._serial_neuron_name: self._transform(
                inputs[self._serial_connection_name], **kwargs
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
                arguments for the connection's forward call. Defaults to None.
            neuron_kwargs (dict[str, dict[str, Any]] | None, optional): keyword
                arguments for the neuron's forward call. Defaults to None.
            capture_intermediate (bool, optional): if output from the connections should
                also be returned. Defaults to False.
            **kwargs (Any): keyword arguments passed to :py:meth:`wiring`.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: output from the neurons,
            if ``capture_intermediate``, this is the first element of a tuple, the second
            being the output from the connection.
        """
        # wrap non-empty dictionaries
        ckw = (
            {self._serial_connection_name: connection_kwargs}
            if connection_kwargs
            else connection_kwargs
        )
        nkw = (
            {self._serial_neuron_name: neuron_kwargs}
            if neuron_kwargs
            else neuron_kwargs
        )

        # call parent forward
        res = Layer.forward(
            self,
            {self._serial_connection_name: inputs},
            connection_kwargs=ckw,
            neuron_kwargs=nkw,
            capture_intermediate=capture_intermediate,
            **kwargs,
        )

        # unpack to sensible output
        if capture_intermediate:
            return (
                res[0][self._serial_neuron_name],
                res[1][self._serial_connection_name],
            )
        else:
            return res[self._serial_neuron_name]
