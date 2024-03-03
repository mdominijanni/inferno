from __future__ import annotations
from . import Connection, Neuron, Synapse
from .modeling import Updater
from .hooks import Normalization, Clamping  # noqa:F401; ignore, used for docs
from .. import Module
from ..infernotypes import OneToOne
from abc import ABC, abstractmethod
from collections.abc import Iterator, Iterable
import einops as ein
from functools import partial
from inferno._internal import argtest, rgetattr
from inferno.observe import ManagedMonitor, MonitorConstructor
from itertools import chain
import torch
import torch.nn as nn
from typing import Any, Callable, Literal


class Cell(Module):
    def __init__(
        self,
        connection: Connection,
        neuron: Neuron,
        add_monitor_callback: Callable | None = None,
        del_monitor_callback: Callable | None = None,
    ):
        # call superclass constructor
        Module.__init__(self)

        # component elements
        self.connection_ = connection
        self.neuron_ = neuron

        # conditionally add the default updater to the connection
        if not self.connection.updatable:
            self.connection = self.connection.defaultupdater()

        # callbacks
        self._add_monitor_callback = add_monitor_callback
        self._del_monitor_callback = del_monitor_callback

    def add_monitor(
        self,
        caller: Module,
        name: str,
        attr: str,
        monitor: MonitorConstructor,
        unpooled: bool = False,
    ) -> ManagedMonitor:
        r"""Adds a managed monitor associated with a trainer.

        This works in conjunction with :py:class:`Layer` to ensure that the added
        monitors are not duplicated if it is unneeded. This non-duplication is only
        enforced across a single layer and single trainer.

        For example, if a layer goes from two connections to one neuron, and both
        resultant cells are trained with the same trainer, both monitors have
        equivalent attribute chains (defined by ``attr``), and the same name as
        defined by ``name``, then rather than creating a new monitor, the existing one
        will be returned.

        Because of this, ``name`` must also capture any information which may be unique
        to a specific trainable.

        All monitor's added this way will be added to the lifecycle of the ``Layer``
        which created them.

        Args:
            caller (Module): module which will use the monitor.
            name (str): name of the monitor to add.
            attr (str): dot-separated attribute path, relative to this cell, to monitor.
            monitor (MonitorConstructor): partial constructor for the monitor to add.
            unpooled (bool): if the monitor should not be aliased from the pool
                regardless. Defaults to False.

        Raises:
            RuntimeError: attribute must be a member of this trainable.
            RuntimeError: 'updater.connection' is the only valid head of the attribute
                chain starting with 'updater'.

        Returns:
            ManagedMonitor: added monitor.

        Important:
            All monitors added this way will be hooked to the :py:class:`Layer` which
            owns this ``Cell`, and therefore will be associated with its ``forward``
            call. If a monitor targeting an updater should trigger on updating, it
            should be directly added to the updater.

        Tip:
            If the monitor's behavior for the targeted attribute may vary with
            hyperparameters or other configuration state, ``unpooled`` should be
            set to ``True``. This does not keep this monitor from being aliased however,
            so the setting of ``unpooled`` should be consistent across all monitors
            in the pool with the same name.
        """
        # check that the attribute is a valid dot-chain identifier
        _ = argtest.nestedidentifier("attr", attr)

        # split the identifier and check for ownership
        attrchain = attr.split(".")

        # ensure the top-level attribute is in this trainable
        if not hasattr(self, attrchain[0]):
            raise RuntimeError(
                f"this trainable does not have an attribute '{attrchain[0]}'"
            )

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
        if unpooled:
            target, attr = "trainable", ".".join(attrchain)
        else:
            match attrchain[0]:
                case "connection":
                    target, attr = "connection", ".".join(attrchain[1:])
                case "neuron":
                    target, attr = "neuron", ".".join(attrchain[1:])
                case _:
                    target, attr = "cell", ".".join(attrchain)

        # use layer callback to add the monitor to its pool and return
        return self._add_monitor_callback(caller, name, target, attr, monitor)

    def del_monitor(self, caller: Module, name: str) -> None:
        r"""Deletes a managed monitor associated with a trainer.

        This "frees" a monitor from the enclosing :py:class:`Layer` that is associated
        with this trainable.

        Args:
            caller (Module): instance of the module associated with the monitor.
            name (str): name of the monitor to remove.
        """
        self._del_monitor_callback(caller, name)

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
    r"""Representation of simultaneously processed connections and neurons.

    Important:
        :py:class:`ManagedMonitor` objects added through :py:meth:`Cell.add_monitor` are
        not exported with the state dictionary. :py:class:`Cell` objects additionally
        are not exported with the state dictionary, although their component
        :py:class:`Connection` and :py:class:`Neuron` objects are.
    """

    def __init__(self, trainable: bool = True):
        # call superclass constructor
        Module.__init__(self)

        # inner modules
        self.connections_ = nn.ModuleDict()
        self.neurons_ = nn.ModuleDict()
        self.cells_ = {}
        self.monitors_ = {}

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
                        self.cells_[name[0]] = {}

                    if name[1] in self.neurons_:
                        # create cell for neuron if it does not exist
                        if name[1] not in self.cells_[name[0]]:
                            self.cells_[name[0]][name[1]] = Cell(
                                self.connections_[name[0]],
                                self.neurons_[name[1]],
                                partial(
                                    self._add_monitor,
                                    connection=name[0],
                                    neuron=name[1],
                                ),
                                partial(
                                    self._del_monitor,
                                    connection=name[0],
                                    neuron=name[1],
                                ),
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
            if not module.updatable:
                module.updater = module.defaultupdater
            self.connections_[name] = module

        elif isinstance(module, Neuron):
            self.neurons_[name] = module

        else:
            _ = argtest.instance("module", module, (Connection, Neuron))

    def __delitem__(self, name: str | tuple[str, str]) -> None:
        """Deletes a connection, neuron, or cell and associated monitors and cells.

        Args:
            name (str | tuple[str, str]): name of the component module to delete.

        Raises:
            AttributeError: name does not specify a registered connection, neuron, or cell.

        Note:
            When ``name`` is a tuple, that cell and its monitors will be deleted but
            the connection and neuron will not be.
        """
        pools = list(self.monitors_.keys())
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
            connections.extend(self.connections__.keys())
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

        # clean up monitors
        for p in pools:

            for c in connections:
                if c in self.monitors_[p]:

                    for n in neurons:
                        if n in self.monitors_[p][c]:
                            del self.monitors_[p][c][n]

                    if not len(self.monitors_[p][c]):
                        del self.monitors_[p][c]

            if not len(self.monitors_[p]):
                del self.monitors_[p]

        else:
            raise AttributeError(
                f"'name' ('{name}') is not a registered connection or neuron"
            )

    def _add_monitor(
        self,
        pool: str,
        name: str,
        target: str,
        attr: str,
        monitor: MonitorConstructor,
        connection: str,
        neuron: str,
    ) -> ManagedMonitor:
        r"""Used as a callback to add monitors from a cell.

        This will create a monitor if it doesn't exist, otherwise it will create a
        reference to the existing monitor and return it.

        Args:
            pool (str): name of the pool to which the monitor will be added.
            name (str): name of the monitor.
            target (str): shorthand for the top-level attribute being targeted.
            attr (str): dot-separated attribute to monitor.
            monitor (MonitorConstructor): partial constructor for managed monitor.
            connection (str): name of the associated connection.
            neuron (str): name of the associated neuron.

        Returns:
            ManagedMonitor: created or retrieved monitor.

        Note:
            Valid targets are "neuron", "connection", and "cell".
        """
        # check if input and output names exist
        if connection not in self.connections_:
            raise AttributeError(
                f"'connection' ('{connection}') is not an added connection"
            )
        if neuron not in self.neurons_:
            raise AttributeError(f"'neuron' ('{neuron}') is not an added neuron")

        # create the pool if it doesn't exist
        if pool not in self.monitors_:
            self.monitors_[pool] = {}

        # create connection group if it doesn't exist
        if connection not in self.monitors_[pool]:
            self.monitors_[pool][connection] = {}

        # create neuron group if it doesn't exist
        if neuron not in self.monitors_[pool][connection]:
            self.monitors_[pool][connection][neuron] = {}

        # alias the monitor
        match target:

            case "connection":
                # set correct attribute relative to the layer
                attr = f"connections_.{connection}.{attr}"

                # alias the monitor if it does not exist
                if name not in self.monitors_[pool][connection][neuron]:
                    for n in self.monitors_[pool][connection]:
                        if name in self.monitors_[pool][connection][n]:
                            self.monitors_[pool][connection][neuron][name] = (
                                self.monitors_[pool][connection][n][name]
                            )
                            break

            case "neuron":
                # set correct attribute relative to the layer
                attr = f"neurons_.{neuron}.{attr}"

                # alias the monitor if it does not exist
                if name not in self.monitors_[pool][connection][neuron]:
                    for c in self.monitors_[pool]:
                        if neuron in self.monitors_[pool][c]:
                            if name in self.monitors_[pool][c][neuron]:
                                self.monitors_[pool][connection][neuron][name] = (
                                    self.monitors_[pool][c][neuron][name]
                                )
                            break

            case "cell":
                # set correct attribute relative to the layer
                attr = f"cell_.{connection}.{neuron}.{attr}"

            case _:
                raise ValueError(
                    f"invalid 'target' ('{target}') specified, expected one of: "
                    "'neuron', 'connection', 'cell'"
                )

        # create the monitor if it does not exist and could not be aliased
        if name not in self.monitors_[pool][connection][neuron]:
            self.monitors_[pool][connection][neuron][name] = monitor(attr, self)

        # return the monitor
        return self.monitors_[pool][connection][neuron][name]

    def _del_monitor(self, pool: str, name: str, connection: str, neuron: str) -> None:
        r"""Used as a callback to free monitors from a cell.

        This will only delete the alias associated with that :py:class:`Cell`.
        If the monitor has been aliased, that alias will persist and be accessible
        as normal.

        Args:
            pool (str): name of the pool to which the monitor will be added.
            name (str): name of the monitor.
            connection (str): name of the associated connection.
            neuron (str): name of the associated neuron.
        """
        # check if the pool exists
        if pool in self.monitors_:

            # check if the connection exists
            if connection in self.monitors_[pool]:

                # check if the neuron exists
                if neuron in self.monitors_[pool][connection]:

                    # delete the monitor if it exists
                    if name in self.monitors_[pool][connection][neuron]:
                        del self.monitors_[pool][connection][neuron][name]

                    # delete neuron container if empty
                    if not len(self.monitors_[pool][connection][neuron]):
                        del self.monitors_[pool][connection][neuron]

                # delete connection container if empty
                if not len(self.monitors_[pool][connection]):
                    del self.monitors_[pool][connection]

            # delete pool container if empty
            if not len(self.monitors_[pool]):
                del self.monitors_[pool]

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
        return ((k, v.updater) for k, v in self.connections_.items())

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
            k: rgetattr(self.connection_, k)(*v, **ckw.get(k, {})) for k, v in inputs
        }

        if capture_intermediate:
            outputs = self.wiring(res, **kwargs)
            outputs = {
                k: rgetattr(self.neurons_, k)(v, **nkw.get(k, {})) for k, v in outputs
            }
            return (outputs, res)
        else:
            res = self.wiring(res, **kwargs)
            res = {k: rgetattr(self.neurons_, k)(v, **nkw.get(k, {})) for k, v in res}
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
            for k, v in self.pre_output
        }


class Serial(Layer):
    r"""Layer with a single connection and single neuron group.

    This wraps :py:class:`Layer` to provid

    Args:
        connection (Connection): module which receives input to the layer.
        neuron (Neuron): module which generates output from the layer.
        transform (OneToOne[torch.Tensor] | None, optional): function
            to apply to connection output before passing into neurons. Defaults to None.

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
    ):
        # call superclass constructor
        Layer.__init__(self)

        # add connection and neuron
        Layer.__setitem__("serial_c", connection)
        Layer.__setitem__("serial_n", neuron)

        # set transformation used
        if transform:
            self._transform = transform
        else:

            def transfn(tensor, **kwargs):
                return tensor

            self._transform = transfn

    @property
    def connection(self) -> Connection:
        r"""Registered connection.

        Returns:
            Connection: registered connection.
        """
        return self["serial_c"]

    @property
    def neuron(self) -> Neuron:
        r"""Registered neuron.

        Returns:
            Neuron: registered neuron.
        """
        return self["serial_n"]

    @property
    def synapse(self) -> Synapse:
        r"""Registered synapse.

        Returns:
            Synapse: registered connection's synapse.
        """
        return self["serial_c"].synapse

    @property
    def updater(self) -> Updater:
        r"""Registered updater.

        Returns:
            Updater: registered connection's updater.
        """
        return self["serial_c"].updater

    @property
    def cell(self) -> Cell:
        r"""Registered cell.

        Returns:
            Cell: registered cell.
        """
        return self["serial_c", "serial_n"]

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
        return {"serial_n": self._transform(inputs["serial_c"], **kwargs)}

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
            {"serial_c": connection_kwargs} if connection_kwargs else connection_kwargs
        )
        nkw = {"serial_n": neuron_kwargs} if neuron_kwargs else neuron_kwargs

        # call parent forward
        res = Layer.forward(
            self,
            {"main": inputs},
            connection_kwargs=ckw,
            neuron_kwargs=nkw,
            capture_intermediate=capture_intermediate,
            **kwargs,
        )

        # unpack to sensible output
        if capture_intermediate:
            return res[0]["serial_n"], res[1]["serial_c"]
        else:
            return res["serial_n"]
