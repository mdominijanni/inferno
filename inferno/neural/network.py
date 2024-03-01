from __future__ import annotations
from . import Connection, Neuron, Synapse
from .modeling import Updater
from .hooks import Normalization, Clamping  # noqa:F401; ignore, used for docs
from .. import Module, ModuleDict
from abc import ABC, abstractmethod
from collections.abc import Iterable
import einops as ein
from functools import partial
from inferno._internal import argtest, rgetattr, Proxy
from inferno.observe import ManagedMonitor, MonitorConstructor
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
        *,
        updatable: bool = True,
    ):
        # call superclass constructor
        Module.__init__(self)

        # component elements
        self.connection_ = connection
        self.neuron_ = neuron

        # conditionally add the default updater to the connection
        if updatable and not self.connection.updatable:
            _ = self.connection.defaultupdater()

        # callbacks
        self._add_monitor_callback = add_monitor_callback
        self._del_monitor_callback = del_monitor_callback

        # reserve state for trainers
        self._remote_storage = ModuleDict()

    def __getitem__(self, key: Module) -> Module:
        r"""Retrieves additional storage the cell.

        If the state doesn't exist, an empty :py:class:`Module` is created and set as
        the state for the given trainer and is returned.

        Args:
            key (Module): module for which to get storage.

        Returns:
            Module: module containing the added state.
        """
        if key not in self._remote_storage:
            self._remote_storage[key] = Module()
        return self._remote_storage[key]

    def __setitem__(self, key: Module, value: Module) -> None:
        r"""Sets additional trainer storage for the cell.

        Args:
            key (Module): module for which to set storage.
            value (Module): module containing the added state.
        """
        self._remote_storage[key] = value

    def __delitem__(self, key: Module) -> None:
        r"""Deletes additional trainer storage for the cell.

        Args:
            key (Module): module for which to delete storage.
        """
        if key in self._remote_storage:
            del self._remote_storage[key]

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
        resultant trainables are trained with the same trainer, both monitors have
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
            attr (str): dot-seperated attribute path, relative to this trainable, to
                monitor.
            monitor (MonitorConstructor): partial constructor for the monitor to add.
            unpooled (bool): if the monitor should not be aliased from the pool
                regardless. Defaults to False.

        Raises:
            RuntimeError: attribute must be a member of this trainable.
            RuntimeError: 'updater.connection' is the only valid head of the attribute
                chain starting with 'updater'.

        Returns:
            ManagedMonitor: added monitor.

        Tip:
            If the monitor's behavior for the targeted attribute may vary with
            hyperparameters or other configuration state, ``unpooled`` should be
            set to ``True``. This does not keep this monitor from being aliased however,
            so the setting of ``unpooled`` should be consistent across all monitors
            with the same name.
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
                    target, attr = "trainable", ".".join(attrchain)

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
    def trainable(self) -> bool:
        """If the cell is trainable.

        Returns:
            bool: if the cell is trainable.
        """
        return self.connection_.updatable

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

    Args:
        trainable (bool, optional): if updaters should automatically be added to
            connections without them. Defaults to True.
    """

    def __init__(self, trainable: bool = True):
        # call superclass constructor
        Module.__init__(self)

        # set trainable property
        self._trainable = trainable

        # inner modules
        self.connections_ = ModuleDict()
        self.neurons_ = ModuleDict()
        self.cells_ = ModuleDict()
        self.monitors_ = ModuleDict()

    @property
    def trainable(self) -> bool:
        r"""If updaters will automatically be added to connections without one.

        Args:
            value (bool): if updaters will automatically be added to connections.

        Returns:
            bool: if updaters will automatically be added to connections.
        """
        return self._trainable

    @trainable.setter
    def trainable(self, value: bool) -> None:
        self._trainable = value

    def __getattr__(self, name: str | tuple[str, str]) -> Connection | Neuron | Cell:
        r"""Retrieves a previously added connection, neuron, or cell.

        The given ``name`` can either be a string (in which case it checks first for
        a :py:class:`Connection` then a :py:class:`Neuron`) or a 2-tuple of strings,
        in which case it expects the first string to be the name of a ``Connection`` and
        the second to be the name of a ``Neuron``, and the corresponding :py:class`Cell`
        is retrieved.

        Args:
            name (str | tuple[str, str]): name of the component module to retrieve.

        Returns:
            Connection | Neuron | Cell: retrieved module.
        """
        try:
            # specifies a cell
            if isinstance(name, tuple):
                if name[0] in self.cells_:
                    if name[1] in self.cells_[name[0]]:
                        return self.cells_[name[0]][name[1]]
                    else:
                        raise AttributeError(
                            f"'name' ('{name}') is not a registered neuron"
                        )
                else:
                    raise AttributeError(
                        f"'name' ('{name}') is not a registered connection"
                    )
            # specifies a connection or neuron
            else:
                if name in self.connections_:
                    return self.connections_[name]
                elif name in self.neurons_:
                    return self.neurons_[name]
                else:
                    raise AttributeError(
                        f"'name' ('{name}') is not a registered connection or neuron"
                    )
        except IndexError:
            raise ValueError("tuple 'name' must have exactly two elements")

    def __setattr__(self, name: str, module: Connection | Neuron) -> None:
        r"""Registers a new connection or neuron.

        The specified ``name`` must be a valid Python identifier, and the ``Cell``
        must not already have a :py:class:`Connection` or :py:class:`Neuron`
        registered with the same name.

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

        # connection
        if isinstance(module, Connection):
            if self.trainable and not module.updatable:
                _ = module.defaultupdater()
            self.connections_[name] = module

            # add all possible cells (i.e. if the layer is a biclique)
            self.cells_[name] = ModuleDict()
            for nname in self.neurons_:
                self.cells_[name][nname] = Cell(
                    self.connections_[name],
                    self.neurons_[nname],
                    partial(self._add_monitor, inputn=name, outputn=nname),
                    partial(self._del_monitor, inputn=name, outputn=nname),
                )

        # neuron
        elif isinstance(module, Neuron):
            self.neurons_[name] = module

            # add all possible cells (i.e. if the layer is a biclique)
            for cname in self.connections_:
                self.cells_[cname][name] = ModuleDict()
                self.cells_[cname][name] = Cell(
                    self.connections_[cname],
                    self.neurons_[name],
                    partial(self._add_monitor, inputn=cname, outputn=name),
                    partial(self._del_monitor, inputn=cname, outputn=name),
                )

        # invalid type
        else:
            _ = argtest.instance("module", module, (Connection, Neuron))

    def add_input(
        self, name: str, module: Connection, updatable: bool = True
    ) -> Updater:
        r"""Adds a module that receives input from outside the layer.

        This registers either a :py:class:`Connection` as a module that receives input
        from outside of the layer.

        This can be accessed later as an ``Updater`` via :py:attr:`updaters`,

        .. code-block:: python

            layer.updaters.name

        a ``Connection`` via :py:attr:`connections`,

        .. code-block:: python

            layer.connections.name

        or a ``Synapse`` via :py:attr:`synapses`.

        .. code-block:: python

            layer.synapses.name

        All possible :py:class:`Cell` objects are also constructed from this input to
        existing outputs. For each output with name ``output_name``, it can be accessed
        via :py:attr:`cells` as follows.

        .. code-block:: python

            layer.trainables.name.output_name

        Args:
            name (str): attribute name of the module receiving input from
                outside the layer.
            module (Connection): module which receives the input and generates
                intermediate output.
            updatable (bool, optional): if an updater should be added to the connection
                if it does not already exist. Defaults to True.

        Raises:
            RuntimeError: name must be unique amongst added inputs.

        Returns:
            Updater: added input module.
        """
        # test that the name is a valid identifier
        _ = argtest.identifier("name", name)

        # check that the name is not taken
        if name in self.updaters_:
            raise RuntimeError(f"'name' ('{name}') already assigned to an input")

        # wraps connection if it is not an updater and assigns
        if not isinstance(module, Updater):
            self.updaters_[name] = Updater(module)
        else:
            self.updaters_[name] = module

        # automatically add trainables
        if name in self.trainables_:
            raise RuntimeError(f"'name' ('{name}') already a first-order trainable key")
        else:
            self.trainables_[name] = nn.ModuleDict()
            for oname in self.neurons_:
                self.trainables_[name][oname] = Trainable(
                    self.updaters_[name],
                    self.neurons_[oname],
                    partial(self._add_monitor, inputn=name, outputn=oname),
                    partial(self._del_monitor, inputn=name, outputn=oname),
                )

        # return assigned value
        return self.updaters_[name]

    def add_output(self, name: str, module: Neuron) -> Neuron:
        r"""Adds a module that generates output from input modules.

        This registers a :py:class:`Neuron` as a module that receives intermediate
        input and will generate output external to the layer. This will be visible to
        PyTorch as a submodule.


        This can be accessed later as a ``Neuron`` via :py:attr:`neurons`.

        .. code-block:: python

            layer.neurons.name

        Args:
            name (str): attribute name of the module generating output to
                outside the layer.
            module (Neuron): module which receives intermediate output and generates
                the final output.

        Raises:
            RuntimeError: the name must be unique amongst added outputs.

        Returns:
            Neuron: added output module.
        """
        # test that the name is a valid identifier
        _ = argtest.identifier("name", name)

        # check that the name is not taken
        if name in self.neurons_:
            raise RuntimeError(f"'name' ('{name}') already assigned to an output")

        # assigns value
        self.neurons_[name] = module

        # automatically add trainables
        for iname in self.updaters_.items():
            if name in self.trainables_[iname]:
                raise RuntimeError(
                    f"'name' ('{name}') already a second-order trainable key in '{iname}'"
                )

            else:
                self.trainables_[iname][name] = Trainable(
                    self.updaters_[iname],
                    self.neurons_[name],
                    partial(self._add_monitor, inputn=iname, outputn=name),
                    partial(self._del_monitor, inputn=iname, outputn=name),
                )

        # return assigned value
        return self.neurons_[name]

    def _add_monitor(
        self,
        pool: str,
        name: str,
        target: str,
        attr: str,
        monitor: MonitorConstructor,
        inputn: str,
        outputn: str,
    ) -> ManagedMonitor:
        r"""Used as a callback to add monitors from a Trainable.

        This will create a monitor if it doesn't exist, otherwise it will create a
        reference to the existing monitor and return it.

        Args:
            pool (str): name of the pool to which the monitor will be added.
            name (str): name of the monitor.
            target (str): shorthand for the top-level attribute being targeted.
            attr (str): dot-seperated attribute to monitor.
            monitor (MonitorConstructor): partial constructor for managed monitor.
            inputn (str): name of the associated input.
            outputn (str): name of the associated output.

        Returns:
            ManagedMonitor: created or retrieved monitor.

        Note:
            Valid targets are "neuron" (with alias "output"), "connection" (with alias
            "input"), and "trainable".
        """
        # check if input and output names exist
        if inputn not in self.innames:
            raise AttributeError(f"input name ('{inputn}') is not an added input")
        if outputn not in self.outnames:
            raise AttributeError(f"output name ('{outputn}') is not an added output")

        # create the pool if it doesn't exist
        if pool not in self.monitors_:
            self.monitors_[pool] = nn.ModuleDict()

        # create input group if it doesn't exist
        if inputn not in self.monitors_[pool]:
            self.monitors_[pool][inputn] = nn.ModuleDict()

        # create input group if it doesn't exist
        if outputn not in self.monitors_[pool][inputn]:
            self.monitors_[pool][inputn][outputn] = nn.ModuleDict()

        # alias the monitor
        match target:

            case "neuron" | "output":
                # set correct attribute relative to the layer
                attr = f"updaters_.connection.{outputn}.{attr}"

                # alias the monitor if it does not exist
                if name not in self.monitors_[pool][inputn][outputn]:
                    for inkey in self.monitors_[pool]:
                        if (
                            outputn in self.monitors_[pool][inkey]
                            and name in self.monitors_[pool][inkey][outputn]
                        ):
                            self.monitors_[pool][inputn][outputn][name] = (
                                self.monitors_[pool][inkey][outputn][name]
                            )
                            break

            case "connection" | "input":
                # set correct attribute relative to the layer
                attr = f"neurons_.{inputn}.{attr}"

                # alias the monitor if it does not exist
                if name not in self.monitors_[pool][inputn][outputn]:
                    for outkey in self.monitors_[pool][inputn]:
                        if name in self.monitors_[pool][inputn][outkey]:
                            self.monitors_[pool][inputn][outputn][name] = (
                                self.monitors_[pool][inputn][outkey][name]
                            )
                            break

            case "trainable":
                # set correct attribute relative to the layer
                attr = f"trainables_.{inputn}.{outputn}.{attr}"

            case _:
                raise ValueError(
                    f"invalid 'target' ('{target}') specified, expected one of: "
                    "'neuron', 'connection', 'trainable'"
                )

        # create the monitor if it does not exist and could not be aliased
        if name not in self.monitors_[pool][inputn][outputn]:
            self.monitors_[pool][inputn][outputn][name] = monitor(attr, self)

        # return the monitor
        return self.monitors_[pool][inputn][outputn][name]

    def _del_monitor(self, pool: str, name: str, inputn: str, outputn: str) -> None:
        r"""Used as a callback to free monitors from a Trainable.

        This will only delete the alias associated with that :py:class:`Trainable`.
        If the monitor has been aliased, that alias will persist and be accessible
        as normal.

        Args:
            pool (str): name of the pool to which the monitor will be added.
            name (str): name of the monitor.
            inputn (str): name of the associated input.
            outputn (str): name of the associated output.
        """
        # check if the pool exists
        if pool in self.monitors_:

            # check if the input exists
            if inputn in self.monitors_[pool]:

                # check if the output exists
                if outputn in self.monitors_[pool][inputn]:

                    # delete the monitor if it exists
                    if name in self.monitors_[pool][inputn][outputn]:
                        del self.monitors_[pool][inputn][outputn][name]

                    # delete output container if empty
                    if not len(self.monitors_[pool][inputn][outputn]):
                        del self.monitors_[pool][inputn][outputn]

                # delete input container if empty
                if not len(self.monitors_[pool][inputn]):
                    del self.monitors_[pool][inputn]

            # delete pool container if empty
            if not len(self.monitors_[pool]):
                del self.monitors_[pool]

    @property
    def innames(self) -> Iterable[str]:
        r"""Registered input names.

        Yields:
            str: name of a registered input.
        """
        return (k for k in self.updaters_.keys())

    @property
    def outnames(self) -> Iterable[str]:
        r"""Registered output names.

        Yields:
            str: name of a registered output.
        """
        return (k for k in self.neurons_.keys())

    @property
    def connections(self) -> Proxy:
        r"""Registred connections.

        For a given ``name`` registered with :py:meth:`add_input`, its corresponding
        :py:class:`Connection` can be accessed as.

        .. code-block:: python

            layer.connections.name

        And is equivalent to the following.

        .. code-block:: python

            layer.updaters.name.connection

        It can be modified in-place (including setting other attributes, adding
        monitors, etc), but it can neither be deleted nor reassigned.

        Returns:
            Proxy: safe access to registered connections.
        """
        return Proxy(self.updaters_, "connection")

    @property
    def named_connections(self) -> Iterable[tuple[str, Connection]]:
        r"""Iterable of registered connections and their names.

        Yields:
            tuple[str, Connection]: tuple of a registered connection and its name.
        """
        return ((k, v.connection) for k, v in self.updaters_.items())

    @property
    def neurons(self) -> Proxy:
        r"""Registered neurons.

        For a given ``name`` registered with :py:meth:`add_output`, its corresponding
        :py:class:`Neuron` can be accessed as.

        .. code-block:: python

            layer.neurons.name

        It can be modified in-place (including setting other attributes, adding
        monitors, etc), but it can neither be deleted nor reassigned.

        Returns:
            Proxy: safe access to registered neurons.
        """
        return Proxy(self.neurons_, "")

    @property
    def named_neurons(self) -> Iterable[tuple[str, Neuron]]:
        r"""Iterable of registered neurons and their names.

        Yields:
            tuple[str, Neuron]: tuple of a registered neuron and its name.
        """
        return ((k, v) for k, v in self.neurons_.items())

    @property
    def synapses(self) -> Proxy:
        r"""Registered synapses.

        For a given ``name`` registered with :py:meth:`add_input`, its corresponding
        :py:class:`Synapse` can be accessed as.

        .. code-block:: python

            layer.synapses.name

        And is equivalent to the following.

        .. code-block:: python

            layer.updaters.name.connection.synapse

        It can be modified in-place (including setting other attributes, adding
        monitors, etc), but it can neither be deleted nor reassigned.

        Returns:
            Proxy: safe access to registered synapses.
        """
        return Proxy(self.updaters_, "connection.synapse")

    @property
    def named_synapses(self) -> Iterable[tuple[str, Synapse]]:
        r"""Iterable of registered synapses and their names.

        Yields:
            tuple[str, Synapse]: tuple of a registered synapse and its name.
        """
        return ((k, v.connection.synapse) for k, v in self.updaters_.items())

    @property
    def trainables(self) -> Proxy:
        r"""Registered trainables.

        For a given ``input_name`` and ``output_name``, its corresponding
        :py:class:`Trainable` can be accessed as.

        .. code-block:: python

            layer.trainables.input_name.output_name

        Returns:
            Proxy: _description_
        """
        return Proxy(self.trainables_, "", "")

    @property
    def named_trainables(self) -> Iterable[tuple[tuple[str, str], Trainable]]:
        r"""Iterable of registered trainables and tuples of the input and output name.

        Yields:
            tuple[tuple[str, str], torch.Tensor]: tuple of a registered connection and
            a tuple of the input name and output name corresponding to it.
        """
        return ((k, v.connection) for k, v in self.updaters_.items())

    @property
    def updaters(self) -> Proxy:
        r"""Registred updaters.

        For a given ``name`` registered with :py:meth:`add_input`, its corresponding
        :py:class:`Updater` can be accessed as.

        .. code-block:: python

            layer.updaters.name

        It can be modified in-place (including setting other attributes, adding
        monitors, etc), but it can neither be deleted nor reassigned.

        Returns:
            Proxy: safe access to registered synapses.
        """
        return Proxy(self.updaters_, "")

    @property
    def named_updaters(self) -> Iterable[tuple[str, Updater]]:
        r"""Iterable of registered updaters and their names.

        Yields:
            tuple[str, Updater]: tuple of a registered updater and its name.
        """
        return ((k, v) for k, v in self.updaters_.items())

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

    def update(self) -> None:
        r"""Applies all cumulative updates.

        This calls every updated which applies cumulative updates and any updater
        hooks are automatically called (e.g. parameter clamping).
        """
        for updater in self.updaters_.values():
            updater()

    def forward(
        self,
        inputs: dict[str, tuple[torch.Tensor, ...]],
        inkwargs: dict[str, dict[str, Any]] | None = None,
        outkwargs: dict[str, dict[str, Any]] | None = None,
        capture_intermediate: bool = False,
        **kwargs: Any,
    ) -> (
        dict[str, torch.Tensor]
        | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]
    ):
        r"""Computes a forward pass.

        The keys for ``inputs`` and ``inkwargs`` are the names of registered
        :py:class:`Updater` objects correspond to elements in :py:attr`innames`.
        The keys for ``outkwargs`` are the names of the registered :py:class`Neuron`
        objects and correspond to elements in :py:attr:`outnames`.

        Underlying :py:class:`Connection` and :py:class:`Neuron` objects are called
        using :py:meth:`~torch.nn.Module.__call__`, which in turn call
        :py:meth:`Connection.forward` and :py:meth:`Neuron.forward` respectively.
        The keyword argument dictionaries will be unpacked for each call automatically,
        and the inputs will be unpacked as positional arguments for each call.

        Only input modules which have keys in ``inputs`` will be run and added to
        the positional argument of :py:meth:`wiring`.

        Args:
            inputs (dict[str, tuple[torch.Tensor, ...]]): inputs passed to the
                registered connections' forward calls.
            inkwargs (dict[str, dict[str, Any]] | None, optional): keyword arguments
                passed to registered connections' forward calls. Defaults to None.
            outkwargs (dict[str, dict[str, Any]] | None, optional): keyword arguments
                passed to registered neurons' forward calls. Defaults to None.
            capture_intermediate (bool, optional): if output from the connections should
                also be returned. Defaults to False.
            **kwargs (Any): keyword arguments passed to :py:meth:`wiring`.

        Returns:
            dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
            tensors from neurons and the associated neuron names, if ``capture_intermediate``,
            this is the second element of a tuple, the first being a tuple of tensors from
            connections and the associated connection names.
        """
        # replace none with empty dictionaries
        inkwargs = inkwargs if inkwargs else {}
        outkwargs = outkwargs if outkwargs else {}

        # get connection outputs
        res = {
            k: rgetattr(self.updaters_, f"{k}.connection")(*v, **inkwargs.get(k, {}))
            for k, v in inputs
        }

        if capture_intermediate:
            outputs = self.wiring(res, **kwargs)
            outputs = {
                k: rgetattr(self.neurons_, k)(*v, **outkwargs.get(k, {}))
                for k, v in outputs
            }
            return (res, outputs)
        else:
            res = self.wiring(res, **kwargs)
            res = {
                k: rgetattr(self.neurons_, k)(*v, **outkwargs.get(k, {}))
                for k, v in res
            }
            return res


class Biclique(Layer):
    r"""Layer structured as a complete bipartite graph.

    Each input is processed by its corresponding connection, with an optional
    transformation applied, before being combined with the results of all other
    connections. These are then, for each group of neurons, optionally transformed
    and then passed in.

    Each element of ``inputs`` and ``outputs`` must be a tuple with at least two
    elements and at most three. The first of these is a name, which must be a
    Python identifier and unique to the set of inputs or outputs respectively. The
    second is the module representing the input or output
    (:py:class:`Updater`/:py:class:`Connection` or :py:class:`Neuron` respectively).
    The third is optionally a function which takes a :py:class`~torch.Tensor` and
    returns a ``Tensor``. This will be applied to the output of, or input to, the
    modules, respectively. This may be used, for example, to reshape or pad a tensor.

    Either a function to combine the tensors from the modules in ``inputs`` to be passed
    into ``outputs`` or a string literal may be provided. These may be "sum", "mean",
    "prod", "min", "max", or "stack". All except for "stack" use ``einops`` to reduce
    them, "stack" will stack the tensors along a new final dimension. When providing
    a function, it must take a tuple of tensors (equal to the number of inputs) and
    produce a single tensor output.

    Args:
        inputs (tuple[tuple[str, Updater | Connection] | tuple[str, Updater | Connection, Callable[[torch.Tensor], torch.Tensor]], ...]):
            modules which receive inputs given to the layer.
        outputs (tuple[tuple[str, Neuron] | tuple[str, Neuron, Callable[[torch.Tensor], torch.Tensor]], ...]):
            modules which produce output from the layer.
        combine (Callable[[dict[str, torch.Tensor]], torch.Tensor] | Literal["stack", "sum", "mean", "prod", "min", "max"], optional):
            function to combine tensors from inputs into a single tensor for ouputs.
            Defaults to "stack".

    Caution:
        When a string literal is used as an argument for ``combine``, especially
        important when using ``stack``, the tensors are used in "insertion order" based
        on the dictionary passed into ``inputs`` in :py:meth:`Layer.forward`.
    """

    def __init__(
        self,
        inputs: tuple[
            tuple[str, Updater | Connection]
            | tuple[str, Updater | Connection, Callable[[torch.Tensor], torch.Tensor]],
            ...,
        ],
        outputs: tuple[
            tuple[str, Neuron]
            | tuple[str, Neuron, Callable[[torch.Tensor], torch.Tensor]],
            ...,
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

                def combinefn(tensors):
                    return torch.stack(list(tensors.values()), dim=-1)

                self._combine = combinefn

            case "sum" | "mean" | "prod" | "min" | "max":

                def combinefn(tensors):
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

        # add inputs
        for idx, input_ in enumerate(inputs):
            match len(input_):
                case 2:
                    Layer.add_input(self, *input_)
                    self.post_input[input_[0]] = lambda x: x
                case 3:
                    Layer.add_input(self, *input_[:-1])
                    self.post_input[input_[0]] = input_[2]
                case _:
                    raise ValueError(
                        f"element at position {idx} in 'inputs' has invalid "
                        f"number of elements {len(input_)}"
                    )

        # add outputs
        for idx, output_ in enumerate(outputs):
            match len(output_):
                case 2:
                    Layer.add_output(self, *output_)
                    self.pre_output[output_[0]] = lambda x: x
                case 3:
                    Layer.add_output(self, *output_[:-1])
                    self.pre_output[output_[0]] = output_[2]
                case _:
                    raise ValueError(
                        f"element at position {idx} in 'outputs' has invalid "
                        f"number of elements {len(output_)}"
                    )

    def add_input(self, *args, **kwargs):
        r"""Overrides function to add inputs.

        Raises:
            RuntimeError: inputs for a biclique layer are fixed on construction.
        """
        raise RuntimeError(
            f"'add_input' of {type(self).__name__}(Biclique) cannot be called."
        )

    def add_output(self, *args, **kwargs):
        r"""Overrides function to add outputs.

        Raises:
            RuntimeError: outputs for a biclique layer are fixed on construction.
        """
        raise RuntimeError(
            f"'add_output' of {type(self).__name__}(Biclique) cannot be called."
        )

    def wiring(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        r"""Connection logic between connection outputs and neuron inputs.

        This implements the forward logic of the biclique topology where the tensors
        from the inputs are transformed, combined, and transformed again before
        being passed to the outputs. Transforms which were unspecified are assumed to
        be identity.

        Args:
            inputs (dict[str, torch.Tensor]): dictionary of input names to tensors.

        Returns:
            dict[str, torch.Tensor]: dictionary of output names to tensors.
        """
        return {
            k: v(self._combine({k: self.post_input[v] for k, v in inputs.items()}))
            for k, v in self.pre_output
        }


class Serial(Layer):
    r"""Layer with a single connection and single neuron group.

    This wraps :py:class:`Layer` to provid

    Args:
        inputs (Updater | Connection): module which receives input to the layer.
        outputs (Neuron): module which generates output from the layer.
        transform (Callable[[torch.Tensor], torch.Tensor] | None, optional): function
            to apply to connection output before passing into neurons. Defaults to None.

    Note:
        When ``transform`` is not specified, the identity function is used.

    Note:
        The :py:class:`Layer` object underlying a ``Serial`` object has the input
        and output (:py:class`Connection`/py:class:`Updater` and :py:class:`Neuron`
        respectively) registered with the name "main".
    """

    def __init__(
        self,
        inputs: Updater | Connection,
        outputs: Neuron,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        """_summary_

        Args:
            inputs (Updater | Connection): _description_
            outputs (Neuron): _description_
            transform (Callable[[torch.Tensor], torch.Tensor] | None, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        # call superclass constructor
        Layer.__init__(self)

        # add connection and neuron
        Layer.add_input(self, "main", inputs)
        Layer.add_output(self, "main", outputs)

        # set transformation used
        if transform:
            self._transform = transform
        else:

            def transfn(tensor):
                return tensor

            self._transform = transfn

    @property
    def connection(self) -> Connection:
        r"""Registered connection.

        Returns:
            Connection: registered connection.
        """
        return self.connections.main

    @property
    def neuron(self) -> Neuron:
        r"""Registered neuron.

        Returns:
            Neuron: registered neuron.
        """
        return self.neuron.main

    @property
    def synapse(self) -> Synapse:
        r"""Registered synapse.

        Returns:
            Synapse: registered synapse.
        """
        return self.synapse.main

    @property
    def trainable(self) -> Trainable:
        r"""Registered trainable.

        Returns:
            Trainable: registered trainable.
        """
        return self.trainable.main

    @property
    def updater(self) -> Updater:
        r"""Registered updater.

        Returns:
            Updater: registered updater.
        """
        return self.updater.main

    def add_input(self, *args, **kwargs):
        r"""Overrides function to add inputs.

        Raises:
            RuntimeError: inputs for a serial layer are fixed on construction.
        """
        raise RuntimeError(
            f"'add_input' of {type(self).__name__}(Serial) cannot be called."
        )

    def add_output(self, *args, **kwargs):
        r"""Overrides function to add outputs.

        Raises:
            RuntimeError: outputs for a serial layer are fixed on construction.
        """
        raise RuntimeError(
            f"'add_output' of {type(self).__name__}(Serial) cannot be called."
        )

    def wiring(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        r"""Connection logic between connection outputs and neuron inputs.

        This implements the forward logic of the serial topology. The ``transform`` is
        applied to the result of the connection before being passed to the neuron. If
        not specified, it is assumed to be identity.

        Args:
            inputs (dict[str, torch.Tensor]): dictionary of input names to tensors.

        Returns:
            dict[str, torch.Tensor]: dictionary of output names to tensors.
        """
        return {"main": self._transform(inputs["main"])}

    def forward(
        self,
        *inputs: torch.Tensor,
        inkwargs: dict[str, dict[str, Any]] | None = None,
        outkwargs: dict[str, dict[str, Any]] | None = None,
        capture_intermediate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        r"""Computes a forward pass.

        Args:
            *inputs (torch.Tensor): values passed to the connection.
            inkwargs (dict[str, dict[str, Any]] | None, optional): keyword arguments
                for the connection's forward call. Defaults to None.
            outkwargs (dict[str, dict[str, Any]] | None, optional): keyword arguments
                for the neuron's forward call. Defaults to None.
            capture_intermediate (bool, optional): if output from the connections should
                also be returned. Defaults to False.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: output from the neurons,
            if ``capture_intermediate``, this is th second element of a tuple, the first
            being the output from the connection.
        """
        # call parent forward
        res = Layer.forward(
            self,
            {"main": inputs},
            inkwargs=inkwargs,
            outkwargs=outkwargs,
            capture_intermediate=capture_intermediate,
        )

        # unpack to sensible output
        if capture_intermediate:
            return res[0]["main"], res[1]["main"]
        else:
            return res["main"]
