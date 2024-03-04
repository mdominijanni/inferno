from abc import ABC, abstractmethod
from collections.abc import Generator, Iterator, MutableMapping
from .. import Module
from inferno._internal import fzip, unique
from inferno.neural import Cell, Layer, Trainable
from inferno.observe import Monitor, ManagedMonitor, MonitorConstructor
import itertools
from itertools import chain
import torch  # noqa:F401, for docstrings
import torch.nn as nn
from typing import Any, Callable
import weakref


class CellTrainer(Module):
    def __init__(self, **kwargs):
        r"""Base trainer for updating cell parameters.

        Important:
            The :py:class:`Layer` object "owns" the :py:class:`Cell` objects but only
            contains the :py:class:`ManagedMonitor` objects.

            If applying a function to :py:class:`Module` objects, e.g.
            via :py:meth:`CellTrainer.to` ``Cell`` objects will not be altered but
            ``ManagedMonitor`` objects will be.

            Likewise, :py:meth:`Layer.to` will alter ``Cell`` objects but not
            ``ManagedMonitor`` objects.
        """
        # call superclass
        Module.__init__(self, **kwargs)

        # interal storage for cells and monitors
        self.cells_: MutableMapping[str, Cell] = {}
        self.states_: MutableMapping[str, MutableMapping[str, nn.Module]] = (
            nn.ModuleDict()
        )
        self.monitors_: MutableMapping[str, MutableMapping[str, ManagedMonitor]] = (
            nn.ModuleDict()
        )

        # delete monitors on final dereference
        self.__finalizer = weakref.finalize(self, self.__finalize)

    def __finalize(self) -> None:
        r"""Deletes monitors from their layers."""
        # evaluate named monitors
        nm = tuple(self.named_monitors)

        # call deletion callback
        for (cell, name), _ in nm:
            self.del_monitor(cell, name)

    @property
    def monitors(self) -> Iterator[ManagedMonitor]:
        r"""Added monitors.

        Because monitors for each ``CellTrainer`` are pooled together for each trainer
        and are managed by their :py:class:`Layer`, duplicate monitors are not created
        where possible. The number of monitors here may be less than the number of
        added monitors.

        Yields:
            ManagedMonitor: added monitors.
        """
        return unique(chain.from_iterable(m.values() for m in self.monitors_.values()))

    @property
    def named_monitors(self) -> Iterator[tuple[tuple[str, str], ManagedMonitor]]:
        r"""Iterable of added monitors and tuples of the cell and monitor name.

        Yields:
            tuple[tuple[str, str], ManagedMonitor]: tuple of an added monitor and a tuple
            of the cell name and monitor name corresponding to it.
        """
        return chain.from_iterable(
            (((c, n), m) for n, m in md.items()) for c, md in self.monitors_.items()
        )

    def cellmonitors(self, cell: str) -> Iterator[tuple[str, ManagedMonitor]]:
        r"""Monitors associated with a given trainable.

        Args:
            cell (str): name of the cell to get associted monitors of.

        Yields:
            tuple[str, ManagedMonitor]: associated monitors and their names.
        """
        return ((n, m) for n, m in self.monitors_.get(cell, {}).items())

    @property
    def cells(self) -> Iterator[Cell]:
        r"""Added cells.

        Yields:
            Cell: added cells.
        """
        return (c for c in self.cells_.values())

    @property
    def named_cells(self) -> Iterator[tuple[str, Cell]]:
        r"""Added cells and their names.

        Yields:
            tuple[str, Cell]: tuple of an added cell and its name.
        """
        return ((n, c) for n, c in self.cells_.items())

    @property
    def cellstates(self) -> Iterator[tuple[str, nn.Module]]:
        """Additional state associated with cells.

        Yields:
            tuple[str, nn.Module]: tuple of added state and the associated cell name.
        """
        return ((c, s) for c, s in self.states_.items())

    def add_cell(
        self, name: str, cell: Cell, state: nn.Module | None = None, **kwargs
    ) -> Cell:
        r"""Adds a cell.

        Args:
            name (str): name of the cell to add.
            cell (Cell): cell to add.
            state (nn.Module | None, optional): any extra state to add. Defaults to None.

        Raises:
            ValueError: a cell with the specified name already exists.
            RuntimeError: given cell has no updater.

        Returns:
            Cell: added cell.
        """
        # ensure a trainable with the given name does not exist
        if name in self.cells_:
            raise ValueError(f"'name' ('{name}') is already the name of an added cell")

        # ensure the cell is updatable
        if not cell.updater:
            raise RuntimeError(
                "'cell' is not updatable, add an updater to the connection"
            )

        # delete any existing monitors (should never occur)
        if name in self.monitors_:
            for monitor in self.monitors_[name]:
                self.del_monitor(name, monitor)
            del self.monitors_[name]

        # delete any existing state (should never occur)
        if name in self.states_:
            del self.states_[name]

        # add the cell
        self.cells_[name] = cell
        if state is not None:
            self.states_[name] = state

        return self.cells_[name]

    def del_cell(self, name: str) -> None:
        r"""Deletes an added cell.

        Args:
            name (str): name of the cell to delete.

        Raises:
            AttributeError: specified cell does not exist.

        Important:
            This does not strictly delete the cell, it is still owned by its
            :py:class:`Layer` and can be added again. It is only removed from the
            trainer. However its associated state and monitors are deleted.
        """
        # check that the trainable exists
        if name not in self.cells_:
            raise AttributeError(f"'name' ('{name}') is not the name of an added cell")

        # delete any existing monitors
        if name in self.monitors_:
            for monitor in self.monitors_[name]:
                self.del_monitor(name, monitor)
            del self.monitors_[name]

        # delete any existing state
        if name in self.states_:
            del self.states_[name]

        # delete the cell
        del self.cells_[name]

    def get_cell(self, name: str) -> tuple[Cell | None, nn.Module | None]:
        r"""Gets an added cell and any added state.

        Args:
            name (str): name of the trainable to get.

        Returns:
            Trainable | None: specified trainable, if it exists.
        """
        return self.cells_.get(name, None), self.states_.get(name, None)

    def add_monitor(
        self,
        cell: str,
        name: str,
        attr: str,
        monitor: MonitorConstructor,
        unpooled: bool = False,
        **kwargs,
    ) -> ManagedMonitor:
        r"""Adds a monitor to a trainable.

        Args:
            cell (str): name of the cell to which the monitor will be added.
            name (str): name of the monitor to add (unique to the cell).
            attr (str): dot-seperated attribute to monitor, relative to the cell.
            monitor (MonitorConstructor): partial constructor for the monitor.
            unpooled (bool): if the monitor should not be aliased from the pool
                regardless. Defaults to False.

        Raises:
            AttributeError: specified cell does not exist.

        Returns:
            ManagedMonitor: added or retrieved monitor.

        Important:
            If the monitor exists, it will be retrieved even if the specifications
            (attribute and/or partial constructor) are different.

        Tip:
            If the monitor's behavior for the targeted attribute may vary with
            hyperparameters or other configuration state, ``unpooled`` should be
            set to ``True``. This does not keep this monitor from being aliased however,
            so the setting of ``unpooled`` should be consistent across all monitors
            with the same name.
        """
        # test if the monitor already exists and return if it does
        maybemon = self.monitors_.get(cell, {}).get(name, None)
        if maybemon:
            return maybemon

        # get underlying cell from given key
        if cell not in self.cells_:
            raise AttributeError(f"'cell' ('{cell}') is not the name of an added cell")

        # create monitor via the cell
        monitor = self.cells_[cell].add_monitor(self, name, attr, monitor, unpooled)

        # add monitor to the trainer
        if cell not in self.monitors_:
            self.monitors_[cell] = nn.ModuleDict()
        self.monitors_[cell][name] = monitor

        # return the monitor
        return monitor

    def del_monitor(self, cell: str, name: str) -> None:
        r"""Deletes an added monitor.

        Args:
            cell (str): name of the cell to which the monitor was added.
            name (str): name of the monitor.

        Raises:
            AttributeError: specified cell does not exist, or does not have a
                monitor with the specified name added to it.
        """
        # check that the cell has monitors
        if cell not in self.monitors_ or cell not in self.cells_:
            raise AttributeError(
                f"'cell' ('{cell}') is either not the name of an added "
                "cell or is a cell with no added monitors"
            )

        # check that the monitor to delete exists
        if name not in self.monitors_[cell]:
            raise AttributeError(
                f"'name' ('{name}') is not the name of a monitor added on cell "
                f"with name '{cell}'"
            )

        # delete the monitor
        del self.monitors_[cell][name]
        self.cells_[cell].del_monitor(self, name)

    def get_monitor(self, trainable: str, name: str) -> ManagedMonitor | None:
        r"""Gets an added monitor.

        Args:
            trainable (str): name of the trainable to which the monitor was added.
            name (str): name of the monitor.

        Returns:
            ManagedMonitor | None: specified monitor, if it exists.
        """
        return self.monitors_.get(trainable, {}).get(name, None)

    def attach(self, clear: bool = True, **kwargs) -> None:
        """Registers all of the monitors for the trainer.

        This additionally sets the trainer into training mode with ``self.train()``.

        Args:
            clear (bool, optional): If the monitors should be cleared before registering
                with submodules. Defaults to True.

        Note:
            Keyword arguments are passed to :py:meth:`Monitor.clear` call.
        """
        self.train()
        for monitor in self.monitors:
            if clear:
                monitor.clear(**kwargs)
            monitor.register()

    def detach(self, clear: bool = False, **kwargs) -> None:
        """Deregisters all of the monitors for the trainer.

        This additionally sets the trainer into evaluation mode with ``self.eval()``.

        Args:
            clear (bool, optional): If the monitors should be cleared before
                deregistering with submodules. Defaults to False.

        Note:
            Keyword arguments are passed to :py:meth:`Monitor.clear` call.
        """
        self.eval()
        for monitor in self.monitors:
            if clear:
                monitor.clear(**kwargs)
            monitor.deregister()

    def clear(self, **kwargs):
        """Clears all of the monitors for the trainer.

        Note:
            Keyword arguments are passed to :py:meth:`Monitor.clear` call.

        Note:
            If a subclassed trainer has additional state, this should be overridden
            to delete that state as well. This however doesn't delete updater state
            as it may be shared across trainers.
        """
        for monitor in self.monitors:
            monitor.clear(**kwargs)

    def update(self, **kwargs) -> None:
        r"""Applies all cumulative updates.

        This calls every updater which applies cumulative updates and any updater
        hooks are automatically called (e.g. parameter clamping). The updaters will
        each be called once, even if present in multiple cells.
        """
        for updater in unique(
            filter(lambda c: c is not None, map(lambda c: c.updater, self.cells))
        ):
            updater(**kwargs)

    def forward(self, *inputs, **kwargs):
        """Processes a training step.

        Raises:
            NotImplementedError: ``forward`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__}(CellTrainer) must implement "
            "the method `forward`."
        )


class LayerwiseTrainer(Module):
    def __init__(self, **kwargs):
        r"""Trainer for update methods without inter-layer dependencies.

        Important:
            The :py:class:`Layer` object "owns" the :py:class:`Cell` objects but only
            contains the :py:class:`ManagedMonitor` objects.

            If applying a function to :py:class:`Module` objects, e.g.
            via :py:meth:`CellTrainer.to` ``Cell`` objects will not be altered but
            ``ManagedMonitor`` objects will be.

            Likewise, :py:meth:`Layer.to` will alter ``Cell`` objects but not
            ``ManagedMonitor`` objects.
        """
        # call superclass
        CellTrainer.__init__(self, **kwargs)

    def add_layer(
        self,
        prefix: str,
        layer: Layer,
        state_constructor: Callable[[Cell], nn.Module] | None = None,
        **kwargs,
    ) -> dict[str, tuple[Cell, nn.Module | None]]:
        r"""Adds all trainables from a given layer.

        Args:
            prefix (str): string to prepend to trainable name.
            layer (Layer): layer from which trainables should be added.
            state_constructor (Callable[[Cell], nn.Module] | None, optional): function
                to construct extra state module for a cell. Defaults to None.

        Returns:
            dict[str, Trainable]: names of added trainables and those trainables.

        Note:
            As this calls :py:meth:`add_cell`, by default, keyword arguments are added
            as attributes the associated state. If no constructor is given, state is
            only added if keyword arguments are given, in which case they are added
            to an empty Module.

        Note:
            The names used are automatically generated based on the name of the input
            and name of the output in the layer. For example, with an input named
            ``"linear"`` and an output named ``"alif"``, and a ``prefix`` of ``"l0"``,
            the trainable name will be ``"l0:linear-alif"``.
        """
        prefix = f"{prefix}:" if prefix else ""
        state_constructor = state_constructor if state_constructor else lambda c: None
        return {
            n: self.add_cell(n, c, state_constructor(c), **kwargs)
            for n, c in fzip(
                layer.named_cells,
                lambda nc: f"{prefix}{nc[0][0]}-{nc[0][1]}",
                lambda nc: nc[1],
                identity=False,
            )
        }

    def add_cell(
        self, name: str, cell: Cell, state: nn.Module | None = None, **kwargs
    ) -> tuple[Cell, nn.Module | None]:
        r"""Adds a cell with associated state.

        Args:
            name (str): name of the cell to add.
            cell (Cell): cell to add.
            state (nn.Module | None, optional): optional state to add, if None and
                keyword arguments are given, an empty Module will be constructed.
                Defaults to None.

        Raises:
            ValueError: a cell with the specified name already exists.
            RuntimeError: given cell has no updater.

        Returns:
            tuple[Cell, nn.Module | None]: added cell and additional state.

        Important:
            Keyword arguments are automatically added as non-persistant state to the
            cell's associate state.
        """
        # determine associated state
        if kwargs and not state:
            state = Module()

        # call superclass method
        cell, state = CellTrainer.add_cell(self, name, Cell, state, **kwargs)

        # adds keyword arguments to state
        if kwargs:
            for k, v in kwargs.items():
                setattr(state, k, v)

        return cell, state


class LayerwiseUpdater(Module, ABC):
    r"""Updater for layers without interdependencies.

    Args:
        *layers (Layer): layers to add to the updater on initialization.

    Caution:
        The property :py:attr:`training` should be managed using
        the updater's :py:meth:`attach` and :py:meth:`detach` methods rather than
        :py:meth:`train` and :py:meth:`eval`.

    Note:
        Registered :py:class:`Layer` and :py:class:`Monitor` objects have their
        parameters excluded from :py:attr:`state_dict` using a hook registered with
        :py:meth:`_register_state_dict_hook`.
    """

    def __init__(
        self,
        *layers: Layer,
        **kwargs,
    ):
        # call superclass constructor
        Module.__init__(self, **kwargs)

        # register modules
        self.trainables_ = nn.ModuleDict()
        self.monitors_ = nn.ModuleDict()

        # go through pairs
        for layer in layers:
            self.add_trainable(layer)

        # add hook for removal of trainables from state_dict export
        def state_dict_submodule_removal(
            obj, state_dict, prefix, local_metadata
        ) -> dict[str, Any]:
            for key in list(state_dict.keys()):
                match key.partition(".")[0]:
                    case "trainables_":
                        del state_dict[key]
                    case "monitors_":
                        del state_dict[key]

            return state_dict

        self._sd_removal_handle = self._register_state_dict_hook(
            state_dict_submodule_removal
        )

    @property
    def trainables(self) -> Generator[Layer]:
        r"""Registered layers to be trained.

        Yields:
            Layer: registered trainables.
        """
        return (layer for layer in self.trainables_.values())

    @property
    def monitors(self) -> Generator[tuple[Monitor, Layer]]:
        r"""Registered monitors for capturing layer state.

        Yields:
            tuple[Monitor, Layer]: registered monitors and their associated layer.
        """
        return itertools.chain.from_iterable(
            ((monitor, self.trainables_[key]) for monitor in monitors.values())
            for key, monitors in self.monitors_.items()
        )

    def add_trainable(
        self, trainable: Layer, add_monitors: bool = True, **kwarge
    ) -> bool:
        r"""Registers a layer as a trainable.

        Args:
            trainable (Layer): layer to register as a trainable.
            add_monitors (bool, optional): if default monitors should be added after
                registring. Defaults to True.

        Returns:
            bool: if the layer was successfully registered.
        """
        # use hexadecimal of object id as the key
        key = hex(id(trainable))

        # add to trainables list for accessing
        if key in self.trainables_:
            return False
        self.trainables_[key] = trainable

        # add to mapping from layers to monitors
        if key in self.monitors_:
            del self.monitors_[key]
        self.monitors_[key] = nn.ModuleDict()

        # add default monitor layout to trainable
        if add_monitors and not len(self.monitors_[key]):
            self.add_monitors(trainable)

        return True

    def del_trainable(self, trainable: Layer, **kwarge) -> bool:
        r"""Deletes a registered layer and its associated monitors.

        Args:
            trainable (Layer): trainable to delete.

        Returns:
            bool: if the layer was successfully deleted.
        """
        # use hexadecimal of object id as the key
        key = hex(id(trainable))

        # try to delete trainable
        try:
            del self.trainables_[key]
            del self.monitors_[key]
            return True
        except KeyError:
            return False

    def add_monitor(
        self, trainable: Layer, name: str, monitor: Monitor, **kwargs
    ) -> bool:
        r"""Adds and associates a monitor with a registered layer.

        Args:
            trainable (Layer): layer to which the monitor should be registered.
            name (str): identifier of the monitor to add.
            monitor (Monitor): monitor to register.

        Returns:
            bool: if the monitor was successfully added.
        """
        # try to get the moduledict with monitors
        try:
            monitors = self.monitors_[hex(id(trainable))]
        except KeyError:
            return False

        # check if name is already associated with trainable
        if name in monitors:
            return False

        # check if this monitor is already registered
        if monitor in (mon for mon, _ in self.monitors):
            return False

        monitor.deregister()
        monitors[name] = monitor
        if self.training:
            monitor.register(trainable)

        return True

    def del_monitor(self, trainable: Layer, name: str) -> bool:
        r"""Deletes a monitor associated with a registered layer.

        Args:
            trainable (Layer): layer from which the monitor should be deleted.
            name (str): dentifier of the monitor to delete.

        Returns:
            bool: if the monitor was successfully deleted.
        """
        # try to get the moduledict with monitors
        try:
            monitors = self.monitors_[hex(id(trainable))]
        except KeyError:
            return False

        # try to delete monitor
        try:
            del monitors[name]
            return True
        except KeyError:
            return False

    def get_monitor(self, trainable: Layer, name: str) -> Monitor | None:
        r"""Retrieves a monitor associated with a registered layer.

        Args:
            trainable (Layer): layer with which the monitor is associated.
            name (str): name of the monitor to get.

        Returns:
            Monitor | None: monitor specified by ``trainable`` and name if it exists.
        """
        # try to get the moduledict with monitors
        try:
            monitors = self.monitors_[hex(id(trainable))]
        except KeyError:
            return None

        # try to retrieve monitor
        try:
            return monitors[name]
        except KeyError:
            return None

    @abstractmethod
    def add_monitors(self, trainable: Layer) -> bool:
        r"""Associates base layout of monitors required by the updater with the layer.

        Args:
            trainable (Layer): layer to which monitors should be added.

        Returns:
            bool: if the monitors were successfully added.

        Raises:
            NotImplementedError: ``add_monitors`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__}(LayerwiseUpdater) must implement "
            "the method `add_monitors`."
        )

    def del_monitors(self, trainable: Layer) -> bool:
        r"""Deletes all monitors associated with a registered layer.

        Args:
            trainable (Layer): layer from which monitors should be deleted.

        Returns:
            bool: if the monitors were successfully deleted.
        """
        key = hex(id(trainable))

        # try to get the moduledict with monitors
        try:
            monitors = self.monitors_[key]
        except KeyError:
            return False

        # remove all elements
        monitors.clear()

        return True

    def get_monitors(self, trainable: Layer) -> Generator[tuple[Monitor, str]]:
        r"""Retrieves all monitors associated with a registered layer.

        Args:
            trainable (Layer): layer for which monitors should be retrieved.

        Yields:
            tuple[Monitor, str]: registered monitors and their names.
        """
        # use hexadecimal of object id as the key
        key = hex(id(trainable))

        # try to get the moduledict with monitors
        try:
            monitors = self.monitors_[key]
        except KeyError:
            return None

        # construct generator
        return ((monitor, name) for name, monitor in monitors.items())

    def attach(self, clear: bool = True, **kwargs) -> None:
        """Registers all of the monitors for the updater.

        Args:
            clear (bool, optional): If the monitors should be cleared before registering
                with submodules. Defaults to True.

        Note:
            Keyword arguments are passed to :py:meth:`Monitor.clear` call.
        """
        _ = self.train()
        for monitor, layer in self.monitors:
            if clear:
                monitor.clear(**kwargs)
            monitor.register(layer)

    def detach(self, clear: bool = False, **kwargs) -> None:
        """Deregisters all of the monitors for the updater.

        Args:
            clear (bool, optional): If the monitors should be cleared before
                deregistering with submodules. Defaults to False.

        Note:
            Keyword arguments are passed to :py:meth:`Monitor.clear` call.
        """
        _ = self.eval()
        for monitor, _ in self.monitors:
            if clear:
                monitor.clear(**kwargs)
            monitor.deregister()

    def clear(self, **kwargs):
        """Clears all of the monitors for the updater.

        Note:
            Keyword arguments are passed to :py:meth:`Monitor.clear` call.

        Note:
            If a subclassed updater has additional state, this should be overridden
            to delete that state as well.
        """

        for monitor, _ in self.monitors:
            monitor.clear(**kwargs)

    @abstractmethod
    def forward(self, *inputs, **kwargs):
        """Processes update for given layers based on current monitor stored data.

        Raises:
            NotImplementedError: ``forward`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__}(LayerwiseUpdater) must implement "
            "the method `forward`."
        )
