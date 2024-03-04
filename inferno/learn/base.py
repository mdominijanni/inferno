from abc import ABC, abstractmethod
from collections.abc import Generator, Iterator
from .. import Module
from inferno._internal import fzip, unique
from inferno.neural import Cell, Layer
from inferno.observe import Monitor, ManagedMonitor, MonitorConstructor
import itertools
from itertools import chain
import torch  # noqa:F401, for docstrings
import torch.nn as nn
from typing import Any, Callable


class CellTrainer(Module):
    def __init__(self, **kwargs):
        r"""Base trainer for updating cell parameters.

        Important:
            The :py:class:`Layer` object "owns" the :py:class:`Cell` objects whereas
            the :py:class:`CellTrainer` owns the :py:class:`ManagedMonitor` objects.

            If applying a function to :py:class:`Module` objects, e.g.
            via :py:meth:`CellTrainer.to` ``Cell`` objects will not be altered but
            ``ManagedMonitor`` objects will be.

            Likewise, :py:meth:`Layer.to` will alter ``Cell`` objects but not
            ``ManagedMonitor`` objects.
        """
        # call superclass
        Module.__init__(self, **kwargs)

        # interal storage for cells and monitors
        self.cells_ = {}
        self.auxiliary_ = nn.ModuleDict()
        self.monitors_ = nn.ModuleDict()

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
    def aux(self) -> Iterator[nn.Module]:
        r"""Auxiliary state associated with cells.

        Yields:
            nn.Module: tuple of added state and the associated cell name.
        """
        return (s for s in self.auxiliary_.values())

    @property
    def named_aux(self) -> Iterator[tuple[str, nn.Module]]:
        r"""Auxiliary state associated with cells and their names.

        Yields:
            tuple[str, nn.Module]: tuple of added state and the associated cell name.
        """
        return ((n, s) for n, s in self.auxiliary_.items())

    def cellaux(self, cell: str) -> nn.Module | None:
        """Auxiliary state associated with a cell, if any.

        Args:
            cell (str): name of the cell to get auxiliary state of.

        Returns:
            nn.Module | None: auxiliary state of the cell if it has any.
        """
        return self.auxiliary_.get(cell, None)

    def add_cell(
        self, name: str, cell: Cell, aux: nn.Module | None = None, **kwargs
    ) -> tuple[Cell, nn.Module | None]:
        r"""Adds a cell.

        Args:
            name (str): name of the cell to add.
            cell (Cell): cell to add.
            aux (nn.Module | None, optional): any extra state to add. Defaults to None.

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
                "'cell' is not updatable, add an updater to 'cell.connection'"
            )

        # delete any existing monitors (should never occur)
        if name in self.monitors_:
            for monitor in self.monitors_[name]:
                self.del_monitor(name, monitor)
            del self.monitors_[name]

        # delete any existing state (should never occur)
        if name in self.auxiliary_:
            del self.auxiliary_[name]

        # add the cell
        self.cells_[name] = cell
        if aux is not None:
            self.auxiliary_[name] = aux

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
        if name in self.auxiliary_:
            del self.auxiliary_[name]

        # delete the cell
        del self.cells_[name]

    def get_cell(self, name: str) -> tuple[Cell | None, nn.Module | None]:
        r"""Gets an added cell and any added state.

        Args:
            name (str): name of the trainable to get.

        Returns:
            Trainable | None: specified trainable, if it exists.
        """
        return self.cells_.get(name, None), self.auxiliary_.get(name, None)

    def add_monitor(
        self,
        cell: str,
        name: str,
        attr: str,
        monitor: MonitorConstructor,
        unique: bool = False,
        **tags: Any,
    ) -> ManagedMonitor:
        r"""Adds a monitor to a trainable.

        Args:
            cell (str): name of the cell to which the monitor will be added.
            name (str): name of the monitor to add (unique to the cell).
            attr (str): dot-seperated attribute to monitor, relative to the cell.
            monitor (MonitorConstructor): partial constructor for the monitor.
            unique (bool): if the monitor should not be aliased from the pool
                regardless. Defaults to False.
            **tags (Any): tags to determine if the monitor is unique amongst monitors
                with the same name, class, and reducer class.

        Raises:
            AttributeError: specified cell does not exist.

        Returns:
            ManagedMonitor: added or retrieved monitor.

        Important:
            Monitors are aliased based off of their name, resolved attribute path
            (i.e. that which is used by the layer), class, the class of their reducer,
            This attribute is added as a tag ``"attr"``.


        Tip:
            Setting ``unique`` to ``True`` will bypass any test on ``tags`` and
            guarantee the monitor will be unique. This is less efficient, but if,
            for instance, monitor properties may change over the course of training,
            this should be used. When ``True`` is also guaranteed to delete an existing
            monitor and return the new one.
        """
        # create expanded tag set
        tags = {"monitor": monitor.monitor, "reducer": monitor.reducer, **tags}

        # get underlying cell from given key
        if cell not in self.cells_:
            raise AttributeError(f"'cell' ('{cell}') is not the name of an added cell")

        # test if the monitor already exists and return if it does, delete if unique
        qmon = self.monitors_.get(cell, {}).get(name, None)
        if qmon:
            if unique:
                del self.monitors_[cell][name]
            else:
                return qmon

        # create monitor group if it doesn't exist
        if cell not in self.monitors_:
            self.monitors_[cell] = nn.ModuleDict()

        # construct the pool
        if unique:
            pool = None
        else:
            pool = (
                (c, {mn: m for mn, m in self.monitors_[cn].items()})
                for cn, c in self.cells_.items()
                if cn in self.monitors_
            )

        # create monitor via the cell
        monitor = self.cells_[cell].get_monitor(name, attr, monitor, pool)

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
        self.monitors_[cell][name].deregister()
        del self.monitors_[cell][name]

    def get_monitor(self, cell: str, name: str) -> ManagedMonitor | None:
        r"""Gets an added monitor.

        Args:
            cell (str): name of the cell to which the monitor was added.
            name (str): name of the monitor.

        Returns:
            ManagedMonitor | None: specified monitor, if it exists.
        """
        return self.monitors_.get(cell, {}).get(name, None)

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


class LayerwiseTrainer(CellTrainer):
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
        aux_constructor: Callable[[Cell], nn.Module] | None = None,
        **kwargs,
    ) -> dict[str, tuple[Cell, nn.Module | None]]:
        r"""Adds all cells from a given layer.

        Args:
            prefix (str): string to prepend to cell name.
            layer (Layer): layer from which cells should be added.
            aux_constructor (Callable[[Cell], nn.Module] | None, optional): function
                to construct auxiliary state module for a cell. Defaults to None.

        Returns:
            dict[str, tuple[Cell, nn.Module | None]]: names of added cells, those cells,
            and any auxiliary state.

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
        aux_constructor = aux_constructor if aux_constructor else lambda c: None
        return {
            n: self.add_cell(n, c, aux_constructor(c), **kwargs)
            for n, c in fzip(
                layer.named_cells,
                lambda nc: f"{prefix}{nc[0][0]}-{nc[0][1]}",
                lambda nc: nc[1],
                identity=False,
            )
        }

    def add_cell(
        self, name: str, cell: Cell, aux: nn.Module | None = None, **kwargs
    ) -> tuple[Cell, nn.Module | None]:
        r"""Adds a cell with associated state.

        Args:
            name (str): name of the cell to add.
            cell (Cell): cell to add.
            aux (nn.Module | None, optional): optional state to add, if None and
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
        if kwargs and not aux:
            aux = Module()

        # call superclass method
        cell, aux = CellTrainer.add_cell(self, name, Cell, aux, **kwargs)

        # adds keyword arguments to state
        if kwargs:
            for k, v in kwargs.items():
                setattr(aux, k, v)

        return cell, aux


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
