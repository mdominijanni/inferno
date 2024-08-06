from __future__ import annotations
from .. import Module
from .._internal import getitem, unique
from ..neural import Cell
from ..observe import Monitor, MonitorConstructor, MonitorPool
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
import torch.nn as nn
from typing import Any
import weakref


class CellTrainer(Module):
    r"""Base trainer for updating cell parameters.

    Important:
        The :py:class:`~inferno.neural.Layer` object "owns" the
        py:class:`~inferno.neural.Cell` objects whereas the ``CellTrainer`` owns the
        :py:class:`~inferno.observe.Monitor` objects.

        If applying a function to :py:class:`~inferno.Module` objects, e.g.
        via calling :py:meth:`~torch.nn.Module.to` on the ``CellTrainer``,
        ``~inferno.neural.Cell`` objects will not be altered but
        ``~inferno.observe.Monitor`` and auxiliary state objects will be.

        Likewise, calling :py:meth:`~torch.nn.Module.to` on the
        :py:class:`~inferno.neural.Layer` will alter ``~inferno.neural.Cell`` objects
        but not ``~inferno.observe.Monitor`` and auxiliary state objects.
    """

    def __init__(self, **kwargs):
        # call superclass constructor
        Module.__init__(self, **kwargs)

        # interval storage for cells and monitors
        self.cells_ = weakref.WeakValueDictionary()
        self.aux_states_ = nn.ModuleDict()
        self.monitor_pool_ = MonitorPool()

    @property
    def monitors(self) -> Iterator[Monitor]:
        r"""Added monitors.

        Because monitors for each ``~inferno.neural.CellTrainer`` are pooled together for each trainer,
        duplicate monitors are not created where possible. The number of monitors here
        may be less than the number of added monitors.

        Yields:
            Monitor: added monitors.
        """
        return self.monitor_pool_.monitors()

    @property
    def named_monitors(self) -> Iterator[tuple[tuple[str, str], Monitor]]:
        r"""Iterable of added monitors and tuples of the cell and monitor name.

        Yields:
            tuple[tuple[str, str], Monitor]: tuple of an added monitor and a tuple
            of the cell name and monitor name corresponding to it.
        """
        return self.monitor_pool_.named_monitors()

    def named_monitors_of(self, cell: str) -> Iterator[tuple[str, Monitor]]:
        r"""Monitors associated with a given cell.

        Args:
            cell (str): name of the cell to get associated monitors of.

        Yields:
            tuple[str, Monitor]: associated monitors and their names.
        """
        return self.monitor_pool_.named_monitors_of(cell)

    @property
    def cells(self) -> Iterator[tuple[Cell, nn.Module | None]]:
        r"""Added cells and their auxiliary states.

        Yields:
            Cell: added cells and their auxiliary states.
        """
        return ((c, getattr(self.aux_states_, n, None)) for n, c in self.cells_.items())

    @property
    def named_cells(self) -> Iterator[tuple[str, tuple[Cell, nn.Module | None]]]:
        r"""Added cell, their auxiliary states, and their names.

        Yields:
            tuple[str, Cell]: tuple of an added cell, their auxiliary states, and its name.
        """
        return (
            (n, (c, getattr(self.aux_states_, n, None))) for n, c in self.cells_.items()
        )

    def add_cell(
        self,
        name: str,
        cell: Cell,
        state: nn.Module | None = None,
        params: Sequence[str] | None = None,
    ) -> tuple[Cell, nn.Module | None]:
        r"""Adds a cell and any auxiliary state.

        Args:
            name (str): name of the cell to add.
            cell (Cell): cell to add.
            state (nn.Module | None, optional): any extra state to add.
                Defaults to ``None``.
            params (Sequence[str] | None, optional): trainable parameters the cell
                updater must have. Defaults to ``None``.


        Raises:
            ValueError: a cell with the specified name already exists.
            RuntimeError: given cell has no updater.

        Returns:
            tuple[Cell | None, nn.Module | None]: added cell and auxiliary state.
        """
        # ensure a trainable with the given name does not exist
        if name in self.cells_:
            raise ValueError(f"'name' ('{name}') is already the name of an added cell")

        # ensure the cell is updatable
        if not cell.updater:
            raise RuntimeError(
                "'cell' is not updatable, add an updater to 'cell.connection'"
            )

        # ensure the cell has required trainable parameters
        if params:
            for p in params:
                if not hasattr(cell.updater, p):
                    raise RuntimeError(
                        f"'cell' does not contain required parameter '{p}'"
                    )

        # delete any existing monitors (should only occur if a cell died w/o removal)
        self.monitor_pool_.del_observed(name)

        # delete any existing state (should only occur if a cell died w/o removal)
        if name in self.aux_states_:
            del self.aux_states_[name]

        # add the cell
        self.cells_[name] = cell
        self.monitor_pool_.add_observed(name, cell)

        if state is not None:
            self.aux_states_[name] = state

        return getitem(self.cells_, name), getitem(self.aux_states_, name, None)

    def get_cell(self, name: str) -> tuple[Cell, nn.Module | None]:
        r"""Gets an added cell and any added state.

        Args:
            name (str): name of the trainable to get.

        Returns:
            tuple[Cell, nn.Module | None]: specified trainable, if it exists.
        """
        return getitem(self.cells_, name), getitem(self.aux_states_, name, None)

    def del_cell(self, name: str) -> None:
        r"""Deletes an added cell.

        Args:
            name (str): name of the cell to delete.

        Raises:
            AttributeError: specified cell does not exist.

        Important:
            This does not strictly delete the cell, it is still owned by its
            :py:class:`~inferno.neural.Layer` and can be added again. It is only removed from the
            trainer. However its associated state and monitors are deleted.
        """
        # check that the trainable exists
        if name not in self.cells_:
            raise AttributeError(f"'name' ('{name}') is not the name of an added cell")

        # delete any existing monitors
        self.monitor_pool_.del_observed(name)

        # delete any existing state
        if name in self.aux_states_:
            del self.aux_states_[name]

        # delete the cell
        del self.cells_[name]

    def add_monitor(
        self,
        cell: str,
        name: str,
        attr: str,
        monitor: MonitorConstructor,
        unique: bool = False,
        /,
        **tags: Any,
    ) -> Monitor:
        r"""Adds a monitor to a trainable.

        Args:
            cell (str): name of the cell to which the monitor will be added.
            name (str): name of the monitor to add (unique to the cell).
            attr (str): dot-separated attribute to monitor, relative to the cell.
            monitor (MonitorConstructor): partial constructor for the monitor.
            unique (bool): if the monitor should not be aliased from the pool
                regardless. Defaults to ``False``.
            **tags (Any): tags to determine if the monitor is unique amongst monitors
                with the same name, class, and reducer class.

        Raises:
            AttributeError: specified cell does not exist.

        Returns:
            Monitor: added or retrieved monitor.

        Important:
            Monitors are aliased based off of their name, resolved attribute path
            (i.e. that which is used by the layer), and any added tags.
            This attribute is added as a tag ``"attr"``.

        Tip:
            Setting ``unique`` to ``True`` will bypass any test on ``tags`` and
            guarantee the monitor will be unique. This is less efficient, but if,
            for instance, monitor properties may change over the course of training,
            this should be used. When ``True`` is also guaranteed to delete an existing
            monitor and return the new one.
        """
        # get underlying cell from given key
        if cell not in self.cells_:
            raise AttributeError(f"'cell' ('{cell}') is not the name of an added cell")

        return self.monitor_pool_.add_monitor(cell, name, attr, monitor, unique, **tags)

    def get_monitor(self, cell: str, name: str) -> Monitor | None:
        r"""Gets an added monitor.

        Args:
            cell (str): name of the cell to which the monitor was added.
            name (str): name of the monitor.

        Returns:
            Monitor | None: specified monitor, if it exists.
        """
        return self.monitor_pool_.get_monitor(cell, name)

    def del_monitor(self, cell: str, name: str) -> None:
        r"""Deletes an added monitor.

        Args:
            cell (str): name of the cell to which the monitor was added.
            name (str): name of the monitor.

        Raises:
            AttributeError: specified cell does not exist, or does not have a
                monitor with the specified name added to it.
        """
        self.monitor_pool_.del_monitor(cell, name)

    def train(self, mode: bool = True) -> CellTrainer:
        r"""Override of module's train method.

        Automatically registers and deregisters monitors. Does not put :py:class:`~inferno.neural.Cell`
        objects into training mode. Does not clear monitor state (call :py:meth:`clear`
        to do so).

        Args:
            mode (bool, optional): if the trainer should be set in training mode
                (``True``) or evaluation mode (``False``). Defaults to ``True``.

        Returns:
            CellTrainer: self.
        """
        # call superclass function
        Module.train(self, mode)

        # attach/detach monitors
        if mode:
            for monitor in self.monitor_pool_.monitors:
                monitor.register()
        else:
            for monitor in self.monitor_pool_.monitors:
                monitor.deregister()

        return self

    def clear(self, **kwargs) -> None:
        r"""Clears all of the monitors for the trainer.

        Note:
            Keyword arguments are passed to :py:meth:`~inferno.observe.Monitor.clear` call.

        Note:
            If a subclassed trainer has additional state, this should be overridden
            to delete that state as well. This however doesn't delete updater state
            as it may be shared across trainers.
        """
        for monitor in self.monitor_pool_.monitors:
            monitor.clear(**kwargs)

    def update(self, **kwargs) -> None:
        r"""Applies all cumulative updates.

        This calls every updater which applies cumulative updates and any updater
        hooks are automatically called (e.g. parameter clamping). The updaters will
        each be called once, even if present in multiple cells.

        This will apply all updates, not only those created by this trainer.
        """
        for updater in unique(
            filter(lambda c: c is not None, map(lambda c: c.updater, self.cells))
        ):
            updater(**kwargs)

    def forward(self, *inputs, **kwargs):
        r"""Processes a training step.

        Raises:
            NotImplementedError: ``forward`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__}(CellTrainer) must implement the method 'forward'"
        )


class IndependentCellTrainer(CellTrainer, ABC):
    r"""Trainer for update methods without inter-cell dependencies.

    Important:
        The :py:class:`~inferno.neural.Layer` object "owns" the :py:class:`~inferno.neural.Cell` objects but not
        the :py:class:`~inferno.observe.Monitor` objects.

        If applying a function to :py:class:`~inferno.Module` objects, e.g.
        via calling :py:meth:`~torch.nn.Module.to` on the ``CellTrainer``,
        ``~inferno.neural.Cell`` objects will not be altered but
        ``~inferno.observe.Monitor`` and auxiliary state objects will be.

        Likewise, calling :py:meth:`~torch.nn.Module.to` on the
        :py:class:`~inferno.neural.Layer` will alter ``~inferno.neural.Cell`` objects
        but not ``~inferno.observe.Monitor`` and auxiliary state objects.
    """

    class Unit(Module):
        r"""Trainable units.

        Attributes:
            cell (Cell): registered cell.
            state (nn.Module): auxiliary state.
            monitors (nn.ModuleDict): associated monitors.

        Args:
            cell (Cell): registered cell.
            state (nn.Module): auxiliary state.
            monitors (dict[str, Monitor]): associated monitors.
        """

        def __init__(self, cell: Cell, state: nn.Module, monitors: dict[str, Monitor]):
            Module.__init__(self)
            self.cell = cell
            self.state = state
            self.monitors = nn.ModuleDict(monitors)

    def __init__(self, **kwargs):
        # call superclass constructor
        CellTrainer.__init__(self, **kwargs)

    def __iter__(
        self,
    ) -> Iterator[tuple[Cell, nn.Module | None, dict[str, Monitor]]]:
        r"""Iterates over trainer contents.

        Yields:
            tuple[Cell, nn.Module | None, dict[str, Monitor]]: trainable unit, containing
            a cell, the auxiliary state for it, and a dictionary of applicable monitors.
        """
        for name in self.cells_:
            yield (
                getitem(self.cells_, name),
                getitem(self.aux_states_, name, None),
                dict(self.monitor_pool_.named_monitors_of(name)),
            )

    def get_unit(self, name: str) -> IndependentCellTrainer.Unit:
        r"""Gets a trainable unit.

        This can be used if a single trainer should handle training on different
        devices. The returned module dictionary can then be altered in-place with
        :py:meth:`~torch.nn.Module.to` in order to move a cell (``cell``), its auxiliary
        state (``state``), and monitors (``monitors``) used by the trainer to a
        different device, or change the used data type.

        Calling :py:meth:`~torch.nn.Module.to` on the trainer itself will apply to all
        monitors and auxiliary states, but not to any cells.

        Args:
            name (str): name of the cell to get alongside its auxiliary state.

        Returns:
            IndependentCellTrainer.Unit: specified cell, auxiliary state, and monitors.
        """
        return self.Unit(
            getitem(self.cells_, name),
            getitem(self.aux_states_, name, None),
            dict(self.monitor_pool_.named_monitors_of(name)),
        )

    @abstractmethod
    def register_cell(
        self, name: str, cell: Cell, **kwargs
    ) -> IndependentCellTrainer.Unit:
        r"""Adds a cell with any required state.

        Args:
            name (str): name of the cell to add.
            cell (Cell): cell to add.

        Returns:
            IndependentCellTrainer.Unit: specified cell, auxiliary state, and monitors,
            as returned by :py:meth:`get_unit`.

        Raises:
            NotImplementedError: ``register_cell`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__}(IndependentCellTrainer) must implement "
            "the method 'register_cell'"
        )
