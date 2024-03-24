from __future__ import annotations
from .. import Module
from .._internal import fzip, getitem, unique
from ..neural import Cell, Layer
from ..observe import Monitor, MonitorConstructor, MonitorPool
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
import torch  # noqa:F401, for docstrings
import torch.nn as nn
from typing import Any
import weakref


class CellTrainer(Module):
    def __init__(self, **kwargs):
        r"""Base trainer for updating cell parameters.

        Important:
            The :py:class:`Layer` object "owns" the :py:class:`Cell` objects whereas
            the :py:class:`CellTrainer` owns the :py:class:`Monitor` objects.

            If applying a function to :py:class:`Module` objects, e.g.
            via :py:meth:`CellTrainer.to` ``Cell`` objects will not be altered but
            ``Monitor`` objects will be.

            Likewise, :py:meth:`Layer.to` will alter ``Cell`` objects but not
            ``Monitor`` objects.
        """
        # call superclass
        Module.__init__(self, **kwargs)

        # interal storage for cells and monitors
        self.cells_ = weakref.WeakValueDictionary()
        self.aux_states_ = nn.ModuleDict()
        self.monitor_pool_ = MonitorPool()

    @property
    def monitors(self) -> Iterator[Monitor]:
        r"""Added monitors.

        Because monitors for each ``CellTrainer`` are pooled together for each trainer,
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
            cell (str): name of the cell to get associted monitors of.

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
        self, name: str, cell: Cell, state: nn.Module | None = None
    ) -> tuple[Cell, nn.Module | None]:
        r"""Adds a cell and any auxiliary state.

        Args:
            name (str): name of the cell to add.
            cell (Cell): cell to add.
            state (nn.Module | None, optional): any extra state to add. Defaults to None.

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

        return getitem(self.cells_, name), getitem(self.aux_states_, name)

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
        self.monitor_pool_.del_observed(name)

        # delete any existing state
        if name in self.aux_states_:
            del self.aux_states_[name]

        # delete the cell
        del self.cells_[name]

    def get_cell(self, name: str) -> tuple[Cell | None, nn.Module | None]:
        r"""Gets an added cell and any added state.

        Args:
            name (str): name of the trainable to get.

        Returns:
            tuple[Cell | None, nn.Module | None]: specified trainable, if it exists.
        """
        return getitem(self.cells_, name), getitem(self.aux_states_, name)

    def add_monitor(
        self,
        cell: str,
        name: str,
        attr: str,
        monitor: MonitorConstructor,
        unique: bool = False,
        tags: Mapping[str, Any] | None = None,
    ) -> Monitor:
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

        return self.monitor_pool_.add_monitor(cell, name, attr, monitor, unique, tags)

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

    def get_monitor(self, cell: str, name: str) -> Monitor | None:
        r"""Gets an added monitor.

        Args:
            cell (str): name of the cell to which the monitor was added.
            name (str): name of the monitor.

        Returns:
            Monitor | None: specified monitor, if it exists.
        """
        return self.monitor_pool_.get_monitor(cell, name)

    def train(self, mode: bool = True) -> CellTrainer:
        r"""Override of module's train method.

        Automatically registers and deregisters monitors. Does not put :py:class:`Cell`
        objects into training mode. Does not clear monitor state (call :py:meth:`clear`
        to do so).

        Args:
            mode (bool, optional): if the trainer should be set in training mode (True)
                or evaluation mode (False). Defaults to True.

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

    def clear(self, **kwargs):
        """Clears all of the monitors for the trainer.

        Note:
            Keyword arguments are passed to :py:meth:`Monitor.clear` call.

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


class LayerwiseTrainer(CellTrainer, ABC):
    def __init__(self, **kwargs):
        r"""Trainer for update methods without inter-layer dependencies.

        Important:
            The :py:class:`Layer` object "owns" the :py:class:`Cell` objects but not
            the :py:class:`Monitor` objects.

            If applying a function to :py:class:`Module` objects, e.g.
            via :py:meth:`CellTrainer.to` ``Cell`` objects will not be altered but
            ``Monitor`` objects will be.

            Likewise, :py:meth:`Layer.to` will alter ``Cell`` objects but not
            ``Monitor`` objects.
        """
        # call superclass
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
                getitem(self.aux_states_, name),
                dict(self.monitor_pool_.named_monitors_of(name)),
            )

    def _add_cell(
        self, name: str, cell: Cell, state: nn.Module | None = None
    ) -> tuple[Cell, nn.Module | None]:
        r"""Alias to CellTrainer's add_cell function.

        This aliases py:meth:`CellTrainer.add_cell` for simplified access in subclasses.

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
        return CellTrainer.add_cell(self, name, cell, state)

    @abstractmethod
    def add_cell(self, name: str, cell: Cell, **kwargs) -> str:
        r"""Adds a cell with any required state.

        Args:
            name (str): name of the cell to add.
            cell (Cell): cell to add.

        Returns:
            str: name of the added cell.

        Raises:
            NotImplementedError: ``add_cell`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__}(LayerwiseTrainer) must implement "
            "the method `add_cell`."
        )

    def add_layer(
        self,
        prefix: str,
        layer: Layer,
        **kwargs: Any,
    ) -> tuple[str, ...]:
        r"""Adds all cells from a given layer.

        Args:
            prefix (str): string to prepend to cell name.
            layer (Layer): layer from which cells should be added.
            **kwargs (Any): keyword arguments passed to each :py:meth:`add_cell` call.

        Returns:
            tuple[str, ...]: names of the added cells.

        Note:
            The names used are automatically generated based on the name of the cell in
            the layer. For example, with a connection named ``"linear"`` and a neuron
            named ``"alif"``, and a ``prefix`` of ``"l0"``, the cell name will be
            ``"l0:linear-alif"``. If ``prefix`` is an empty string, the name would be
            ``"linear-alif"``.
        """
        prefix = f"{prefix}:" if prefix else ""
        return tuple(
            self.add_cell(n, c, **kwargs)
            for n, c in fzip(
                layer.named_cells,
                lambda nc: f"{prefix}{nc[0][0]}-{nc[0][1]}",
                lambda nc: nc[1],
                identity=False,
            )
        )
