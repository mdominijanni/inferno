from abc import ABC, abstractmethod
from collections.abc import Generator, Iterator
from .. import Module
from inferno._internal import fzip, unique
from inferno.neural import Layer, Trainable
from inferno.observe import Monitor, ManagedMonitor, MonitorConstructor
import itertools
from itertools import chain
import torch  # noqa:F401, for docstrings
import torch.nn as nn
from typing import Any
import uuid


class Trainer(Module):
    def __init__(self, name: str | None, **kwargs):
        r"""Base trainer for updating connection parameters.

        Args:
            name (str | None): name of the trainer, for layer monitor pooling.

        Important:
            The :py:class:`Layer` object "owns" the :py:class:`Trainable` and
            :py:class:`ManagedMonitor` objects, not the trainer. If applying a function
            to :py:class:`Module` objects, e.g. via :py:meth:`~torch.nn.Module.to`, they
            will be excluded when performed on the trainer.

        Tip:
            If a function does need to be applied to :py:class:`Trainable` or
            :py:meth:`ManagedMonitor` objects where it is sensible to do at the
            trainer-level, the properties :py:attr:`monitors` and :py:attr:`trainables`
            can be iterated over.
        """
        # call superclass
        Module.__init__(self, **kwargs)

        # set name automatically if not provided
        if name is None:
            name = f"trainer_{uuid.uuid4().hex}"
        self.register_extra("_pool_name", name)

        # interal storage for trainables and monitors
        self.trainables_: dict[str, Trainable] = {}
        self.monitors_: dict[str, ManagedMonitor] = {}

    @property
    def name(self) -> str:
        r"""Name of the trainer.

        This name is used by :py:class:`Layer` to identify a specific pool of monitors.
        This is done to avoid duplicate monitors in the case of non-serial layers.
        Once set on a trainer, it cannot be changed, and will persist via the
        state dictionary.

        Returns:
            str: name of the trainer.
        """
        return self._pool_name

    @property
    def monitors(self) -> Iterator[ManagedMonitor]:
        r"""Added monitors.

        Because monitors are owned and managed by the :py:class:`Layer` for each
        trainable and duplicate monitors are not created where possible, the number
        of monitors here may be less than the number of added monitors.

        Yields:
            ManagedMonitor: added monitors.
        """
        return unique(chain.from_iterable(m.values() for m in self.monitors_.values()))

    def monitors_of(self, trainable: str) -> Iterator[ManagedMonitor]:
        r"""Monitors associated with a given trainable.

        Args:
            trainable (str): name of the trainable to get associted monitors of.

        Yields:
            ManagedMonitor: associated monitors.
        """
        return (m for m in self.monitors_.get(trainable, {}).values())

    def named_monitors_of(self, trainable: str) -> Iterator[tuple[str, ManagedMonitor]]:
        r"""Monitors associated with a given trainable and their names.

        Args:
            trainable (str): name of the trainable to get associted monitors of.

        Yields:
            tuple[str, ManagedMonitor]: tuple of an associated monitor and its name.
        """
        return (nm for nm in self.monitors_.get(trainable, {}).items())

    @property
    def trainables(self) -> Iterator[Trainable]:
        r"""Added trainables.

        Yields:
            Trainable: added trainables.
        """
        return (t for t in self.trainables_.values())

    @property
    def named_trainables(self) -> Iterator[tuple[str, Trainable]]:
        r"""Added trainables and their names.

        Yields:
            tuple[str, Trainable]: tuple of an added trainable and its name.
        """
        return (nt for nt in self.trainables_.items())

    @property
    def named_state_trainables(self) -> Iterator[tuple[str, Trainable, Module]]:
        r"""Added trainables, their states, and their names.

        Yields:
            tuple[str, Trainable, Module]: tuple of an added trainable,
                its state, and its name.
        """

        return fzip(
            self.trainables_,
            lambda n: self.get_trainable(n),
            lambda n: self.get_state(n),
        )

    def add_trainable(self, name: str, trainable: Trainable, **kwargs) -> Trainable:
        r"""Adds a trainable.

        Args:
            name (str): name of the trainable to add.
            trainable (Trainable): trainable to add.

        Raises:
            ValueError: a trainable with the specified name already exists.

        Returns:
            Trainable: added trainable.
        """
        # ensure a trainable with the given name does not exist
        if name in self.trainables_:
            raise ValueError(
                f"'name' ('{name}') is already the name of an added trainable"
            )

        # delete any existing monitors (should never occur)
        if name in self.monitors_:
            for monitor in self.monitors_[name]:
                self.del_monitor(name, monitor)
            del self.monitors_[name]

        # add the trainable
        self.trainables_[name] = trainable

        return self.trainables_[name]

    def del_trainable(self, name: str) -> None:
        r"""Deletes an added trainable.

        Args:
            name (str): name of the trainable to delete.

        Raises:
            AttributeError: specified trainable does not exist.

        Important:
            This does not strictly delete the trainable, it is still owned by its
            :py:class:`Layer` and can be added again. It is only removed from the
            trainer.
        """
        # check that the trainable exists
        if name not in self.trainables_:
            raise AttributeError(
                f"'name' ('{name}') is not the name of an added trainable"
            )

        # delete any existing monitors
        if name in self.monitors_:
            for monitor in self.monitors_[name]:
                self.del_monitor(name, monitor)
            del self.monitors_[name]

        # clear trainable state and delete the trainable
        self.trainables_[name].del_trainer_state(self.name)
        del self.trainables_[name]

    def get_trainable(self, name: str) -> Trainable | None:
        r"""Gets an added trainable.

        Args:
            name (str): name of the trainable to get.

        Returns:
            Trainable | None: specified trainable, if it exists.
        """
        return self.trainables_.get(name, None)

    def get_state(self, name: str) -> Module | None:
        r"""Gets the additional state for an added trainable.

        Args:
            name (str): name of the trainable to get.

        Returns:
            Trainable | None: specified trainable's state, if it exists.
        """
        if name in self.trainables_:
            return self.trainables_[name].trainer_state(self.name)

    def add_monitor(
        self,
        trainable: str,
        name: str,
        attr: str,
        monitor: MonitorConstructor,
        **kwargs,
    ) -> ManagedMonitor:
        r"""Adds a monitor to a trainable.

        Args:
            trainable (str): name of the trainable to which the monitor will be added.
            name (str): name of the monitor to add (unique to the trainable).
            attr (str): dot-seperated attribute to monitor, relative to the trainable.
            monitor (MonitorConstructor): partial constructor for the monitor.

        Raises:
            AttributeError: specified trainable does not exist.

        Returns:
            ManagedMonitor: added or retrieved monitor.

        Important:
            If the monitor exists, it will be retrieved even if the specifications
            (attribute and/or partial constructor) are different.
        """
        # test if the monitor already exists and return if it does
        maybemon = self.monitors_.get(trainable, {}).get(name, None)
        if maybemon:
            return maybemon

        # get underlying trainable from given key
        if trainable not in self.trainables_:
            raise AttributeError(
                f"'trainable' ('{trainable}') is not the name of an added trainable"
            )

        # create monitor via the trainable
        monitor = self.trainables_[trainable].add_monitor(
            self._pool_name, name, attr, monitor
        )

        # add monitor to the trainer
        if trainable not in self.monitors_:
            self.monitors_[trainable] = {}
        self.monitors_[trainable][name] = monitor

        # return the monitor
        return monitor

    def del_monitor(self, trainable: str, name: str) -> None:
        r"""Deletes an added monitor.

        Args:
            trainable (str): name of the trainable to which the monitor was added.
            name (str): name of the monitor.

        Raises:
            AttributeError: specified trainable does not exist, or does not have a
                monitor with the specified name added to it.
        """
        # check that the trainable has monitors
        if trainable not in self.monitors_ or trainable not in self.trainables_:
            raise AttributeError(
                f"'trainable' ('{trainable}') is either not the name of an added "
                "trainable or is a trainable with no added monitors"
            )

        # check that the monitor to delete exists
        if name not in self.monitors_[trainable]:
            raise AttributeError(
                f"'name' ('{name}') is not the name of a monitor added on trainable "
                f"with name '{trainable}'"
            )

        # delete the monitor
        del self.monitors_[trainable][name]
        self.trainables_[trainable].del_monitor(self._pool_name, name)

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
        _ = self.train()
        for monitor in self.monitors:
            if clear:
                monitor.clear(**kwargs)
            monitor.register()

    def detach(self, clear: bool = False, **kwargs) -> None:
        """Deregisters all of the monitors for the updater.

        This additionally sets the trainer into evaluation mode with ``self.eval()``.

        Args:
            clear (bool, optional): If the monitors should be cleared before
                deregistering with submodules. Defaults to False.

        Note:
            Keyword arguments are passed to :py:meth:`Monitor.clear` call.
        """
        _ = self.eval()
        for monitor in self.monitors:
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
        for monitor in self.monitors:
            monitor.clear(**kwargs)

    def update(self) -> None:
        r"""Applies all cumulative updates.

        This calls every updated which applies cumulative updates and any updater
        hooks are automatically called (e.g. parameter clamping). The updaters will
        each be called once, even if present in multiple trainables.
        """
        for updater in unique(map(lambda t: t.updater, self.trainables)):
            updater()

    def forward(self, *inputs, **kwargs):
        """Processes a training step for given layers based on stored data.

        Raises:
            NotImplementedError: ``forward`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__}(LayerwiseUpdater) must implement "
            "the method `forward`."
        )


class LayerwiseTrainer(Trainer):
    def __init__(self, name: str | None, **kwargs):
        r"""Trainer for update methods without inter-layer dependencies.

        Args:
            name (str | None): name of the trainer, for layer monitor pooling.

        Important:
            The :py:class:`Layer` object "owns" the :py:class:`Trainable` and
            :py:class:`ManagedMonitor` objects, not the trainer. If applying a function
            to :py:class:`Module` objects, e.g. via :py:meth:`~torch.nn.Module.to`, they
            will be excluded when performed on the trainer.

        Tip:
            If a function does need to be applied to :py:class:`Trainable` or
            :py:meth:`ManagedMonitor` objects where it is sensible to do at the
            trainer-level, the properties :py:attr:`monitors` and :py:attr:`trainables`
            can be iterated over.
        """
        # call superclass
        Trainer.__init__(self, name, **kwargs)

    def add_layer(
        self, layer: Layer, prefix: str | None = None, **kwargs
    ) -> dict[str, Trainable]:
        r"""Adds all trainables from a given layer.

        Args:
            layer (Layer): layer from which trainables should be added.
            prefix (str | None, optional): string to prepend to trainable name.
                Defaults to None.

        Returns:
            dict[str, Trainable]: names of added trainables and those trainables.

        Note:
            The names used are automatically generated based on the name of the input
            and name of the output in the layer. For example, with an input named
            ``"linear"`` and an output named ``"alif"``, the trainable name will be
            named ``"linear-alif"``. If a ``prefix`` is given, say ``"l0"``, then the
            name will be ``"l0-linear-alif"``.
        """
        prefix = f"{prefix}-" if prefix else ""
        return {
            n: self.add_trainable(n, t, **kwargs)
            for n, t in fzip(
                layer.named_trainables,
                lambda nt: f"{prefix}{nt[0][0]}-{nt[0][1]}",
                lambda nt: nt[1],
                identity=False,
            )
        }

    def add_trainable(self, name: str, trainable: Trainable, **kwargs) -> Trainable:
        r"""Adds a trainable.

        Args:
            name (str): name of the trainable to add.
            trainable (Trainable): trainable to add.

        Raises:
            ValueError: a trainable with the specified name already exists.

        Returns:
            Trainable: added trainable.

        Important:
            Keyword arguments are automatically added as non-persistant state to the
            trainable.
        """
        # call superclass method
        trainable = Trainer.add_trainable(self, name, trainable, **kwargs)

        # adds keyword arguments to state
        if kwargs:
            for k, v in kwargs.items():
                setattr(self.get_state(name), k, v)

        return trainable


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
