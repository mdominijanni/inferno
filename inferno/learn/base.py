from abc import ABC, abstractmethod
from collections.abc import Generator
from .. import Module
from inferno.neural import Layer
from inferno.observe import Monitor
import itertools
import torch.nn as nn
from typing import Any


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
