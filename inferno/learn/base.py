from abc import ABC, abstractmethod
from collections.abc import Generator
from inferno import Module
from inferno.neural import Layer
from inferno.observe import Monitor
import itertools
import torch.nn as nn
from typing import Any


class LayerwiseUpdater(Module, ABC):
    r"""Updater for layers without interdependencies.

    Args:
        *layers (Layer): layers to add to the updater on initialization.

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
        self._trainables = nn.ModuleDict()
        self._monitors = nn.ModuleDict()

        # go through pairs
        for layer in layers:
            self.add_trainable(layer)

        # add hook for removal of trainables from state_dict export
        def state_dict_submodule_removal(
            obj, state_dict, prefix, local_metadata
        ) -> dict[str, Any]:
            for key in list(state_dict.keys()):
                match key.partition(".")[0]:
                    case "_trainables":
                        del state_dict[key]
                    case "_monitors":
                        del state_dict[key]

            return state_dict

        self._sd_removal_handle = self._register_state_dict_hook(
            state_dict_submodule_removal
        )

    @property
    def trainables(self) -> Generator[Layer]:
        r"""Registered layers to be trained.

        Yields:
            Generator[Layer]: layers registered as trainable for updating.
        """
        return (layer for layer in self._trainables.values())

    @property
    def monitors(self) -> Generator[tuple[Monitor, Layer]]:
        r"""Registered monitors for capturing layer state.

        Yields:
            Generator[tuple[Monitor, Layer]]: monitor and layer to which it is
                registered.
        """
        return itertools.chain.from_iterable(
            ((monitor, self._trainables[key]) for monitor in monitors.values())
            for key, monitors in self._monitors.items()
        )

    def add_trainable(self, trainable: Layer, **kwarge):
        r"""Registers a layer as a trainable.

        Args:
            trainable (Layer): layer to register as trainable.

        Note:
            This automatically calls :py:meth:`add_monitors` with the given
            layer if not previously registered.
        """
        # use hexadecimal of object id as the key
        key = hex(id(trainable))

        # add to trainables list for accessing
        if key not in self._trainables:
            self._trainables[key] = trainable

        # add to mapping from layers to monitors
        if key not in self._monitors:
            self._monitors[key] = nn.ModuleDict()

        # add default monitor layout to trainable
        if not len(self._monitors[key]):
            self.add_monitors(trainable)

    def del_trainable(self, trainable: Layer, **kwarge):
        r"""Deletes a registered layer and the associated monitors.

        Args:
            trainable (Layer): layer to delete.
        """
        # use hexadecimal of object id as the key
        key = hex(id(trainable))

        # add to trainables list for accessing
        if key in self._trainables:
            del self._trainables[key]

        # add to mapping from layers to monitors
        if key in self._monitors:
            del self._monitors[key]

    def add_monitor(
        self, trainable: Layer, name: str, monitor: Monitor, **kwargs
    ) -> None:

        # check if trainable is valid
        if trainable not in self._trainables.values():
            raise RuntimeError(
                f"{type(trainable).__name__} `trainable` is not a registered trainable."
            )

        key = hex(id(trainable))

        # check if name is already associated with trainable
        if name in self._monitors[key]:
            raise KeyError(
                f"Monitor with name '{name}' already associated "
                "with specified trainable."
            )

        # check if this monitor is already registered
        for layer in self._trainables:
            if monitor in self._monitors[hex(id(layer))].values():
                raise RuntimeError(
                    f"specified {type(monitor).__name__} already associated with "
                    f"a trainable {layer}."
                )

        monitor.deregister()
        self._monitors[key][name] = monitor
        if self.training:
            monitor.register(trainable)

    def del_monitor(self, trainable: Layer, name: str) -> None:
        # check if trainable is valid
        if trainable not in self._trainables.values():
            raise RuntimeError(
                f"specified {type(trainable).__name__} instance not in updater."
            )

        key = hex(id(trainable))

        # check if name is not associated with trainable
        if name not in self._monitors[key]:
            raise KeyError(
                f"monitor with name {name} not registered to specified trainable."
            )

        del self._monitors[key][name]

    def get_monitor(self, trainable: Layer, name: str) -> Monitor:
        # check if trainable is valid
        if trainable not in self._trainables.values():
            raise RuntimeError(
                f"specified {type(trainable).__name__} instance not in updater."
            )

        key = hex(id(trainable))

        # check if name is not associated with trainable
        if name not in self._monitors[key]:
            raise KeyError(
                f"monitor with name {name} not registered to specified trainable."
            )

        return self._monitors[key][name]

    @abstractmethod
    def add_monitors(self, trainable: Layer) -> None:
        raise NotImplementedError(
            f"LayerwiseUpdater `{type(self).__name__}` must implement "
            "the method `add_monitors`."
        )

    def get_monitors(self, trainable: Layer) -> Generator[tuple[str, Monitor]]:
        # check if trainable is valid
        if trainable not in self._trainables.values():
            raise RuntimeError(
                f"specified {type(trainable).__name__} instance not in updater."
            )

        return (
            (name, monitor)
            for name, monitor in self._monitors[hex(id(trainable))].items()
        )

    def del_monitors(self, trainable: Layer):
        # check if trainable is valid
        if trainable not in self._trainables.values():
            raise RuntimeError(
                f"specified {type(trainable).__name__} instance not in updater."
            )

        for name in self._monitors[trainable]:
            del self._monitors[trainable][name]

    def connect(self, clear: bool = True, **kwargs) -> None:
        """Registers all of the monitors for the updater.

        Args:
            clear (bool, optional): If the monitors should be cleared before registering
                with submodules. Defaults to True.

        Note:
            Keyword arguments are passed to :py:meth:`Monitor.clear` call.
        """
        self.train()
        for monitor, layer in self.monitors:
            if clear:
                monitor.clear(**kwargs)
            monitor.register(layer)

    def disconnect(self, clear: bool = False, **kwargs) -> None:
        """Deregisters all of the monitors for the updater.

        Args:
            clear (bool, optional): If the monitors should be cleared before deregistering
                with submodules. Defaults to False.

        Note:
            Keyword arguments are passed to :py:meth:`Monitor.clear` call.
        """
        self.eval()
        for monitor, _ in self.monitors:
            if clear:
                monitor.clear(**kwargs)
            monitor.deregister()

    def clear(self, **kwargs):
        """Clears all of the monitors for the updater.

        Note:
            Keyword arguments are passed to :py:meth:`Monitor.clear` call.
        """

        for monitor, _ in self.monitors:
            monitor.clear(**kwargs)

    @abstractmethod
    def forward(self, *inputs, **kwargs):
        """Processes update for given layers based on current monitor stored data.

        Raises:
            NotImplementedError: py:meth:`forward` is abstract and must be implemented by the subclass.
        """
        raise NotImplementedError(
            "'LayerwiseUpdater.forward()' is abstract, "
            f"{type(self).__name__} must implement forward()."
        )
