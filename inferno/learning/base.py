from abc import ABC
from collections.abc import Generator
from inferno import Module
from inferno.neural import Layer
from inferno.observe import Monitor
import itertools
import torch.nn as nn
from typing import Any


class LayerwiseUpdater(Module, ABC):
    def __init__(
        self,
        *pairs: tuple[Layer, Layer],
        **kwargs,
    ):
        # call superclass constructor
        Module.__init__(self)

        # register modules
        self._trainables = nn.ModuleList()
        self._monitors = nn.ModuleDict()
        self._pairs = []

        # go through pairs
        for source, target in pairs:
            # add to trainables list for accessing
            if source not in self._trainables:
                self._trainables.append(source)
            if target not in self._trainables:
                self._trainables.append(target)

            # add to mapping from layers to monitors
            if source not in self._monitors:
                self._monitors[source] = nn.ModuleDict()
            if target not in self._monitors:
                self._monitors[target] = nn.ModuleDict()

            # add pairs to list
            if (source, target) not in self.pairs:
                self.pairs.append((source, target))

        # add hook for removal of trainables from state_dict export
        def state_dict_submodule_removal(
            obj, state_dict, prefix, local_metadata
        ) -> dict[str, Any]:
            for key in list(state_dict.keys()):
                match key.partition(".")[0]:
                    case "trainables":
                        del state_dict[key]
                    case "monitors":
                        del state_dict[key]

            return state_dict

        self._sd_removal_handle = self._register_state_dict_hook(
            state_dict_submodule_removal
        )

    @property
    def trainables(self) -> Generator[Layer]:
        return (layer for layer in self._trainables)

    @property
    def monitors(self) -> Generator[Monitor]:
        return itertools.chain.from_iterable(
            (monitor for monitor in module_monitor.values())
            for module_monitor in self._monitors.values()
        )

    @property
    def pairs(self) -> Generator[tuple[Layer, Layer]]:
        return ((source, target) for source, target in self._pairs)

    def add_monitor(self, trainable: Layer, name: str, monitor: Monitor) -> None:
        if trainable not in self._trainables:
            raise RuntimeError(
                f"specified {type(trainable).__name__} instance not in updater."
            )
        if name in self._monitors[trainable]:
            raise KeyError(
                f"monitor with name {name} already registered to specified trainable."
            )
        if monitor in self._monitors[trainable].values():
            raise RuntimeError(
                f"specified {type(monitor).__name__} already registered to specified trainable."
            )

        self._monitors[trainable][name] = monitor

    def del_monitor(self, trainable: Layer, name: str) -> None:
        if trainable not in self._trainables:
            raise RuntimeError(
                f"specified {type(trainable).__name__} instance not in updater."
            )
        if name not in self._monitors[trainable]:
            raise KeyError(
                f"monitor with name {name} not registered to specified trainable."
            )

        del self._monitors[trainable][name]

    def get_monitor(self, trainable: Layer, name: str) -> Monitor:
        if trainable not in self._trainables:
            raise RuntimeError(
                f"specified {type(trainable).__name__} instance not in updater."
            )
        if name not in self._monitors[trainable]:
            raise KeyError(
                f"monitor with name {name} not registered to specified trainable."
            )

        return self._monitors[trainable][name]
