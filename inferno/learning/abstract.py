from abc import abstractmethod
from functools import partial

import torch.nn as nn

from inferno.neural import AbstractLayer
from inferno.monitoring import AbstractMonitor, AbstractPreMonitor


class AbstractLayerUpdater(nn.Module):
    """Abstract class for updaters which use a training method that is only applied to a single layer.
    """
    def __init__(
        self,
        trainable: AbstractLayer
    ):
        # call superclass constructor
        nn.Module.__init__(self)

        # register modules
        self.submodule = trainable

        # register monitor dictionary
        self.monitors = nn.ModuleDict()

    def connect(self, clear: bool = True) -> None:
        """Registers all of the monitors for the updater.

        Args:
            clear (bool, optional): If the monitors should be cleared before registering with submodules. Defaults to True.
        """
        def single_connect(monitor, module, clear):
            if isinstance(monitor, (AbstractMonitor, AbstractPreMonitor)):
                if clear:
                    monitor.clear()
                monitor.register(module)
        self.monitors.apply(partial(single_connect, module=self.submodule, clear=clear))

    def disconnect(self, clear: bool = False) -> None:
        """Deregisters all of the monitors for the updater.

        Args:
            clear (bool, optional): If the monitors should be cleared before deregistering with submodules. Defaults to False.
        """
        def single_disconnect(monitor, clear):
            if isinstance(monitor, (AbstractMonitor, AbstractPreMonitor)):
                if clear:
                    monitor.clear()
                monitor.deregister()
        self.monitors.apply(partial(single_disconnect, clear=clear))

    def clear(self):
        """Clears all of the monitors for the updater.
        """
        def single_clear(monitor):
            if isinstance(monitor, (AbstractMonitor, AbstractPreMonitor)):
                monitor.clear()
        self.monitors.apply(partial(single_clear))

    @abstractmethod
    def forward(self, *inputs, **kwargs):
        """Processes update for given layers based on current monitor stored data.

        Raises:
            NotImplementedError: py:meth:`forward` is abstract and must be implemented by the subclass.
        """
        raise NotImplementedError(f"'AbstractLayerUpdater.clear()' is abstract, {type(self).__name__} must implement the 'forward' method")


class AbstractLayerwiseUpdater(nn.Module):
    """Abstract class for updaters which use a training method that is applied to a sequence of layers.
    """
    def __init__(
        self,
        trainables: tuple[AbstractLayer, ...] | AbstractLayer,
    ):
        # call superclass constructor
        nn.Module.__init__(self)

        # register modules
        self.submodules = nn.ModuleList([trainables] if isinstance(trainables, AbstractLayer) else trainables)

        # register monitor dictionary
        self.monitors = nn.ModuleDict([(hex(id(sm)), nn.ModuleDict()) for sm in self.submodules])

    def connect(self, clear: bool = True) -> None:
        """Registers all of the monitors for the updater.

        Args:
            clear (bool, optional): If the monitors should be cleared before registering with submodules. Defaults to True.
        """
        def single_connect(monitor, module, clear):
            if isinstance(monitor, (AbstractMonitor, AbstractPreMonitor)):
                if clear:
                    monitor.clear()
                monitor.register(module)
        for sm in self.submodules:
            if self.monitors[hex(id(sm))]:
                self.monitors[hex(id(sm))].apply(partial(single_connect, module=sm, clear=clear))

    def disconnect(self, clear: bool = False) -> None:
        """Deregisters all of the monitors for the updater.

        Args:
            clear (bool, optional): If the monitors should be cleared before deregistering with submodules. Defaults to False.
        """
        def single_disconnect(monitor, clear):
            if isinstance(monitor, (AbstractMonitor, AbstractPreMonitor)):
                if clear:
                    monitor.clear()
                monitor.deregister()
        self.monitors.apply(partial(single_disconnect, clear=clear))

    def clear(self):
        """Clears all of the monitors for the updater.
        """
        def single_clear(monitor):
            if isinstance(monitor, (AbstractMonitor, AbstractPreMonitor)):
                monitor.clear()
        self.monitors.apply(partial(single_clear))

    @abstractmethod
    def forward(self, *inputs, **kwargs):
        """Processes update for given layers based on current monitor stored data.

        Raises:
            NotImplementedError: py:meth:`forward` is abstract and must be implemented by the subclass.
        """
        raise NotImplementedError(f"'AbstractLayerwiseUpdater.clear()' is abstract, {type(self).__name__} must implement the 'forward' method")
