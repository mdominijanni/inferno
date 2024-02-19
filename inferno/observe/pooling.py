from .monitors import ManagedMonitor, MonitorConstructor
from .. import Module
from inferno._internal import argtest
import torch.nn as nn
import uuid


class PoolManager(Module):

    def __init__(self, reference: Module | None = None, requiresids: bool = True):
        # call superclass constructor
        Module.__init__(self)

        # sets attributes
        self.monitors_ = nn.ModuleDict()
        self._ref = reference
        self._require_ids = requiresids


    @staticmethod
    def uniquepool() -> str:
        r"""Generates a unique string name for a monitor pool.

        This uses :py:mod:`uuid` to generate a UUID4 hex value, appended to "unique_".

        Returns:
            str: unique string identifier.
        """
        return "unique_" + uuid.uuid4().hex

    def add_monitor(
        self, pool: str, name: str, attr: str, monitor: MonitorConstructor
    ) -> ManagedMonitor:
        # enforce valid identifiers
        if self._require_ids:
            _ = argtest.identifier('pool', pool)
            _ = argtest.identifier('name', name)

        # add pool if it doesn't exist
        if pool not in self.monitors_:
            self.monitors_[pool] = nn.ModuleDict()



    def tag_monitor(
        self, monitor: ManagedMonitor, pool: str, name: str, attr: str
    ) -> ManagedMonitor:
        r"""Uses reflections to add pooling data to a managed monitor.

        Args:
            monitor (ManagedMonitor): monitor to add pooling metadata to.
            pool (str): _description_
            name (str): _description_
            attr (str): _description_

        Returns:
            ManagedMonitor: _description_
        """
        pass
