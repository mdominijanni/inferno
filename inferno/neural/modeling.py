from abc import ABC, abstractproperty
import functools
import operator
from typing import Callable

import torch
import torch.nn as nn

from inferno.neural import AbstractConnection, AbstractDynamics


class AbstractLayer(nn.Module, ABC):

    def __init__(self):
        nn.Module.__init__(self)

    @abstractproperty
    def connections(self) -> tuple[AbstractConnection, ...]:
        raise NotImplementedError(f"'AbstractLayer.connections' is abstract, {type(self).__name__} must implement the 'connections' property")

    @abstractproperty
    def dynamics(self) -> tuple[AbstractDynamics, ...]:
        raise NotImplementedError(f"'AbstractLayer.dynamics' is abstract, {type(self).__name__} must implement the 'dynamics' property")


class Layer(AbstractLayer):

    def __init__(
        self,
        connection: AbstractConnection,
        dynamics: AbstractDynamics
    ):
        nn.Module.__init__(self)
        if not isinstance(connection, (type(None), AbstractConnection)):
            raise TypeError(f'connection must be an instance of {AbstractConnection.__name__}, received {type(connection).__name__}')
        if not isinstance(dynamics, (type(None), AbstractDynamics)):
            raise TypeError(f'dynamics must be an instance of {AbstractDynamics.__name__}, received {type(dynamics).__name__}')
        self._connections = nn.ModuleList([connection])
        self._dynamics = dynamics

    @property
    def connections(self) -> tuple[AbstractConnection, ...]:
        return self._connections

    @property
    def connection(self) -> AbstractConnection:
        return self._connections[0]

    @property
    def dynamics(self) -> AbstractDynamics:
        return self._dynamics

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.dynamics(self.connections[0](inputs))


class MultiLayer(AbstractLayer):

    def __init__(
        self,
        connections: tuple[AbstractConnection, ...],
        dynamics: AbstractDynamics,
        fold_func: Callable[[list[torch.Tensor]], torch.Tensor] = lambda t: functools.reduce(operator.add, t)
    ):
        nn.Module.__init__(self)
        for conn in connections:
            if not isinstance(conn, (type(None), AbstractConnection)):
                raise TypeError(f'each element of connections must be an instance of {AbstractConnection.__name__}, received {type(conn).__name__}')
        if not isinstance(dynamics, (type(None), AbstractDynamics)):
            raise TypeError(f'dynamics must be an instance of {AbstractDynamics.__name__}, received {type(dynamics).__name__}')
        self._connections = nn.ModuleList(connections)
        self._dynamics = dynamics
        self.fold_func = fold_func

    def forward(self, inputs: torch.Tensor):
        return self.dynamics(
            functools.reduce(
                self.fold_func,
                (conn(inputs) for conn in self.connections)
            )
        )

    @property
    def connections(self) -> tuple[AbstractConnection, ...]:
        return self._connections

    @property
    def dynamics(self) -> AbstractDynamics:
        return self._dynamics
