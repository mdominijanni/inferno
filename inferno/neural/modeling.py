from inferno import Module
import warnings
from . import Connection, Neuron


class Layer(Module):

    def __init__(self, connection: Connection, neuron: Neuron):
        Module.__init__(self)

        if connection.dt != neuron.dt:
            warnings.warn(f"inconsistent step times, {connection.dt} ms "
                          f"for connection and {neuron.dt} ms for neuron.")

        self.register_module('connection', connection)
        self.register_module('neuron', neuron)

    def clear(self, **kwargs):
        self.connection.clear(**kwargs)
        self.neuron.clear(**kwargs)

    def forward(self, inputs, **kwargs):
        return self.neuron(self.connection(inputs, **kwargs), **kwargs)

    @property
    def dt(self) -> float:
        return self.neuron.dt

    @dt.setter
    def dt(self, value: float):
        self.connection.dt = value
        self.neuron.dt = value
