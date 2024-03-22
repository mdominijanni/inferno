from functools import reduce
from itertools import chain
import pytest
import random
import torch
import torch.nn as nn

from inferno import Module
from inferno.observe import Monitor, Reducer


class MockReducer(Reducer):

    def __init__(self, step_time, duration):
        Reducer.__init__(self, step_time, duration)
        self.latest_inputs = None

    def clear(self, **kwargs) -> None:
        return kwargs

    def view(self, *args, **kwargs) -> torch.Tensor | None:
        return (args, kwargs)

    def dump(self, *args, **kwargs) -> torch.Tensor | None:
        return (args, kwargs)

    def peek(self, *args, **kwargs) -> torch.Tensor | None:
        return (args, kwargs)

    def push(self, inputs: torch.Tensor, **kwargs) -> None:
        return (inputs, kwargs)

    def forward(self, *inputs: torch.Tensor, **kwargs) -> None:
        self.latest_inputs = inputs


class TestMonitor:

    def test_register_on_init(self):
        identity = lambda *args: args  # noqa:E731;

        monitor = Monitor(MockReducer(1.0, 0.0), Module(), identity, identity)
        assert monitor.registered

        monitor = Monitor(MockReducer(1.0, 0.0), None, identity, identity)
        assert not monitor.registered

    def test_register_registered(self):
        identity = lambda *args: args  # noqa:E731;
        sentinel = Module()
        monitor = Monitor(MockReducer(1.0, 0.0), sentinel, identity, identity)

        with pytest.raises(RuntimeError) as excinfo:
            monitor.register(Module())

        assert (
            f"{type(monitor).__name__}(Monitor) is already registered to a module "
            "so register() was ignored" in str(excinfo.value)
        )
        assert id(sentinel) == id(monitor._observed())

        monitor.register()
        assert id(sentinel) == id(monitor._observed())

    def test_register_unregistered():
        identity = lambda *args: args  # noqa:E731;
        sentinel = Module()
        monitor = Monitor(MockReducer(1.0, 0.0), None, identity, identity)

        monitor.register(sentinel)
        assert monitor.registered
