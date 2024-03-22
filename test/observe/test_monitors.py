from functools import reduce
import pytest
import random
import torch
import weakref

from inferno import Module
from inferno.observe import Monitor, Reducer, InputMonitor, StateMonitor


class MockReducer(Reducer):

    def __init__(self):
        Reducer.__init__(self, 1.0, 0.0)
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
        if len(inputs) == 1:
            self.latest_inputs = inputs[0]
        else:
            self.latest_inputs = inputs


class MockModule(Module):

    def __init__(self, op=None):
        Module.__init__(self)
        if op is None:
            self.op = lambda a, b: a + b  # noqa:E731;
        else:
            self.op = op
        self.data = None

    def forward(self, *inputs):
        if inputs:
            if self.data is not None:
                self.data = reduce(self.op, (self.data, *inputs))
            elif len(inputs) > 1:
                self.data = reduce(self.op, inputs)
            else:
                self.data = inputs[0]

        return self.data


class TestMonitor:

    class MockMonitor(Monitor):

        def __init__(
            self,
            reducer: Reducer,
            module: Module | None = None,
        ):
            Monitor.__init__(
                self,
                reducer=reducer,
                module=module,
                prehook="noop",
                posthook="noop",
            )

        def noop(self, *args):
            pass

    def test_register_on_init(self):
        monitor = self.MockMonitor(MockReducer(), Module())
        assert monitor.registered

        monitor = self.MockMonitor(MockReducer(), None)
        assert not monitor.registered

    def test_register_registered(self):
        sentinel = Module()
        monitor = self.MockMonitor(MockReducer(), sentinel)

        with pytest.raises(RuntimeError) as excinfo:
            monitor.register(Module())

        assert (
            f"{type(monitor).__name__}(Monitor) is already registered to a module "
            "so register() was ignored" in str(excinfo.value)
        )
        assert id(sentinel) == id(monitor._observed())

        monitor.register()
        assert id(sentinel) == id(monitor._observed())

    def test_register_unregistered(self):
        sentinel = Module()
        monitor = self.MockMonitor(MockReducer(), None)

        monitor.register(sentinel)
        assert monitor.registered


class TestInputMonitor:

    @staticmethod
    def random_shape(mindims=1, maxdims=9, minsize=1, maxsize=9):
        return tuple(
            random.randint(mindims, maxdims)
            for _ in range(random.randint(minsize, maxsize))
        )

    def test_finalizer_unregistered(self):
        reducer = MockReducer()
        module = MockModule()
        monitor = InputMonitor(reducer, module)

        monitor.deregister()

        monitorref = weakref.ref(monitor)
        del monitor

        assert monitorref() is None
        assert len(module._forward_pre_hooks) == 0

    def test_finalizer_registered(self):
        reducer = MockReducer()
        module = MockModule()
        monitor = InputMonitor(reducer, module)

        assert len(module._forward_pre_hooks) == 1

        monitorref = weakref.ref(monitor)
        del monitor

        assert monitorref() is None
        assert len(module._forward_pre_hooks) == 0

    def test_filter(self):
        shape = self.random_shape()
        reducer = MockReducer()
        module = MockModule()

        monitor = InputMonitor(reducer, None)
        monitor.register(module)

        sentinel = torch.rand(shape)

        module(sentinel)
        assert id(reducer.latest_inputs) == id(sentinel)

        module()
        assert id(reducer.latest_inputs) == id(sentinel)

        sentinel = torch.rand(shape)
        module(sentinel)
        assert id(reducer.latest_inputs) == id(sentinel)

    def test_custom_filter(self):
        shape = self.random_shape()
        reducer = MockReducer()
        module = MockModule()

        monitor = InputMonitor(reducer, None, filter_=lambda x: bool(len(x) % 2))
        monitor.register(module)

        pass_data = (torch.rand(shape), torch.rand(shape), torch.rand(shape))
        fail_data = (torch.rand(shape), torch.rand(shape))

        module(*pass_data)
        assert all(
            (id(ds) == id(di) for ds, di in zip(reducer.latest_inputs, pass_data))
        )

        module(*fail_data)
        assert all(
            (id(ds) == id(di) for ds, di in zip(reducer.latest_inputs, pass_data))
        )


class TestStateMonitor:

    @staticmethod
    def random_shape(mindims=1, maxdims=9, minsize=1, maxsize=9):
        return tuple(
            random.randint(mindims, maxdims)
            for _ in range(random.randint(minsize, maxsize))
        )

    @pytest.mark.parametrize(
        "prehook",
        (True, False),
        ids=("prehook", "posthook"),
    )
    def test_finalizer_unregistered(self, prehook):
        reducer = MockReducer()
        module = MockModule()
        monitor = StateMonitor(reducer, "data", module, as_prehook=prehook)

        monitor.deregister()

        monitorref = weakref.ref(monitor)
        del monitor

        assert monitorref() is None
        if prehook:
            assert len(module._forward_pre_hooks) == 0
        else:
            assert len(module._forward_hooks) == 0

    @pytest.mark.parametrize(
        "prehook",
        (True, False),
        ids=("prehook", "posthook"),
    )
    def test_finalizer_registered(self, prehook):
        reducer = MockReducer()
        module = MockModule()
        monitor = StateMonitor(reducer, "data", module, as_prehook=prehook)

        if prehook:
            assert len(module._forward_pre_hooks) == 1
        else:
            assert len(module._forward_hooks) == 1

        monitorref = weakref.ref(monitor)
        del monitor

        assert monitorref() is None
        if prehook:
            assert len(module._forward_pre_hooks) == 0
        else:
            assert len(module._forward_hooks) == 0
