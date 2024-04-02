from functools import reduce
import pytest
import random
import torch
import weakref

from inferno import Module
from inferno.observe import (
    Monitor,
    RecordReducer,
    InputMonitor,
    OutputMonitor,
    StateMonitor,
    MultiStateMonitor,
)


class MockReducer(RecordReducer):

    def __init__(self):
        RecordReducer.__init__(self, 1.0, 0.0)
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
            reducer: RecordReducer,
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

    def test_custom_map(self):
        shape = self.random_shape()
        reducer = MockReducer()
        module = MockModule()

        map_ = lambda inputs: tuple(x * 2 for x in inputs)  # noqa:E731;
        monitor = InputMonitor(reducer, None, map_=map_)
        monitor.register(module)

        data = (torch.rand(shape) for _ in range(random.randint(1, 9)))

        module(*data)
        assert all(
            torch.all(ds == di) for ds, di in zip(reducer.latest_inputs, map_(data))
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

    def test_filter(self):
        shape = self.random_shape()
        reducer = MockReducer()
        module = MockModule()

        monitor = StateMonitor(reducer, "data", None)
        monitor.register(module)

        data = torch.rand(shape)

        module(data)
        assert id(reducer.latest_inputs) == id(module.data)

        module()
        assert id(reducer.latest_inputs) == id(module.data)

        data = torch.rand(shape)
        module(data)
        assert id(reducer.latest_inputs) == id(module.data)


class TestOutputMonitor:

    @staticmethod
    def random_shape(mindims=1, maxdims=9, minsize=1, maxsize=9):
        return tuple(
            random.randint(mindims, maxdims)
            for _ in range(random.randint(minsize, maxsize))
        )

    def test_finalizer_unregistered(self):
        reducer = MockReducer()
        module = MockModule()
        monitor = OutputMonitor(reducer, module)

        monitor.deregister()

        monitorref = weakref.ref(monitor)
        del monitor

        assert monitorref() is None
        assert len(module._forward_hooks) == 0

    def test_finalizer_registered(self):
        reducer = MockReducer()
        module = MockModule()
        monitor = OutputMonitor(reducer, module)

        assert len(module._forward_hooks) == 1

        monitorref = weakref.ref(monitor)
        del monitor

        assert monitorref() is None
        assert len(module._forward_hooks) == 0

    def test_filter(self):
        shape = self.random_shape()
        reducer = MockReducer()
        module = MockModule()

        monitor = OutputMonitor(reducer, None)
        monitor.register(module)

        sentinel = torch.rand(shape)

        res = module(sentinel)
        assert id(reducer.latest_inputs) == id(res)
        assert torch.all(reducer.latest_inputs == res)

        module()
        assert id(reducer.latest_inputs) == id(res)

        sentinel = torch.rand(shape)
        res = module(sentinel)
        assert id(reducer.latest_inputs) == id(res)
        assert torch.all(reducer.latest_inputs == res)


class TestMultiStateMonitor:

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
        module.nested = MockModule()
        module.nested.nested = MockModule()

        monitor = MultiStateMonitor(
            reducer, "nested", ("data", "nested.data"), module, as_prehook=prehook
        )

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
        module.nested = MockModule()
        module.nested.nested = MockModule()

        monitor = MultiStateMonitor(
            reducer, "nested", ("data", "nested.data"), module, as_prehook=prehook
        )

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

    def test_monitored_forward(self):
        shape = self.random_shape()
        reducer = MockReducer()
        module = MockModule()
        module.nested = MockModule()
        module.nested.nested = MockModule()

        module.nested.data = torch.rand(3, 3)
        module.nested.nested.data = torch.rand(3, 3)

        _ = MultiStateMonitor(
            reducer, "nested", ("data", "nested.data"), module
        )

        data = torch.rand(shape)

        module(data)
        assert id(reducer.latest_inputs[0]) == id(module.nested.data)
        assert id(reducer.latest_inputs[1]) == id(module.nested.nested.data)
