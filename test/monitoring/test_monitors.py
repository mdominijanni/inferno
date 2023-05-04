import random

import pytest
import torch
import torch.nn as nn

import sys
sys.path.insert(0, '../..')

from inferno.monitoring import AbstractReducer, InputMonitor, OutputMonitor, StateMonitor, StatePreMonitor


class SimpleReducer(AbstractReducer):

    def __init__(self):
        AbstractReducer.__init__(self)
        self.data = None

    def clear(self) -> None:
        del self.data
        self.data = None

    def peak(self):
        return self.data

    def pop(self):
        res = self.peak()
        self.clear()
        return res

    def forward(self, inputs, **kwargs):
        self.clear()
        self.data = inputs


class SingleIOModule(nn.Module):

    def __init__(self, forward_func=(lambda x: x)):
        super().__init__()
        self.forward_func = forward_func
        self.historic = None

    def forward(self, inputs):
        self.historic = self.forward_func(inputs)
        return self.historic


class MultiIOModule(nn.Module):

    def __init__(self, forward_func=(lambda x: x)):
        super().__init__()
        self.forward_func = forward_func
        self.historic = None

    def forward(self, *inputs):
        self.historic = tuple(self.forward_func(i) for i in inputs)
        return self.historic


@pytest.fixture(scope='function')
def inputs():
    return [torch.rand((3, 3), dtype=torch.float32, requires_grad=False) for _ in range(20)]


@pytest.fixture(scope='function')
def reducer():
    return SimpleReducer()


@pytest.fixture(scope='function')
def single_module():
    return SingleIOModule(forward_func=(lambda x: x * 2))


@pytest.fixture(scope='function')
def multi_module():
    return MultiIOModule()


class TestInputMonitor:
    @pytest.fixture(scope='function')
    def make_monitor(self, reducer):
        return lambda index=0, train_update=True, eval_update=True, module=None: InputMonitor(reducer, index, train_update, eval_update, module)

    def test_index_selection(self, make_monitor, multi_module, inputs):
        for t in inputs:
            power = random.randint(1, 4)
            monitor = make_monitor(index=(power - 1))
            monitor.register(multi_module)
            multi_module(t, t**2, t**3, t**4)
            assert torch.all(monitor.reducer.peak() == t**power)
            monitor.deregister()

    def test_update_train_true(self, make_monitor, single_module, inputs):
        monitor = make_monitor(module=single_module, train_update=True)
        single_module.train()
        for t in inputs:
            single_module(t)
            assert torch.all(monitor.reducer.peak() == t)

    def test_update_train_false(self, make_monitor, single_module, inputs):
        monitor = make_monitor(module=single_module, train_update=False)
        single_module.train()
        for t in inputs:
            single_module(t)
            assert monitor.reducer.peak() is None

    def test_update_eval_true(self, make_monitor, single_module, inputs):
        monitor = make_monitor(module=single_module, eval_update=True)
        single_module.eval()
        for t in inputs:
            single_module(t)
            assert torch.all(monitor.reducer.peak() == t)

    def test_update_eval_false(self, make_monitor, single_module, inputs):
        monitor = make_monitor(module=single_module, eval_update=False)
        single_module.eval()
        for t in inputs:
            single_module(t)
            assert monitor.reducer.peak() is None


class TestOutputMonitor:
    @pytest.fixture(scope='function')
    def make_monitor(self, reducer):
        return lambda index=None, train_update=True, eval_update=True, module=None: OutputMonitor(reducer, index, train_update, eval_update, module)

    def test_index_selection(self, make_monitor, multi_module, inputs):
        for t in inputs:
            power = random.randint(1, 4)
            monitor = make_monitor(index=(power - 1))
            monitor.register(multi_module)
            res = multi_module(t, t**2, t**3, t**4)
            assert torch.all(monitor.reducer.peak() == res[power - 1])
            monitor.deregister()

    def test_update_train_true(self, make_monitor, single_module, inputs):
        monitor = make_monitor(module=single_module, train_update=True)
        single_module.train()
        for t in inputs:
            res = single_module(t)
            assert torch.all(monitor.reducer.peak() == res)

    def test_update_train_false(self, make_monitor, single_module, inputs):
        monitor = make_monitor(module=single_module, train_update=False)
        single_module.train()
        for t in inputs:
            single_module(t)
            assert monitor.reducer.peak() is None

    def test_update_eval_true(self, make_monitor, single_module, inputs):
        monitor = make_monitor(module=single_module, eval_update=True)
        single_module.eval()
        for t in inputs:
            res = single_module(t)
            assert torch.all(monitor.reducer.peak() == res)

    def test_update_eval_false(self, make_monitor, single_module, inputs):
        monitor = make_monitor(module=single_module, eval_update=False)
        single_module.eval()
        for t in inputs:
            single_module(t)
            assert monitor.reducer.peak() is None


class TestStateMonitor:
    @pytest.fixture(scope='function')
    def make_monitor(self, reducer):
        return lambda index=None, train_update=True, eval_update=True, module=None: StateMonitor('historic', reducer, train_update, eval_update, module)

    def test_update_train_true(self, make_monitor, single_module, inputs):
        monitor = make_monitor(module=single_module, train_update=True)
        single_module.train()
        for t in inputs:
            single_module(t)
            assert torch.all(monitor.reducer.peak() == single_module.historic)

    def test_update_train_false(self, make_monitor, single_module, inputs):
        monitor = make_monitor(module=single_module, train_update=False)
        single_module.train()
        for t in inputs:
            single_module(t)
            assert monitor.reducer.peak() is None

    def test_update_eval_true(self, make_monitor, single_module, inputs):
        monitor = make_monitor(module=single_module, eval_update=True)
        single_module.eval()
        for t in inputs:
            single_module(t)
            assert torch.all(monitor.reducer.peak() == single_module.historic)

    def test_update_eval_false(self, make_monitor, single_module, inputs):
        monitor = make_monitor(module=single_module, eval_update=False)
        single_module.eval()
        for t in inputs:
            single_module(t)
            assert monitor.reducer.peak() is None


class TestStatePreMonitor:
    @pytest.fixture(scope='function')
    def make_monitor(self, reducer):
        return lambda index=None, train_update=True, eval_update=True, module=None: StatePreMonitor('historic', reducer, train_update, eval_update, module)

    def test_update_train_true(self, make_monitor, single_module, inputs):
        monitor = make_monitor(module=single_module, train_update=True)
        single_module.train()
        for t in inputs:
            res = single_module.historic
            if isinstance(res, torch.Tensor):
                single_module(t)
                assert torch.all(monitor.reducer.peak() == res)
            else:
                single_module(t)
                assert monitor.reducer.peak() == res

    def test_update_train_false(self, make_monitor, single_module, inputs):
        monitor = make_monitor(module=single_module, train_update=False)
        single_module.train()
        for t in inputs:
            single_module(t)
            assert monitor.reducer.peak() is None

    def test_update_eval_true(self, make_monitor, single_module, inputs):
        monitor = make_monitor(module=single_module, eval_update=True)
        single_module.eval()
        for t in inputs:
            res = single_module.historic
            if isinstance(res, torch.Tensor):
                single_module(t)
                assert torch.all(monitor.reducer.peak() == res)
            else:
                single_module(t)
                assert monitor.reducer.peak() == res

    def test_update_eval_false(self, make_monitor, single_module, inputs):
        monitor = make_monitor(module=single_module, eval_update=False)
        single_module.eval()
        for t in inputs:
            single_module(t)
            assert monitor.reducer.peak() is None
