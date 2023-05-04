import pytest
import torch
# import torch.nn as nn

import sys
sys.path.insert(0, '../..')

from inferno.monitoring import PassthroughReducer, SinglePassthroughReducer, SMAReducer, CMAReducer, EMAReducer


@pytest.fixture(scope='function')
def timesteps():
    return 20


@pytest.fixture(scope='function')
def inputs(timesteps):
    return [torch.rand((3, 3), dtype=torch.float32, requires_grad=False) for _ in range(timesteps)]


@pytest.fixture(scope='module')
def epsilon():
    return 1e-6


def generic_test_pop(p_reducer, p_inputs):
    for t in p_inputs:
        p_reducer(t)
        peak_res = p_reducer.peak().clone()
        pop_res = p_reducer.pop()
        assert torch.all(peak_res == pop_res)
        assert p_reducer.peak() is None
        assert p_reducer.pop() is None


def generic_test_clear(p_reducer, p_inputs):
    for t in p_inputs:
        p_reducer(t)
        assert p_reducer.peak() is not None
        p_reducer.clear()
        assert p_reducer.peak() is None


class TestPassthroughReducer:

    @pytest.fixture(scope='function')
    def window(self):
        return 10

    @pytest.fixture(scope='function')
    def reducer(self, window):
        return PassthroughReducer(window)

    def test_forward_peak(self, reducer, window, inputs):
        assert reducer.peak() is None
        for idx, t in enumerate(inputs):
            reducer(t)
            res_test = reducer.peak()
            res_true = torch.stack([inputs[tidx] for tidx in range(max(0, idx - window + 1), idx + 1)], -1)
            assert torch.all(res_test == res_true)

    def test_pop(self, reducer, inputs):
        generic_test_pop(reducer, inputs)

    def test_clear(self, reducer, inputs):
        generic_test_clear(reducer, inputs)


class TestSinglePassthroughReducer:

    @pytest.fixture(scope='function')
    def reducer(self):
        return SinglePassthroughReducer()

    def test_forward_peak(self, reducer, inputs):
        assert reducer.peak() is None
        for t in inputs:
            reducer(t)
            res_test = reducer.peak()
            res_true = t
            assert torch.all(res_test == res_true)

    def test_pop(self, reducer, inputs):
        generic_test_pop(reducer, inputs)

    def test_clear(self, reducer, inputs):
        generic_test_clear(reducer, inputs)


class TestSMAReducer:

    @pytest.fixture(scope='function')
    def window(self):
        return 10

    @pytest.fixture(scope='function')
    def reducer(self, window):
        return SMAReducer(window)

    def test_forward_peak(self, reducer, window, inputs, epsilon):
        assert reducer.peak() is None
        for idx, t in enumerate(inputs):
            reducer(t)
            res_test = reducer.peak()
            res_true = torch.mean(torch.stack([inputs[tidx] for tidx in range(max(0, idx - window + 1), idx + 1)], -1), -1)
            assert torch.all(torch.abs(res_test - res_true) <= epsilon)

    def test_pop(self, reducer, inputs):
        generic_test_pop(reducer, inputs)

    def test_clear(self, reducer, inputs):
        generic_test_clear(reducer, inputs)


class TestCMAReducer:

    @pytest.fixture(scope='function')
    def reducer(self):
        return CMAReducer()

    def test_forward_peak(self, reducer, inputs, epsilon):
        assert reducer.peak() is None
        for idx, t in enumerate(inputs):
            reducer(t)
            res_test = reducer.peak()
            res_true = torch.mean(torch.stack([inputs[tidx] for tidx in range(0, idx + 1)], -1), -1)
            assert torch.all(torch.abs(res_test - res_true) <= epsilon)

    def test_pop(self, reducer, inputs):
        generic_test_pop(reducer, inputs)

    def test_clear(self, reducer, inputs):
        generic_test_clear(reducer, inputs)


class TestEMAReducer:

    @pytest.fixture(scope='function')
    def make_reducer(self):
        return lambda alpha: EMAReducer(alpha)

    @pytest.mark.parametrize('alpha', [0.0, 0.5, 1.0])
    def test_forward_peak(self, make_reducer, alpha, inputs):
        reducer = make_reducer(alpha)
        assert reducer.peak() is None
        res_true = torch.zeros_like(inputs[0])
        for t in inputs:
            reducer(t)
            res_test = reducer.peak()
            res_true = reducer.alpha * t + res_true * (1 - reducer.alpha)
            assert torch.all(res_test == res_true)

    @pytest.mark.parametrize('alpha', [0.0, 0.5, 1.0])
    def test_pop(self, make_reducer, alpha, inputs):
        generic_test_pop(make_reducer(alpha), inputs)

    @pytest.mark.parametrize('alpha', [0.0, 0.5, 1.0])
    def test_clear(self, make_reducer, alpha, inputs):
        generic_test_clear(make_reducer(alpha), inputs)
