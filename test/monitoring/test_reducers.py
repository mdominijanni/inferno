import pytest
import torch

import sys
sys.path.insert(0, '../..')

from inferno._internal import create_tensor

from inferno.monitoring import (
    PassthroughReducer, SinglePassthroughReducer,
    LastEventReducer, FuzzyLastEventReducer,
    SMAReducer, CMAReducer, EMAReducer,
    TraceReducer, AdditiveTraceReducer, ScalingTraceReducer
)


@pytest.fixture(scope='function')
def timesteps():
    return 20


@pytest.fixture(scope='function')
def shape():
    return (3, 3)


@pytest.fixture(scope='function')
def inputs(timesteps, shape):
    return [torch.rand(shape, dtype=torch.float32, requires_grad=False) for _ in range(timesteps)]


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


class TestLastEventReducer:

    @pytest.fixture(scope='function')
    def make_reducer(self):
        return lambda target: LastEventReducer(target)

    @pytest.fixture(scope='function')
    def inputs(self, request, timesteps, shape):
        return [torch.randint(request.param['min'], request.param['max'], shape, dtype=request.param['dtype'], requires_grad=False) for _ in range(timesteps)]

    @pytest.fixture(scope='function')
    def target(self, request, shape):
        if request.param.get('dtype') is None:
            return request.param['value']
        elif request.param.get('value') is not None:
            return torch.tensor(request.param['value'], dtype=request.param['dtype'])
        else:
            return torch.randint(request.param['min'], request.param['max'], shape, dtype=request.param['dtype'])

    @pytest.mark.parametrize('target,inputs',
        [({'value': True}, {'min': 0, 'max': 2, 'dtype': torch.bool}),
         ({'value': 3, 'dtype': torch.int32}, {'min': 0, 'max': 10, 'dtype': torch.int64}),
         ({'min': 0, 'max': 2, 'dtype': torch.bool}, {'min': 0, 'max': 10, 'dtype': torch.int64})],
        indirect=['target', 'inputs']
    )
    def test_forward_peak(self, make_reducer, target, inputs):
        reducer = make_reducer(target)
        assert reducer.peak() is None
        res_true = torch.full_like(inputs[0], float('inf'), dtype=torch.float32)
        for t in inputs:
            reducer(t)
            res_test = reducer.peak()
            res_true = res_true.add(1)
            res_true = res_true.masked_fill(t == target, 0)
            assert torch.all(res_test == res_true)

    @pytest.mark.parametrize('target,inputs',
        [({'value': True}, {'min': 0, 'max': 2, 'dtype': torch.bool}),
         ({'value': 3, 'dtype': torch.int32}, {'min': 0, 'max': 10, 'dtype': torch.int64}),
         ({'min': 0, 'max': 2, 'dtype': torch.bool}, {'min': 0, 'max': 10, 'dtype': torch.int64})],
        indirect=['target', 'inputs']
    )
    def test_pop(self, make_reducer, target, inputs):
        generic_test_pop(make_reducer(target), inputs)

    @pytest.mark.parametrize('target,inputs',
        [({'value': True}, {'min': 0, 'max': 2, 'dtype': torch.bool}),
         ({'value': 3, 'dtype': torch.int32}, {'min': 0, 'max': 10, 'dtype': torch.int64}),
         ({'min': 0, 'max': 2, 'dtype': torch.bool}, {'min': 0, 'max': 10, 'dtype': torch.int64})],
        indirect=['target', 'inputs']
    )
    def test_clear(self, make_reducer, target, inputs):
        generic_test_clear(make_reducer(target), inputs)


class TestFuzzyLastEventReducer:

    @pytest.fixture(scope='function')
    def make_reducer(self):
        return lambda target, epsilon: FuzzyLastEventReducer(target, epsilon)

    @pytest.fixture(scope='function')
    def target(self, request, shape):
        if request.param.get('dtype') is None:
            return request.param['value']
        elif request.param.get('value') is not None:
            return torch.tensor(request.param['value'], dtype=request.param['dtype'])
        else:
            return torch.rand(shape, dtype=request.param['dtype'])

    @pytest.fixture(scope='function')
    def epsilon(self, request):
        return request.param

    @pytest.mark.parametrize('target,epsilon',
        [({'value': 0.3}, 0.2),
         ({'value': 0.7, 'dtype': torch.float32}, 0.2),
         ({'dtype': torch.float64}, 0.2)],
        indirect=['target', 'epsilon']
    )
    def test_forward_peak(self, make_reducer, target, epsilon, inputs):
        reducer = make_reducer(target, epsilon)
        assert reducer.peak() is None
        res_true = torch.full_like(inputs[0], float('inf'), dtype=torch.float32)
        for t in inputs:
            reducer(t)
            res_test = reducer.peak()
            res_true = res_true.add(1)
            res_true = res_true.masked_fill(torch.abs(t - target) <= epsilon, 0)
            assert torch.all(res_test == res_true)

    @pytest.mark.parametrize('target,epsilon',
        [({'value': 0.3}, 0.2),
         ({'value': 0.7, 'dtype': torch.float32}, 0.2),
         ({'dtype': torch.float64}, 0.2)],
        indirect=['target', 'epsilon']
    )
    def test_pop(self, make_reducer, target, epsilon, inputs):
        generic_test_pop(make_reducer(target, epsilon), inputs)

    @pytest.mark.parametrize('target,epsilon',
        [({'value': 0.3}, 0.2),
         ({'value': 0.7, 'dtype': torch.float32}, 0.2),
         ({'dtype': torch.float64}, 0.2)],
        indirect=['target', 'epsilon']
    )
    def test_clear(self, make_reducer, target, epsilon, inputs):
        generic_test_clear(make_reducer(target, epsilon), inputs)


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


class TestTraceReducer:

    @pytest.fixture(scope='function')
    def amplitude(self):
        return 5.0

    @pytest.fixture(scope='function')
    def decay(self, request):
        return request.param

    @pytest.fixture(scope='function')
    def step_time(self, request):
        return request.param

    @pytest.fixture(scope='function')
    def time_constant(self, request):
        return request.param

    @pytest.fixture(scope='function')
    def make_reducer_oneparam(self, amplitude):
        return lambda target, decay: TraceReducer(amplitude, decay=decay)

    @pytest.fixture(scope='function')
    def make_reducer_twoparam(self, amplitude):
        return lambda target, step_time, time_constant: TraceReducer(amplitude, step_time=step_time, time_constant=time_constant)

    @pytest.fixture(scope='function')
    def inputs(self, request, timesteps, shape):
        return [torch.randint(request.param['min'], request.param['max'], shape, dtype=request.param['dtype'], requires_grad=False) for _ in range(timesteps)]

    @pytest.fixture(scope='function')
    def target(self, request, shape):
        if request.param.get('dtype') is None:
            return request.param['value']
        elif request.param.get('value') is not None:
            return torch.tensor(request.param['value'], dtype=request.param['dtype'])
        else:
            return torch.randint(request.param['min'], request.param['max'], shape, dtype=request.param['dtype'])

    @pytest.mark.parametrize('decay,step_time,time_constant,target,inputs',
        [pytest.param(torch.exp(create_tensor(-1.0 / 20.0)), 1.0, 20.0, {'value': True}, {'min': 0, 'max': 2, 'dtype': torch.bool}, id='singleton_equal_decays'),
         pytest.param(torch.exp(create_tensor(-1.2 / 20.0)), 1.0, 20.0, {'value': True}, {'min': 0, 'max': 2, 'dtype': torch.bool}, marks=pytest.mark.xfail, id='unequal_decays'),
         pytest.param(torch.exp(create_tensor(-1.0)), 1.0, -1.0, {'value': True}, {'min': 0, 'max': 2, 'dtype': torch.bool}, marks=pytest.mark.xfail, id='bad_decays')],
        indirect=['decay', 'step_time', 'time_constant', 'target', 'inputs']
    )
    def test_forward_peak_decay_calc(self, make_reducer_oneparam, make_reducer_twoparam, amplitude, decay, step_time, time_constant, target, inputs):
        reducer1p = make_reducer_oneparam(target, decay)
        reducer2p = make_reducer_twoparam(target, step_time, time_constant)
        assert reducer1p.peak() is None
        assert reducer2p.peak() is None
        res_true = torch.zeros_like(inputs[0], dtype=torch.float32)
        for t in inputs:
            reducer1p(t)
            reducer2p(t)
            res_test1p = reducer1p.peak()
            res_test2p = reducer2p.peak()
            res_true = res_true.mul(decay)
            res_true = res_true.masked_fill(t == target, amplitude)
            assert torch.all(res_test1p == res_true)
            assert torch.all(res_test2p == res_true)

    @pytest.mark.parametrize('decay,target,inputs',
        [pytest.param(torch.exp(create_tensor(-1.0 / 20.0)), {'value': True}, {'min': 0, 'max': 2, 'dtype': torch.bool}, id='standard')],
        indirect=['decay', 'target', 'inputs']
    )
    def test_pop(self, make_reducer_oneparam, decay, target, inputs):
        generic_test_pop(make_reducer_oneparam(target, decay), inputs)

    @pytest.mark.parametrize('decay,target,inputs',
        [pytest.param(torch.exp(create_tensor(-1.0 / 20.0)), {'value': True}, {'min': 0, 'max': 2, 'dtype': torch.bool}, id='standard')],
        indirect=['decay', 'target', 'inputs']
    )
    def test_clear(self, make_reducer_oneparam, decay, target, inputs):
        generic_test_clear(make_reducer_oneparam(target, decay), inputs)


class TestAdditiveTraceReducer:

    @pytest.fixture(scope='function')
    def amplitude(self):
        return 5.0

    @pytest.fixture(scope='function')
    def decay(self, request):
        return request.param

    @pytest.fixture(scope='function')
    def step_time(self, request):
        return request.param

    @pytest.fixture(scope='function')
    def time_constant(self, request):
        return request.param

    @pytest.fixture(scope='function')
    def make_reducer_oneparam(self, amplitude):
        return lambda target, decay: AdditiveTraceReducer(amplitude, decay=decay)

    @pytest.fixture(scope='function')
    def make_reducer_twoparam(self, amplitude):
        return lambda target, step_time, time_constant: AdditiveTraceReducer(amplitude, step_time=step_time, time_constant=time_constant)

    @pytest.fixture(scope='function')
    def inputs(self, request, timesteps, shape):
        return [torch.randint(request.param['min'], request.param['max'], shape, dtype=request.param['dtype'], requires_grad=False) for _ in range(timesteps)]

    @pytest.fixture(scope='function')
    def target(self, request, shape):
        if request.param.get('dtype') is None:
            return request.param['value']
        elif request.param.get('value') is not None:
            return torch.tensor(request.param['value'], dtype=request.param['dtype'])
        else:
            return torch.randint(request.param['min'], request.param['max'], shape, dtype=request.param['dtype'])

    @pytest.mark.parametrize('decay,step_time,time_constant,target,inputs',
        [pytest.param(torch.exp(create_tensor(-1.0 / 20.0)), 1.0, 20.0, {'value': True}, {'min': 0, 'max': 2, 'dtype': torch.bool}, id='singleton_equal_decays'),
         pytest.param(torch.exp(create_tensor(-1.2 / 20.0)), 1.0, 20.0, {'value': True}, {'min': 0, 'max': 2, 'dtype': torch.bool}, marks=pytest.mark.xfail, id='unequal_decays'),
         pytest.param(torch.exp(create_tensor(-1.0)), 1.0, -1.0, {'value': True}, {'min': 0, 'max': 2, 'dtype': torch.bool}, marks=pytest.mark.xfail, id='bad_decays')],
        indirect=['decay', 'step_time', 'time_constant', 'target', 'inputs']
    )
    def test_forward_peak_decay_calc(self, make_reducer_oneparam, make_reducer_twoparam, amplitude, decay, step_time, time_constant, target, inputs):
        reducer1p = make_reducer_oneparam(target, decay)
        reducer2p = make_reducer_twoparam(target, step_time, time_constant)
        assert reducer1p.peak() is None
        assert reducer2p.peak() is None
        res_true = torch.zeros_like(inputs[0], dtype=torch.float32)
        for t in inputs:
            reducer1p(t)
            reducer2p(t)
            res_test1p = reducer1p.peak()
            res_test2p = reducer2p.peak()
            res_true = res_true.mul(decay)
            res_true = res_true + amplitude * (t == target)
            assert torch.all(res_test1p == res_true)
            assert torch.all(res_test2p == res_true)

    @pytest.mark.parametrize('decay,target,inputs',
        [pytest.param(torch.exp(create_tensor(-1.0 / 20.0)), {'value': True}, {'min': 0, 'max': 2, 'dtype': torch.bool}, id='standard')],
        indirect=['decay', 'target', 'inputs']
    )
    def test_pop(self, make_reducer_oneparam, decay, target, inputs):
        generic_test_pop(make_reducer_oneparam(target, decay), inputs)

    @pytest.mark.parametrize('decay,target,inputs',
        [pytest.param(torch.exp(create_tensor(-1.0 / 20.0)), {'value': True}, {'min': 0, 'max': 2, 'dtype': torch.bool}, id='standard')],
        indirect=['decay', 'target', 'inputs']
    )
    def test_clear(self, make_reducer_oneparam, decay, target, inputs):
        generic_test_clear(make_reducer_oneparam(target, decay), inputs)


class TestScalingTraceReducer:

    @pytest.fixture(scope='function')
    def amplitude(self):
        return 5.0

    @pytest.fixture(scope='function')
    def decay(self, request):
        return request.param

    @pytest.fixture(scope='function')
    def step_time(self, request):
        return request.param

    @pytest.fixture(scope='function')
    def time_constant(self, request):
        return request.param

    @pytest.fixture(scope='function')
    def make_reducer_oneparam(self, amplitude):
        return lambda decay: ScalingTraceReducer(amplitude, decay=decay)

    @pytest.fixture(scope='function')
    def make_reducer_twoparam(self, amplitude):
        return lambda step_time, time_constant: ScalingTraceReducer(amplitude, step_time=step_time, time_constant=time_constant)

    @pytest.fixture(scope='function')
    def inputs(self, request, timesteps, shape):
        return [torch.randint(request.param['min'], request.param['max'], shape, dtype=request.param['dtype'], requires_grad=False) for _ in range(timesteps)]

    @pytest.mark.parametrize('decay,step_time,time_constant,inputs',
        [pytest.param(torch.exp(create_tensor(-1.0 / 20.0)), 1.0, 20.0, {'min': 0, 'max': 9, 'dtype': torch.int64}, id='singleton_equal_decays'),
         pytest.param(torch.exp(create_tensor(-1.2 / 20.0)), 1.0, 20.0, {'min': 0, 'max': 9, 'dtype': torch.int64}, marks=pytest.mark.xfail, id='unequal_decays'),
         pytest.param(torch.exp(create_tensor(-1.0)), 1.0, -1.0, {'min': 0, 'max': 9, 'dtype': torch.int64}, marks=pytest.mark.xfail, id='bad_decays')],
        indirect=['decay', 'step_time', 'time_constant', 'inputs']
    )
    def test_forward_peak_decay_calc(self, make_reducer_oneparam, make_reducer_twoparam, amplitude, decay, step_time, time_constant, inputs):
        reducer1p = make_reducer_oneparam(decay)
        reducer2p = make_reducer_twoparam(step_time, time_constant)
        assert reducer1p.peak() is None
        assert reducer2p.peak() is None
        res_true = torch.zeros_like(inputs[0], dtype=torch.float32)
        for t in inputs:
            reducer1p(t)
            reducer2p(t)
            res_test1p = reducer1p.peak()
            res_test2p = reducer2p.peak()
            res_true = res_true.mul(decay)
            res_true = res_true + amplitude * t
            assert torch.all(res_test1p == res_true)
            assert torch.all(res_test2p == res_true)

    @pytest.mark.parametrize('decay,inputs',
        [pytest.param(torch.exp(create_tensor(-1.0 / 20.0)), {'min': 0, 'max': 9, 'dtype': torch.int64}, id='standard')],
        indirect=['decay', 'inputs']
    )
    def test_pop(self, make_reducer_oneparam, decay, inputs):
        generic_test_pop(make_reducer_oneparam(decay), inputs)

    @pytest.mark.parametrize('decay,inputs',
        [pytest.param(torch.exp(create_tensor(-1.0 / 20.0)), {'min': 0, 'max': 9, 'dtype': torch.int64}, id='standard')],
        indirect=['decay', 'inputs']
    )
    def test_clear(self, make_reducer_oneparam, decay, inputs):
        generic_test_clear(make_reducer_oneparam(decay), inputs)
