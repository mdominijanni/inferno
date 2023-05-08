import pytest
import torch
import torch.nn as nn

import sys
sys.path.insert(0, '../..')

from inferno.neural import ParameterNormalization, ParameterClamping


class RandomizingModule(nn.Module):

    def __init__(self, min, max, shape, dtype):
        nn.Module.__init__(self)
        self.min = min
        self.max = max
        self.register_parameter('state', nn.Parameter(torch.empty(shape, dtype=dtype), False))
        self.randomize()

    def randomize(self):
        if self.state.is_floating_point():
            self.state.uniform_(self.min, self.max)
        else:
            self.state.random_(self.min, self.max)

    def forward(self):
        prior = self.state.clone()
        self.randomize()
        return prior


@pytest.fixture(scope='function')
def shape():
    return (2, 3, 4, 5)


@pytest.fixture(scope='function')
def dtype():
    return torch.float32


@pytest.fixture(scope='function')
def timesteps():
    return 20


@pytest.fixture(scope='function')
def min():
    return 0.0


@pytest.fixture(scope='function')
def max():
    return 1.0


@pytest.fixture(scope='function')
def targetmod(min, max, shape, dtype):
    return RandomizingModule(min, max, shape, dtype)


class TestParameterClamping:

    @pytest.fixture(scope='function')
    def make_hook(self):
        return lambda clamp_min, clamp_max, train_update, eval_update: ParameterClamping('state', clamp_min, clamp_max, train_update, eval_update)

    @pytest.fixture(scope='function')
    def clamp_min(self, request):
        return request.param

    @pytest.fixture(scope='function')
    def clamp_max(self, request):
        return request.param

    @pytest.mark.parametrize('clamp_min,clamp_max',
        [pytest.param(0.20, 0.70, id='minmax'),
         pytest.param(None, 0.70, id='maxonly'),
         pytest.param(0.20, None, id='minonly'),
         pytest.param(None, None, id='nobound', marks=pytest.mark.xfail)],
        indirect=['clamp_min', 'clamp_max']
    )
    def test_update_train_true(self, make_hook, targetmod, timesteps, clamp_min, clamp_max):
        hook = make_hook(clamp_min, clamp_max, True, False)
        hook.register(targetmod)
        targetmod.train()
        for _ in range(timesteps):
            res = targetmod()
            if clamp_min is not None:
                assert res.min() >= clamp_min
            if clamp_max is not None:
                assert res.max() <= clamp_max

    @pytest.mark.parametrize('clamp_min,clamp_max',
        [pytest.param(0.20, 0.70, id='minmax')],
        indirect=['clamp_min', 'clamp_max']
    )
    def test_update_train_false(self, make_hook, targetmod, timesteps, clamp_min, clamp_max):
        hook = make_hook(clamp_min, clamp_max, False, True)
        hook.register(targetmod)
        targetmod.train()
        for _ in range(timesteps):
            prior = targetmod.state.clone()
            res = targetmod()
            assert torch.all(res == prior)

    @pytest.mark.parametrize('clamp_min,clamp_max',
        [pytest.param(0.20, 0.70, id='minmax'),
         pytest.param(None, 0.70, id='maxonly'),
         pytest.param(0.20, None, id='minonly'),
         pytest.param(None, None, id='nobound', marks=pytest.mark.xfail)],
        indirect=['clamp_min', 'clamp_max']
    )
    def test_update_eval_true(self, make_hook, targetmod, timesteps, clamp_min, clamp_max):
        hook = make_hook(clamp_min, clamp_max, False, True)
        hook.register(targetmod)
        targetmod.eval()
        for _ in range(timesteps):
            res = targetmod()
            if clamp_min is not None:
                assert res.min() >= clamp_min
            if clamp_max is not None:
                assert res.max() <= clamp_max

    @pytest.mark.parametrize('clamp_min,clamp_max',
        [pytest.param(0.20, 0.70, id='minmax')],
        indirect=['clamp_min', 'clamp_max']
    )
    def test_update_eval_false(self, make_hook, targetmod, timesteps, clamp_min, clamp_max):
        hook = make_hook(clamp_min, clamp_max, True, False)
        hook.register(targetmod)
        targetmod.eval()
        for _ in range(timesteps):
            prior = targetmod.state.clone()
            res = targetmod()
            assert torch.all(res == prior)


class TestParameterNormalization:

    @pytest.fixture(scope='function')
    def epsilon(self):
        return 1e-6

    @pytest.fixture(scope='function')
    def make_hook(self):
        return lambda dims, norm, scale, train_update, eval_update: ParameterNormalization('state', dims, norm, scale, train_update, eval_update)

    @pytest.fixture(scope='function')
    def dims(self, request):
        return request.param

    @pytest.fixture(scope='function')
    def norm(self, request):
        return request.param

    @pytest.fixture(scope='function')
    def scale(self, request):
        return request.param

    @pytest.mark.parametrize('dims,norm,scale',
        [pytest.param(0, 1, 1, id='absolute_dim'),
         pytest.param(-1, 1, 1, id='relative_dim'),
         pytest.param((1, 2), 2, 2, id='multiple_dims')],
        indirect=['dims', 'norm', 'scale']
    )
    def test_update_train_true(self, make_hook, targetmod, timesteps, dims, norm, scale, epsilon):
        hook = make_hook(dims, norm, scale, True, False)
        hook.register(targetmod)
        targetmod.train()
        for _ in range(timesteps):
            res = torch.pow(torch.sum(torch.pow(torch.abs(targetmod()), norm), dim=dims), 1 / norm)
            assert torch.all((res - scale) <= epsilon)

    @pytest.mark.parametrize('dims,norm,scale',
        [pytest.param(0, 1, 1, id='absolute_dim')],
        indirect=['dims', 'norm', 'scale']
    )
    def test_update_train_false(self, make_hook, targetmod, timesteps, dims, norm, scale, epsilon):
        hook = make_hook(dims, norm, scale, False, True)
        hook.register(targetmod)
        targetmod.train()
        for _ in range(timesteps):
            prior = targetmod.state.clone()
            res = targetmod()
            assert torch.all(res == prior)

    @pytest.mark.parametrize('dims,norm,scale',
        [pytest.param(0, 1, 1, id='absolute_dim'),
         pytest.param(-1, 1, 1, id='relative_dim'),
         pytest.param((1, 2), 2, 2, id='multiple_dims')],
        indirect=['dims', 'norm', 'scale']
    )
    def test_update_eval_true(self, make_hook, targetmod, timesteps, dims, norm, scale, epsilon):
        hook = make_hook(dims, norm, scale, False, True)
        hook.register(targetmod)
        targetmod.eval()
        for _ in range(timesteps):
            res = torch.pow(torch.sum(torch.pow(torch.abs(targetmod()), norm), dim=dims), 1 / norm)
            assert torch.all((res - scale) <= epsilon)

    @pytest.mark.parametrize('dims,norm,scale',
        [pytest.param(0, 1, 1, id='absolute_dim')],
        indirect=['dims', 'norm', 'scale']
    )
    def test_update_eval_false(self, make_hook, targetmod, timesteps, dims, norm, scale, epsilon):
        hook = make_hook(dims, norm, scale, True, False)
        hook.register(targetmod)
        targetmod.eval()
        for _ in range(timesteps):
            prior = targetmod.state.clone()
            res = targetmod()
            assert torch.all(res == prior)
