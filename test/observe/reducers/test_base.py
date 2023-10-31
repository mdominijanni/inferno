import pytest
import math
import random
import torch

import sys

sys.path.insert(0, "../../..")

from inferno.observe import MapReducer, FoldReducer


@pytest.fixture(scope="class")
def observation_ndim():
    return random.randint(3, 7)


@pytest.fixture(scope="class")
def observation_shape(observation_ndim):
    return tuple([random.randint(4, 9) for _ in range(observation_ndim)])


class TestMapReducer:
    @pytest.fixture(scope="class")
    def window_size(self):
        return random.randint(15, 30)

    @pytest.fixture(scope="class")
    def num_samples_underfill(self, window_size):
        return random.randint(5, window_size - 5)

    @pytest.fixture(scope="class")
    def num_samples_overfill(self, window_size):
        return random.randint(window_size, window_size * 2)

    def test_clear_peek(self, window_size, observation_shape):
        reducer = MapReducer(window=window_size)
        inputs = [torch.rand(*observation_shape).float() for _ in range(window_size)]
        for sample in inputs:
            reducer(sample.clone().detach())
        test_res_ini = reducer.peek().clone().detach()

        reducer.clear(keepshape=False)

        for sample in inputs:
            reducer(sample.clone().detach())
        test_res_fin = reducer.peek().clone().detach()

        assert torch.all(test_res_ini == test_res_fin)
        assert test_res_ini.shape == test_res_fin.shape
        assert test_res_ini.device == test_res_fin.device
        assert test_res_ini.dtype == test_res_fin.dtype

    def test_clear_keepshape(self, window_size, observation_shape):
        reducer = MapReducer(window=window_size)
        inputs = torch.rand(*observation_shape).float()

        test_res_start = tuple(reducer._data.shape)
        reducer(inputs.clone().detach())
        test_res_input = tuple(reducer._data.shape)
        reducer.clear(keepshape=True)
        test_res_reset = tuple(reducer._data.shape)

        assert test_res_start != test_res_input
        assert test_res_start != test_res_reset
        assert test_res_input == test_res_reset

    def test_clear_delshape(self, window_size, observation_shape):
        reducer = MapReducer(window=window_size)
        inputs = torch.rand(*observation_shape).float()

        test_res_start = tuple(reducer._data.shape)
        reducer(inputs.clone().detach())
        test_res_input = tuple(reducer._data.shape)
        reducer.clear(keepshape=False)
        test_res_reset = tuple(reducer._data.shape)

        assert test_res_start != test_res_input
        assert test_res_start == test_res_reset
        assert test_res_input != test_res_reset

    def test_peek_shape(self, window_size, observation_shape, observation_ndim):
        inputs = [torch.rand(*observation_shape).float() for _ in range(window_size)]
        for dim in range(0, observation_ndim):
            reducer = MapReducer(window=window_size)
            for sample in inputs:
                reducer(sample.clone().detach())
            test_res = tuple(reducer.peek(dim=dim).shape)
            true_res, obs_shape = [], list(observation_shape)
            for d in range(0, observation_ndim + 1):
                if d == dim:
                    true_res.append(window_size)
                else:
                    true_res.append(obs_shape[0])
                    del obs_shape[0]
            true_res = tuple(true_res)
            assert test_res == true_res

    @pytest.mark.parametrize(
        "num_samples",
        ["num_samples_overfill", "num_samples_underfill"],
        ids=["overfill", "underfill"],
    )
    def test_peek(self, window_size, observation_shape, num_samples, request):
        num_samples = request.getfixturevalue(num_samples)
        reducer = MapReducer(window=window_size)
        inputs = [torch.rand(*observation_shape).float() for _ in range(num_samples)]
        for sample in inputs:
            reducer(sample.clone().detach())
        test_res = reducer.peek(dim=-1)
        if num_samples > window_size:
            true_res = torch.stack(inputs[-window_size:], dim=-1)
        else:
            true_res = torch.stack(inputs, dim=-1)
        assert torch.all(test_res == true_res)
        assert tuple(test_res.shape) == tuple(true_res.shape)

    @pytest.mark.parametrize(
        "num_samples",
        ["num_samples_overfill", "num_samples_underfill"],
        ids=["overfill", "underfill"],
    )
    def test_pop_keepshape(self, window_size, observation_shape, num_samples, request):
        num_samples = request.getfixturevalue(num_samples)
        reducer = MapReducer(window=window_size)
        inputs = [torch.rand(*observation_shape).float() for _ in range(num_samples)]
        for sample in inputs:
            reducer(sample.clone().detach())
        test_res = reducer.pop(dim=-1, keepshape=True)
        if num_samples > window_size:
            true_res = torch.stack(inputs[-window_size:], dim=-1)
        else:
            true_res = torch.stack(inputs, dim=-1)
        assert torch.all(test_res == true_res)
        assert tuple(test_res.shape) == tuple(true_res.shape)
        assert reducer.pop(keepshape=True) is None

        for sample in inputs:
            reducer(sample.clone().detach())
        test_res = reducer.pop(dim=-1, keepshape=True)
        assert torch.all(test_res == true_res)
        assert tuple(test_res.shape) == tuple(true_res.shape)

    @pytest.mark.parametrize(
        "num_samples",
        ["num_samples_overfill", "num_samples_underfill"],
        ids=["overfill", "underfill"],
    )
    def test_pop_delshape(self, window_size, observation_shape, num_samples, request):
        num_samples = request.getfixturevalue(num_samples)
        reducer = MapReducer(window=window_size)
        inputs = [torch.rand(*observation_shape).float() for _ in range(num_samples)]
        for sample in inputs:
            reducer(sample.clone().detach())
        test_res = reducer.pop(dim=-1, keepshape=False)
        if num_samples > window_size:
            true_res = torch.stack(inputs[-window_size:], dim=-1)
        else:
            true_res = torch.stack(inputs, dim=-1)
        assert torch.all(test_res == true_res)
        assert tuple(test_res.shape) == tuple(true_res.shape)
        assert reducer.pop() is None

        inputs = [
            torch.rand(*[os + 1 for os in observation_shape]).float()
            for _ in range(num_samples)
        ]

        for sample in inputs:
            reducer(sample.clone().detach())

    def test_nondefault_map(self, window_size, observation_shape):
        mapfn = lambda x1, x2: x1**2 + x2
        reducer = MapReducer(window=window_size, mapfn=mapfn)
        inputs = [
            (
                torch.rand(*observation_shape).float(),
                torch.rand(*observation_shape).float(),
            )
            for _ in range(window_size)
        ]
        for sample in inputs:
            reducer(*[s.clone().detach() for s in sample])
        test_res = reducer.peek()
        true_res = torch.stack([mapfn(x1, x2) for x1, x2 in inputs], dim=-1)
        assert torch.all(test_res == true_res)

    def test_nondefault_filter(self, window_size, observation_shape):
        count = 0

        def isprime(n):
            if n <= 3:
                return n > 1
            if n % 2 == 0 or n % 3 == 0:
                return False
            for k in range(5, math.isqrt(n) + 1, 6):
                if n % k == 0 or n % (k + 2) == 0:
                    return False
            return True

        def filterfn(inputs):
            nonlocal count
            count += 1
            return isprime(count)

        reducer = MapReducer(window=window_size, filterfn=filterfn)
        inputs = [torch.rand(*observation_shape).float() for _ in range(window_size)]

        for sample in inputs:
            reducer(sample.clone().detach())
        test_res = reducer.peek()
        true_res = []
        for idx, sample in enumerate(inputs):
            if isprime(idx + 1):
                true_res.append(sample.clone().detach())
        true_res = torch.stack(true_res, dim=-1)
        assert torch.all(test_res == true_res)


class TestFoldReducer:
    @pytest.fixture(scope="class")
    def num_samples(self):
        return random.randint(15, 30)

    def test_clear_peek(self, observation_shape, num_samples):
        reducer = FoldReducer(foldfn=lambda x, s: x**2 + s)
        inputs = [
            torch.randint(0, 10, tuple(observation_shape)).float()
            for _ in range(num_samples)
        ]
        for sample in inputs:
            reducer(sample.clone().detach())
        test_res_ini = reducer.peek().clone().detach()

        reducer.clear()

        for sample in inputs:
            reducer(sample.clone().detach())
        test_res_fin = reducer.peek().clone().detach()

        assert torch.all(test_res_ini == test_res_fin)
        assert test_res_ini.shape == test_res_fin.shape
        assert test_res_ini.device == test_res_fin.device
        assert test_res_ini.dtype == test_res_fin.dtype

    def test_peek(self, observation_shape, num_samples):
        reducer = reducer = FoldReducer(foldfn=lambda x, s: x**2 + s)
        inputs = [
            torch.randint(0, 10, tuple(observation_shape)).float()
            for _ in range(num_samples)
        ]
        for sample in inputs:
            reducer(sample.clone().detach())
        test_res = reducer.peek()
        true_res = torch.sum(torch.stack(inputs, dim=-1) ** 2, dim=-1)
        assert torch.all(test_res == true_res)
        assert tuple(test_res.shape) == tuple(true_res.shape)

    def test_pop(self, observation_shape, num_samples):
        reducer = FoldReducer(foldfn=lambda x, s: x**2 + s)
        inputs = [
            torch.randint(0, 10, tuple(observation_shape)).float()
            for _ in range(num_samples)
        ]
        for sample in inputs:
            reducer(sample.clone().detach())
        test_res_ini = reducer.pop()

        reducer.clear()

        for sample in inputs:
            reducer(sample.clone().detach())
        test_res_fin = reducer.pop()

        assert torch.all(test_res_ini == test_res_fin)
        assert test_res_ini.shape == test_res_fin.shape
        assert test_res_ini.device == test_res_fin.device
        assert test_res_ini.dtype == test_res_fin.dtype

    def test_nondefault_map(self, observation_shape, num_samples):
        mapfn = lambda x1, x2: x1 + x2 * 2
        reducer = FoldReducer(foldfn=lambda x, s: x**2 + s, mapfn=mapfn)
        inputs = [
            (
                torch.randint(0, 10, tuple(observation_shape)).float(),
                torch.randint(0, 10, tuple(observation_shape)).float(),
            )
            for _ in range(num_samples)
        ]
        for sample in inputs:
            reducer(*[s.clone().detach() for s in sample])
        test_res = reducer.peek()
        true_res = None
        for sample in inputs:
            val = sample[0] + sample[1] * 2
            if true_res is None:
                true_res = val**2
            else:
                true_res += val**2
        assert torch.all(test_res == true_res)

    def test_nondefault_filter(self, observation_shape, num_samples):
        count = 0

        def isprime(n):
            if n <= 3:
                return n > 1
            if n % 2 == 0 or n % 3 == 0:
                return False
            for k in range(5, math.isqrt(n) + 1, 6):
                if n % k == 0 or n % (k + 2) == 0:
                    return False
            return True

        def filterfn(inputs):
            nonlocal count
            count += 1
            return isprime(count)

        reducer = FoldReducer(foldfn=lambda x, s: x**2 + s, filterfn=filterfn)
        inputs = [
            torch.randint(0, 10, tuple(observation_shape)).float()
            for _ in range(num_samples)
        ]

        for sample in inputs:
            reducer(sample.clone().detach())
        test_res = reducer.peek()
        true_res = None
        for idx, sample in enumerate(inputs):
            if isprime(idx + 1):
                if true_res is None:
                    true_res = sample.clone().detach() ** 2
                else:
                    true_res += sample**2
        assert torch.all(test_res == true_res)

    def test_nondefault_init(self, observation_shape, num_samples):
        reducer = reducer = FoldReducer(
            foldfn=lambda x, s: x**2 + s, initfn=lambda x: torch.full_like(x, 5)
        )
        inputs = [
            torch.randint(0, 10, tuple(observation_shape)).float()
            for _ in range(num_samples)
        ]
        for sample in inputs:
            reducer(sample.clone().detach())
        test_res = reducer.peek()
        true_res = torch.sum(torch.stack(inputs, dim=-1) ** 2, dim=-1) + 5
        assert torch.all(test_res == true_res)
        assert tuple(test_res.shape) == tuple(true_res.shape)
