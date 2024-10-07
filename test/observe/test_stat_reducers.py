import pytest
import random
import torch

from inferno.observe import CAReducer


def aaeq(t0, t1, eps=1e-6) -> bool:
    try:
        return bool(t0 == t1)
    except RuntimeError:
        if torch.all(t0 == t1):
            return True
        else:
            return torch.all((t0 - t1).abs() < eps)


class TestCAReducer:

    def test_clear(self):
        dt = random.uniform(0.4, 1.7)
        duration = 20 * dt
        data = torch.rand(20, random.randint(3, 5), random.randint(3, 5))

        reducer = CAReducer(step_time=dt, duration=duration, inclusive=False)
        for k in range(data.shape[0]):
            reducer(data[k])

        assert reducer._count != 0
        reducer.clear()
        assert reducer._count == 0

    @pytest.mark.parametrize(
        "dtype", (torch.float32, torch.bool), ids=("float32", "bool")
    )
    def test_forward_peek(self, dtype):
        dt = random.uniform(0.4, 1.7)
        duration = 12 * dt
        data = torch.rand(20, random.randint(3, 5), random.randint(3, 5))
        if dtype == torch.bool:
            data = data > 0.5
        else:
            data = data.to(dtype=dtype)

        reducer = CAReducer(step_time=dt, duration=duration, inclusive=False)
        for k in range(data.shape[0]):
            reducer(data[k])
            assert aaeq(reducer.peek(), data[slice(0, k + 1)].float().mean(0))
