from functools import reduce
import pytest
import random
import torch

from inferno.neural import Synapse, DeltaCurrent, DeltaPlusCurrent


def random_shape(mindims=1, maxdims=9, minsize=1, maxsize=9):
    return tuple(
        random.randint(mindims, maxdims)
        for _ in range(random.randint(minsize, maxsize))
    )


def validate_batchsz(synapse: Synapse, batchsz: int) -> None:
    assert synapse.batchsz == batchsz


def validate_shape(synapse: Synapse, shape: tuple[int, ...]) -> None:
    assert synapse.shape == shape


def validate_count(synapse: Synapse, shape: tuple[int, ...]) -> None:
    assert synapse.count == reduce(lambda a, b: a * b, shape)


def validate_batchedshape(
    synapse: Synapse, batchsz: int, shape: tuple[int, ...]
) -> None:
    assert synapse.batchedshape == (batchsz,) + shape


def validate_duration(synapse: Synapse, delay: float) -> None:
    assert synapse.delay == delay


class TestDeltaCurrent:

    @staticmethod
    def random_hyper(delayed=False):
        hyper = {}
        hyper["step_time"] = random.uniform(0.6, 1.8)
        hyper["spike_q"] = random.uniform(20, 30)
        hyper["delay"] = random.uniform(hyper["step_time"], 3) if delayed else 0.0
        hyper["interp_mode"] = ("nearest", "previous")[random.randint(0, 1)]
        hyper["interp_tol"] = 0.0
        hyper["current_overbound"] = 0.0
        hyper["spike_overbound"] = False
        hyper["batch_size"] = random.randint(1, 9)
        return hyper

    def test_batchsz(self):
        shape = random_shape()
        hyper = self.random_hyper(True)
        synapse = DeltaCurrent(shape, **hyper)

        validate_batchsz(synapse, hyper["batch_size"])

    def test_shape(self):
        shape = random_shape()
        hyper = self.random_hyper(True)
        synapse = DeltaCurrent(shape, **hyper)

        validate_shape(synapse, shape)

    def test_count(self):
        shape = random_shape()
        hyper = self.random_hyper(True)
        synapse = DeltaCurrent(shape, **hyper)

        validate_count(synapse, shape)

    def test_batchedshape(self):
        shape = random_shape()
        hyper = self.random_hyper(True)
        synapse = DeltaCurrent(shape, **hyper)

        validate_batchedshape(synapse, hyper["batch_size"], shape)

    def test_duration(self):
        shape = random_shape()
        hyper = self.random_hyper(True)
        synapse = DeltaCurrent(shape, **hyper)

        validate_duration(synapse, hyper["delay"])

    @pytest.mark.parametrize(
        "delayed",
        (True, False),
        ids=("delayed", "undelayed"),
    )
    def test_current_integration(self, delayed):
        shape = random_shape(maxdims=5)
        hyper = self.random_hyper(delayed)
        synapse = DeltaCurrent(shape, **hyper)
        spikes = torch.rand(hyper["batch_size"], *shape) > 0.5

        res = synapse(spikes)
        assert torch.all(
            (res - spikes * (hyper["spike_q"] / hyper["step_time"])).abs() <= 1e-7
        )
        assert torch.all(
            (synapse.current - spikes * (hyper["spike_q"] / hyper["step_time"])).abs()
            <= 1e-7
        )

    @pytest.mark.parametrize(
        "delayed",
        (True, False),
        ids=("delayed", "undelayed"),
    )
    def test_current_overbounding(self, delayed):
        shape = random_shape(maxdims=5)
        hyper = self.random_hyper(delayed) | {"current_overbound": -20.0}
        synapse = DeltaCurrent(shape, **hyper)
        spikes = torch.rand(hyper["batch_size"], *shape) > 0.5
        selector = (torch.rand(hyper["batch_size"], *shape) > 0.5) * (
            hyper["delay"] + 1
        )

        _ = synapse(spikes)

        assert torch.all(
            torch.abs(
                synapse.current_at(selector)
                - spikes * (hyper["spike_q"] / hyper["step_time"])
            )[selector == 0]
            <= 1e-7
        )
        assert torch.all(
            synapse.current_at(selector)[selector != 0] == hyper["current_overbound"]
        )
        assert torch.all(
            synapse.current_at(-selector)[selector != 0] == hyper["current_overbound"]
        )


class TestDeltaPlusCurrent:

    @staticmethod
    def random_hyper(delayed=False):
        hyper = {}
        hyper["step_time"] = random.uniform(0.6, 1.8)
        hyper["spike_q"] = random.uniform(20, 30)
        hyper["delay"] = random.uniform(hyper["step_time"], 3) if delayed else 0.0
        hyper["interp_mode"] = ("nearest", "previous")[random.randint(0, 1)]
        hyper["interp_tol"] = 0.0
        hyper["current_overbound"] = 0.0
        hyper["spike_overbound"] = False
        hyper["batch_size"] = random.randint(1, 9)
        return hyper

    def test_batchsz(self):
        shape = random_shape()
        hyper = self.random_hyper(True)
        synapse = DeltaPlusCurrent(shape, **hyper)

        validate_batchsz(synapse, hyper["batch_size"])

    def test_shape(self):
        shape = random_shape()
        hyper = self.random_hyper(True)
        synapse = DeltaPlusCurrent(shape, **hyper)

        validate_shape(synapse, shape)

    def test_count(self):
        shape = random_shape()
        hyper = self.random_hyper(True)
        synapse = DeltaPlusCurrent(shape, **hyper)

        validate_count(synapse, shape)

    def test_batchedshape(self):
        shape = random_shape()
        hyper = self.random_hyper(True)
        synapse = DeltaPlusCurrent(shape, **hyper)

        validate_batchedshape(synapse, hyper["batch_size"], shape)

    def test_duration(self):
        shape = random_shape()
        hyper = self.random_hyper(True)
        synapse = DeltaPlusCurrent(shape, **hyper)

        validate_duration(synapse, hyper["delay"])

    @pytest.mark.parametrize(
        "delayed",
        (True, False),
        ids=("delayed", "undelayed"),
    )
    @pytest.mark.parametrize(
        "ninjects",
        (0, 1, 2),
        ids=("ninjects=0", "ninjects=1", "ninjects=2"),
    )
    def test_current_integration(self, delayed, ninjects):
        shape = random_shape(maxdims=5)
        hyper = self.random_hyper(delayed)
        synapse = DeltaPlusCurrent(shape, **hyper)
        spikes = torch.rand(hyper["batch_size"], *shape) > 0.5
        injects = tuple(
            torch.rand(hyper["batch_size"], *shape) for _ in range(ninjects)
        )

        res = synapse(spikes, *injects)
        assert torch.all(
            torch.abs(
                res - sum((spikes * (hyper["spike_q"] / hyper["step_time"]), *injects))
            )
            <= 5e-6
        )
        assert torch.all(
            torch.abs(
                synapse.current
                - sum((spikes * (hyper["spike_q"] / hyper["step_time"]), *injects))
            )
            <= 5e-6
        )

    @pytest.mark.parametrize(
        "delayed",
        (True, False),
        ids=("delayed", "undelayed"),
    )
    @pytest.mark.parametrize(
        "ninjects",
        (0, 1),
        ids=("ninjects=0", "ninjects=1"),
    )
    def test_current_overbounding(self, delayed, ninjects):
        shape = random_shape(maxdims=5)
        hyper = self.random_hyper(delayed) | {"current_overbound": -20.0}
        synapse = DeltaPlusCurrent(shape, **hyper)
        spikes = torch.rand(hyper["batch_size"], *shape) > 0.5
        injects = tuple(
            torch.rand(hyper["batch_size"], *shape) for _ in range(ninjects)
        )
        selector = (torch.rand(hyper["batch_size"], *shape) > 0.5) * (
            hyper["delay"] + 1
        )

        _ = synapse(spikes, *injects)

        assert torch.all(
            torch.abs(
                synapse.current_at(selector)
                - sum((spikes * (hyper["spike_q"] / hyper["step_time"]), *injects))
            )[selector == 0]
            <= 5e-6
        )
        assert torch.all(
            synapse.current_at(selector)[selector != 0] == hyper["current_overbound"]
        )
        assert torch.all(
            synapse.current_at(-selector)[selector != 0] == hyper["current_overbound"]
        )
