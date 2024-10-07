from functools import reduce
import math
import pytest
import random
import torch

from inferno.neural import Synapse, SingleExponentialCurrent, DoubleExponentialCurrent


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


class TestSingleExponentialCurrent:

    @staticmethod
    def random_hyper(delayed=False, inplace=False):
        hyper = {}
        hyper["step_time"] = random.uniform(0.6, 1.8)
        hyper["spike_charge"] = random.uniform(20, 30)
        hyper["time_constant"] = random.uniform(20, 30)
        hyper["delay"] = (
            random.uniform(hyper["step_time"], 3 * hyper["step_time"])
            if delayed
            else 0.0
        )
        hyper["spike_interp_mode"] = ("nearest", "previous")[random.randint(0, 1)]
        hyper["interp_tol"] = 0.0
        hyper["current_overbound"] = 0.0
        hyper["spike_overbound"] = False
        hyper["batch_size"] = random.randint(1, 9)
        hyper["inplace"] = inplace
        return hyper

    def test_batchsz(self):
        shape = random_shape()
        hyper = self.random_hyper(True)
        synapse = SingleExponentialCurrent(shape, **hyper)

        validate_batchsz(synapse, hyper["batch_size"])

    def test_shape(self):
        shape = random_shape()
        hyper = self.random_hyper(True)
        synapse = SingleExponentialCurrent(shape, **hyper)

        validate_shape(synapse, shape)

    def test_count(self):
        shape = random_shape()
        hyper = self.random_hyper(True)
        synapse = SingleExponentialCurrent(shape, **hyper)

        validate_count(synapse, shape)

    def test_batchedshape(self):
        shape = random_shape()
        hyper = self.random_hyper(True)
        synapse = SingleExponentialCurrent(shape, **hyper)

        validate_batchedshape(synapse, hyper["batch_size"], shape)

    def test_duration(self):
        shape = random_shape()
        hyper = self.random_hyper(True)
        synapse = SingleExponentialCurrent(shape, **hyper)

        validate_duration(synapse, hyper["delay"])

    @pytest.mark.parametrize(
        "delayed",
        (True, False),
        ids=("delayed", "undelayed"),
    )
    @pytest.mark.parametrize(
        "inplace",
        (True, False),
        ids=("inplace", "realloc"),
    )
    def test_current_integration(self, delayed, inplace):
        shape = random_shape(maxdims=5)
        hyper = self.random_hyper(delayed, inplace)
        synapse = SingleExponentialCurrent(shape, **hyper)

        currents = torch.zeros(hyper["batch_size"], *shape)

        for _ in range(100):
            spikes = torch.rand(hyper["batch_size"], *shape) < 0.1
            res = synapse(spikes)
            currents = currents * math.exp(-hyper["step_time"] / hyper["time_constant"])
            currents = (
                currents + (hyper["spike_charge"] / hyper["time_constant"]) * spikes
            )

            assert torch.all((res - currents).abs() <= 1e-7)
            assert torch.all((synapse.current - currents).abs() <= 1e-7)

    @pytest.mark.parametrize(
        "delayed",
        (True, False),
        ids=("delayed", "undelayed"),
    )
    @pytest.mark.parametrize(
        "inplace",
        (True, False),
        ids=("inplace", "realloc"),
    )
    def test_current_overbounding(self, delayed, inplace):
        shape = random_shape(maxdims=5)
        hyper = self.random_hyper(delayed, inplace) | {"current_overbound": -20.0}
        synapse = SingleExponentialCurrent(shape, **hyper)
        selector = (torch.rand(hyper["batch_size"], *shape) > 0.5) * (
            hyper["delay"] + hyper["step_time"]
        )

        for _ in range(30):
            _ = synapse(torch.rand(hyper["batch_size"], *shape) < 0.2)

        assert torch.all(
            synapse.current_at(selector)[selector == 0]
            == synapse.current[selector == 0]
        )
        assert torch.all(
            synapse.current_at(selector)[selector != 0] == hyper["current_overbound"]
        )
        assert torch.all(
            synapse.current_at(-selector)[selector != 0] == hyper["current_overbound"]
        )

    @pytest.mark.parametrize(
        "inplace",
        (True, False),
        ids=("inplace", "realloc"),
    )
    def test_current_interpolation(self, inplace):
        shape = random_shape(maxdims=5)
        hyper = self.random_hyper(True, inplace)
        hyper = hyper | {"delay": hyper["step_time"]}
        synapse = SingleExponentialCurrent(shape, **hyper)

        for _ in range(30):
            _ = synapse(torch.rand(hyper["batch_size"], *shape) < 0.2)

        selector = torch.rand(hyper["batch_size"], *shape) * hyper["step_time"]

        currents = synapse.current_.read(2) * torch.exp(
            (-(hyper["step_time"] - selector)) / hyper["time_constant"]
        )

        assert torch.all((currents - synapse.current_at(selector)).abs() <= 1e-7)


class TestDoubleExponentialCurrent:

    @staticmethod
    def random_hyper(delayed=False, inplace=False):
        hyper = {}
        hyper["step_time"] = random.uniform(0.6, 1.8)
        hyper["spike_charge"] = random.uniform(20, 30)
        hyper["tc_decay"] = random.uniform(40, 60)
        hyper["tc_rise"] = hyper["tc_decay"] / random.uniform(7, 12)
        hyper["delay"] = (
            random.uniform(hyper["step_time"], 3 * hyper["step_time"])
            if delayed
            else 0.0
        )
        hyper["spike_interp_mode"] = ("nearest", "previous")[random.randint(0, 1)]
        hyper["interp_tol"] = 0.0
        hyper["current_overbound"] = 0.0
        hyper["spike_overbound"] = False
        hyper["batch_size"] = random.randint(1, 9)
        hyper["inplace"] = inplace
        return hyper

    def test_batchsz(self):
        shape = random_shape()
        hyper = self.random_hyper(True)
        synapse = DoubleExponentialCurrent(shape, **hyper)

        validate_batchsz(synapse, hyper["batch_size"])

    def test_shape(self):
        shape = random_shape()
        hyper = self.random_hyper(True)
        synapse = DoubleExponentialCurrent(shape, **hyper)

        validate_shape(synapse, shape)

    def test_count(self):
        shape = random_shape()
        hyper = self.random_hyper(True)
        synapse = DoubleExponentialCurrent(shape, **hyper)

        validate_count(synapse, shape)

    def test_batchedshape(self):
        shape = random_shape()
        hyper = self.random_hyper(True)
        synapse = DoubleExponentialCurrent(shape, **hyper)

        validate_batchedshape(synapse, hyper["batch_size"], shape)

    def test_duration(self):
        shape = random_shape()
        hyper = self.random_hyper(True)
        synapse = DoubleExponentialCurrent(shape, **hyper)

        validate_duration(synapse, hyper["delay"])

    @pytest.mark.parametrize(
        "delayed",
        (True, False),
        ids=("delayed", "undelayed"),
    )
    @pytest.mark.parametrize(
        "inplace",
        (True, False),
        ids=("inplace", "realloc"),
    )
    def test_current_integration(self, delayed, inplace):
        shape = random_shape(maxdims=5)
        hyper = self.random_hyper(delayed, inplace)
        synapse = DoubleExponentialCurrent(shape, **hyper)

        pos_currents = torch.zeros(hyper["batch_size"], *shape)
        neg_currents = torch.zeros(hyper["batch_size"], *shape)

        for _ in range(100):
            spikes = torch.rand(hyper["batch_size"], *shape) < 0.1
            res = synapse(spikes)

            pos_currents = pos_currents * math.exp(
                -hyper["step_time"] / hyper["tc_decay"]
            )
            pos_currents = (
                pos_currents
                + (hyper["spike_charge"] / (hyper["tc_decay"] - hyper["tc_rise"]))
                * spikes
            )

            neg_currents = neg_currents * math.exp(
                -hyper["step_time"] / hyper["tc_rise"]
            )
            neg_currents = (
                neg_currents
                + (hyper["spike_charge"] / (hyper["tc_decay"] - hyper["tc_rise"]))
                * spikes
            )

            currents = pos_currents - neg_currents

            assert torch.all((res - currents).abs() <= 1e-7)
            assert torch.all((synapse.current - currents).abs() <= 1e-7)

    @pytest.mark.parametrize(
        "delayed",
        (True, False),
        ids=("delayed", "undelayed"),
    )
    @pytest.mark.parametrize(
        "inplace",
        (True, False),
        ids=("inplace", "realloc"),
    )
    def test_current_overbounding(self, delayed, inplace):
        shape = random_shape(maxdims=5)
        hyper = self.random_hyper(delayed, inplace) | {"current_overbound": -20.0}
        synapse = DoubleExponentialCurrent(shape, **hyper)
        selector = (torch.rand(hyper["batch_size"], *shape) > 0.5) * (
            hyper["delay"] + hyper["step_time"]
        )

        for _ in range(30):
            _ = synapse(torch.rand(hyper["batch_size"], *shape) < 0.2)

        assert torch.all(
            synapse.current_at(selector)[selector == 0]
            == synapse.current[selector == 0]
        )
        assert torch.all(
            synapse.current_at(selector)[selector != 0] == hyper["current_overbound"]
        )
        assert torch.all(
            synapse.current_at(-selector)[selector != 0] == hyper["current_overbound"]
        )

    @pytest.mark.parametrize(
        "inplace",
        (True, False),
        ids=("inplace", "realloc"),
    )
    def test_current_interpolation(self, inplace):
        shape = random_shape(maxdims=5)
        hyper = self.random_hyper(True, inplace)
        hyper = hyper | {"delay": hyper["step_time"]}
        synapse = DoubleExponentialCurrent(shape, **hyper)

        for _ in range(30):
            _ = synapse(torch.rand(hyper["batch_size"], *shape) < 0.2)

        selector = torch.rand(hyper["batch_size"], *shape) * hyper["step_time"]

        currents = (
            synapse.pos_current_.read(2)
            * torch.exp((-(hyper["step_time"] - selector)) / hyper["tc_decay"])
        ) - (
            synapse.neg_current_.read(2)
            * torch.exp((-(hyper["step_time"] - selector)) / hyper["tc_rise"])
        )

        assert torch.all((currents - synapse.current_at(selector)).abs() <= 1e-7)
