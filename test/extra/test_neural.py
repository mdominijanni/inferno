import math
import pytest
import random
import torch

from inferno.extra import ExactNeuron


def random_shape(nd_min=1, nd_max=9, sz_min=1, sz_max=9):
    return tuple(
        random.randint(nd_min, nd_max) for _ in range(random.randint(sz_min, sz_max))
    )


class TestExactNeuron:

    @staticmethod
    def random_hyper():
        return {
            "step_time": random.uniform(0.5, 1.5),
            "rest_v": random.uniform(-70.0, -60.0),
            "thresh_v": random.uniform(-50.0, -40.0),
            "batch_size": random.randint(1, 9),
        }

    @pytest.fixture
    def shape(self):
        return random_shape(1, 4)

    @pytest.fixture
    def hyper(self):
        return self.random_hyper()

    def test_batchsz(self, shape, hyper):
        neuron = ExactNeuron(shape, **hyper)
        assert neuron.batchsz == hyper["batch_size"]

    def test_shape(self, shape, hyper):
        neuron = ExactNeuron(shape, **hyper)
        assert neuron.shape == shape

    def test_count(self, shape, hyper):
        neuron = ExactNeuron(shape, **hyper)
        assert neuron.count == math.prod(shape)

    def test_batchedshape(self, shape, hyper):
        neuron = ExactNeuron(shape, **hyper)
        assert neuron.batchedshape == (hyper["batch_size"],) + shape

    def test_spikes(self, shape, hyper):
        neuron = ExactNeuron(shape, **hyper)
        batchshp = (100, hyper["batch_size"]) + shape
        inputs = torch.randn(batchshp) - torch.rand(batchshp)

        for x in inputs:
            res = neuron(x)
            assert torch.all(neuron.spike == (x > 0))
            assert torch.all(res == (x > 0))

    def test_voltages(self, shape, hyper):
        neuron = ExactNeuron(shape, **hyper)
        batchshp = (100, hyper["batch_size"]) + shape
        inputs = torch.randn(batchshp) - torch.rand(batchshp)

        for x in inputs:
            _ = neuron(x)
            assert torch.all(neuron.voltage[x > 0] == hyper["thresh_v"])
            assert torch.all(neuron.voltage[x <= 0] == hyper["rest_v"])

    def test_refracs(self, shape, hyper):
        neuron = ExactNeuron(shape, **hyper)
        assert torch.all(neuron.refrac == 0)

        neuron.refrac = torch.rand_like(neuron.refrac)
        assert torch.all(neuron.refrac == 0)
