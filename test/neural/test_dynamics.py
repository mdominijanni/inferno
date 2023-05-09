from functools import reduce
import random
import pytest
import torch

import sys
sys.path.insert(0, '../..')

from inferno.neural import PIFDynamics, LIFDynamics, AdaptiveLIFDynamics


@pytest.fixture(scope='function')
def shape():
    return (4, 3)


@pytest.fixture(scope='function')
def step_time():
    return 1.0


@pytest.fixture(scope='function')
def single_batch_size():
    return 1


@pytest.fixture(scope='function')
def mini_batch_size():
    return 7


class TestLinearIFDynamics:

    @pytest.fixture(scope='function')
    def v_rest(self):
        return -70.0

    @pytest.fixture(scope='function')
    def v_reset(self):
        return -65.0

    @pytest.fixture(scope='function')
    def v_thresh(self):
        return -50.0

    @pytest.fixture(scope='function')
    def ts_refrac(self):
        return 2

    @pytest.fixture(scope='function')
    def tc_membrane(self):
        return 20.0

    @pytest.fixture(scope='function')
    def timesteps(self):
        return 200

    @pytest.fixture(scope='function')
    def inputs(self, shape, single_batch_size, timesteps):
        bshape = [single_batch_size]
        bshape.extend(shape)
        return [torch.rand(bshape) * 5 for _ in range(timesteps)]

    @pytest.fixture(scope='function')
    def batched_inputs(self, shape, mini_batch_size, timesteps):
        bshape = [mini_batch_size]
        bshape.extend(shape)
        return [torch.rand(bshape) * 5 for _ in range(timesteps)]

    class TestPIFDynamics:

        @pytest.fixture(scope='function')
        def make_dynamics(self, shape, step_time, v_rest, v_reset, v_thresh, ts_refrac):
            return lambda batch_size: PIFDynamics(shape=shape, step_time=step_time, v_rest=v_rest, v_reset=v_reset, v_threshold=v_thresh, ts_abs_refrac=ts_refrac, batch_size=batch_size)

        def test_size_getter(self, make_dynamics, shape, single_batch_size):
            dynamics = make_dynamics(single_batch_size)
            assert dynamics.size == reduce(lambda a, b: a * b, shape)

        def test_shape_getter(self, make_dynamics, shape, single_batch_size):
            dynamics = make_dynamics(single_batch_size)
            assert dynamics.shape == shape

        def test_batched_shape_getter(self, make_dynamics, shape, single_batch_size):
            dynamics = make_dynamics(single_batch_size)
            bshape = [single_batch_size]
            bshape.extend(shape)
            assert dynamics.batched_shape == tuple(bshape)

        def test_batch_size_getter(self, make_dynamics, single_batch_size):
            dynamics = make_dynamics(single_batch_size)
            assert dynamics.batch_size == single_batch_size

        def test_batch_size_setter(self, make_dynamics, single_batch_size, mini_batch_size):
            dynamics = make_dynamics(single_batch_size)
            assert dynamics.batch_size == single_batch_size
            dynamics.batch_size = mini_batch_size
            assert dynamics.batch_size == mini_batch_size

        def test_clear(self, make_dynamics, single_batch_size):
            dynamics = make_dynamics(single_batch_size)
            dynamics.v_membranes.add_(3.0)
            dynamics.ts_refrac_membranes.add_(1)
            dynamics.clear()
            assert torch.all(dynamics.v_membranes == dynamics.v_rest)
            assert torch.all(dynamics.ts_refrac_membranes == 0)

        def test_forward_unbatched(self, make_dynamics, v_reset, v_thresh, ts_refrac, single_batch_size, inputs):
            dynamics = make_dynamics(single_batch_size)
            voltages = dynamics.v_membranes.clone().detach()
            refracs = dynamics.ts_refrac_membranes.clone().detach()
            for t in inputs:
                voltages = voltages.masked_fill(refracs == ts_refrac, v_reset)
                voltages = voltages + (refracs == 0) * t
                spikes = (voltages >= v_thresh).masked_fill(refracs != 0, 0)
                refracs = torch.clamp_min(refracs - 1, 0)
                refracs = refracs.masked_fill(spikes, ts_refrac)
                res_spikes = dynamics(t)
                assert torch.all(voltages == dynamics.v_membranes)
                assert torch.all(refracs == dynamics.ts_refrac_membranes)
                assert torch.all(spikes == res_spikes)

        def test_forward_batched(self, make_dynamics, v_reset, v_thresh, ts_refrac, mini_batch_size, batched_inputs):
            dynamics = make_dynamics(mini_batch_size)
            voltages = dynamics.v_membranes.clone().detach()
            refracs = dynamics.ts_refrac_membranes.clone().detach()
            for t in batched_inputs:
                voltages = voltages.masked_fill(refracs == ts_refrac, v_reset)
                voltages = voltages + (refracs == 0) * t
                spikes = (voltages >= v_thresh).masked_fill(refracs != 0, 0)
                refracs = torch.clamp_min(refracs - 1, 0)
                refracs = refracs.masked_fill(spikes, ts_refrac)
                res_spikes = dynamics(t)
                assert torch.all(voltages == dynamics.v_membranes)
                assert torch.all(refracs == dynamics.ts_refrac_membranes)
                assert torch.all(spikes == res_spikes)

    class TestLIFDynamics:

        @pytest.fixture(scope='function')
        def make_dynamics(self, shape, step_time, v_rest, v_reset, v_thresh, ts_refrac, tc_membrane):
            return lambda batch_size: LIFDynamics(shape=shape, step_time=step_time, v_rest=v_rest, v_reset=v_reset, v_threshold=v_thresh, ts_abs_refrac=ts_refrac, tc_membrane=tc_membrane, batch_size=batch_size)

        def test_size_getter(self, make_dynamics, shape, single_batch_size):
            dynamics = make_dynamics(single_batch_size)
            assert dynamics.size == reduce(lambda a, b: a * b, shape)

        def test_shape_getter(self, make_dynamics, shape, single_batch_size):
            dynamics = make_dynamics(single_batch_size)
            assert dynamics.shape == shape

        def test_batched_shape_getter(self, make_dynamics, shape, single_batch_size):
            dynamics = make_dynamics(single_batch_size)
            bshape = [single_batch_size]
            bshape.extend(shape)
            assert dynamics.batched_shape == tuple(bshape)

        def test_batch_size_getter(self, make_dynamics, shape, single_batch_size):
            dynamics = make_dynamics(single_batch_size)
            assert dynamics.batch_size == single_batch_size

        def test_batch_size_setter(self, make_dynamics, shape, single_batch_size, mini_batch_size):
            dynamics = make_dynamics(single_batch_size)
            assert dynamics.batch_size == single_batch_size
            dynamics.batch_size = mini_batch_size
            assert dynamics.batch_size == mini_batch_size

        def test_clear(self, make_dynamics, shape, single_batch_size):
            dynamics = make_dynamics(single_batch_size)
            dynamics.v_membranes.add_(3.0)
            dynamics.ts_refrac_membranes.add_(1)
            dynamics.clear()
            assert torch.all(dynamics.v_membranes == dynamics.v_rest)
            assert torch.all(dynamics.ts_refrac_membranes == 0)

        def test_forward_unbatched(self, make_dynamics, step_time, v_rest, v_reset, v_thresh, ts_refrac, tc_membrane, single_batch_size, inputs):
            dynamics = make_dynamics(single_batch_size)
            voltages = dynamics.v_membranes.clone().detach()
            refracs = dynamics.ts_refrac_membranes.clone().detach()
            for t in inputs:
                voltages = voltages.masked_fill(refracs == ts_refrac, v_reset)
                voltages = (voltages - v_rest) * torch.exp(-torch.tensor(step_time / tc_membrane)) + v_rest + (refracs == 0) * t
                spikes = (voltages >= v_thresh).masked_fill(refracs != 0, 0)
                refracs = torch.clamp_min(refracs - 1, 0)
                refracs = refracs.masked_fill(spikes, ts_refrac)
                res_spikes = dynamics(t)
                assert torch.all(voltages == dynamics.v_membranes)
                assert torch.all(refracs == dynamics.ts_refrac_membranes)
                assert torch.all(spikes == res_spikes)

        def test_forward_batched(self, make_dynamics, step_time, v_rest, v_reset, v_thresh, ts_refrac, tc_membrane, mini_batch_size, batched_inputs):
            dynamics = make_dynamics(mini_batch_size)
            voltages = dynamics.v_membranes.clone().detach()
            refracs = dynamics.ts_refrac_membranes.clone().detach()
            for t in batched_inputs:
                voltages = voltages.masked_fill(refracs == ts_refrac, v_reset)
                voltages = (voltages - v_rest) * torch.exp(-torch.tensor(step_time / tc_membrane)) + v_rest + (refracs == 0) * t
                spikes = (voltages >= v_thresh).masked_fill(refracs != 0, 0)
                refracs = torch.clamp_min(refracs - 1, 0)
                refracs = refracs.masked_fill(spikes, ts_refrac)
                res_spikes = dynamics(t)
                assert torch.all(voltages == dynamics.v_membranes)
                assert torch.all(refracs == dynamics.ts_refrac_membranes)
                assert torch.all(spikes == res_spikes)

    class TestAdaptiveLIFDynamics:

        @pytest.fixture(scope='function')
        def tc_theta(self):
            return 30.0

        @pytest.fixture(scope='function')
        def theta_plus(self):
            return 0.1

        @pytest.fixture(scope='function')
        def make_dynamics(self, shape, step_time, v_rest, v_reset, v_thresh, ts_refrac, tc_membrane, tc_theta, theta_plus):
            return lambda batch_size: AdaptiveLIFDynamics(shape=shape, step_time=step_time, v_rest=v_rest, v_reset=v_reset, v_threshold=v_thresh, ts_abs_refrac=ts_refrac, tc_membrane=tc_membrane, batch_size=batch_size, tc_theta=tc_theta, theta_plus=theta_plus)

        def test_size_getter(self, make_dynamics, shape, single_batch_size):
            dynamics = make_dynamics(single_batch_size)
            assert dynamics.size == reduce(lambda a, b: a * b, shape)

        def test_shape_getter(self, make_dynamics, shape, single_batch_size):
            dynamics = make_dynamics(single_batch_size)
            assert dynamics.shape == shape

        def test_batched_shape_getter(self, make_dynamics, shape, single_batch_size):
            dynamics = make_dynamics(single_batch_size)
            bshape = [single_batch_size]
            bshape.extend(shape)
            assert dynamics.batched_shape == tuple(bshape)

        def test_batch_size_getter(self, make_dynamics, shape, single_batch_size):
            dynamics = make_dynamics(single_batch_size)
            assert dynamics.batch_size == single_batch_size

        def test_batch_size_setter(self, make_dynamics, shape, single_batch_size, mini_batch_size):
            dynamics = make_dynamics(single_batch_size)
            assert dynamics.batch_size == single_batch_size
            dynamics.batch_size = mini_batch_size
            assert dynamics.batch_size == mini_batch_size

        def test_clear(self, make_dynamics, shape, single_batch_size):
            dynamics = make_dynamics(single_batch_size)
            dynamics.v_membranes.add_(3.0)
            dynamics.ts_refrac_membranes.add_(1)
            dynamics.clear()
            assert torch.all(dynamics.v_membranes == dynamics.v_rest)
            assert torch.all(dynamics.ts_refrac_membranes == 0)

        def test_forward_unbatched(self, make_dynamics, step_time, v_rest, v_reset, v_thresh, ts_refrac, tc_membrane, single_batch_size, inputs, tc_theta, theta_plus):
            dynamics = make_dynamics(single_batch_size)
            voltages = dynamics.v_membranes.clone().detach()
            refracs = dynamics.ts_refrac_membranes.clone().detach()
            theta = dynamics.theta.clone().detach()
            for t in inputs:
                if random.randint(0, 1):
                    theta_mode = True
                    dynamics.train()
                else:
                    theta_mode = False
                    dynamics.eval()
                voltages = voltages.masked_fill(refracs == ts_refrac, v_reset)
                voltages = (voltages - v_rest) * torch.exp(-torch.tensor(step_time / tc_membrane)) + v_rest + (refracs == 0) * t
                if theta_mode:
                    theta = theta * torch.exp(-torch.tensor(step_time / tc_theta))
                spikes = (voltages >= v_thresh + theta).masked_fill(refracs != 0, 0)
                if theta_mode:
                    theta = theta + theta_plus * spikes
                refracs = torch.clamp_min(refracs - 1, 0)
                refracs = refracs.masked_fill(spikes, ts_refrac)
                res_spikes = dynamics(t)
                assert torch.all(voltages == dynamics.v_membranes)
                assert torch.all(refracs == dynamics.ts_refrac_membranes)
                assert torch.all(spikes == res_spikes)

        def test_forward_batched(self, make_dynamics, step_time, v_rest, v_reset, v_thresh, ts_refrac, tc_membrane, mini_batch_size, batched_inputs, tc_theta, theta_plus):
            dynamics = make_dynamics(mini_batch_size)
            voltages = dynamics.v_membranes.clone().detach()
            refracs = dynamics.ts_refrac_membranes.clone().detach()
            theta = dynamics.theta.clone().detach()
            for t in batched_inputs:
                if random.randint(0, 1):
                    theta_mode = True
                    dynamics.train()
                else:
                    theta_mode = False
                    dynamics.eval()
                voltages = voltages.masked_fill(refracs == ts_refrac, v_reset)
                voltages = (voltages - v_rest) * torch.exp(-torch.tensor(step_time / tc_membrane)) + v_rest + (refracs == 0) * t
                if theta_mode:
                    theta = theta * torch.exp(-torch.tensor(step_time / tc_theta))
                spikes = (voltages >= v_thresh + theta).masked_fill(refracs != 0, 0)
                if theta_mode:
                    theta = theta + theta_plus * spikes
                refracs = torch.clamp_min(refracs - 1, 0)
                refracs = refracs.masked_fill(spikes, ts_refrac)
                res_spikes = dynamics(t)
                assert torch.all(voltages == dynamics.v_membranes)
                assert torch.all(refracs == dynamics.ts_refrac_membranes)
                assert torch.all(spikes == res_spikes)
