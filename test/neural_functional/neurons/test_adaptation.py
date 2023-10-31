import pytest
import math
import random
import torch

import sys

sys.path.insert(0, "../../..")

from inferno.neural.functional import (
    adaptive_currents_linear,
    adaptive_thresholds_linear_voltage,
    adaptive_thresholds_linear_spike,
)


@pytest.fixture(scope="class")
def num_parameters():
    return 15


@pytest.fixture(scope="class")
def num_batches():
    return 7


@pytest.fixture(scope="class")
def primitive_shape():
    return None


@pytest.fixture(scope="class")
def singleton_shape():
    return []


@pytest.fixture(scope="class")
def vector_1_shape():
    return [1]


@pytest.fixture(scope="class")
def vector_K_shape(num_parameters):
    return [num_parameters]


@pytest.fixture(scope="class")
def dynamics_shape():
    return [18, 19, 20]


@pytest.fixture(scope="class")
def batch_dynamics_shape(num_batches):
    return [num_batches, 18, 19, 20]


@pytest.fixture(scope="class")
def dynamics_like_2d_shape():
    return [19, 20]


@pytest.fixture(scope="class")
def adaptation_shape(num_parameters):
    return [18, 19, 20, num_parameters]


@pytest.fixture(scope="class")
def adaptation_like_2d_shape(num_parameters):
    return [20, num_parameters]


@pytest.fixture(scope="class")
def batch_adaptation_shape(num_parameters, num_batches):
    return [num_batches, 18, 19, 20, num_parameters]


@pytest.fixture(scope="class")
def float_error_tolerance():
    return 3e-6


class TestAdaptiveCurrentsLinear:
    @pytest.mark.parametrize(
        "postsyn_spikes_shape,voltages_shape,refracs_shape,target_adaptations_shape",
        [
            ["batch_dynamics_shape"] * 3 + ["batch_adaptation_shape"],
            ["dynamics_shape"] * 3 + ["adaptation_shape"],
            ["batch_dynamics_shape"] * 2
            + ["primitive_shape", "batch_adaptation_shape"],
            ["dynamics_shape"] * 2 + ["primitive_shape", "adaptation_shape"],
        ],
        ids=["dynBatched", "dynNoBatch", "dynBatchNoRefrac", "dynNoBatchNoRefrac"],
    )
    @pytest.mark.parametrize("adaptations_shape", ["adaptation_shape"], ids=["adapt"])
    @pytest.mark.parametrize(
        "rest_v_shape",
        [
            "primitive_shape",
            "singleton_shape",
            "vector_1_shape",
            "dynamics_shape",
            "dynamics_like_2d_shape",
        ],
        ids=[
            "Vrest.primitive",
            "Vrest.singleton",
            "Vrest.vec1",
            "Vrest.dyn",
            "Vrest.dyn2D",
        ],
    )
    @pytest.mark.parametrize(
        "step_time_shape",
        [
            "primitive_shape",
            "singleton_shape",
            "vector_1_shape",
            "vector_K_shape",
            "adaptation_shape",
            "adaptation_like_2d_shape",
        ],
        ids=[
            "DT.primitive",
            "DT.singleton",
            "DT.vec1",
            "DT.vecK",
            "DT.adapt",
            "DT.adapt2D",
        ],
    )
    @pytest.mark.parametrize(
        "time_constant_shape",
        [
            "primitive_shape",
            "singleton_shape",
            "vector_1_shape",
            "vector_K_shape",
            "adaptation_shape",
            "adaptation_like_2d_shape",
        ],
        ids=[
            "TC.primitive",
            "TC.singleton",
            "TC.vec1",
            "TC.vecK",
            "TC.adapt",
            "TC.adapt2D",
        ],
    )
    @pytest.mark.parametrize(
        "voltage_coupling_shape",
        [
            "primitive_shape",
            "singleton_shape",
            "vector_1_shape",
            "vector_K_shape",
            "adaptation_shape",
            "adaptation_like_2d_shape",
        ],
        ids=[
            "VC.primitive",
            "VC.singleton",
            "VC.vec1",
            "VC.vecK",
            "VC.adapt",
            "VC.adapt2D",
        ],
    )
    @pytest.mark.parametrize(
        "spike_increment_shape",
        [
            "primitive_shape",
            "singleton_shape",
            "vector_1_shape",
            "vector_K_shape",
            "adaptation_shape",
            "adaptation_like_2d_shape",
        ],
        ids=[
            "Incr.primitive",
            "Incr.singleton",
            "Incr.vec1",
            "Incr.vecK",
            "Incr.adapt",
            "Incr.adapt2D",
        ],
    )
    def test_shape_compatibility(
        self,
        target_adaptations_shape,
        adaptations_shape,
        postsyn_spikes_shape,
        voltages_shape,
        rest_v_shape,
        step_time_shape,
        time_constant_shape,
        voltage_coupling_shape,
        spike_increment_shape,
        refracs_shape,
        request,
    ):
        target_adaptations_shape = request.getfixturevalue(target_adaptations_shape)
        adaptations_shape = request.getfixturevalue(adaptations_shape)
        postsyn_spikes_shape = request.getfixturevalue(postsyn_spikes_shape)
        voltages_shape = request.getfixturevalue(voltages_shape)
        rest_v_shape = request.getfixturevalue(rest_v_shape)
        step_time_shape = request.getfixturevalue(step_time_shape)
        time_constant_shape = request.getfixturevalue(time_constant_shape)
        voltage_coupling_shape = request.getfixturevalue(voltage_coupling_shape)
        spike_increment_shape = request.getfixturevalue(spike_increment_shape)
        refracs_shape = request.getfixturevalue(refracs_shape)

        adaptations = torch.rand(adaptations_shape) * 10
        postsyn_spikes = torch.rand(postsyn_spikes_shape) > 0.3
        voltages = torch.rand(voltages_shape) * -60
        rest_v = (
            random.uniform(-60, -55)
            if rest_v_shape is None
            else torch.rand(rest_v_shape) * -60
        )
        step_time = (
            random.uniform(0.1, 3.0)
            if step_time_shape is None
            else torch.rand(step_time_shape) * 2
        )
        time_constant = (
            random.uniform(10, 30)
            if time_constant_shape is None
            else torch.rand(time_constant_shape) * 30
        )
        voltage_coupling = (
            random.uniform(0.1, 3.0)
            if voltage_coupling_shape is None
            else torch.rand(voltage_coupling_shape) * 2
        )
        spike_increment = (
            random.uniform(5, 15)
            if spike_increment_shape is None
            else torch.rand(spike_increment_shape) + 5 * 3
        )
        refracs = None if refracs_shape is None else torch.randint(0, 3, refracs_shape)

        adaptations_fd = adaptations.clone().detach().expand(*target_adaptations_shape)
        postsyn_spikes_fd = postsyn_spikes.clone().detach()
        voltages_fd = voltages.clone().detach()
        rest_v_fd = (
            rest_v
            if not isinstance(rest_v, torch.Tensor)
            else rest_v.clone().detach().expand(*voltages_shape)
        )
        step_time_fd = (
            step_time
            if not isinstance(step_time, torch.Tensor)
            else step_time.clone().detach().expand(*target_adaptations_shape)
        )
        time_constant_fd = (
            time_constant
            if not isinstance(time_constant, torch.Tensor)
            else time_constant.clone().detach().expand(*target_adaptations_shape)
        )
        voltage_coupling_fd = (
            voltage_coupling
            if not isinstance(voltage_coupling, torch.Tensor)
            else voltage_coupling.clone().detach().expand(*target_adaptations_shape)
        )
        spike_increment_fd = (
            spike_increment
            if not isinstance(spike_increment, torch.Tensor)
            else spike_increment.clone().detach().expand(*target_adaptations_shape)
        )
        refracs_fd = (
            refracs if not isinstance(refracs, torch.Tensor) else refracs.clone()
        )

        res = adaptive_currents_linear(
            adaptations=adaptations,
            postsyn_spikes=postsyn_spikes,
            voltages=voltages,
            rest_v=rest_v,
            step_time=step_time,
            time_constant=time_constant,
            voltage_coupling=voltage_coupling,
            spike_increment=spike_increment,
            refracs=refracs,
        )

        res_fd = adaptive_currents_linear(
            adaptations=adaptations_fd,
            postsyn_spikes=postsyn_spikes_fd,
            voltages=voltages_fd,
            rest_v=rest_v_fd,
            step_time=step_time_fd,
            time_constant=time_constant_fd,
            voltage_coupling=voltage_coupling_fd,
            spike_increment=spike_increment_fd,
            refracs=refracs_fd,
        )

        assert tuple(res.shape) == tuple(target_adaptations_shape)
        assert torch.all(res_fd == res)

    def test_nospike_update(
        self, adaptation_shape, dynamics_shape, float_error_tolerance
    ):
        adaptations = torch.rand(adaptation_shape) * 10
        postsyn_spikes = torch.zeros(dynamics_shape).bool()
        voltages = torch.rand(dynamics_shape) * -60
        rest_v = -60
        step_time = 0.75
        time_constant = 30.0
        voltage_coupling = 0.6
        spike_increment = 7.6
        refracs = None

        res = adaptive_currents_linear(
            adaptations=adaptations.clone().detach(),
            postsyn_spikes=postsyn_spikes.clone().detach(),
            voltages=voltages.clone().detach(),
            rest_v=rest_v,
            step_time=step_time,
            time_constant=time_constant,
            voltage_coupling=voltage_coupling,
            spike_increment=spike_increment,
            refracs=refracs,
        )

        assert torch.all(
            (res - adaptations)
            - (
                (step_time / time_constant)
                * (voltage_coupling * (voltages - rest_v).unsqueeze(-1) - adaptations)
            )
            < float_error_tolerance
        )

    def test_nospike_update_no_vc_coupling(
        self, adaptation_shape, dynamics_shape, float_error_tolerance
    ):
        adaptations = torch.rand(adaptation_shape) * 10
        postsyn_spikes = torch.zeros(dynamics_shape).bool()
        voltages = torch.rand(dynamics_shape) * -60
        rest_v = -60
        step_time = 0.75
        time_constant = 30.0
        voltage_coupling = 0.0
        spike_increment = 7.6
        refracs = None

        res = adaptive_currents_linear(
            adaptations=adaptations.clone().detach(),
            postsyn_spikes=postsyn_spikes.clone().detach(),
            voltages=voltages.clone().detach(),
            rest_v=rest_v,
            step_time=step_time,
            time_constant=time_constant,
            voltage_coupling=voltage_coupling,
            spike_increment=spike_increment,
            refracs=refracs,
        )

        assert torch.all(
            (res - adaptations) - ((step_time / time_constant) * -adaptations)
            < float_error_tolerance
        )

    def test_increment(self, adaptation_shape, dynamics_shape, float_error_tolerance):
        adaptations = torch.rand(adaptation_shape) * 10
        postsyn_spikes = torch.rand(dynamics_shape) > 0.5
        voltages = torch.rand(dynamics_shape) * -60
        rest_v = -60
        step_time = 0.75
        time_constant = 30.0
        voltage_coupling = 0.6
        spike_increment = 7.6
        refracs = None

        res = adaptive_currents_linear(
            adaptations=adaptations.clone().detach(),
            postsyn_spikes=postsyn_spikes.clone().detach(),
            voltages=voltages.clone().detach(),
            rest_v=rest_v,
            step_time=step_time,
            time_constant=time_constant,
            voltage_coupling=voltage_coupling,
            spike_increment=spike_increment,
            refracs=refracs,
        )

        assert torch.all(
            res
            - (
                adaptations
                + (
                    (step_time / time_constant)
                    * (
                        voltage_coupling * (voltages - rest_v).unsqueeze(-1)
                        - adaptations
                    )
                )
                + (spike_increment * postsyn_spikes.unsqueeze(-1))
            )
            < float_error_tolerance
        )

    def test_refrac_blocking(
        self, adaptation_shape, dynamics_shape, float_error_tolerance
    ):
        adaptations = torch.rand(adaptation_shape) * 10
        postsyn_spikes = torch.rand(dynamics_shape) > 0.5
        voltages = torch.rand(dynamics_shape) * -60
        rest_v = -60.0
        step_time = 0.75
        time_constant = 30.0
        voltage_coupling = 0.6
        spike_increment = 7.6
        refracs = (torch.rand(dynamics_shape) > 0.5).int()

        res = adaptive_currents_linear(
            adaptations=adaptations.clone().detach(),
            postsyn_spikes=postsyn_spikes.clone().detach(),
            voltages=voltages.clone().detach(),
            rest_v=rest_v,
            step_time=step_time,
            time_constant=time_constant,
            voltage_coupling=voltage_coupling,
            spike_increment=spike_increment,
            refracs=refracs,
        )

        assert torch.all(
            res
            - (
                torch.where(
                    refracs.unsqueeze(-1) == 0,
                    adaptations
                    + (
                        (step_time / time_constant)
                        * (
                            voltage_coupling * (voltages - rest_v).unsqueeze(-1)
                            - adaptations
                        )
                    ),
                    adaptations,
                )
                + (spike_increment * postsyn_spikes.unsqueeze(-1))
            )
            < float_error_tolerance
        )

    def test_non_destructive(self, adaptation_shape, dynamics_shape):
        adaptations = torch.rand(adaptation_shape) * 10
        postsyn_spikes = torch.rand(dynamics_shape) > 0.5
        voltages = torch.rand(dynamics_shape) * -60
        rest_v = torch.tensor(-60.0)
        step_time = torch.tensor(0.75)
        time_constant = torch.tensor(30.0)
        voltage_coupling = torch.tensor(0.6)
        spike_increment = torch.tensor(7.6)
        refracs = (torch.rand(dynamics_shape) > 0.5).int()

        adaptations_orig = adaptations.clone().detach()
        postsyn_spikes_orig = postsyn_spikes.clone().detach()
        voltages_orig = voltages.clone().detach()
        rest_v_orig = rest_v.clone().detach()
        step_time_orig = step_time.clone().detach()
        time_constant_orig = time_constant.clone().detach()
        voltage_coupling_orig = voltage_coupling.clone().detach()
        spike_increment_orig = spike_increment.clone().detach()
        refracs_orig = refracs.clone().detach()

        _ = adaptive_currents_linear(
            adaptations=adaptations,
            postsyn_spikes=postsyn_spikes,
            voltages=voltages,
            rest_v=rest_v,
            step_time=step_time,
            time_constant=time_constant,
            voltage_coupling=voltage_coupling,
            spike_increment=spike_increment,
            refracs=refracs,
        )

        assert torch.all(adaptations == adaptations_orig)
        assert torch.all(postsyn_spikes == postsyn_spikes_orig)
        assert torch.all(voltages == voltages_orig)
        assert torch.all(rest_v == rest_v_orig)
        assert torch.all(step_time == step_time_orig)
        assert torch.all(time_constant == time_constant_orig)
        assert torch.all(voltage_coupling == voltage_coupling_orig)
        assert torch.all(spike_increment == spike_increment_orig)
        assert torch.all(refracs == refracs_orig)


class TestAdaptiveThresholdsLinearVoltage:
    @pytest.mark.parametrize(
        "postsyn_spikes_shape,voltages_shape,refracs_shape,target_adaptations_shape",
        [
            ["batch_dynamics_shape"] * 3 + ["batch_adaptation_shape"],
            ["dynamics_shape"] * 3 + ["adaptation_shape"],
            ["batch_dynamics_shape"] * 2
            + ["primitive_shape", "batch_adaptation_shape"],
            ["dynamics_shape"] * 2 + ["primitive_shape", "adaptation_shape"],
            ["primitive_shape"]
            + ["batch_dynamics_shape"] * 2
            + ["batch_adaptation_shape"],
            ["primitive_shape"] + ["dynamics_shape"] * 2 + ["adaptation_shape"],
            [
                "primitive_shape",
                "batch_dynamics_shape",
                "primitive_shape",
                "batch_adaptation_shape",
            ],
            [
                "primitive_shape",
                "dynamics_shape",
                "primitive_shape",
                "adaptation_shape",
            ],
        ],
        ids=[
            "dynBatched",
            "dynNoBatch",
            "dynBatchNoRefrac",
            "dynNoBatchNoRefrac",
            "dynBatchedNoPostsyn",
            "dynNoBatchNoPostsyn",
            "dynBatchNoRefracNoPostsyn",
            "dynNoBatchNoRefracNoPostsyn",
        ],
    )
    @pytest.mark.parametrize("adaptations_shape", ["adaptation_shape"], ids=["adapt"])
    @pytest.mark.parametrize(
        "rest_v_shape",
        [
            "primitive_shape",
            "singleton_shape",
            "vector_1_shape",
            "dynamics_shape",
            "dynamics_like_2d_shape",
        ],
        ids=[
            "Vrest.primitive",
            "Vrest.singleton",
            "Vrest.vec1",
            "Vrest.dyn",
            "Vrest.dyn2D",
        ],
    )
    @pytest.mark.parametrize(
        "step_time_shape",
        [
            "primitive_shape",
            "singleton_shape",
            "vector_1_shape",
            "vector_K_shape",
            "adaptation_shape",
            "adaptation_like_2d_shape",
        ],
        ids=[
            "DT.primitive",
            "DT.singleton",
            "DT.vec1",
            "DT.vecK",
            "DT.adapt",
            "DT.adapt2D",
        ],
    )
    @pytest.mark.parametrize(
        "adaptation_rate_shape",
        [
            "primitive_shape",
            "singleton_shape",
            "vector_1_shape",
            "vector_K_shape",
            "adaptation_shape",
            "adaptation_like_2d_shape",
        ],
        ids=[
            "AdaR.primitive",
            "AdaR.singleton",
            "AdaR.vec1",
            "AdaR.vecK",
            "AdaR.adapt",
            "AdaR.adapt2D",
        ],
    )
    @pytest.mark.parametrize(
        "rebound_rate_shape",
        [
            "primitive_shape",
            "singleton_shape",
            "vector_1_shape",
            "vector_K_shape",
            "adaptation_shape",
            "adaptation_like_2d_shape",
        ],
        ids=[
            "RebR.primitive",
            "RebR.singleton",
            "RebR.vec1",
            "RebR.vecK",
            "RebR.adapt",
            "RebR.adapt2D",
        ],
    )
    @pytest.mark.parametrize(
        "adaptation_reset_min_shape",
        [
            "primitive_shape",
            "singleton_shape",
            "vector_1_shape",
            "vector_K_shape",
            "adaptation_shape",
            "adaptation_like_2d_shape",
        ],
        ids=[
            "Reset.primitive",
            "Reset.singleton",
            "Reset.vec1",
            "Reset.vecK",
            "Reset.adapt",
            "Reset.adapt2D",
        ],
    )
    def test_shape_compatibility(
        self,
        target_adaptations_shape,
        adaptations_shape,
        voltages_shape,
        rest_v_shape,
        step_time_shape,
        adaptation_rate_shape,
        rebound_rate_shape,
        adaptation_reset_min_shape,
        postsyn_spikes_shape,
        refracs_shape,
        request,
    ):
        target_adaptations_shape = request.getfixturevalue(target_adaptations_shape)
        adaptations_shape = request.getfixturevalue(adaptations_shape)
        voltages_shape = request.getfixturevalue(voltages_shape)
        rest_v_shape = request.getfixturevalue(rest_v_shape)
        step_time_shape = request.getfixturevalue(step_time_shape)
        adaptation_rate_shape = request.getfixturevalue(adaptation_rate_shape)
        rebound_rate_shape = request.getfixturevalue(rebound_rate_shape)
        adaptation_reset_min_shape = request.getfixturevalue(adaptation_reset_min_shape)
        postsyn_spikes_shape = request.getfixturevalue(postsyn_spikes_shape)
        refracs_shape = request.getfixturevalue(refracs_shape)

        adaptations = torch.rand(adaptations_shape) * 10
        voltages = torch.rand(voltages_shape) * -60
        rest_v = (
            random.uniform(-60, -55)
            if rest_v_shape is None
            else torch.rand(rest_v_shape) * -60
        )
        step_time = (
            random.uniform(0.1, 3.0)
            if step_time_shape is None
            else torch.rand(step_time_shape) * 2
        )
        adaptation_rate = (
            random.uniform(0.0, 1.0)
            if adaptation_rate_shape is None
            else torch.rand(adaptation_rate_shape) * 30
        )
        rebound_rate = (
            random.uniform(0.0, 1.0)
            if rebound_rate_shape is None
            else torch.rand(rebound_rate_shape)
        )
        adaptation_reset_min = (
            random.uniform(5, 15)
            if adaptation_reset_min_shape is None
            else torch.rand(adaptation_reset_min_shape) * 10 + 5
        )
        refracs = None if refracs_shape is None else torch.randint(0, 3, refracs_shape)
        postsyn_spikes = (
            None
            if postsyn_spikes_shape is None
            else torch.rand(postsyn_spikes_shape) > 0.3
        )

        adaptations_fd = adaptations.clone().detach().expand(*target_adaptations_shape)
        voltages_fd = voltages.clone().detach()
        rest_v_fd = (
            rest_v
            if not isinstance(rest_v, torch.Tensor)
            else rest_v.clone().detach().expand(*voltages_shape)
        )
        step_time_fd = (
            step_time
            if not isinstance(step_time, torch.Tensor)
            else step_time.clone().detach().expand(*target_adaptations_shape)
        )
        adaptation_rate_fd = (
            adaptation_rate
            if not isinstance(adaptation_rate, torch.Tensor)
            else adaptation_rate.clone().detach().expand(*target_adaptations_shape)
        )
        rebound_rate_fd = (
            rebound_rate
            if not isinstance(rebound_rate, torch.Tensor)
            else rebound_rate.clone().detach().expand(*target_adaptations_shape)
        )
        adaptation_reset_min_fd = (
            adaptation_reset_min
            if not isinstance(adaptation_reset_min, torch.Tensor)
            else adaptation_reset_min.clone().detach().expand(*target_adaptations_shape)
        )
        postsyn_spikes_fd = (
            postsyn_spikes
            if not isinstance(postsyn_spikes, torch.Tensor)
            else postsyn_spikes.clone().detach()
        )
        refracs_fd = (
            refracs
            if not isinstance(refracs, torch.Tensor)
            else refracs.clone().detach()
        )

        res = adaptive_thresholds_linear_voltage(
            adaptations=adaptations,
            voltages=voltages,
            rest_v=rest_v,
            step_time=step_time,
            adaptation_rate=adaptation_rate,
            rebound_rate=rebound_rate,
            adaptation_reset_min=adaptation_reset_min,
            postsyn_spikes=postsyn_spikes,
            refracs=refracs,
        )

        res_fd = adaptive_thresholds_linear_voltage(
            adaptations=adaptations_fd,
            voltages=voltages_fd,
            rest_v=rest_v_fd,
            step_time=step_time_fd,
            adaptation_rate=adaptation_rate_fd,
            rebound_rate=rebound_rate_fd,
            adaptation_reset_min=adaptation_reset_min_fd,
            postsyn_spikes=postsyn_spikes_fd,
            refracs=refracs_fd,
        )

        assert tuple(res.shape) == tuple(target_adaptations_shape)
        assert torch.all(res_fd == res)

    def test_nospike_update(
        self, adaptation_shape, dynamics_shape, float_error_tolerance
    ):
        adaptations = torch.rand(adaptation_shape) * 10
        voltages = torch.rand(dynamics_shape) * -60
        rest_v = -60.0
        step_time = 0.75
        adaptation_rate = 0.5
        rebound_rate = 0.5
        adaptation_reset_min = None
        postsyn_spikes = None
        refracs = None

        res = adaptive_thresholds_linear_voltage(
            adaptations=adaptations.clone().detach(),
            voltages=voltages.clone().detach(),
            rest_v=rest_v,
            step_time=step_time,
            adaptation_rate=adaptation_rate,
            rebound_rate=rebound_rate,
            adaptation_reset_min=adaptation_reset_min,
            postsyn_spikes=postsyn_spikes,
            refracs=refracs,
        )
        assert torch.all(
            (res - adaptations)
            - (
                step_time
                * (
                    adaptation_rate * (voltages - rest_v).unsqueeze(-1)
                    - rebound_rate * adaptations
                )
            )
            < float_error_tolerance
        )

    def test_nospike_update_no_adapt(
        self, adaptation_shape, dynamics_shape, float_error_tolerance
    ):
        adaptations = torch.rand(adaptation_shape) * 10
        voltages = torch.rand(dynamics_shape) * -60
        rest_v = -60.0
        step_time = 0.75
        adaptation_rate = 0.0
        rebound_rate = 0.5
        adaptation_reset_min = None
        postsyn_spikes = None
        refracs = None

        res = adaptive_thresholds_linear_voltage(
            adaptations=adaptations,
            voltages=voltages,
            rest_v=rest_v,
            step_time=step_time,
            adaptation_rate=adaptation_rate,
            rebound_rate=rebound_rate,
            adaptation_reset_min=adaptation_reset_min,
            postsyn_spikes=postsyn_spikes,
            refracs=refracs,
        )

        assert torch.all(
            (res - adaptations) - (step_time * (-rebound_rate * adaptations))
            < float_error_tolerance
        )

    def test_nospike_update_no_rebound(
        self, adaptation_shape, dynamics_shape, float_error_tolerance
    ):
        adaptations = torch.rand(adaptation_shape) * 10
        voltages = torch.rand(dynamics_shape) * -60
        rest_v = -60.0
        step_time = 0.75
        adaptation_rate = 0.5
        rebound_rate = 0.0
        adaptation_reset_min = None
        postsyn_spikes = None
        refracs = None

        res = adaptive_thresholds_linear_voltage(
            adaptations=adaptations,
            voltages=voltages,
            rest_v=rest_v,
            step_time=step_time,
            adaptation_rate=adaptation_rate,
            rebound_rate=rebound_rate,
            adaptation_reset_min=adaptation_reset_min,
            postsyn_spikes=postsyn_spikes,
            refracs=refracs,
        )

        assert torch.all(
            (res - adaptations)
            - (step_time * (adaptation_rate * (voltages - rest_v).unsqueeze(-1)))
            < float_error_tolerance
        )

    def test_increment(self, adaptation_shape, dynamics_shape):
        adaptations = torch.rand(adaptation_shape) * 10
        voltages = torch.rand(dynamics_shape) * -60
        rest_v = -60.0
        step_time = 0.75
        adaptation_rate = 0.0
        rebound_rate = 0.0
        adaptation_reset_min = 7.5
        postsyn_spikes = torch.rand(dynamics_shape) > 0.5
        refracs = None

        res = adaptive_thresholds_linear_voltage(
            adaptations=adaptations.clone().detach(),
            voltages=voltages.clone().detach(),
            rest_v=rest_v,
            step_time=step_time,
            adaptation_rate=adaptation_rate,
            rebound_rate=rebound_rate,
            adaptation_reset_min=adaptation_reset_min,
            postsyn_spikes=postsyn_spikes.clone().detach(),
            refracs=refracs,
        )

        assert torch.all(
            res
            == adaptations.where(
                postsyn_spikes.unsqueeze(-1) == 0,
                adaptations.clamp_min(adaptation_reset_min),
            )
        )

    def test_refrac_blocking(
        self, adaptation_shape, dynamics_shape, float_error_tolerance
    ):
        adaptations = torch.rand(adaptation_shape) * 10
        voltages = torch.rand(dynamics_shape) * -60
        rest_v = -60.0
        step_time = 0.75
        adaptation_rate = 0.5
        rebound_rate = 0.5
        adaptation_reset_min = 7.5
        postsyn_spikes = torch.rand(dynamics_shape) > 0.5
        refracs = (torch.rand(dynamics_shape) > 0.5).int()

        res = adaptive_thresholds_linear_voltage(
            adaptations=adaptations.clone().detach(),
            voltages=voltages.clone().detach(),
            rest_v=rest_v,
            step_time=step_time,
            adaptation_rate=adaptation_rate,
            rebound_rate=rebound_rate,
            adaptation_reset_min=adaptation_reset_min,
            postsyn_spikes=postsyn_spikes.clone().detach(),
            refracs=refracs.clone().detach(),
        )

        res_comp = adaptations.where(
            refracs.unsqueeze(-1) > 0,
            adaptations
            + step_time
            * (
                adaptation_rate * (voltages - rest_v).unsqueeze(-1)
                - rebound_rate * adaptations
            ),
        )

        assert torch.all(
            res
            - (
                res_comp.where(
                    postsyn_spikes.unsqueeze(-1) == 0,
                    res_comp.clamp_min(adaptation_reset_min),
                )
            )
            < float_error_tolerance
        )

    def test_non_destructive(self, adaptation_shape, dynamics_shape):
        adaptations = torch.rand(adaptation_shape) * 10
        voltages = torch.rand(dynamics_shape) * -60
        rest_v = torch.tensor(-60.0)
        step_time = torch.tensor(0.75)
        adaptation_rate = torch.tensor(0.0)
        rebound_rate = torch.tensor(0.0)
        adaptation_reset_min = torch.tensor(7.5)
        postsyn_spikes = torch.rand(dynamics_shape) > 0.5
        refracs = (torch.rand(dynamics_shape) > 0.5).int()

        adaptations_orig = adaptations.clone().detach()
        voltages_orig = voltages.clone().detach()
        rest_v_orig = rest_v.clone().detach()
        step_time_orig = step_time.clone().detach()
        adaptation_rate_orig = adaptation_rate.clone().detach()
        rebound_rate_orig = rebound_rate.clone().detach()
        adaptation_reset_min_orig = adaptation_reset_min.clone().detach()
        postsyn_spikes_orig = postsyn_spikes.clone().detach()
        refracs_orig = refracs.clone().detach()

        _ = adaptive_thresholds_linear_voltage(
            adaptations=adaptations.clone().detach(),
            voltages=voltages.clone().detach(),
            rest_v=rest_v,
            step_time=step_time,
            adaptation_rate=adaptation_rate,
            rebound_rate=rebound_rate,
            adaptation_reset_min=adaptation_reset_min,
            postsyn_spikes=postsyn_spikes.clone().detach(),
            refracs=refracs.clone().detach(),
        )

        assert torch.all(adaptations == adaptations_orig)
        assert torch.all(voltages == voltages_orig)
        assert torch.all(rest_v == rest_v_orig)
        assert torch.all(step_time == step_time_orig)
        assert torch.all(adaptation_rate == adaptation_rate_orig)
        assert torch.all(rebound_rate == rebound_rate_orig)
        assert torch.all(adaptation_reset_min == adaptation_reset_min_orig)
        assert torch.all(postsyn_spikes == postsyn_spikes_orig)
        assert torch.all(refracs == refracs_orig)


class TestAdaptiveThresholdsLinearSpikes:
    @pytest.mark.parametrize(
        "target_adaptations_shape,postsyn_spikes_shape,refracs_shape",
        [
            ["batch_adaptation_shape"] + ["batch_dynamics_shape"] * 2,
            ["adaptation_shape"] + ["dynamics_shape"] * 2,
            ["batch_adaptation_shape", "batch_dynamics_shape", "primitive_shape"],
            ["adaptation_shape", "dynamics_shape", "primitive_shape"],
        ],
        ids=[
            "dynBatched",
            "dynNoBatch",
            "dynBatchNoRefrac",
            "dynNoBatchNoRefrac",
        ],
    )
    @pytest.mark.parametrize("adaptations_shape", ["adaptation_shape"], ids=["adapt"])
    @pytest.mark.parametrize(
        "step_time_shape",
        [
            "primitive_shape",
            "singleton_shape",
            "vector_1_shape",
            "vector_K_shape",
            "adaptation_shape",
            "adaptation_like_2d_shape",
        ],
        ids=[
            "DT.primitive",
            "DT.singleton",
            "DT.vec1",
            "DT.vecK",
            "DT.adapt",
            "DT.adapt2D",
        ],
    )
    @pytest.mark.parametrize(
        "time_constant_shape",
        [
            "primitive_shape",
            "singleton_shape",
            "vector_1_shape",
            "vector_K_shape",
            "adaptation_shape",
            "adaptation_like_2d_shape",
        ],
        ids=[
            "RebR.primitive",
            "RebR.singleton",
            "RebR.vec1",
            "RebR.vecK",
            "RebR.adapt",
            "RebR.adapt2D",
        ],
    )
    @pytest.mark.parametrize(
        "spike_increment_shape",
        [
            "primitive_shape",
            "singleton_shape",
            "vector_1_shape",
            "vector_K_shape",
            "adaptation_shape",
            "adaptation_like_2d_shape",
        ],
        ids=[
            "Reset.primitive",
            "Reset.singleton",
            "Reset.vec1",
            "Reset.vecK",
            "Reset.adapt",
            "Reset.adapt2D",
        ],
    )
    def test_shape_compatibility(
        self,
        target_adaptations_shape,
        adaptations_shape,
        postsyn_spikes_shape,
        step_time_shape,
        time_constant_shape,
        spike_increment_shape,
        refracs_shape,
        request,
    ):
        target_adaptations_shape = request.getfixturevalue(target_adaptations_shape)
        adaptations_shape = request.getfixturevalue(adaptations_shape)
        postsyn_spikes_shape = request.getfixturevalue(postsyn_spikes_shape)
        step_time_shape = request.getfixturevalue(step_time_shape)
        time_constant_shape = request.getfixturevalue(time_constant_shape)
        spike_increment_shape = request.getfixturevalue(spike_increment_shape)
        refracs_shape = request.getfixturevalue(refracs_shape)

        adaptations = torch.rand(adaptations_shape) * 10
        postsyn_spikes = torch.rand(postsyn_spikes_shape) > 0.5
        step_time = (
            random.uniform(0.1, 3.0)
            if step_time_shape is None
            else torch.rand(step_time_shape) * 2
        )
        time_constant = (
            random.uniform(10.0, 30.0)
            if time_constant_shape is None
            else torch.rand(time_constant_shape) * 20 + 10
        )
        spike_increment = (
            random.uniform(0.1, 0.5)
            if spike_increment_shape is None
            else torch.rand(spike_increment_shape) / 2 + 0.1
        )
        refracs = None if refracs_shape is None else torch.randint(0, 3, refracs_shape)

        adaptations_fd = adaptations.clone().detach().expand(*target_adaptations_shape)
        postsyn_spikes_fd = postsyn_spikes.clone().detach()
        step_time_fd = (
            step_time
            if not isinstance(step_time, torch.Tensor)
            else step_time.clone().detach().expand(*target_adaptations_shape)
        )
        time_constant_fd = (
            time_constant
            if not isinstance(time_constant, torch.Tensor)
            else time_constant.clone().detach().expand(*target_adaptations_shape)
        )
        spike_increment_fd = (
            spike_increment
            if not isinstance(spike_increment, torch.Tensor)
            else spike_increment.clone().detach().expand(*target_adaptations_shape)
        )
        refracs_fd = (
            refracs
            if not isinstance(refracs, torch.Tensor)
            else refracs.clone().detach()
        )

        res = adaptive_thresholds_linear_spike(
            adaptations=adaptations,
            postsyn_spikes=postsyn_spikes,
            step_time=step_time,
            time_constant=time_constant,
            spike_increment=spike_increment,
            refracs=refracs,
        )

        res_fd = adaptive_thresholds_linear_spike(
            adaptations=adaptations_fd,
            postsyn_spikes=postsyn_spikes_fd,
            step_time=step_time_fd,
            time_constant=time_constant_fd,
            spike_increment=spike_increment_fd,
            refracs=refracs_fd,
        )

        assert tuple(res.shape) == tuple(target_adaptations_shape)
        assert torch.all(res_fd == res)

    def test_nospike_update(
        self, adaptation_shape, dynamics_shape, float_error_tolerance
    ):
        adaptations = torch.rand(adaptation_shape) * 10
        postsyn_spikes = torch.zeros(dynamics_shape).bool()
        step_time = 0.5
        time_constant = 25.0
        spike_increment = 1.0
        refracs = None

        res = adaptive_thresholds_linear_spike(
            adaptations=adaptations.clone().detach(),
            postsyn_spikes=postsyn_spikes.clone().detach(),
            step_time=step_time,
            time_constant=time_constant,
            spike_increment=spike_increment,
            refracs=refracs,
        )
        assert torch.all(
            (res / adaptations) - (math.exp(-step_time / time_constant))
            < float_error_tolerance
        )

    def test_nodecay_update(
        self, adaptation_shape, dynamics_shape, float_error_tolerance
    ):
        adaptations = torch.rand(adaptation_shape) * 10
        postsyn_spikes = torch.rand(dynamics_shape) > 0.5
        step_time = 0.0
        time_constant = 25.0
        spike_increment = 1.0
        refracs = None

        res = adaptive_thresholds_linear_spike(
            adaptations=adaptations.clone().detach(),
            postsyn_spikes=postsyn_spikes.clone().detach(),
            step_time=step_time,
            time_constant=time_constant,
            spike_increment=spike_increment,
            refracs=refracs,
        )

        assert torch.all(
            (res - adaptations) - (spike_increment * postsyn_spikes.unsqueeze(-1))
            < float_error_tolerance
        )

    def test_refrac_blocking(
        self, adaptation_shape, dynamics_shape, float_error_tolerance
    ):
        adaptations = torch.rand(adaptation_shape) * 10
        postsyn_spikes = torch.rand(dynamics_shape) > 0.5
        step_time = 0.6
        time_constant = 25.0
        spike_increment = 1.0
        refracs = (torch.rand(dynamics_shape) > 0.5).int()

        res = adaptive_thresholds_linear_spike(
            adaptations=adaptations.clone().detach(),
            postsyn_spikes=postsyn_spikes.clone().detach(),
            step_time=step_time,
            time_constant=time_constant,
            spike_increment=spike_increment,
            refracs=refracs.clone().detach(),
        )

        assert torch.all(
            res
            - (
                adaptations.where(
                    refracs.unsqueeze(-1) > 0,
                    math.exp(-step_time / time_constant) * adaptations,
                )
                + (spike_increment * postsyn_spikes.unsqueeze(-1))
            )
            < float_error_tolerance
        )

    def test_non_destructive(self, adaptation_shape, dynamics_shape):
        adaptations = torch.rand(adaptation_shape) * 10
        postsyn_spikes = torch.rand(dynamics_shape) > 0.5
        step_time = torch.tensor(0.6)
        time_constant = torch.tensor(25.0)
        spike_increment = torch.tensor(1.0)
        refracs = (torch.rand(dynamics_shape) > 0.5).int()

        adaptations_orig = adaptations.clone().detach()
        postsyn_spikes_orig = postsyn_spikes.clone().detach()
        step_time_orig = step_time.clone().detach()
        time_constant_orig = time_constant.clone().detach()
        spike_increment_orig = spike_increment.clone().detach()
        refracs_orig = refracs.clone().detach()

        _ = adaptive_thresholds_linear_spike(
            adaptations=adaptations,
            postsyn_spikes=postsyn_spikes,
            step_time=step_time,
            time_constant=time_constant,
            spike_increment=spike_increment,
            refracs=refracs,
        )

        assert torch.all(adaptations == adaptations_orig)
        assert torch.all(postsyn_spikes == postsyn_spikes_orig)
        assert torch.all(step_time == step_time_orig)
        assert torch.all(time_constant == time_constant_orig)
        assert torch.all(spike_increment == spike_increment_orig)
        assert torch.all(refracs == refracs_orig)
