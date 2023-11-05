import pytest
import random
import torch

import sys

sys.path.insert(0, "../../..")


from inferno.neural.functional import (
    _voltage_thresholding_discrete,
    _voltage_thresholding_slope_intercept_discrete
)


@pytest.fixture(scope="class")
def n_ndim():
    return random.randint(3, 5)


@pytest.fixture(scope="class")
def n_shape(n_ndim):
    return [random.randint(4, 7) for _ in range(n_ndim)]


@pytest.fixture(scope="class")
def n_m1_shape(n_shape):
    return n_shape[1:]


@pytest.fixture(scope="class")
def n_m2_shape(n_shape):
    return n_shape[2:]


@pytest.fixture(scope="class")
def primitive_shape():
    return None


@pytest.fixture(scope="class")
def singleton_shape():
    return []


class Test_VoltageThresholding:
    @pytest.mark.parametrize("dynamics_shape", ["n_shape"], ids=["dyn"])
    @pytest.mark.parametrize(
        "reset_v_shape",
        [
            "primitive_shape",
            "singleton_shape",
            "n_m1_shape",
            "n_m2_shape",
            "n_shape",
        ],
        ids=[
            "Vreset.primitive",
            "Vreset.singleton",
            "Vreset.dynMinus1",
            "Vreset.dynMinus2",
            "Vreset.dyn",
        ],
    )
    @pytest.mark.parametrize(
        "thresh_v_shape",
        [
            "primitive_shape",
            "singleton_shape",
            "n_m1_shape",
            "n_m2_shape",
            "n_shape",
        ],
        ids=[
            "Vthresh.primitive",
            "Vthresh.singleton",
            "Vthresh.dynMinus1",
            "Vthresh.dynMinus2",
            "Vthresh.dyn",
        ],
    )
    @pytest.mark.parametrize(
        "refrac_ts_shape",
        [
            "primitive_shape",
            "singleton_shape",
            "n_m1_shape",
            "n_m2_shape",
            "n_shape",
        ],
        ids=[
            "RefracTS.primitive",
            "RefracTS.singleton",
            "RefracTS.dynMinus1",
            "RefracTS.dynMinus2",
            "RefracTS.dyn",
        ],
    )
    @pytest.mark.parametrize(
        "voltages_shape",
        [
            "primitive_shape",
            "n_shape",
        ],
        ids=[
            "Voltages.none",
            "Voltages.dyn",
        ],
    )
    def test_shape_compatibility(
        self,
        dynamics_shape,
        reset_v_shape,
        thresh_v_shape,
        refrac_ts_shape,
        voltages_shape,
        request,
    ):
        dynamics_shape = request.getfixturevalue(dynamics_shape)
        reset_v_shape = request.getfixturevalue(reset_v_shape)
        thresh_v_shape = request.getfixturevalue(thresh_v_shape)
        refrac_ts_shape = request.getfixturevalue(refrac_ts_shape)
        voltages_shape = request.getfixturevalue(voltages_shape)

        inputs = torch.rand(dynamics_shape, dtype=torch.float32) * 10
        refracs = torch.randint(0, 3, dynamics_shape, dtype=torch.int64)
        reset_v = (
            random.uniform(-70, -65)
            if reset_v_shape is None
            else 5 * torch.rand(reset_v_shape) - 70
        )
        thresh_v = (
            random.uniform(-51, -49)
            if thresh_v_shape is None
            else torch.rand(thresh_v_shape) * 2 - 51
        )
        refrac_ts = (
            2
            if refrac_ts_shape is None
            else torch.randint(1, 4, refrac_ts_shape, dtype=torch.int64)
        )
        voltages = (
            None
            if voltages_shape is None
            else torch.rand(dynamics_shape, dtype=torch.float32) * 10 - 70
        )

        vfn_mod = torch.rand(dynamics_shape, dtype=torch.float32)
        voltage_fn = lambda x: (vfn_mod * 5 - 60) + x

        inputs_fd = inputs.clone().detach()
        refracs_fd = refracs.clone().detach()
        reset_v_fd = (
            reset_v
            if not isinstance(reset_v, torch.Tensor)
            else reset_v.clone().detach().expand(*dynamics_shape)
        )
        thresh_v_fd = (
            thresh_v
            if not isinstance(thresh_v, torch.Tensor)
            else thresh_v.clone().detach().expand(*dynamics_shape)
        )
        refrac_ts_fd = (
            refrac_ts
            if not isinstance(refrac_ts, torch.Tensor)
            else refrac_ts.clone().detach().expand(*dynamics_shape)
        )
        voltages_fd = (
            voltages
            if not isinstance(voltages, torch.Tensor)
            else voltages.clone().detach()
        )

        res = _voltage_thresholding_discrete(
            inputs=inputs,
            refracs=refracs,
            voltage_fn=voltage_fn,
            reset_v=reset_v,
            thresh_v=thresh_v,
            refrac_ts=refrac_ts,
            voltages=voltages,
        )

        res_fd = _voltage_thresholding_discrete(
            inputs=inputs_fd,
            refracs=refracs_fd,
            voltage_fn=voltage_fn,
            reset_v=reset_v_fd,
            thresh_v=thresh_v_fd,
            refrac_ts=refrac_ts_fd,
            voltages=voltages_fd,
        )

        assert tuple(res[0].shape) == tuple(dynamics_shape)
        assert tuple(res[1].shape) == tuple(dynamics_shape)
        assert tuple(res[2].shape) == tuple(dynamics_shape)
        assert torch.all(res_fd[0] == res[0])
        assert torch.all(res_fd[1] == res[1])
        assert torch.all(res_fd[2] == res[2])

    def test_refrac_blocking(self, n_shape):
        dynamics_shape = [3 * ns for ns in n_shape]
        inputs = torch.rand(dynamics_shape, dtype=torch.float32) * 10
        refracs = torch.randint(0, 4, dynamics_shape, dtype=torch.int64)
        reset_v = -70
        thresh_v = -58.5
        refrac_ts = 3
        voltages = torch.rand(dynamics_shape, dtype=torch.float32) * 10 - 70
        voltage_fn = lambda x: voltages + x

        res_spikes, res_volts, _ = _voltage_thresholding_discrete(
            inputs=inputs.clone().detach(),
            refracs=refracs.clone().detach(),
            voltage_fn=voltage_fn,
            reset_v=reset_v,
            thresh_v=thresh_v,
            refrac_ts=refrac_ts,
            voltages=voltages.clone().detach(),
        )

        assert torch.all(~res_spikes[refracs > 1])
        assert torch.all((res_volts == voltages)[refracs > 1])
        assert torch.all((res_volts != voltages)[refracs <= 1])

    def test_non_destructive(self, n_shape):
        dynamics_shape = n_shape
        inputs = torch.rand(dynamics_shape, dtype=torch.float32) * 10
        refracs = torch.randint(0, 4, dynamics_shape, dtype=torch.int64)
        reset_v = torch.tensor(-70.0, dtype=torch.float32)
        thresh_v = torch.tensor(-58.5, dtype=torch.float32)
        refrac_ts = torch.tensor(3, dtype=torch.int64)
        voltages = torch.rand(dynamics_shape, dtype=torch.float32) * 10 - 70
        voltage_fn = lambda x: voltages + x

        inputs_orig = inputs.clone().detach()
        refracs_orig = refracs.clone().detach()
        reset_v_orig = reset_v.clone().detach()
        thresh_v_orig = thresh_v.clone().detach()
        refrac_ts_orig = refrac_ts.clone().detach()
        voltages_orig = voltages.clone().detach()

        _ = _voltage_thresholding_discrete(
            inputs=inputs,
            refracs=refracs,
            voltage_fn=voltage_fn,
            reset_v=reset_v,
            thresh_v=thresh_v,
            refrac_ts=refrac_ts,
            voltages=voltages,
        )

        assert torch.all(inputs == inputs_orig)
        assert torch.all(refracs == refracs_orig)
        assert torch.all(reset_v == reset_v_orig)
        assert torch.all(thresh_v == thresh_v_orig)
        assert torch.all(refrac_ts == refrac_ts_orig)
        assert torch.all(voltages == voltages_orig)

    def test_reset_and_integrate_behavior(self, n_shape):
        dynamics_shape = [3 * ns for ns in n_shape]
        inputs = torch.rand(dynamics_shape, dtype=torch.float32) + 9
        refracs = torch.randint(0, 4, dynamics_shape, dtype=torch.int64)
        reset_v = -70.0
        thresh_v = -55.0
        refrac_ts = 3
        voltages = torch.rand(dynamics_shape, dtype=torch.float32) * 10 - 70
        voltage_fn = lambda x: voltages.clone().detach() + x

        res_spikes, res_volts, res_refracs = _voltage_thresholding_discrete(
            inputs=inputs.clone().detach(),
            refracs=refracs.clone().detach(),
            voltage_fn=voltage_fn,
            reset_v=reset_v,
            thresh_v=thresh_v,
            refrac_ts=refrac_ts,
            voltages=None,
        )

        assert torch.all((res_refracs == refrac_ts)[res_spikes])
        assert torch.all((res_volts == reset_v)[res_spikes])
        assert torch.all(
            (res_volts == (voltages + inputs))[
                torch.logical_and(~res_spikes, refracs <= 1)
            ]
        )
        assert torch.all(
            res_spikes
            == torch.logical_and((voltages + inputs) >= thresh_v, refracs <= 1)
        )

    def test_adversarially_typed_primitives(self, n_shape):
        dynamics_shape = n_shape
        inputs = torch.rand(dynamics_shape, dtype=torch.float32) * 10
        refracs = torch.randint(0, 4, dynamics_shape, dtype=torch.int64)
        reset_v = int(-70)
        thresh_v = int(-60)
        refrac_ts = float(3.0)
        voltages = torch.rand(dynamics_shape, dtype=torch.float32) * 10 - 70
        voltage_fn = lambda x: voltages + x

        res_spikes, res_volts, res_refracs = _voltage_thresholding_discrete(
            inputs=inputs,
            refracs=refracs,
            voltage_fn=voltage_fn,
            reset_v=reset_v,
            thresh_v=thresh_v,
            refrac_ts=refrac_ts,
            voltages=voltages,
        )

        assert res_spikes.dtype == torch.bool
        assert res_volts.dtype == torch.float32
        assert res_refracs.dtype == torch.int64


class Test_VoltageThresholdingSlopeIntercept:
    @pytest.mark.parametrize("dynamics_shape", ["n_shape"], ids=["dyn"])
    @pytest.mark.parametrize(
        "rest_v_shape",
        [
            "primitive_shape",
            "singleton_shape",
            "n_m1_shape",
            "n_m2_shape",
            "n_shape",
        ],
        ids=[
            "Vrest.primitive",
            "Vrest.singleton",
            "Vrest.dynMinus1",
            "Vrest.dynMinus2",
            "Vrest.dyn",
        ],
    )
    @pytest.mark.parametrize(
        "v_slope_shape",
        [
            "primitive_shape",
            "singleton_shape",
            "n_m1_shape",
            "n_m2_shape",
            "n_shape",
        ],
        ids=[
            "SlopeV.primitive",
            "SlopeV.singleton",
            "SlopeV.dynMinus1",
            "SlopeV.dynMinus2",
            "SlopeV.dyn",
        ],
    )
    @pytest.mark.parametrize(
        "v_intercept_shape",
        [
            "primitive_shape",
            "singleton_shape",
            "n_m1_shape",
            "n_m2_shape",
            "n_shape",
        ],
        ids=[
            "InterceptV.primitive",
            "InterceptV.singleton",
            "InterceptV.dynMinus1",
            "InterceptV.dynMinus2",
            "InterceptV.dyn",
        ],
    )
    @pytest.mark.parametrize(
        "thresh_v_shape",
        [
            "primitive_shape",
            "singleton_shape",
            "n_m1_shape",
            "n_m2_shape",
            "n_shape",
        ],
        ids=[
            "Vthresh.primitive",
            "Vthresh.singleton",
            "Vthresh.dynMinus1",
            "Vthresh.dynMinus2",
            "Vthresh.dyn",
        ],
    )
    @pytest.mark.parametrize(
        "refrac_ts_shape",
        [
            "primitive_shape",
            "singleton_shape",
            "n_m1_shape",
            "n_m2_shape",
            "n_shape",
        ],
        ids=[
            "RefracTS.primitive",
            "RefracTS.singleton",
            "RefracTS.dynMinus1",
            "RefracTS.dynMinus2",
            "RefracTS.dyn",
        ],
    )
    @pytest.mark.parametrize(
        "voltages_shape",
        [
            "primitive_shape",
            "n_shape",
        ],
        ids=[
            "Voltages.none",
            "Voltages.dyn",
        ],
    )
    def test_shape_compatibility(
        self,
        dynamics_shape,
        rest_v_shape,
        v_slope_shape,
        v_intercept_shape,
        thresh_v_shape,
        refrac_ts_shape,
        voltages_shape,
        request,
    ):
        dynamics_shape = request.getfixturevalue(dynamics_shape)
        rest_v_shape = request.getfixturevalue(rest_v_shape)
        v_slope_shape = request.getfixturevalue(v_slope_shape)
        v_intercept_shape = request.getfixturevalue(v_intercept_shape)
        thresh_v_shape = request.getfixturevalue(thresh_v_shape)
        refrac_ts_shape = request.getfixturevalue(refrac_ts_shape)
        voltages_shape = request.getfixturevalue(voltages_shape)

        inputs = torch.rand(dynamics_shape, dtype=torch.float32) * 10
        refracs = torch.randint(0, 3, dynamics_shape, dtype=torch.int64)
        rest_v = (
            random.uniform(-65, -60)
            if rest_v_shape is None
            else 5 * torch.rand(rest_v_shape) - 65
        )
        v_slope = (
            random.uniform(-1.25, -0.75)
            if v_slope_shape is None
            else 5 * torch.rand(v_slope_shape) - 10
        )
        v_intercept = (
            random.uniform(-10, -5)
            if v_intercept_shape is None
            else 5 * torch.rand(v_intercept_shape) - 10
        )
        thresh_v = (
            random.uniform(-51, -49)
            if thresh_v_shape is None
            else torch.rand(thresh_v_shape) * 2 - 51
        )
        refrac_ts = (
            2
            if refrac_ts_shape is None
            else torch.randint(1, 4, refrac_ts_shape, dtype=torch.int64)
        )
        voltages = (
            None
            if voltages_shape is None
            else torch.rand(dynamics_shape, dtype=torch.float32) * 10 - 70
        )

        vfn_mod = torch.rand(dynamics_shape, dtype=torch.float32)
        voltage_fn = lambda x: (vfn_mod * 5 - 60) + x

        inputs_fd = inputs.clone().detach()
        refracs_fd = refracs.clone().detach()
        rest_v_fd = (
            rest_v
            if not isinstance(rest_v, torch.Tensor)
            else rest_v.clone().detach().expand(*dynamics_shape)
        )
        v_slope_fd = (
            v_slope
            if not isinstance(v_slope, torch.Tensor)
            else v_slope.clone().detach().expand(*dynamics_shape)
        )
        v_intercept_fd = (
            v_intercept
            if not isinstance(v_intercept, torch.Tensor)
            else v_intercept.clone().detach().expand(*dynamics_shape)
        )
        thresh_v_fd = (
            thresh_v
            if not isinstance(thresh_v, torch.Tensor)
            else thresh_v.clone().detach().expand(*dynamics_shape)
        )
        refrac_ts_fd = (
            refrac_ts
            if not isinstance(refrac_ts, torch.Tensor)
            else refrac_ts.clone().detach().expand(*dynamics_shape)
        )
        voltages_fd = (
            voltages
            if not isinstance(voltages, torch.Tensor)
            else voltages.clone().detach()
        )

        res = _voltage_thresholding_slope_intercept_discrete(
            inputs=inputs,
            refracs=refracs,
            voltage_fn=voltage_fn,
            rest_v=rest_v,
            v_slope=v_slope,
            v_intercept=v_intercept,
            thresh_v=thresh_v,
            refrac_ts=refrac_ts,
            voltages=voltages,
        )

        res_fd = _voltage_thresholding_slope_intercept_discrete(
            inputs=inputs_fd,
            refracs=refracs_fd,
            voltage_fn=voltage_fn,
            rest_v=rest_v_fd,
            v_slope=v_slope_fd,
            v_intercept=v_intercept_fd,
            thresh_v=thresh_v_fd,
            refrac_ts=refrac_ts_fd,
            voltages=voltages_fd,
        )

        assert tuple(res[0].shape) == tuple(dynamics_shape)
        assert tuple(res[1].shape) == tuple(dynamics_shape)
        assert tuple(res[2].shape) == tuple(dynamics_shape)
        assert torch.all(res_fd[0] == res[0])
        assert torch.all(res_fd[1] == res[1])
        assert torch.all(res_fd[2] == res[2])

    def test_refrac_blocking(self, n_shape):
        dynamics_shape = [3 * ns for ns in n_shape]
        inputs = torch.rand(dynamics_shape, dtype=torch.float32) * 10
        refracs = torch.randint(0, 4, dynamics_shape, dtype=torch.int64)
        rest_v = -70.0
        v_slope = -1.1
        v_intercept = -7.5
        thresh_v = -58.5
        refrac_ts = 3
        voltages = torch.rand(dynamics_shape, dtype=torch.float32) * 10 - 70
        voltage_fn = lambda x: voltages + x

        res_spikes, res_volts, _ = _voltage_thresholding_slope_intercept_discrete(
            inputs=inputs.clone().detach(),
            refracs=refracs.clone().detach(),
            voltage_fn=voltage_fn,
            rest_v=rest_v,
            v_slope=v_slope,
            v_intercept=v_intercept,
            thresh_v=thresh_v,
            refrac_ts=refrac_ts,
            voltages=voltages.clone().detach(),
        )

        assert torch.all(~res_spikes[refracs > 1])
        assert torch.all((res_volts == voltages)[refracs > 1])
        assert torch.all((res_volts != voltages)[refracs <= 1])

    def test_non_destructive(self, n_shape):
        dynamics_shape = n_shape
        inputs = torch.rand(dynamics_shape, dtype=torch.float32) * 10
        refracs = torch.randint(0, 4, dynamics_shape, dtype=torch.int64)
        rest_v = torch.tensor(-70.0, dtype=torch.float32)
        v_slope = torch.tensor(-1.1, dtype=torch.float32)
        v_intercept = torch.tensor(-7.5, dtype=torch.float32)
        thresh_v = torch.tensor(-58.5, dtype=torch.float32)
        refrac_ts = torch.tensor(3, dtype=torch.int64)
        voltages = torch.rand(dynamics_shape, dtype=torch.float32) * 10 - 70
        voltage_fn = lambda x: voltages + x

        inputs_orig = inputs.clone().detach()
        refracs_orig = refracs.clone().detach()
        rest_v_orig = rest_v.clone().detach()
        v_slope_orig = v_slope.clone().detach()
        v_intercept_orig = v_intercept.clone().detach()
        thresh_v_orig = thresh_v.clone().detach()
        refrac_ts_orig = refrac_ts.clone().detach()
        voltages_orig = voltages.clone().detach()

        _ = _voltage_thresholding_slope_intercept_discrete(
            inputs=inputs.clone().detach(),
            refracs=refracs.clone().detach(),
            voltage_fn=voltage_fn,
            rest_v=rest_v,
            v_slope=v_slope,
            v_intercept=v_intercept,
            thresh_v=thresh_v,
            refrac_ts=refrac_ts,
            voltages=voltages.clone().detach(),
        )

        assert torch.all(inputs == inputs_orig)
        assert torch.all(refracs == refracs_orig)
        assert torch.all(rest_v == rest_v_orig)
        assert torch.all(v_slope == v_slope_orig)
        assert torch.all(v_intercept == v_intercept_orig)
        assert torch.all(thresh_v == thresh_v_orig)
        assert torch.all(refrac_ts == refrac_ts_orig)
        assert torch.all(voltages == voltages_orig)

    def test_reset_and_integrate_behavior(self, n_shape):
        dynamics_shape = [3 * ns for ns in n_shape]
        inputs = torch.rand(dynamics_shape, dtype=torch.float32) + 9
        refracs = torch.randint(0, 4, dynamics_shape, dtype=torch.int64)
        rest_v = -70.0
        v_slope = -1.1
        v_intercept = -7.5
        thresh_v = -58.5
        refrac_ts = 3
        voltages = torch.rand(dynamics_shape, dtype=torch.float32) * 10 - 70
        voltage_fn = lambda x: voltages.clone().detach() + x

        res_spikes, res_volts, res_refracs = _voltage_thresholding_slope_intercept_discrete(
            inputs=inputs.clone().detach(),
            refracs=refracs.clone().detach(),
            voltage_fn=voltage_fn,
            rest_v=rest_v,
            v_slope=v_slope,
            v_intercept=v_intercept,
            thresh_v=thresh_v,
            refrac_ts=refrac_ts,
            voltages=None,
        )

        assert torch.all((res_refracs == refrac_ts)[res_spikes])
        assert torch.all((res_volts == (rest_v + v_slope * ((voltages + inputs) - rest_v) - v_intercept))[res_spikes])
        assert torch.all(
            (res_volts == (voltages + inputs))[
                torch.logical_and(~res_spikes, refracs <= 1)
            ]
        )
        assert torch.all(
            res_spikes
            == torch.logical_and((voltages + inputs) >= thresh_v, refracs <= 1)
        )

    def test_adversarially_typed_primitives(self, n_shape):
        dynamics_shape = n_shape
        inputs = torch.rand(dynamics_shape, dtype=torch.float32) * 10
        refracs = torch.randint(0, 4, dynamics_shape, dtype=torch.int64)
        rest_v = int(-70)
        v_slope = int(-2)
        v_intercept = int(-7)
        thresh_v = int(-58)
        refrac_ts = float(3.0)
        voltages = torch.rand(dynamics_shape, dtype=torch.float32) * 10 - 70
        voltage_fn = lambda x: voltages + x

        res_spikes, res_volts, res_refracs = _voltage_thresholding_slope_intercept_discrete(
            inputs=inputs,
            refracs=refracs,
            voltage_fn=voltage_fn,
            rest_v=rest_v,
            v_slope=v_slope,
            v_intercept=v_intercept,
            thresh_v=thresh_v,
            refrac_ts=refrac_ts,
            voltages=None,
        )

        assert res_spikes.dtype == torch.bool
        assert res_volts.dtype == torch.float32
        assert res_refracs.dtype == torch.int64
