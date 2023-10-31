import pytest
import random
import torch

import sys

sys.path.insert(0, "../../..")

from inferno.observe import NearestTraceReducer, CumulativeTraceReducer


@pytest.fixture(scope="class")
def observation_ndim():
    return random.randint(3, 7)


@pytest.fixture(scope="class")
def observation_shape(observation_ndim):
    return tuple([random.randint(4, 9) for _ in range(observation_ndim)])


@pytest.fixture(scope="class")
def num_samples():
    return random.randint(13, 27)


@pytest.fixture(scope="class")
def float_error_tolerance():
    return 1e-7


@pytest.fixture(scope="class")
def step_time():
    return random.random()


@pytest.fixture(scope="class")
def step_time_multival(observation_shape):
    return torch.rand(*observation_shape)


@pytest.fixture(scope="class")
def time_constant():
    return random.random() * 29 + 1


@pytest.fixture(scope="class")
def time_constant_multival(observation_shape):
    return torch.rand(*observation_shape) * 29 + 1


@pytest.fixture(scope="class")
def amplitude():
    return random.random()


@pytest.fixture(scope="class")
def amplitude_multival(observation_shape):
    return torch.rand(*observation_shape) * 4 - 2


@pytest.fixture(scope="class")
def target():
    return random.randint(0, 3)


@pytest.fixture(scope="class")
def target_multival(observation_shape):
    return torch.randint(0, 4, observation_shape)


@pytest.fixture(scope="class")
def fuzzy_target():
    return random.random()


@pytest.fixture(scope="class")
def fuzzy_target_multival(observation_shape):
    return torch.rand(*observation_shape)


@pytest.fixture(scope="class")
def match_tolerance():
    return 0.1


@pytest.fixture(scope="class")
def match_tolerance_multival(observation_shape):
    return torch.rand(*observation_shape) / 5


@pytest.fixture(scope="class")
def exact_tolerance():
    return None


class TestNearestTraceReducer:
    @pytest.mark.parametrize(
        "target_val, tolerance",
        [("target", "exact_tolerance"), ("fuzzy_target", "match_tolerance")],
        ids=["exact", "fuzzy"],
    )
    def test_tensor_primitive_equivalence(
        self,
        observation_shape,
        num_samples,
        step_time,
        time_constant,
        amplitude,
        target_val,
        tolerance,
        request,
        float_error_tolerance,
    ):
        target_val = request.getfixturevalue(target_val)
        tolerance = request.getfixturevalue(tolerance)

        if tolerance is not None:
            inputs = [
                torch.rand(*observation_shape).float() for _ in range(num_samples)
            ]
        else:
            inputs = [
                torch.randint(0, 4, observation_shape).float()
                for _ in range(num_samples)
            ]

        reducer = NearestTraceReducer(
            step_time, time_constant, amplitude, target_val, tolerance
        )

        for sample in inputs:
            reducer(sample)

        prim_res = reducer.peek().clone().detach()

        tolerance_expd = (
            None
            if tolerance is None
            else torch.tensor(tolerance).expand(*observation_shape).clone().detach()
        )
        reducer = NearestTraceReducer(
            torch.tensor(step_time).expand(*observation_shape).clone().detach(),
            torch.tensor(time_constant).expand(*observation_shape).clone().detach(),
            torch.tensor(amplitude).expand(*observation_shape).clone().detach(),
            torch.tensor(target_val).expand(*observation_shape).clone().detach(),
            tolerance_expd,
        )

        for sample in inputs:
            reducer(sample)

        tens_res = reducer.peek().clone().detach()
        print(
            f"Absolute Difference: max: {(prim_res - tens_res).abs().max().item()}, "
            f"avg: {(prim_res - tens_res).abs().mean().item()}"
        )
        assert torch.all(torch.abs(prim_res - tens_res) <= float_error_tolerance)

    @pytest.mark.parametrize(
        "target_val, tolerance",
        [
            ("target_multival", "exact_tolerance"),
            ("fuzzy_target_multival", "match_tolerance_multival"),
        ],
        ids=["exact", "fuzzy"],
    )
    def test_heterogeneous_config(
        self,
        observation_shape,
        num_samples,
        step_time_multival,
        time_constant_multival,
        amplitude_multival,
        target_val,
        tolerance,
        request,
    ):
        target_val = request.getfixturevalue(target_val)
        tolerance = request.getfixturevalue(tolerance)

        if tolerance is not None:
            inputs = [
                torch.rand(*observation_shape).float() for _ in range(num_samples)
            ]
        else:
            inputs = [
                torch.randint(0, 4, observation_shape).float()
                for _ in range(num_samples)
            ]

        reducer = NearestTraceReducer(
            step_time_multival,
            time_constant_multival,
            amplitude_multival,
            target_val,
            tolerance,
        )

        for sample in inputs:
            reducer(sample.clone().detach())

        test_res = reducer.peek().clone().detach()

        true_res = torch.zeros_like(inputs[0])
        for sample in inputs:
            if tolerance is not None:
                mask = torch.abs(sample - target_val) <= tolerance
            else:
                mask = sample == target_val
            true_res = torch.where(
                mask,
                amplitude_multival,
                torch.exp(-step_time_multival / time_constant_multival) * true_res,
            )

        assert torch.all(test_res == true_res)

    @pytest.mark.parametrize(
        "target_val, tolerance",
        [
            ("target_multival", "exact_tolerance"),
            ("fuzzy_target_multival", "match_tolerance_multival"),
        ],
        ids=["exact", "fuzzy"],
    )
    def test_non_destructive(
        self,
        observation_shape,
        num_samples,
        step_time_multival,
        time_constant_multival,
        amplitude_multival,
        target_val,
        tolerance,
        request,
    ):
        target_val = request.getfixturevalue(target_val)
        tolerance = request.getfixturevalue(tolerance)

        if tolerance is not None:
            inputs = [
                torch.rand(*observation_shape).float() for _ in range(num_samples)
            ]
        else:
            inputs = [
                torch.randint(0, 4, observation_shape).float()
                for _ in range(num_samples)
            ]

        reducer = NearestTraceReducer(
            step_time_multival,
            time_constant_multival,
            amplitude_multival,
            target_val,
            tolerance,
        )

        for sample in inputs:
            reducer(sample)

        test_res = reducer.peek().clone().detach()

        reducer = NearestTraceReducer(
            step_time_multival,
            time_constant_multival,
            amplitude_multival,
            target_val,
            tolerance,
        )

        for sample in inputs:
            reducer(sample)

        true_res = reducer.peek().clone().detach()

        assert torch.all(test_res == true_res)

    def test_custom_map_method(
        self,
        observation_shape,
        num_samples,
        step_time,
        time_constant,
        amplitude,
        fuzzy_target,
        match_tolerance,
    ):
        mapfn = lambda r, x1, x2: ((x1 * 2 + x2) / 3) + (
            0 if r.peek() is None else r.peek()
        )

        reducer = NearestTraceReducer(
            step_time,
            time_constant,
            amplitude,
            fuzzy_target,
            match_tolerance,
            mapmeth=mapfn,
        )

        inputs = [
            (
                torch.rand(*observation_shape).float(),
                torch.rand(*observation_shape).float(),
            )
            for _ in range(num_samples)
        ]

        for sample in inputs:
            reducer(*[s.clone().detach() for s in sample])
        test_res = reducer.peek().clone().detach()

        reducer = NearestTraceReducer(
            step_time,
            time_constant,
            amplitude,
            fuzzy_target,
            match_tolerance,
        )
        for sample in inputs:
            if reducer.peek() is None:
                add_term = 0
            else:
                add_term = reducer.peek().clone().detach()
            mapres = (sample[0] * 2 + sample[1]) / 3 + add_term
            reducer(mapres)

        true_res = reducer.peek().clone().detach()
        assert torch.all(test_res == true_res)

    def test_custom_filter_method(
        self,
        observation_shape,
        num_samples,
        step_time,
        time_constant,
        amplitude,
        fuzzy_target,
        match_tolerance,
    ):
        filterfn = (
            lambda r, x: int(
                (x + (0 if r.peek() is None else r.peek())).mean().item() * 10
            )
            % 2
            == 0
        )

        reducer = NearestTraceReducer(
            step_time,
            time_constant,
            amplitude,
            fuzzy_target,
            match_tolerance,
            filtermeth=filterfn,
        )

        inputs = [torch.rand(*observation_shape).float() for _ in range(num_samples)]

        for sample in inputs:
            reducer(sample.clone().detach())

        test_res = reducer.peek().clone().detach()

        reducer = NearestTraceReducer(
            step_time,
            time_constant,
            amplitude,
            fuzzy_target,
            match_tolerance,
        )
        for sample in inputs:
            if reducer.peek() is None:
                add_term = 0
            else:
                add_term = reducer.peek().clone().detach()
            filterres = int((sample + add_term).mean().item() * 10) % 2 == 0
            if filterres:
                reducer(sample)

        true_res = reducer.peek().clone().detach()
        assert torch.all(test_res == true_res)


class TestCumulativeTraceReducer:
    @pytest.mark.parametrize(
        "target_val, tolerance",
        [("target", "exact_tolerance"), ("fuzzy_target", "match_tolerance")],
        ids=["exact", "fuzzy"],
    )
    def test_tensor_primitive_equivalence(
        self,
        observation_shape,
        num_samples,
        step_time,
        time_constant,
        amplitude,
        target_val,
        tolerance,
        request,
        float_error_tolerance,
    ):
        target_val = request.getfixturevalue(target_val)
        tolerance = request.getfixturevalue(tolerance)

        if tolerance is not None:
            inputs = [
                torch.rand(*observation_shape).float() for _ in range(num_samples)
            ]
        else:
            inputs = [
                torch.randint(0, 4, observation_shape).float()
                for _ in range(num_samples)
            ]

        reducer = CumulativeTraceReducer(
            step_time, time_constant, amplitude, target_val, tolerance
        )

        for sample in inputs:
            reducer(sample)

        prim_res = reducer.peek().clone().detach()

        tolerance_expd = (
            None
            if tolerance is None
            else torch.tensor(tolerance).expand(*observation_shape).clone().detach()
        )
        reducer = CumulativeTraceReducer(
            torch.tensor(step_time).expand(*observation_shape).clone().detach(),
            torch.tensor(time_constant).expand(*observation_shape).clone().detach(),
            torch.tensor(amplitude).expand(*observation_shape).clone().detach(),
            torch.tensor(target_val).expand(*observation_shape).clone().detach(),
            tolerance_expd,
        )

        for sample in inputs:
            reducer(sample)

        tens_res = reducer.peek().clone().detach()
        print(
            f"Absolute Difference: max: {(prim_res - tens_res).abs().max().item()}, "
            f"avg: {(prim_res - tens_res).abs().mean().item()}"
        )
        assert torch.all(torch.abs(prim_res - tens_res) <= float_error_tolerance)

    @pytest.mark.parametrize(
        "target_val, tolerance",
        [
            ("target_multival", "exact_tolerance"),
            ("fuzzy_target_multival", "match_tolerance_multival"),
        ],
        ids=["exact", "fuzzy"],
    )
    def test_heterogeneous_config(
        self,
        observation_shape,
        num_samples,
        step_time_multival,
        time_constant_multival,
        amplitude_multival,
        target_val,
        tolerance,
        request,
    ):
        target_val = request.getfixturevalue(target_val)
        tolerance = request.getfixturevalue(tolerance)

        if tolerance is not None:
            inputs = [
                torch.rand(*observation_shape).float() for _ in range(num_samples)
            ]
        else:
            inputs = [
                torch.randint(0, 4, observation_shape).float()
                for _ in range(num_samples)
            ]

        reducer = CumulativeTraceReducer(
            step_time_multival,
            time_constant_multival,
            amplitude_multival,
            target_val,
            tolerance,
        )

        for sample in inputs:
            reducer(sample.clone().detach())

        test_res = reducer.peek().clone().detach()

        true_res = torch.zeros_like(inputs[0])
        for sample in inputs:
            if tolerance is not None:
                mask = torch.abs(sample - target_val) <= tolerance
            else:
                mask = sample == target_val
            true_res = true_res * torch.exp(
                -step_time_multival / time_constant_multival
            ) + torch.where(
                mask,
                amplitude_multival,
                0,
            )

        assert torch.all(test_res == true_res)

    @pytest.mark.parametrize(
        "target_val, tolerance",
        [
            ("target_multival", "exact_tolerance"),
            ("fuzzy_target_multival", "match_tolerance_multival"),
        ],
        ids=["exact", "fuzzy"],
    )
    def test_non_destructive(
        self,
        observation_shape,
        num_samples,
        step_time_multival,
        time_constant_multival,
        amplitude_multival,
        target_val,
        tolerance,
        request,
    ):
        target_val = request.getfixturevalue(target_val)
        tolerance = request.getfixturevalue(tolerance)

        if tolerance is not None:
            inputs = [
                torch.rand(*observation_shape).float() for _ in range(num_samples)
            ]
        else:
            inputs = [
                torch.randint(0, 4, observation_shape).float()
                for _ in range(num_samples)
            ]

        reducer = CumulativeTraceReducer(
            step_time_multival,
            time_constant_multival,
            amplitude_multival,
            target_val,
            tolerance,
        )

        for sample in inputs:
            reducer(sample)

        test_res = reducer.peek().clone().detach()

        reducer = CumulativeTraceReducer(
            step_time_multival,
            time_constant_multival,
            amplitude_multival,
            target_val,
            tolerance,
        )

        for sample in inputs:
            reducer(sample)

        true_res = reducer.peek().clone().detach()

        assert torch.all(test_res == true_res)

    def test_custom_map_method(
        self,
        observation_shape,
        num_samples,
        step_time,
        time_constant,
        amplitude,
        fuzzy_target,
        match_tolerance,
    ):
        mapfn = lambda r, x1, x2: ((x1 * 2 + x2) / 3) + (
            0 if r.peek() is None else r.peek()
        )

        reducer = CumulativeTraceReducer(
            step_time,
            time_constant,
            amplitude,
            fuzzy_target,
            match_tolerance,
            mapmeth=mapfn,
        )

        inputs = [
            (
                torch.rand(*observation_shape).float(),
                torch.rand(*observation_shape).float(),
            )
            for _ in range(num_samples)
        ]

        for sample in inputs:
            reducer(*[s.clone().detach() for s in sample])
        test_res = reducer.peek().clone().detach()

        reducer = CumulativeTraceReducer(
            step_time,
            time_constant,
            amplitude,
            fuzzy_target,
            match_tolerance,
        )
        for sample in inputs:
            if reducer.peek() is None:
                add_term = 0
            else:
                add_term = reducer.peek().clone().detach()
            mapres = (sample[0] * 2 + sample[1]) / 3 + add_term
            reducer(mapres)

        true_res = reducer.peek().clone().detach()
        assert torch.all(test_res == true_res)

    def test_custom_filter_method(
        self,
        observation_shape,
        num_samples,
        step_time,
        time_constant,
        amplitude,
        fuzzy_target,
        match_tolerance,
    ):
        filterfn = (
            lambda r, x: int(
                (x + (0 if r.peek() is None else r.peek())).mean().item() * 10
            )
            % 2
            == 0
        )

        reducer = CumulativeTraceReducer(
            step_time,
            time_constant,
            amplitude,
            fuzzy_target,
            match_tolerance,
            filtermeth=filterfn,
        )

        inputs = [torch.rand(*observation_shape).float() for _ in range(num_samples)]

        for sample in inputs:
            reducer(sample.clone().detach())

        test_res = reducer.peek().clone().detach()

        reducer = CumulativeTraceReducer(
            step_time,
            time_constant,
            amplitude,
            fuzzy_target,
            match_tolerance,
        )
        for sample in inputs:
            if reducer.peek() is None:
                add_term = 0
            else:
                add_term = reducer.peek().clone().detach()
            filterres = int((sample + add_term).mean().item() * 10) % 2 == 0
            if filterres:
                reducer(sample)

        true_res = reducer.peek().clone().detach()
        assert torch.all(test_res == true_res)
