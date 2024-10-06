import math
import pytest
import random
import torch

from inferno import Module, RecordTensor
from inferno.functional import (
    interp_previous,
    interp_next,
    interp_nearest,
    interp_linear,
    interp_expdecay,
    interp_expratedecay,
)


@pytest.fixture
def rtmodule():
    module = Module()
    dt = random.uniform(0.5, 1.5)
    RecordTensor.create(
        module,
        "rt",
        dt,
        dt,
        torch.full(
            [random.randint(3, 7) for _ in range(random.randint(2, 4))], float("nan")
        ),
        inclusive=True,
    )

    return module


@pytest.mark.parametrize("bias", (0.1, 0.5, 0.9), ids=("next", "center", "prev"))
@pytest.mark.parametrize("context", (False, True), ids=("standalone", "recordtensor"))
def test_interp_previous(rtmodule, bias, context):

    rt = rtmodule.rt

    prev_data = torch.rand(*rt.shape)
    next_data = torch.rand(*rt.shape)
    selector = torch.full(rt.shape, bias) * rt.dt
    interp = interp_previous

    if context:
        rt.push(prev_data)
        rt.push(next_data)
        res = rt.select(selector, interp, offset=1)

    else:
        res = interp(prev_data, next_data, rt.dt - selector, rt.dt)

    assert torch.all(res == prev_data)


@pytest.mark.parametrize("bias", (0.1, 0.5, 0.9), ids=("next", "center", "prev"))
@pytest.mark.parametrize("context", (False, True), ids=("standalone", "recordtensor"))
def test_interp_next(rtmodule, bias, context):

    rt = rtmodule.rt

    prev_data = torch.rand(*rt.shape)
    next_data = torch.rand(*rt.shape)
    selector = torch.full(rt.shape, bias) * rt.dt
    interp = interp_next

    if context:
        rt.push(prev_data)
        rt.push(next_data)
        res = rt.select(selector, interp, offset=1)

    else:
        res = interp(prev_data, next_data, rt.dt - selector, rt.dt)

    assert torch.all(res == next_data)


@pytest.mark.parametrize("bias", (0.1, 0.5, 0.9), ids=("next", "center", "prev"))
@pytest.mark.parametrize("context", (False, True), ids=("standalone", "recordtensor"))
def test_interp_nearest(rtmodule, bias, context):

    rt = rtmodule.rt

    prev_data = torch.rand(*rt.shape)
    next_data = torch.rand(*rt.shape)
    selector = torch.full(rt.shape, bias) * rt.dt
    interp = interp_nearest

    if context:
        rt.push(prev_data)
        rt.push(next_data)
        res = rt.select(selector, interp, offset=1)

    else:
        res = interp(prev_data, next_data, rt.dt - selector, rt.dt)

    match bias:
        case 0.1:
            assert torch.all(res == next_data)
        case 0.5:
            assert torch.all(res == prev_data)
        case 0.9:
            assert torch.all(res == prev_data)
        case _:
            raise ValueError(f"invalid bias '{bias}' specified")


@pytest.mark.parametrize("bias", (0.1, 0.5, 0.9), ids=("next", "center", "prev"))
@pytest.mark.parametrize("context", (False, True), ids=("standalone", "recordtensor"))
def test_interp_linear(rtmodule, bias, context):

    rt = rtmodule.rt

    prev_data = torch.rand(*rt.shape)
    next_data = torch.rand(*rt.shape)
    selector = torch.full(rt.shape, bias) * rt.dt
    interp = interp_linear

    if context:
        rt.push(prev_data)
        rt.push(next_data)
        res = rt.select(selector, interp, offset=1)

    else:
        res = interp(prev_data, next_data, rt.dt - selector, rt.dt)

    cmp = ((next_data - prev_data) / rt.dt) * (rt.dt - selector) + prev_data

    assert torch.all(res == cmp)


@pytest.mark.parametrize("bias", (0.1, 0.5, 0.9), ids=("next", "center", "prev"))
@pytest.mark.parametrize("context", (False, True), ids=("standalone", "recordtensor"))
def test_interp_expdecay(rtmodule, bias, context):

    rt = rtmodule.rt

    tc = random.uniform(20, 30)
    prev_data = torch.rand(*rt.shape)
    next_data = prev_data * math.exp(-rt.dt / tc)
    selector = torch.full(rt.shape, bias) * rt.dt
    interp = interp_expdecay

    if context:
        rt.push(prev_data)
        rt.push(next_data)
        res = rt.select(selector, interp, offset=1, interp_kwargs={"time_constant": tc})

    else:
        res = interp(prev_data, next_data, rt.dt - selector, rt.dt, time_constant=tc)

    cmp = prev_data * torch.exp(-(rt.dt - selector) / tc)

    assert torch.all(res == cmp)


@pytest.mark.parametrize("bias", (0.1, 0.5, 0.9), ids=("next", "center", "prev"))
@pytest.mark.parametrize("context", (False, True), ids=("standalone", "recordtensor"))
def test_interp_expratedecay(rtmodule, bias, context):

    rt = rtmodule.rt

    tc = random.uniform(20, 30)
    rc = 1.0 / tc
    prev_data = torch.rand(*rt.shape)
    next_data = prev_data * math.exp(-rt.dt / tc)
    selector = torch.full(rt.shape, bias) * rt.dt
    interp = interp_expratedecay

    if context:
        rt.push(prev_data)
        rt.push(next_data)
        res = rt.select(selector, interp, offset=1, interp_kwargs={"rate_constant": rc})

    else:
        res = interp(prev_data, next_data, rt.dt - selector, rt.dt, rate_constant=rc)

    cmp = prev_data * torch.exp(-(rt.dt - selector) / tc)

    assert torch.all(res == cmp)
