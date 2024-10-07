import pytest
import random
import torch

from inferno import Module, RecordTensor
from inferno.functional import (
    extrap_previous,
    extrap_next,
    extrap_neighbors,
    extrap_nearest,
    extrap_linear_forward,
    extrap_linear_backward,
    extrap_expdecay,
    extrap_expratedecay,
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
def test_extrap_previous(rtmodule, bias, context):

    rt = rtmodule.rt

    prev_data = torch.rand(*rt.shape)
    next_data = torch.rand(*rt.shape)
    selector = torch.full(rt.shape, bias) * rt.dt
    insert = torch.rand(*rt.shape)
    extrap = extrap_previous

    if context:
        rt.push(prev_data)
        rt.push(next_data)
        rt.insert(insert, selector, extrap, offset=1)
        prev_res = rt.select(rt.dt)
        next_res = rt.select(0.0)

    else:
        prev_res, next_res = extrap(
            insert, rt.dt - selector, prev_data, next_data, rt.dt
        )

    assert torch.all(prev_res == insert)
    assert torch.all(next_res == next_data)


@pytest.mark.parametrize("bias", (0.1, 0.5, 0.9), ids=("next", "center", "prev"))
@pytest.mark.parametrize("context", (False, True), ids=("standalone", "recordtensor"))
def test_extrap_next(rtmodule, bias, context):

    rt = rtmodule.rt

    prev_data = torch.rand(*rt.shape)
    next_data = torch.rand(*rt.shape)
    selector = torch.full(rt.shape, bias) * rt.dt
    insert = torch.rand(*rt.shape)
    extrap = extrap_next

    if context:
        rt.push(prev_data)
        rt.push(next_data)
        rt.insert(insert, selector, extrap, offset=1)
        prev_res = rt.select(rt.dt)
        next_res = rt.select(0.0)

    else:
        prev_res, next_res = extrap(
            insert, rt.dt - selector, prev_data, next_data, rt.dt
        )

    assert torch.all(prev_res == prev_data)
    assert torch.all(next_res == insert)


@pytest.mark.parametrize("bias", (0.1, 0.5, 0.9), ids=("next", "center", "prev"))
@pytest.mark.parametrize("context", (False, True), ids=("standalone", "recordtensor"))
def test_extrap_neighbors(rtmodule, bias, context):

    rt = rtmodule.rt

    prev_data = torch.rand(*rt.shape)
    next_data = torch.rand(*rt.shape)
    selector = torch.full(rt.shape, bias) * rt.dt
    insert = torch.rand(*rt.shape)
    extrap = extrap_neighbors

    if context:
        rt.push(prev_data)
        rt.push(next_data)
        rt.insert(insert, selector, extrap, offset=1)
        prev_res = rt.select(rt.dt)
        next_res = rt.select(0.0)

    else:
        prev_res, next_res = extrap(
            insert, rt.dt - selector, prev_data, next_data, rt.dt
        )

    assert torch.all(prev_res == insert)
    assert torch.all(next_res == insert)


@pytest.mark.parametrize("bias", (0.1, 0.5, 0.9), ids=("next", "center", "prev"))
@pytest.mark.parametrize("context", (False, True), ids=("standalone", "recordtensor"))
def test_extrap_nearest(rtmodule, bias, context):

    rt = rtmodule.rt

    prev_data = torch.rand(*rt.shape)
    next_data = torch.rand(*rt.shape)
    selector = torch.full(rt.shape, bias) * rt.dt
    insert = torch.rand(*rt.shape)
    extrap = extrap_nearest

    if context:
        rt.push(prev_data)
        rt.push(next_data)
        rt.insert(insert, selector, extrap, offset=1)
        prev_res = rt.select(rt.dt)
        next_res = rt.select(0.0)

    else:
        prev_res, next_res = extrap(
            insert, rt.dt - selector, prev_data, next_data, rt.dt
        )

    match bias:
        case 0.1:
            assert torch.all(prev_res == prev_data)
            assert torch.all(next_res == insert)
        case 0.5:
            assert torch.all(prev_res == insert)
            assert torch.all(next_res == next_data)
        case 0.9:
            assert torch.all(prev_res == insert)
            assert torch.all(next_res == next_data)
        case _:
            raise ValueError(f"invalid bias '{bias}' specified")


@pytest.mark.parametrize("bias", (0.1, 0.5, 0.9), ids=("next", "center", "prev"))
@pytest.mark.parametrize("context", (False, True), ids=("standalone", "recordtensor"))
def test_extrap_linear_forward(rtmodule, bias, context):

    rt = rtmodule.rt

    prev_data = torch.rand(*rt.shape)
    next_data = torch.rand(*rt.shape)
    selector = torch.full(rt.shape, bias) * rt.dt
    insert = torch.rand(*rt.shape)
    extrap = extrap_linear_forward

    if context:
        rt.push(prev_data)
        rt.push(next_data)
        rt.insert(insert, selector, extrap, offset=1)
        prev_res = rt.select(rt.dt)
        next_res = rt.select(0.0)

    else:
        prev_res, next_res = extrap(
            insert, rt.dt - selector, prev_data, next_data, rt.dt
        )

    proj = ((insert - prev_data) / (rt.dt - selector)) * rt.dt + prev_data

    assert torch.all(prev_res == prev_data)
    assert torch.all(next_res == proj)


@pytest.mark.parametrize("bias", (0.1, 0.5, 0.9), ids=("next", "center", "prev"))
@pytest.mark.parametrize("context", (False, True), ids=("standalone", "recordtensor"))
def test_extrap_linear_backward(rtmodule, bias, context):

    rt = rtmodule.rt

    prev_data = torch.rand(*rt.shape)
    next_data = torch.rand(*rt.shape)
    selector = torch.full(rt.shape, bias) * rt.dt
    insert = torch.rand(*rt.shape)
    extrap = extrap_linear_backward

    if context:
        rt.push(prev_data)
        rt.push(next_data)
        rt.insert(insert, selector, extrap, offset=1)
        prev_res = rt.select(rt.dt)
        next_res = rt.select(0.0)

    else:
        prev_res, next_res = extrap(
            insert, rt.dt - selector, prev_data, next_data, rt.dt
        )

    proj = next_data - ((next_data - insert) / selector) * rt.dt

    assert torch.all((prev_res - proj).abs().amax() < 5e-6)
    assert torch.all(next_res == next_data)


@pytest.mark.parametrize("bias", (0.1, 0.5, 0.9), ids=("next", "center", "prev"))
@pytest.mark.parametrize("context", (False, True), ids=("standalone", "recordtensor"))
def test_extrap_expdecay(rtmodule, bias, context):

    rt = rtmodule.rt

    tc = random.uniform(20, 30)
    prev_data = torch.rand(*rt.shape)
    next_data = torch.rand(*rt.shape)
    selector = torch.full(rt.shape, bias) * rt.dt
    insert = torch.rand(*rt.shape)
    extrap = extrap_expdecay

    if context:
        rt.push(prev_data)
        rt.push(next_data)
        rt.insert(
            insert, selector, extrap, offset=1, extrap_kwargs={"time_constant": tc}
        )
        prev_res = rt.select(rt.dt)
        next_res = rt.select(0.0)

    else:
        prev_res, next_res = extrap(
            insert, rt.dt - selector, prev_data, next_data, rt.dt, time_constant=tc
        )

    prev_proj = insert * torch.exp((rt.dt - selector) / tc)
    next_proj = insert * torch.exp(-selector / tc)

    assert torch.all((prev_res - prev_proj).abs().amax() < 1e-6)
    assert torch.all((next_res - next_proj).abs().amax() < 1e-6)


@pytest.mark.parametrize("bias", (0.1, 0.5, 0.9), ids=("next", "center", "prev"))
@pytest.mark.parametrize("context", (False, True), ids=("standalone", "recordtensor"))
def test_extrap_expratedecay(rtmodule, bias, context):

    rt = rtmodule.rt

    tc = random.uniform(20, 30)
    rc = 1.0 / tc
    prev_data = torch.rand(*rt.shape)
    next_data = torch.rand(*rt.shape)
    selector = torch.full(rt.shape, bias) * rt.dt
    insert = torch.rand(*rt.shape)
    extrap = extrap_expratedecay

    if context:
        rt.push(prev_data)
        rt.push(next_data)
        rt.insert(
            insert, selector, extrap, offset=1, extrap_kwargs={"rate_constant": rc}
        )
        prev_res = rt.select(rt.dt)
        next_res = rt.select(0.0)

    else:
        prev_res, next_res = extrap(
            insert, rt.dt - selector, prev_data, next_data, rt.dt, rate_constant=rc
        )

    prev_proj = insert * torch.exp((rt.dt - selector) / tc)
    next_proj = insert * torch.exp(-selector / tc)

    assert torch.all((prev_res - prev_proj).abs().amax() < 1e-6)
    assert torch.all((next_res - next_proj).abs().amax() < 1e-6)
