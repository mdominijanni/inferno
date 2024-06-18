from itertools import repeat
import einops as ein
import pytest
import random
import torch
import torch.nn as nn
import weakref
import uuid

from inferno import Module, ShapedTensor, RecordTensor, VirtualTensor


@pytest.fixture
def infernomodule():
    return Module()


@pytest.fixture
def testattr():
    return f"testvar_{uuid.uuid4().hex}"


def randshape(mindims=1, maxdims=4, minsize=1, maxsize=4):
    return tuple(
        random.randint(minsize, maxsize)
        for _ in range(random.randint(mindims, maxdims))
    )


class TestShapedTensor:

    @pytest.mark.parametrize(
        "persist_data",
        (True, False),
        ids=("persist_data=True", "persist_data=False"),
    )
    @pytest.mark.parametrize(
        "persist_constraints",
        (True, False),
        ids=("persist_constraints=True", "persist_constraints=False"),
    )
    @pytest.mark.parametrize(
        "initialized_param",
        (True, False),
        ids=("param_type=initialized", "param_type=uninitialized"),
    )
    @pytest.mark.parametrize(
        "constrained",
        (True, False),
        ids=("constrained=True", "constrained=False"),
    )
    def test_parameter_init(
        self,
        infernomodule,
        testattr,
        persist_data,
        persist_constraints,
        initialized_param,
        constrained,
    ):
        data = (
            nn.Parameter(torch.rand(randshape()))
            if initialized_param
            else nn.UninitializedParameter()
        )

        constraints = (
            (
                {0: data.shape[0]}
                if initialized_param
                else {random.randint(-3, 3): random.randint(0, 9)}
            )
            if constrained
            else None
        )
        setattr(
            infernomodule,
            testattr,
            ShapedTensor(
                infernomodule,
                testattr,
                data,
                constraints=constraints,
                persist_data=persist_data,
                persist_constraints=persist_constraints,
            ),
        )

        assert id(getattr(infernomodule, testattr).value) == id(data)
        assert hasattr(infernomodule, f"_{testattr}_data")
        assert f"_{testattr}_data" in infernomodule._parameters
        assert hasattr(infernomodule, f"_{testattr}_constraints")
        if persist_constraints:
            assert f"_{testattr}_constraints" in infernomodule._extras

    @pytest.mark.parametrize(
        "persist_data",
        (True, False),
        ids=("persist_data=True", "persist_data=False"),
    )
    @pytest.mark.parametrize(
        "persist_constraints",
        (True, False),
        ids=("persist_constraints=True", "persist_constraints=False"),
    )
    @pytest.mark.parametrize(
        "initialized_buffer",
        (True, False, None),
        ids=(
            "buffer_type=initialized",
            "buffer_type=uninitialized",
            "buffer_type=None",
        ),
    )
    @pytest.mark.parametrize(
        "constrained",
        (True, False),
        ids=("constrained=True", "constrained=False"),
    )
    def test_buffer_init(
        self,
        infernomodule,
        testattr,
        persist_data,
        persist_constraints,
        initialized_buffer,
        constrained,
    ):
        data = (
            torch.rand(randshape())
            if initialized_buffer
            else (None if initialized_buffer is None else nn.UninitializedBuffer())
        )
        constraints = (
            (
                {0: data.shape[0]}
                if initialized_buffer
                else {random.randint(-3, 3): random.randint(0, 9)}
            )
            if constrained
            else None
        )
        setattr(
            infernomodule,
            testattr,
            ShapedTensor(
                infernomodule,
                testattr,
                data,
                constraints=constraints,
                persist_data=persist_data,
                persist_constraints=persist_constraints,
            ),
        )

        assert id(getattr(infernomodule, testattr).value) == id(data)
        assert hasattr(infernomodule, f"_{testattr}_data")
        assert f"_{testattr}_data" in infernomodule._buffers
        if not persist_data:
            assert f"_{testattr}_data" in infernomodule._non_persistent_buffers_set
        assert hasattr(infernomodule, f"_{testattr}_constraints")
        if persist_constraints:
            assert f"_{testattr}_constraints" in infernomodule._extras

    @pytest.mark.parametrize(
        "asparam",
        (False, True),
        ids=("buffer", "parameter"),
    )
    def test_finalizer(self, infernomodule, testattr, asparam):
        if asparam:
            data = nn.Parameter(torch.rand(randshape()), True)
        else:
            data = torch.rand(randshape())

        setattr(infernomodule, testattr, ShapedTensor(infernomodule, testattr, data))

        assert hasattr(infernomodule, f"_{testattr}_data")
        assert hasattr(infernomodule, f"_{testattr}_constraints")

        delattr(infernomodule, testattr)

        assert not hasattr(infernomodule, f"_{testattr}_data")
        assert not hasattr(infernomodule, f"_{testattr}_constraints")

    @pytest.mark.parametrize(
        "asparam",
        (False, True),
        ids=("buffer", "parameter"),
    )
    def test_add_valid_constraint(self, infernomodule, testattr, asparam):
        shape = randshape(2, 5)
        dim = random.randint(-len(shape), len(shape) - 1)
        size = shape[dim]
        data = torch.rand(shape)

        setattr(
            infernomodule,
            testattr,
            ShapedTensor(
                infernomodule,
                testattr,
                (
                    nn.Parameter(data.clone().detach(), True)
                    if asparam
                    else data.clone().detach()
                ),
            ),
        )

        getattr(infernomodule, testattr).reconstrain(dim, size)
        assert torch.all(getattr(infernomodule, testattr).value == data)
        assert getattr(infernomodule, testattr).constraints == {dim: size}

    @pytest.mark.parametrize(
        "asparam",
        (False, True),
        ids=("buffer", "parameter"),
    )
    def test_add_invalid_constraint(self, infernomodule, testattr, asparam):
        shape = randshape(2, 5)
        dim = random.randint(-len(shape), len(shape) - 1)
        size = shape[dim]
        while size == shape[dim]:
            size = random.randint(0, 9)
        data = torch.rand(shape)

        setattr(
            infernomodule,
            testattr,
            ShapedTensor(
                infernomodule,
                testattr,
                (
                    nn.Parameter(data.clone().detach(), True)
                    if asparam
                    else data.clone().detach()
                ),
            ),
        )

        with pytest.raises(ValueError) as excinfo:
            getattr(infernomodule, testattr).reconstrain(dim, size)

        assert (
            "constrained tensor would be invalidated by constraint of "
            f"size {size} on dim {dim}"
        ) in str(excinfo.value)

    @pytest.mark.parametrize(
        "asparam",
        (False, True),
        ids=("buffer", "parameter"),
    )
    def test_delete_valid_constraint(self, infernomodule, testattr, asparam):
        shape = randshape(2, 5)
        dim = random.randint(-len(shape), len(shape) - 1)
        size = shape[dim]
        data = torch.rand(shape)

        setattr(
            infernomodule,
            testattr,
            ShapedTensor(
                infernomodule,
                testattr,
                (
                    nn.Parameter(data.clone().detach(), True)
                    if asparam
                    else data.clone().detach()
                ),
                constraints={dim: size},
            ),
        )

        getattr(infernomodule, testattr).reconstrain(dim, None)
        assert dim not in getattr(infernomodule, testattr).constraints
        assert torch.all(getattr(infernomodule, testattr).value == data)

    @pytest.mark.parametrize(
        "asparam",
        (False, True),
        ids=("buffer", "parameter"),
    )
    def test_delete_now_valid_constraint(self, infernomodule, testattr, asparam):
        shape = randshape(2, 5)
        dim = random.randint(-len(shape), len(shape) - 1)
        size = shape[dim]
        data = torch.rand(shape)

        setattr(
            infernomodule,
            testattr,
            ShapedTensor(
                infernomodule,
                testattr,
                (
                    nn.Parameter(data.clone().detach(), True)
                    if asparam
                    else data.clone().detach()
                ),
                constraints={dim: size},
                live=False,
            ),
        )

        repshp = list(repeat(1, times=len(shape)))
        repshp[dim] = 2

        if asparam:
            getattr(infernomodule, f"_{testattr}_data").data = getattr(
                infernomodule, f"_{testattr}_data"
            ).repeat(repshp)
        else:
            setattr(
                infernomodule,
                f"_{testattr}_data",
                getattr(infernomodule, f"_{testattr}_data").repeat(repshp),
            )

        getattr(infernomodule, testattr).reconstrain(dim, None)
        assert dim not in getattr(infernomodule, testattr).constraints

    @pytest.mark.parametrize(
        "asparam",
        (False, True),
        ids=("buffer", "parameter"),
    )
    def test_delete_invalid_constraint(self, infernomodule, testattr, asparam):
        shape = randshape(2, 5)
        dim = random.randint(0, len(shape) - 1)
        size = shape[dim]
        data = torch.rand(shape)

        setattr(
            infernomodule,
            testattr,
            ShapedTensor(
                infernomodule,
                testattr,
                (
                    nn.Parameter(data.clone().detach(), True)
                    if asparam
                    else data.clone().detach()
                ),
                constraints={
                    dim: size,
                    ((dim + 1) % len(shape)): shape[(dim + 1) % len(shape)],
                },
                live=False,
            ),
        )

        repshp = list(repeat(1, times=len(shape)))
        repshp[(dim + 1) % len(shape)] = 2

        if asparam:
            getattr(infernomodule, f"_{testattr}_data").data = getattr(
                infernomodule, f"_{testattr}_data"
            ).repeat(repshp)
        else:
            setattr(
                infernomodule,
                f"_{testattr}_data",
                getattr(infernomodule, f"_{testattr}_data").repeat(repshp),
            )

        with pytest.raises(RuntimeError) as excinfo:
            getattr(infernomodule, testattr).reconstrain(dim, None)

        assert ("constrained tensor has been invalidated") in str(excinfo.value)

    @pytest.mark.parametrize(
        "asparam",
        (False, True),
        ids=("buffer", "parameter"),
    )
    @pytest.mark.parametrize("zerodim", (False, True), ids=("nonzerodim", "zerodim"))
    def test_edit_valid_constraint(self, infernomodule, testattr, asparam, zerodim):
        shape = list(randshape(2, 5))
        dim = random.randint(-len(shape), len(shape) - 1)

        if zerodim:
            shape[dim] = random.randint(1, 5)
            tgtsize = 0
            tgtshape = list(shape)
            tgtshape[dim] = tgtsize

        else:
            shape[dim] = shape[dim] + 2
            tgtsize = shape[dim] - 1
            tgtshape = list(shape)
            tgtshape[dim] = tgtsize

        size = shape[dim]
        data = torch.rand(shape)

        setattr(
            infernomodule,
            testattr,
            ShapedTensor(
                infernomodule,
                testattr,
                (
                    nn.Parameter(data.clone().detach(), True)
                    if asparam
                    else data.clone().detach()
                ),
                constraints={dim: size},
            ),
        )

        newdata = torch.rand(tgtshape)

        if asparam:
            getattr(infernomodule, f"_{testattr}_data").data = newdata.clone().detach()
        else:
            setattr(infernomodule, f"_{testattr}_data", newdata.clone().detach())

        getattr(infernomodule, testattr).reconstrain(dim, tgtsize)

        assert dim in getattr(infernomodule, testattr).constraints
        assert getattr(infernomodule, testattr).constraints[dim] == tgtsize
        assert torch.all(getattr(infernomodule, testattr).value == newdata)

    @pytest.mark.parametrize(
        "asparam",
        (False, True),
        ids=("buffer", "parameter"),
    )
    @pytest.mark.parametrize("zerodim", (False, True), ids=("nonzerodim", "zerodim"))
    @pytest.mark.parametrize("shrink", (False, True), ids=("expand", "shrink"))
    def test_edit_fixable_constraint(
        self, infernomodule, testattr, asparam, zerodim, shrink
    ):
        shape = list(randshape(2, 5))
        tgtshape = list(shape)
        dim = random.randint(-len(shape), len(shape) - 1)

        if zerodim:
            if shrink:
                shape[dim] = random.randint(1, 5)
                tgtshape[dim] = 0
            else:
                shape[dim] = 0
                tgtshape[dim] = random.randint(1, 5)

        else:
            if shrink:
                shape[dim] = random.randint(2, 5)
                tgtshape[dim] = shape[dim] - 1
            else:
                shape[dim] = random.randint(1, 5)
                tgtshape[dim] = shape[dim] + 1

        size, tgtsize = shape[dim], tgtshape[dim]
        data = torch.rand(shape)

        setattr(
            infernomodule,
            testattr,
            ShapedTensor(
                infernomodule,
                testattr,
                (
                    nn.Parameter(data.clone().detach(), True)
                    if asparam
                    else data.clone().detach()
                ),
                constraints={dim: size},
            ),
        )

        getattr(infernomodule, testattr).reconstrain(dim, tgtsize)
        assert dim in getattr(infernomodule, testattr).constraints
        assert getattr(infernomodule, testattr).constraints[dim] == tgtsize

        if zerodim:
            assert tuple(tgtshape) == tuple(
                getattr(infernomodule, testattr).value.shape
            )
            assert data.dtype == getattr(infernomodule, testattr).value.dtype

        elif shrink:
            slices = list(repeat(slice(None), times=len(shape)))
            slices[dim] = slice(1, None)
            assert torch.all(data[*slices] == getattr(infernomodule, testattr).value)
        else:
            slices = list(repeat(slice(None), times=len(shape)))
            slices[dim] = slice(None, 1)
            assert torch.all(getattr(infernomodule, testattr).value[*slices] == 0)


@pytest.fixture
def infmodule():
    return Module()


@pytest.fixture
def name():
    return f"testattr_{uuid.uuid4().hex}"


class TestRecordTensor:

    @pytest.mark.parametrize(
        "persist",
        (True, False),
        ids=("persist=True", "persist=False"),
    )
    @pytest.mark.parametrize(
        "valuetype",
        (
            None,
            torch.Tensor,
            nn.Parameter,
            nn.UninitializedBuffer,
            nn.UninitializedParameter,
        ),
        ids=(
            "type=None",
            "type=Tensor",
            "type=Parameter",
            "type=UninitializedBuffer",
            "type=UninitializedParameter",
        ),
    )
    def test_init_data_persistence(self, infmodule, name, persist, valuetype):

        match valuetype:
            case None:
                data = None
            case torch.Tensor:
                data = torch.rand(randshape())
                datacopy = data.clone().detach()
            case nn.Parameter:
                data = nn.Parameter(torch.rand(randshape()), True)
                datacopy = nn.Parameter(data.data.clone().detach(), True)
            case nn.UninitializedBuffer:
                data = nn.UninitializedBuffer()
            case nn.UninitializedParameter:
                data = nn.UninitializedParameter()

        setattr(
            infmodule,
            name,
            RecordTensor(
                infmodule,
                name,
                step_time=random.uniform(0.75, 1.25),
                duration=random.uniform(3.25, 5.0),
                value=data,
                persist_data=persist,
                persist_constraints=False,
                persist_temporal=False,
            ),
        )

        assert hasattr(infmodule, getattr(infmodule, name).attributes.data)

        dataname = getattr(infmodule, name).attributes.data

        match valuetype:

            case None:
                assert getattr(infmodule, name).value is None
                assert dataname in infmodule._buffers
                if not persist:
                    assert dataname in infmodule._non_persistent_buffers_set
                else:
                    assert dataname not in infmodule._non_persistent_buffers_set

            case torch.Tensor:
                assert torch.all(
                    getattr(infmodule, name).value == datacopy.unsqueeze(0)
                )
                assert dataname in infmodule._buffers
                if not persist:
                    assert dataname in infmodule._non_persistent_buffers_set
                else:
                    assert dataname not in infmodule._non_persistent_buffers_set

            case nn.Parameter:
                assert torch.all(
                    getattr(infmodule, name).value == datacopy.unsqueeze(0)
                )
                assert dataname in infmodule._parameters

            case nn.UninitializedBuffer:
                assert id(getattr(infmodule, name).value) == id(data)
                assert dataname in infmodule._buffers
                if not persist:
                    assert dataname in infmodule._non_persistent_buffers_set
                else:
                    assert dataname not in infmodule._non_persistent_buffers_set

            case nn.UninitializedParameter:
                assert id(getattr(infmodule, name).value) == id(data)
                assert dataname in infmodule._parameters

    @pytest.mark.parametrize(
        "persist",
        (True, False),
        ids=("persist=True", "persist=False"),
    )
    def test_init_constraint_persistence(self, infmodule, name, persist):
        constraints = {}
        offset_constraints = {}

        for _ in range(random.randint(5, 11)):
            dim, size = random.randint(-4, 5), random.randint(1, 9)
            constraints[dim] = size
            if dim >= 0:
                offset_constraints[dim + 1] = size
            else:
                offset_constraints[dim] = size

        recordsz = random.randint(1, 13)

        setattr(
            infmodule,
            name,
            RecordTensor(
                infmodule,
                name,
                step_time=1.0,
                duration=(recordsz - 1.0),
                value=None,
                constraints=constraints,
                persist_data=False,
                persist_constraints=persist,
                persist_temporal=False,
            ),
        )

        cstrname = getattr(infmodule, name).attributes.constraints

        assert hasattr(infmodule, cstrname)
        assert offset_constraints | {0: recordsz} == getattr(infmodule, cstrname)
        if persist:
            assert cstrname in infmodule._extras
        else:
            assert cstrname not in infmodule._extras

    @pytest.mark.parametrize(
        "persist",
        (True, False),
        ids=("persist=True", "persist=False"),
    )
    def test_init_temporal_persistence(self, infmodule, name, persist):
        constraints = {}
        for _ in range(random.randint(3, 7)):
            constraints[random.randint(-4, 5)] = random.randint(1, 9)

        step_time = random.uniform(0.75, 1.25)
        duration = random.uniform(3.25, 5.0)

        setattr(
            infmodule,
            name,
            RecordTensor(
                infmodule,
                name,
                step_time=step_time,
                duration=duration,
                value=None,
                persist_data=False,
                persist_constraints=False,
                persist_temporal=persist,
            ),
        )

        dtname = getattr(infmodule, name).attributes.dt
        durname = getattr(infmodule, name).attributes.duration

        assert hasattr(infmodule, dtname)
        assert hasattr(infmodule, durname)
        assert step_time == getattr(infmodule, dtname)
        assert duration == getattr(infmodule, durname)
        if persist:
            assert dtname in infmodule._extras
            assert durname in infmodule._extras
        else:
            assert dtname not in infmodule._extras
            assert durname not in infmodule._extras

    @pytest.mark.parametrize(
        "asparam",
        (True, False),
        ids=("parameter", "buffer"),
    )
    def test_init_reshapes(self, infmodule, name, asparam):
        data = torch.rand(randshape())
        if asparam:
            d2 = nn.Parameter(data.clone().detach(), True)
        else:
            d2 = data.clone().detach()

        rt = RecordTensor(
            infmodule,
            name,
            step_time=random.uniform(0.75, 1.25),
            duration=random.uniform(3.25, 5.0),
            value=d2,
            persist_data=True,
            persist_constraints=True,
            persist_temporal=True,
        )
        setattr(infmodule, name, rt)

        assert tuple(rt.value.shape) == (rt.recordsz, *data.shape)

    def test_create(self, infmodule, name):
        RecordTensor.create(
            infmodule,
            name,
            step_time=random.uniform(0.75, 1.25),
            duration=random.uniform(3.25, 5.0),
            value=torch.rand(randshape()),
            persist_data=True,
            persist_constraints=True,
            persist_temporal=True,
        )

        assert hasattr(infmodule, name)
        for attr in getattr(infmodule, name).attributes:
            assert hasattr(infmodule, attr)

        assert getattr(infmodule, name).attributes.pointer in infmodule._extras

    @pytest.mark.parametrize(
        "asparam",
        (True, False),
        ids=("parameter", "buffer"),
    )
    def test_finalizer(self, infmodule, name, asparam):
        if asparam:
            RecordTensor.create(
                infmodule,
                name,
                step_time=random.uniform(0.75, 1.25),
                duration=random.uniform(3.25, 5.0),
                value=nn.Parameter(torch.rand(randshape()), True),
                persist_data=True,
                persist_constraints=True,
                persist_temporal=True,
            )
        else:
            RecordTensor.create(
                infmodule,
                name,
                step_time=random.uniform(0.75, 1.25),
                duration=random.uniform(3.25, 5.0),
                value=torch.rand(randshape()),
                persist_data=True,
                persist_constraints=True,
                persist_temporal=True,
            )

        attributes = (*getattr(infmodule, name).attributes,)

        assert hasattr(infmodule, name)
        for attr in attributes:
            assert hasattr(infmodule, attr)

        delattr(infmodule, name)

        assert not hasattr(infmodule, name)
        for attr in attributes:
            assert not hasattr(infmodule, attr)

    def test_constraint_mapping(self, infmodule, name):
        constraints, adjusted = {}, {}
        for d in range(-3, 4):
            constraints[d] = random.randint(1, 9)
            adjusted[d + (d >= 0)] = constraints[d]

        rt = RecordTensor(
            infmodule,
            name,
            step_time=random.uniform(0.75, 1.25),
            duration=random.uniform(3.25, 5.0),
            value=None,
            constraints=constraints,
            persist_data=True,
            persist_constraints=True,
            persist_temporal=True,
        )
        setattr(infmodule, name, rt)

        assert rt.constraints == constraints
        assert ShapedTensor.constraints.fget(rt) == (adjusted | {0: rt.recordsz})

    def test_tensor_parameter_assignment(self, infmodule, name):
        data = nn.Parameter(torch.rand(randshape()), False)
        rt = RecordTensor(
            infmodule,
            name,
            step_time=random.uniform(0.75, 1.25),
            duration=random.uniform(3.25, 5.0),
            value=data,
            persist_data=True,
            persist_constraints=True,
            persist_temporal=True,
        )
        setattr(infmodule, name, rt)

        assert isinstance(rt.value, nn.Parameter)

        newdata = torch.rand_like(rt.value)
        rt.value = newdata

        assert isinstance(rt.value, nn.Parameter)
        assert torch.all(rt.value == newdata)

    @pytest.mark.parametrize(
        "asparam",
        (True, False),
        ids=("parameter", "buffer"),
    )
    def test_align(self, infmodule, name, asparam):
        rt = RecordTensor(
            infmodule,
            name,
            step_time=random.uniform(0.75, 1.25),
            duration=random.uniform(3.25, 5.0),
            value=(nn.Parameter(torch.empty(0), True) if asparam else torch.empty(0)),
            persist_data=True,
            persist_constraints=True,
            persist_temporal=True,
        )
        setattr(infmodule, name, rt)

        data = torch.rand(rt.recordsz, *randshape())
        rt.value = data.clone().detach()

        index = random.randint(-rt.recordsz, rt.recordsz - 1)
        rt.align(index)

        assert torch.all(rt.value == torch.roll(data, index, 0))

        if asparam:
            assert isinstance(rt.value, nn.Parameter)

    @pytest.mark.parametrize(
        "asparam",
        (True, False),
        ids=("parameter", "buffer"),
    )
    @pytest.mark.parametrize(
        "fill",
        (None, 2.7182818),
        ids=("parameter", "buffer"),
    )
    def test_reset(self, infmodule, name, asparam, fill):
        rt = RecordTensor(
            infmodule,
            name,
            step_time=random.uniform(0.75, 1.25),
            duration=random.uniform(3.25, 5.0),
            value=(nn.Parameter(torch.empty(0), True) if asparam else torch.empty(0)),
            persist_data=True,
            persist_constraints=True,
            persist_temporal=True,
        )
        setattr(infmodule, name, rt)

        data = torch.rand(rt.recordsz, *randshape())
        rt.value = data.clone().detach()

        ptr = random.randint(1, rt.recordsz)
        setattr(infmodule, rt.attributes.pointer, ptr)
        rt.reset(fill)

        assert rt.pointer == 0

        if fill:
            assert torch.all(rt.value == fill)
        else:
            assert torch.all(rt.value == data.roll(-ptr, 0))

        if asparam:
            assert isinstance(rt.value, nn.Parameter)
            assert rt.value.requires_grad

    @pytest.mark.parametrize(
        "valuetype",
        (
            None,
            torch.Tensor,
            nn.Parameter,
            nn.UninitializedBuffer,
            nn.UninitializedParameter,
        ),
        ids=(
            "type=None",
            "type=Tensor",
            "type=Parameter",
            "type=UninitializedBuffer",
            "type=UninitializedParameter",
        ),
    )
    @pytest.mark.parametrize(
        "usespec",
        (True, False),
        ids=("override=True", "override=False"),
    )
    def test_initialize(self, infmodule, name, valuetype, usespec):
        dtylist = [torch.float16, torch.float32, torch.int32, torch.complex64]
        devlist = ["cpu"]
        # if hascuda:
        #    devlist.append("cuda")
        # if hasmps:
        #    devlist.append("mps")

        shape = randshape(mindims=2, maxdims=4)
        dtype = dtylist[random.randint(0, len(dtylist) - 1)]
        device = devlist[random.randint(0, len(devlist) - 1)]

        or_dtype = dtylist[random.randint(0, len(dtylist) - 1)] if usespec else None
        or_device = devlist[random.randint(0, len(devlist) - 1)] if usespec else None

        checktype = (
            or_dtype if usespec else (torch.float32 if valuetype is None else dtype)
        )
        checkdevice = or_device if usespec else ("cpu" if valuetype is None else device)

        match valuetype:
            case None:
                data = None
            case torch.Tensor:
                data = torch.empty(0, dtype=dtype, device=device)
            case nn.Parameter:
                data = nn.Parameter(torch.empty(0, dtype=dtype, device=device), False)
            case nn.UninitializedBuffer:
                data = nn.UninitializedBuffer(dtype=dtype, device=device)
            case nn.UninitializedParameter:
                data = nn.UninitializedParameter(
                    dtype=dtype, device=device, requires_grad=False
                )

        rt = RecordTensor(
            infmodule,
            name,
            step_time=random.uniform(0.75, 1.25),
            duration=random.uniform(3.25, 5.0),
            value=data,
            persist_data=True,
            persist_constraints=True,
            persist_temporal=True,
        )
        setattr(infmodule, name, rt)

        rt.initialize(shape, device=or_device, dtype=or_dtype, fill=1.0)

        assert rt.value.dtype == checktype
        assert str(rt.value.device).partition(":")[0] == checkdevice
        assert (*rt.value.shape,) == (rt.recordsz, *shape)
        assert torch.all(rt.value == 1)

        if valuetype in (nn.Parameter, nn.UninitializedParameter):
            assert isinstance(rt.value, nn.Parameter)
        else:
            assert not isinstance(rt.value, nn.Parameter)
            assert isinstance(rt.value, torch.Tensor)

    @pytest.mark.parametrize(
        "valuetype",
        (
            None,
            torch.Tensor,
            nn.Parameter,
            nn.UninitializedBuffer,
            nn.UninitializedParameter,
        ),
        ids=(
            "type=None",
            "type=Tensor",
            "type=Parameter",
            "type=UninitializedBuffer",
            "type=UninitializedParameter",
        ),
    )
    @pytest.mark.parametrize(
        "useuninit",
        (True, False),
        ids=("useinit=True", "usinit=False"),
    )
    def test_deinitialize(self, infmodule, name, valuetype, useuninit):
        dtylist = [torch.float16, torch.float32, torch.int32, torch.complex64]
        devlist = ["cpu"]
        # if hascuda:
        #    devlist.append("cuda")
        # if hasmps:
        #    devlist.append("mps")

        shape = randshape(mindims=2, maxdims=4)
        dtype = dtylist[random.randint(0, len(dtylist) - 1)]
        device = devlist[random.randint(0, len(devlist) - 1)]

        checktype = torch.float32 if valuetype is None else dtype
        checkdevice = "cpu" if valuetype is None else device

        match valuetype:
            case None:
                data = None
            case torch.Tensor:
                data = torch.empty(shape, dtype=dtype, device=device)
            case nn.Parameter:
                data = nn.Parameter(
                    torch.empty(shape, dtype=dtype, device=device), False
                )
            case nn.UninitializedBuffer:
                data = nn.UninitializedBuffer(dtype=dtype, device=device)
            case nn.UninitializedParameter:
                data = nn.UninitializedParameter(
                    dtype=dtype, device=device, requires_grad=False
                )

        rt = RecordTensor(
            infmodule,
            name,
            step_time=random.uniform(0.75, 1.25),
            duration=random.uniform(3.25, 5.0),
            value=data,
            persist_data=True,
            persist_constraints=True,
            persist_temporal=True,
        )
        setattr(infmodule, name, rt)

        rt.deinitialize(use_uninitialized=useuninit)

        assert rt.value.dtype == checktype
        assert str(rt.value.device).partition(":")[0] == checkdevice

        if valuetype in (nn.Parameter, nn.UninitializedParameter):
            if useuninit:
                assert isinstance(rt.value, nn.UninitializedParameter)
            else:
                assert isinstance(rt.value, nn.Parameter)
                assert not isinstance(rt.value, nn.UninitializedParameter)
        else:
            if useuninit:
                assert isinstance(rt.value, nn.UninitializedBuffer)
            else:
                assert isinstance(rt.value, torch.Tensor)
                assert not isinstance(rt.value, nn.UninitializedBuffer)

    def test_peek_latest_get(self, infmodule, name):
        rt = RecordTensor(
            infmodule,
            name,
            step_time=random.uniform(0.75, 1.25),
            duration=random.uniform(3.25, 5.0),
            value=None,
            persist_data=True,
            persist_constraints=True,
            persist_temporal=True,
        )
        setattr(infmodule, name, rt)

        rt.value = torch.rand(rt.recordsz, *randshape())
        ptr = random.randint(1, rt.recordsz - 1)
        setattr(infmodule, rt.attributes.pointer, ptr)

        assert torch.all(rt.peek() == rt.value[(ptr - 1) % rt.recordsz, ...])
        assert torch.all(rt.latest == rt.value[(ptr - 1) % rt.recordsz, ...])

    def test_pop(self, infmodule, name):
        rt = RecordTensor(
            infmodule,
            name,
            step_time=random.uniform(0.75, 1.25),
            duration=random.uniform(3.25, 5.0),
            value=None,
            persist_data=True,
            persist_constraints=True,
            persist_temporal=True,
        )
        setattr(infmodule, name, rt)

        rt.value = torch.rand(rt.recordsz, *randshape())
        ptr = random.randint(1, rt.recordsz - 1)
        setattr(infmodule, rt.attributes.pointer, ptr)

        assert torch.all(rt.pop() == rt.value[(ptr - 1) % rt.recordsz, ...])
        assert rt.pointer == ptr - 1

        setattr(infmodule, rt.attributes.pointer, ptr)

        assert torch.all(rt.latest == rt.value[(ptr - 1) % rt.recordsz, ...])

    def test_latest_del(self, infmodule, name):
        rt = RecordTensor(
            infmodule,
            name,
            step_time=random.uniform(0.75, 1.25),
            duration=random.uniform(3.25, 5.0),
            value=None,
            persist_data=True,
            persist_constraints=True,
            persist_temporal=True,
        )
        setattr(infmodule, name, rt)

        data = torch.rand(rt.recordsz, *randshape())
        rt.value = data.clone().detach()
        ptr = random.randint(1, rt.recordsz - 1)
        setattr(infmodule, rt.attributes.pointer, ptr)

        del rt.latest

        assert rt.pointer == ptr - 1
        assert torch.all(rt.value == data)

    @pytest.mark.parametrize(
        "inplace",
        (True, False),
        ids=("inplace", "normal"),
    )
    def test_push(self, infmodule, name, inplace):
        rt = RecordTensor(
            infmodule,
            name,
            step_time=random.uniform(0.75, 1.25),
            duration=random.uniform(3.25, 5.0),
            value=None,
            persist_data=True,
            persist_constraints=True,
            persist_temporal=True,
        )
        setattr(infmodule, name, rt)

        shape = randshape()

        data = torch.rand(rt.recordsz, *shape)
        rt.value = data.clone().detach()
        ptr = random.randint(1, rt.recordsz - 1)
        setattr(infmodule, rt.attributes.pointer, ptr)

        obs = torch.rand(shape)
        obsclone = obs.clone().detach()
        obsclone.requires_grad = True
        rt.push(obsclone, inplace)

        assert rt.pointer == (ptr + 1) % rt.recordsz
        for d in range(rt.recordsz):
            if d == ptr:
                assert torch.all(rt.value[d, ...] == obs)
            else:
                assert torch.all(rt.value[d, ...] == data[d, ...])

    @pytest.mark.parametrize(
        "inplace",
        (True, False),
        ids=("inplace", "normal"),
    )
    @pytest.mark.parametrize(
        "asparam",
        (True, False),
        ids=("parameter", "buffer"),
    )
    def test_push_init(self, infmodule, name, inplace, asparam):
        rt = RecordTensor(
            infmodule,
            name,
            step_time=random.uniform(0.75, 1.25),
            duration=random.uniform(3.25, 5.0),
            value=(nn.Parameter(torch.empty(0), True) if asparam else torch.empty(0)),
            persist_data=True,
            persist_constraints=True,
            persist_temporal=True,
        )
        setattr(infmodule, name, rt)

        shape = randshape()

        obs = torch.rand(*shape)
        rt.push(obs.clone().detach(), inplace)

        if asparam:
            assert isinstance(rt.value, nn.Parameter)
            assert rt.value.requires_grad

        assert (*rt.value.shape,) == (rt.recordsz, *shape)
        assert torch.all(rt.value[0, ...] == obs)
        assert torch.all(rt.value[1:, ...] == 0)

    def test_latest_set(self, infmodule, name):
        rt = RecordTensor(
            infmodule,
            name,
            step_time=random.uniform(0.75, 1.25),
            duration=random.uniform(3.25, 5.0),
            value=None,
            persist_data=True,
            persist_constraints=True,
            persist_temporal=True,
        )
        setattr(infmodule, name, rt)

        shape = randshape()

        data = torch.rand(rt.recordsz, *shape)
        rt.value = data.clone().detach()
        ptr = random.randint(1, rt.recordsz - 1)
        setattr(infmodule, rt.attributes.pointer, ptr)

        obs = torch.rand(shape)
        obsclone = obs.clone().detach()
        obsclone.requires_grad = True
        rt.latest = obsclone

        assert rt.pointer == (ptr + 1) % rt.recordsz
        for d in range(rt.recordsz):
            if d == ptr:
                assert torch.all(rt.value[d, ...] == obs)
            else:
                assert torch.all(rt.value[d, ...] == data[d, ...])

    @pytest.mark.parametrize(
        "asparam",
        (True, False),
        ids=("parameter", "buffer"),
    )
    def test_read(self, infmodule, name, asparam):
        rt = RecordTensor(
            infmodule,
            name,
            step_time=random.uniform(0.75, 1.25),
            duration=random.uniform(3.25, 5.0),
            value=None,
            persist_data=True,
            persist_constraints=True,
            persist_temporal=True,
        )
        setattr(infmodule, name, rt)

        shape = randshape(mindims=2, maxdims=4)
        if asparam:
            rt.value = nn.Parameter(torch.zeros(rt.recordsz, *shape).float(), True)
        else:
            rt.value = torch.zeros(rt.recordsz, *shape).float()

        data = [torch.rand(shape) for _ in range(rt.recordsz)]
        for d in data:
            rt.push(d)

        for idx, d in enumerate(data):
            torch.all(d == rt.read(rt.recordsz - idx - 1))

    @pytest.mark.parametrize(
        "inplace",
        (True, False),
        ids=("inplace", "normal"),
    )
    @pytest.mark.parametrize(
        "asparam",
        (True, False),
        ids=("parameter", "buffer"),
    )
    def test_write(self, infmodule, name, asparam, inplace):
        rt = RecordTensor(
            infmodule,
            name,
            step_time=random.uniform(0.75, 1.25),
            duration=random.uniform(3.25, 5.0),
            value=None,
            persist_data=True,
            persist_constraints=True,
            persist_temporal=True,
        )
        setattr(infmodule, name, rt)

        shape = randshape(mindims=2, maxdims=4)
        if asparam:
            rt.value = nn.Parameter(torch.zeros(rt.recordsz, *shape).float(), True)
        else:
            rt.value = torch.zeros(rt.recordsz, *shape).float()

        data = [torch.rand(shape) for _ in range(rt.recordsz)]
        for idx, d in enumerate(data):
            rt.write(d, idx, inplace)

        for idx, d in enumerate(data):
            torch.all(d == rt.value[rt.recordsz - idx - 1, ...])

    @pytest.mark.parametrize(
        "asparam",
        (True, False),
        ids=("parameter", "buffer"),
    )
    @pytest.mark.parametrize(
        "forward",
        (True, False),
        ids=("forward", "backward"),
    )
    def test_readrange_contiguous_int(self, infmodule, name, asparam, forward):
        rt = RecordTensor(
            infmodule,
            name,
            step_time=random.uniform(0.75, 1.25),
            duration=random.uniform(16.25, 20.0),
            value=None,
            persist_data=True,
            persist_constraints=True,
            persist_temporal=True,
        )
        setattr(infmodule, name, rt)

        shape = randshape(mindims=2, maxdims=4)
        data = torch.rand(rt.recordsz, *shape)
        if asparam:
            rt.value = nn.Parameter(data.clone().detach(), True)
        else:
            rt.value = data.clone().detach()
        setattr(infmodule, rt.attributes.pointer, rt.recordsz - 1)

        start = random.randint(3, 6)
        end = start + random.randint(1, 5)

        if forward:
            res = rt.readrange(end - start, rt.pointer - start, True)
        else:
            res = rt.readrange(end - start, rt.pointer - end + 1, False)

        assert torch.all(res == ein.rearrange(data[start:end, ...], "t ... -> ... t"))

    @pytest.mark.parametrize(
        "asparam",
        (True, False),
        ids=("parameter", "buffer"),
    )
    @pytest.mark.parametrize(
        "forward",
        (True, False),
        ids=("forward", "backward"),
    )
    def test_readrange_noncontiguous_int(self, infmodule, name, asparam, forward):
        rt = RecordTensor(
            infmodule,
            name,
            step_time=random.uniform(0.75, 1.25),
            duration=random.uniform(16.25, 20.0),
            value=None,
            persist_data=True,
            persist_constraints=True,
            persist_temporal=True,
        )
        setattr(infmodule, name, rt)

        shape = randshape(mindims=2, maxdims=4)
        data = torch.rand(rt.recordsz, *shape)
        if asparam:
            rt.value = nn.Parameter(data.clone().detach(), True)
        else:
            rt.value = data.clone().detach()
        setattr(infmodule, rt.attributes.pointer, rt.recordsz - 1)

        start = rt.recordsz - random.randint(3, 6)
        end = random.randint(1, 5)

        if forward:
            res = rt.readrange(end + rt.pointer - (start) + 1, rt.pointer - start, True)
        else:
            res = rt.readrange(
                end + rt.pointer - (start) + 1, rt.pointer - end + 1, False
            )

        assert torch.all(
            res
            == ein.rearrange(
                torch.cat((data[start:, ...], data[:end, ...]), 0), "t ... -> ... t"
            )
        )

    @pytest.mark.parametrize(
        "asparam",
        (True, False),
        ids=("parameter", "buffer"),
    )
    @pytest.mark.parametrize(
        "forward",
        (True, False),
        ids=("forward", "backward"),
    )
    def test_readrange_tensor(self, infmodule, name, asparam, forward):
        rt = RecordTensor(
            infmodule,
            name,
            step_time=random.uniform(0.75, 1.25),
            duration=random.uniform(16.25, 20.0),
            value=None,
            persist_data=True,
            persist_constraints=True,
            persist_temporal=True,
        )
        setattr(infmodule, name, rt)

        shape = randshape(mindims=2, maxdims=4)
        data = torch.rand(rt.recordsz, *shape)
        if asparam:
            rt.value = nn.Parameter(data.clone().detach(), True)
        else:
            rt.value = data.clone().detach()

        length = random.randint(3, 11)
        offsets = torch.randint(0, rt.recordsz, shape)
        ptr = random.randint(0, rt.recordsz - 1)

        setattr(infmodule, rt.attributes.pointer, ptr)

        res = rt.readrange(length, offsets, forward)

        offsets = offsets + (length - 1) * (not forward)
        offsets = torch.stack([offsets - k for k in range(length)], 0)

        assert torch.all(
            res
            == ein.rearrange(
                data.gather(0, (ptr - offsets) % rt.recordsz), "t ... -> ... t"
            )
        )

    @pytest.mark.parametrize(
        "asparam",
        (True, False),
        ids=("parameter", "buffer"),
    )
    @pytest.mark.parametrize(
        "forward",
        (True, False),
        ids=("forward", "backward"),
    )
    @pytest.mark.parametrize(
        "inplace",
        (True, False),
        ids=("inplace", "normal"),
    )
    @pytest.mark.parametrize(
        "execnum",
        range(10),
    )
    def test_writerange_int(self, infmodule, name, asparam, forward, inplace, execnum):
        rt = RecordTensor(
            infmodule,
            name,
            step_time=random.uniform(0.75, 1.25),
            duration=random.uniform(16.25, 20.0),
            value=None,
            persist_data=True,
            persist_constraints=True,
            persist_temporal=True,
        )
        setattr(infmodule, name, rt)

        shape = randshape(mindims=2, maxdims=4)
        data = torch.rand(rt.recordsz, *shape)
        if asparam:
            rt.value = nn.Parameter(data.clone().detach(), True)
        else:
            rt.value = data.clone().detach()

        length = random.randint(3, 11)
        offset = random.randint(0, rt.recordsz - 1)
        ptr = random.randint(0, rt.recordsz - 1)

        writedata = torch.rand(*shape, length)

        setattr(infmodule, rt.attributes.pointer, ptr)

        rt.writerange(writedata, offset, forward, inplace)
        offsets = torch.full(shape, offset, dtype=torch.int64)
        offsets = offsets + (length - 1) * (not forward)
        offsets = torch.stack([offsets - k for k in range(length)], 0)

        assert torch.all(
            rt.value
            == data.scatter(
                0,
                (ptr - offsets) % rt.recordsz,
                ein.rearrange(writedata, "... t -> t ..."),
            )
        )

    @pytest.mark.parametrize(
        "asparam",
        (True, False),
        ids=("parameter", "buffer"),
    )
    @pytest.mark.parametrize(
        "forward",
        (True, False),
        ids=("forward", "backward"),
    )
    @pytest.mark.parametrize(
        "inplace",
        (True, False),
        ids=("inplace", "normal"),
    )
    def test_writerange_tensor(self, infmodule, name, asparam, forward, inplace):
        rt = RecordTensor(
            infmodule,
            name,
            step_time=random.uniform(0.75, 1.25),
            duration=random.uniform(16.25, 20.0),
            value=None,
            persist_data=True,
            persist_constraints=True,
            persist_temporal=True,
        )
        setattr(infmodule, name, rt)

        shape = randshape(mindims=2, maxdims=4)
        data = torch.rand(rt.recordsz, *shape)
        if asparam:
            rt.value = nn.Parameter(data.clone().detach(), True)
        else:
            rt.value = data.clone().detach()

        length = random.randint(3, 11)
        offsets = torch.randint(0, rt.recordsz, shape)
        ptr = random.randint(0, rt.recordsz - 1)

        writedata = torch.rand(*shape, length)

        setattr(infmodule, rt.attributes.pointer, ptr)

        rt.writerange(writedata, offsets, forward, inplace)

        offsets = offsets + (length - 1) * (not forward)
        offsets = torch.stack([offsets - k for k in range(length)], 0)

        assert torch.all(
            rt.value
            == data.scatter(
                0,
                (ptr - offsets) % rt.recordsz,
                ein.rearrange(writedata, "... t -> t ..."),
            )
        )

    @pytest.mark.parametrize(
        "offset",
        (1, 3, -2),
        ids=("offset=1", "offset=3", "offset=-2"),
    )
    @pytest.mark.parametrize(
        "asparam",
        (True, False),
        ids=("parameter", "buffer"),
    )
    @pytest.mark.parametrize(
        "nselects",
        (1, 7),
        ids=("nselects=1", "nselects=7"),
    )
    def test_select_tensor_interp_bypass(
        self,
        infmodule,
        name,
        offset,
        asparam,
        nselects,
    ):
        def bad_interp(prev_data, next_data, sample_at, step_time, **kwargs):
            return torch.rand_like(prev_data)

        rt = RecordTensor(
            infmodule,
            name,
            step_time=random.uniform(0.75, 1.25),
            duration=random.uniform(16.25, 20.0),
            value=None,
            persist_data=True,
            persist_constraints=True,
            persist_temporal=True,
        )
        setattr(infmodule, name, rt)

        shape = randshape(mindims=2, maxdims=4)
        data = torch.rand(rt.recordsz, *shape)
        if asparam:
            rt.value = nn.Parameter(data.clone().detach(), True)
        else:
            rt.value = data.clone().detach()

        ptr = random.randint(0, rt.recordsz - 1)
        setattr(infmodule, rt.attributes.pointer, ptr)

        tolerance = 1e-4
        indices = torch.randint(0, rt.recordsz, (*shape, nselects))
        times = (indices * rt.dt) + (
            (tolerance - 1e-6) * torch.randint(-1, 2, indices.shape)
        )
        res = rt.select(
            times.squeeze(-1), bad_interp, tolerance=tolerance, offset=offset
        )
        cmp = ein.rearrange(
            data.gather(
                0,
                (ptr - (ein.rearrange(indices, "... t -> t ...") + offset))
                % rt.recordsz,
            ),
            "t ... -> ... t",
        ).squeeze(-1)
        assert torch.all(res == cmp)

    @pytest.mark.parametrize(
        "offset",
        (1, 3, -2),
        ids=("offset=1", "offset=3", "offset=-2"),
    )
    @pytest.mark.parametrize(
        "asparam",
        (True, False),
        ids=("parameter", "buffer"),
    )
    @pytest.mark.parametrize(
        "boundedupper",
        (True, False),
        ids=("boundedupper", "boundedlower"),
    )
    def test_select_float_interp_bypass(
        self, infmodule, name, offset, asparam, boundedupper
    ):
        def bad_interp(prev_data, next_data, sample_at, step_time, **kwargs):
            return torch.rand_like(prev_data)

        rt = RecordTensor(
            infmodule,
            name,
            step_time=random.uniform(0.75, 1.25),
            duration=random.uniform(16.25, 20.0),
            value=None,
            persist_data=True,
            persist_constraints=True,
            persist_temporal=True,
        )
        setattr(infmodule, name, rt)

        shape = randshape(mindims=2, maxdims=4)
        data = torch.rand(rt.recordsz, *shape)
        if asparam:
            rt.value = nn.Parameter(data.clone().detach(), True)
        else:
            rt.value = data.clone().detach()

        ptr = random.randint(0, rt.recordsz - 1)
        setattr(infmodule, rt.attributes.pointer, ptr)

        tolerance = 1e-4
        index = random.randint(0, rt.recordsz - 1)
        time = index * rt.dt + ((tolerance - 1e-6) * (-1 * (not boundedupper)))
        indices = torch.full((1, *shape), index)

        res = rt.select(time, bad_interp, tolerance=tolerance, offset=offset)
        cmp = data.gather(0, (ptr - (indices + offset)) % rt.recordsz).squeeze(0)
        assert torch.all(res == cmp)

    @pytest.mark.parametrize(
        "offset",
        (1, 3, -2),
        ids=("offset=1", "offset=3", "offset=-2"),
    )
    @pytest.mark.parametrize(
        "asparam",
        (True, False),
        ids=("parameter", "buffer"),
    )
    @pytest.mark.parametrize(
        "nselects",
        (1, 7),
        ids=("nselects=1", "nselects=7"),
    )
    def test_select_tensor_interpolated(
        self,
        infmodule,
        name,
        offset,
        asparam,
        nselects,
    ):
        def interp_linear(prev_data, next_data, sample_at, step_time, **kwargs):
            return ((next_data - prev_data) / step_time) * sample_at

        rt = RecordTensor(
            infmodule,
            name,
            step_time=random.uniform(0.75, 1.25),
            duration=random.uniform(16.25, 20.0),
            value=None,
            persist_data=True,
            persist_constraints=True,
            persist_temporal=True,
        )
        setattr(infmodule, name, rt)

        shape = randshape(mindims=2, maxdims=4)
        data = torch.rand(rt.recordsz, *shape)
        if asparam:
            rt.value = nn.Parameter(data.clone().detach(), True)
        else:
            rt.value = data.clone().detach()

        ptr = random.randint(0, rt.recordsz - 1)
        setattr(infmodule, rt.attributes.pointer, ptr)

        indices = torch.randint(0, rt.recordsz - 1, (*shape, nselects)) + 0.5
        times = indices * rt.dt

        res = rt.select(times.squeeze(-1), interp_linear, tolerance=1e-4, offset=offset)
        indices = ein.rearrange(indices, "... t -> t ...")
        cmp = ein.rearrange(
            interp_linear(
                data.gather(
                    0,
                    (ptr - (indices.ceil().long() + offset)) % rt.recordsz,
                ),
                data.gather(
                    0,
                    (ptr - (indices.floor().long() + offset)) % rt.recordsz,
                ),
                ein.rearrange(times, "... t -> t ...") % rt.dt,
                rt.dt,
            ),
            "t ... -> ... t",
        ).squeeze(-1)
        assert torch.all((res - cmp).abs() < 1e-6)

    @pytest.mark.parametrize(
        "offset",
        (1, 3, -2),
        ids=("offset=1", "offset=3", "offset=-2"),
    )
    @pytest.mark.parametrize(
        "asparam",
        (True, False),
        ids=("parameter", "buffer"),
    )
    def test_select_float_interpolated(
        self,
        infmodule,
        name,
        offset,
        asparam,
    ):
        def interp_linear(prev_data, next_data, sample_at, step_time, **kwargs):
            return ((next_data - prev_data) / step_time) * sample_at

        rt = RecordTensor(
            infmodule,
            name,
            step_time=random.uniform(0.75, 1.25),
            duration=random.uniform(16.25, 20.0),
            value=None,
            persist_data=True,
            persist_constraints=True,
            persist_temporal=True,
        )
        setattr(infmodule, name, rt)

        shape = randshape(mindims=2, maxdims=4)
        data = torch.rand(rt.recordsz, *shape)
        if asparam:
            rt.value = nn.Parameter(data.clone().detach(), True)
        else:
            rt.value = data.clone().detach()

        ptr = random.randint(0, rt.recordsz - 1)
        setattr(infmodule, rt.attributes.pointer, ptr)

        index = random.randint(0, rt.recordsz - 2) + 0.5
        time = index * rt.dt
        indices = torch.full((1, *shape), index)
        times = indices * rt.dt

        res = rt.select(time, interp_linear, tolerance=1e-4, offset=offset)
        cmp = interp_linear(
            data.gather(
                0,
                (ptr - (indices.ceil().long() + offset)) % rt.recordsz,
            ),
            data.gather(
                0,
                (ptr - (indices.floor().long() + offset)) % rt.recordsz,
            ),
            times % rt.dt,
            rt.dt,
        ).squeeze(0)
        assert torch.all((res - cmp).abs() < 1e-6)

    @pytest.mark.parametrize(
        "offset",
        (1, 3, -2),
        ids=("offset=1", "offset=3", "offset=-2"),
    )
    @pytest.mark.parametrize(
        "asparam",
        (True, False),
        ids=("parameter", "buffer"),
    )
    @pytest.mark.parametrize("inplace", (True, False), ids=("inplace", "normal"))
    def test_insert_tensor_bypass_extrap(
        self, infmodule, name, offset, asparam, inplace
    ):
        def bad_extrap(sample, sample_at, prev_data, next_data, step_time, **kwargs):
            return (torch.rand_like(prev_data), torch.rand_like(next_data))

        rt = RecordTensor(
            infmodule,
            name,
            step_time=random.uniform(0.75, 1.25),
            duration=random.uniform(16.25, 20.0),
            value=None,
            persist_data=True,
            persist_constraints=True,
            persist_temporal=True,
        )
        setattr(infmodule, name, rt)

        shape = randshape(mindims=2, maxdims=4)
        data = torch.rand(rt.recordsz, *shape)
        if asparam:
            rt.value = nn.Parameter(data.clone().detach(), True)
        else:
            rt.value = data.clone().detach()

        ptr = random.randint(0, rt.recordsz - 1)
        setattr(infmodule, rt.attributes.pointer, ptr)

        tolerance = 1e-4
        indices = torch.randint(0, rt.recordsz, shape)
        times = (indices * rt.dt) + (
            (tolerance - 1e-6) * torch.randint(-1, 2, indices.shape)
        )
        obs = torch.rand(shape)
        rt.insert(
            obs, times, bad_extrap, tolerance=tolerance, offset=offset, inplace=inplace
        )
        cmp = data.scatter(
            0,
            ((ptr - (indices + offset)) % rt.recordsz).unsqueeze(0),
            obs.unsqueeze(0),
        )

        assert torch.all(rt.value == cmp)

    @pytest.mark.parametrize(
        "offset",
        (1, 3, -2),
        ids=("offset=1", "offset=3", "offset=-2"),
    )
    @pytest.mark.parametrize(
        "asparam",
        (True, False),
        ids=("parameter", "buffer"),
    )
    @pytest.mark.parametrize("inplace", (True, False), ids=("inplace", "normal"))
    @pytest.mark.parametrize(
        "boundedupper",
        (True, False),
        ids=("boundedupper", "boundedlower"),
    )
    def test_insert_float_bypass_extrap(
        self, infmodule, name, offset, asparam, inplace, boundedupper
    ):
        def bad_extrap(sample, sample_at, prev_data, next_data, step_time, **kwargs):
            return (torch.rand_like(prev_data), torch.rand_like(next_data))

        rt = RecordTensor(
            infmodule,
            name,
            step_time=random.uniform(0.75, 1.25),
            duration=random.uniform(16.25, 20.0),
            value=None,
            persist_data=True,
            persist_constraints=True,
            persist_temporal=True,
        )
        setattr(infmodule, name, rt)

        shape = randshape(mindims=2, maxdims=4)
        data = torch.rand(rt.recordsz, *shape)
        if asparam:
            rt.value = nn.Parameter(data.clone().detach(), True)
        else:
            rt.value = data.clone().detach()

        ptr = random.randint(0, rt.recordsz - 1)
        setattr(infmodule, rt.attributes.pointer, ptr)

        tolerance = 1e-4
        index = random.randint(0, rt.recordsz - 1)
        time = index * rt.dt + ((tolerance - 1e-6) * (-1 * (not boundedupper)))
        indices = torch.full(shape, index)

        obs = torch.rand(shape)
        rt.insert(
            obs, time, bad_extrap, tolerance=tolerance, offset=offset, inplace=inplace
        )
        cmp = data.scatter(
            0,
            ((ptr - (indices + offset)) % rt.recordsz).unsqueeze(0),
            obs.unsqueeze(0),
        )

        assert torch.all(rt.value == cmp)

    @pytest.mark.parametrize(
        "offset",
        (1, 3, -2),
        ids=("offset=1", "offset=3", "offset=-2"),
    )
    @pytest.mark.parametrize(
        "asparam",
        (True, False),
        ids=("parameter", "buffer"),
    )
    @pytest.mark.parametrize("inplace", (True, False), ids=("inplace", "normal"))
    def test_insert_tensor_extrapolated(
        self, infmodule, name, offset, asparam, inplace
    ):
        def extrap_expdecay(
            sample, sample_at, prev_data, next_data, step_time, tc, **kwargs
        ):
            return (
                sample * torch.exp(sample_at / tc),
                sample * torch.exp((sample_at - step_time) / tc),
            )

        rt = RecordTensor(
            infmodule,
            name,
            step_time=random.uniform(0.75, 1.25),
            duration=random.uniform(16.25, 20.0),
            value=None,
            persist_data=True,
            persist_constraints=True,
            persist_temporal=True,
        )
        setattr(infmodule, name, rt)

        shape = randshape(mindims=2, maxdims=4)
        data = torch.rand(rt.recordsz, *shape)
        if asparam:
            rt.value = nn.Parameter(data.clone().detach(), True)
        else:
            rt.value = data.clone().detach()

        ptr = random.randint(0, rt.recordsz - 1)
        setattr(infmodule, rt.attributes.pointer, ptr)

        tolerance = 1e-4
        indices = torch.randint(0, rt.recordsz - 1, shape) + 0.5
        times = indices * rt.dt
        obs = torch.rand(shape)

        rt.insert(
            obs,
            times,
            extrap_expdecay,
            tolerance=tolerance,
            offset=offset,
            inplace=inplace,
            extrap_kwargs={"tc": 20.0},
        )
        pred, postd = extrap_expdecay(
            obs.unsqueeze(0),
            times.unsqueeze(0) % rt.dt,
            data.gather(
                0, (ptr - (indices.unsqueeze(0).ceil().long() + offset)) % rt.recordsz
            ),
            data.gather(
                0,
                (ptr - (indices.unsqueeze(0).floor().long() + offset)) % rt.recordsz,
            ),
            rt.dt,
            20.0,
        )
        cmp = data.scatter(
            0,
            (ptr - (indices.unsqueeze(0).ceil().long() + offset)) % rt.recordsz,
            pred,
        ).scatter(
            0,
            (ptr - (indices.unsqueeze(0).floor().long() + offset)) % rt.recordsz,
            postd,
        )
        assert torch.all((rt.value - cmp).abs() < 1e-6)

    @pytest.mark.parametrize(
        "offset",
        (1, 3, -2),
        ids=("offset=1", "offset=3", "offset=-2"),
    )
    @pytest.mark.parametrize(
        "asparam",
        (True, False),
        ids=("parameter", "buffer"),
    )
    @pytest.mark.parametrize("inplace", (True, False), ids=("inplace", "normal"))
    def test_insert_float_extrapolated(self, infmodule, name, offset, asparam, inplace):
        def extrap_expdecay(
            sample, sample_at, prev_data, next_data, step_time, tc, **kwargs
        ):
            return (
                sample * torch.exp(sample_at / tc),
                sample * torch.exp((sample_at - step_time) / tc),
            )

        rt = RecordTensor(
            infmodule,
            name,
            step_time=random.uniform(0.75, 1.25),
            duration=random.uniform(16.25, 20.0),
            value=None,
            persist_data=True,
            persist_constraints=True,
            persist_temporal=True,
        )
        setattr(infmodule, name, rt)

        shape = randshape(mindims=2, maxdims=4)
        data = torch.rand(rt.recordsz, *shape)
        if asparam:
            rt.value = nn.Parameter(data.clone().detach(), True)
        else:
            rt.value = data.clone().detach()

        ptr = random.randint(0, rt.recordsz - 1)
        setattr(infmodule, rt.attributes.pointer, ptr)

        tolerance = 1e-4

        index = random.randint(0, rt.recordsz - 2) + 0.5
        time = index * rt.dt
        indices = torch.full(shape, index)
        times = indices * rt.dt

        obs = torch.rand(shape)

        rt.insert(
            obs,
            time,
            extrap_expdecay,
            tolerance=tolerance,
            offset=offset,
            inplace=inplace,
            extrap_kwargs={"tc": 20.0},
        )
        pred, postd = extrap_expdecay(
            obs.unsqueeze(0),
            times.unsqueeze(0) % rt.dt,
            data.gather(
                0, (ptr - (indices.unsqueeze(0).ceil().long() + offset)) % rt.recordsz
            ),
            data.gather(
                0,
                (ptr - (indices.unsqueeze(0).floor().long() + offset)) % rt.recordsz,
            ),
            rt.dt,
            20.0,
        )
        cmp = data.scatter(
            0,
            (ptr - (indices.unsqueeze(0).ceil().long() + offset)) % rt.recordsz,
            pred,
        ).scatter(
            0,
            (ptr - (indices.unsqueeze(0).floor().long() + offset)) % rt.recordsz,
            postd,
        )
        assert torch.all((rt.value - cmp).abs() < 1e-6)

    @pytest.mark.parametrize(
        "asparam",
        (True, False),
        ids=("parameter", "buffer"),
    )
    def test_reconstrain(self, infmodule, name, asparam):
        rt = RecordTensor(
            infmodule,
            name,
            step_time=random.uniform(0.75, 1.25),
            duration=random.uniform(16.25, 20.0),
            value=None,
            persist_data=True,
            persist_constraints=True,
            persist_temporal=True,
        )
        setattr(infmodule, name, rt)

        shape = randshape(mindims=3, maxdims=5)
        data = torch.rand(rt.recordsz, *shape)
        if asparam:
            rt.value = nn.Parameter(data.clone().detach(), True)
        else:
            rt.value = data.clone().detach()

        rt.reconstrain(-1, shape[-1])
        rt.reconstrain(0, shape[0])
        assert rt.constraints[-1] == shape[-1]
        assert rt.constraints[0] == shape[0]
        assert getattr(infmodule, rt.attributes.constraints)[0] == rt.recordsz
        assert getattr(infmodule, rt.attributes.constraints)[-1] == shape[-1]
        assert getattr(infmodule, rt.attributes.constraints)[1] == shape[0]


class TestVirtualTensor:

    class VirtualModule(Module):
        def __init__(self, mask, scale):
            Module.__init__(self)
            self.register_buffer("mask", mask.bool())
            self.scale = scale

            def matf(module, dtype, device):
                return module.mask.to(dtype=dtype, device=device) * module.scale

            self.matf = matf

        def matm(self, dtype, device):
            return self.mask.to(dtype=dtype, device=device) * self.scale

        def badmatm(self, dtype, device):
            return self.mask * self.scale

    @pytest.fixture
    def shape(self):
        return randshape(2, 3, 3, 5)

    def test_finalizer_method_materializer(self, shape, testattr):
        vm = self.VirtualModule(torch.rand(shape) > 0.7, 2.0)

        setattr(vm, testattr, VirtualTensor(vm, testattr, "matm"))
        wr = weakref.ref(getattr(vm, testattr))

        assert hasattr(vm, f"_{testattr}_ref")

        delattr(vm, testattr)

        assert not hasattr(vm, testattr)
        assert not hasattr(vm, f"_{testattr}_ref")
        assert wr() is None

    def test_finalizer_extfunc_materializer(self, shape, testattr):
        vm = self.VirtualModule(torch.rand(shape) > 0.7, 2.0)

        def materializer(module, dtype, device):
            return module.mask.to(dtype=dtype, device=device) * module.scale

        setattr(vm, testattr, VirtualTensor(vm, testattr, materializer))
        wr = weakref.ref(getattr(vm, testattr))

        assert hasattr(vm, f"_{testattr}_ref")

        delattr(vm, testattr)

        assert not hasattr(vm, testattr)
        assert not hasattr(vm, f"_{testattr}_ref")
        assert wr() is None

    def test_finalizer_intfunc_materializer(self, shape, testattr):
        vm = self.VirtualModule(torch.rand(shape) > 0.7, 2.0)

        setattr(vm, testattr, VirtualTensor(vm, testattr, vm.matf))
        wr = weakref.ref(getattr(vm, testattr))

        assert hasattr(vm, f"_{testattr}_ref")

        delattr(vm, testattr)

        assert not hasattr(vm, testattr)
        assert not hasattr(vm, f"_{testattr}_ref")
        assert wr() is None

    @pytest.mark.parametrize(
        "persist",
        (True, False),
        ids=("persist=True", "persist=False"),
    )
    def test_persistence(self, shape, testattr, persist):
        vm = self.VirtualModule(torch.rand(shape) > 0.7, 2)
        VirtualTensor.create(vm, testattr, "matm", persist=persist)

        assert f"_{testattr}_ref" in vm._buffers
        if not persist:
            assert f"_{testattr}_ref" in vm._non_persistent_buffers_set

    def test_value_dtype_switch(self, shape, testattr):
        vm = self.VirtualModule(torch.rand(shape) > 0.7, 2)
        VirtualTensor.create(vm, testattr, "matm", dtype=torch.float32)
        vt = getattr(vm, testattr)

        vt.to(dtype=torch.bfloat16)

        assert vt.dtype == torch.bfloat16
        assert vt.value.dtype == torch.bfloat16

        vt.dtype = torch.float64

        assert vt.dtype == torch.float64
        assert vt.value.dtype == torch.float64

        vm.to(dtype=torch.complex128)

        assert vm.mask.dtype == torch.bool
        assert getattr(vm, vt.attributes.ref).dtype == torch.complex128
        assert vt.value.dtype == torch.complex128

    def test_dtype_coercion(self, shape, testattr):
        vm = self.VirtualModule(torch.rand(shape) > 0.7, 2)
        VirtualTensor.create(vm, testattr, "badmatm", dtype=torch.float32)
        vt = getattr(vm, testattr)

        vt.to(dtype=torch.bfloat16)

        assert vt.dtype == torch.bfloat16
        assert vt.value.dtype == torch.bfloat16
        assert vm.badmatm(vt.dtype, vt.device).dtype != torch.bfloat16
