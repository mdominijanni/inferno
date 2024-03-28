from itertools import repeat
import pytest
import random
import torch
import torch.nn as nn
import uuid

from inferno import Module, ShapedTensor, RecordTensor


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
                    getattr(infmodule, name).value == datacopy.unsqueeze(-1)
                )
                assert dataname in infmodule._buffers
                if not persist:
                    assert dataname in infmodule._non_persistent_buffers_set
                else:
                    assert dataname not in infmodule._non_persistent_buffers_set

            case nn.Parameter:
                assert torch.all(
                    getattr(infmodule, name).value == datacopy.unsqueeze(-1)
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
            if dim < 0:
                offset_constraints[dim - 1] = size
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
        assert offset_constraints | {-1: recordsz} == getattr(infmodule, cstrname)
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
