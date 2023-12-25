from functools import reduce
import pytest
import random
import torch
import torch.nn as nn

import sys

sys.path.insert(0, "../..")

from inferno import Module, DimensionalModule  # noqa: E402


class TestModule:
    @pytest.fixture(scope="function")
    def module(self):
        return Module()

    @pytest.fixture(scope="class")
    def extra_name(self):
        return "extra_attr"

    @pytest.fixture(scope="class")
    def extra_value(self):
        return 3

    @pytest.fixture(scope="class")
    def submodule_name(self):
        return "sm_attr"

    @pytest.fixture(scope="class")
    def extras_sd_path(self):
        return ("_extra_state",)

    @pytest.fixture(scope="class")
    def invalid_type_register_substr(self):
        return "object to"

    @pytest.fixture(scope="class")
    def duplicate_register_substr(self):
        return "already exists"

    def test_getattr(self, module, extra_name, extra_value):
        module.register_extra(extra_name, extra_value)
        assert getattr(module, extra_name) == extra_value

    def test_setattr(self, module, extra_name, extra_value):
        module.register_extra(extra_name, extra_value)
        setattr(module, extra_name, extra_value**2)
        assert getattr(module, extra_name) == extra_value**2

    def test_delattr(self, module, extra_name, extra_value):
        module.register_extra(extra_name, extra_value)
        delattr(module, extra_name)
        assert not hasattr(module, extra_name)

    @pytest.mark.parametrize(
        "depth",
        (0, 1, 2, 3),
        ids=("depth=0", "depth=1", "depth=2", "depth=3"),
    )
    def test_get_extra(self, module, extra_name, extra_value, submodule_name, depth):
        accessor = extra_name
        current = module

        for _ in range(depth):
            setattr(current, submodule_name, Module())
            accessor = f"{submodule_name}.{accessor}"
            current = getattr(current, submodule_name)

        current.register_extra(extra_name, extra_value)
        assert module.get_extra(accessor) == extra_value

    def test_state_dict_export(self, module, extra_name, extra_value, extras_sd_path):
        module.register_extra(extra_name, extra_value)
        value_from_sd = reduce(
            lambda d, k: d[k], (module.state_dict(),) + extras_sd_path + (extra_name,)
        )
        assert value_from_sd == extra_value

    def test_state_dict_import(self, module, extra_name, extra_value):
        module.register_extra(extra_name, extra_value)
        from_import = Module()
        from_import.load_state_dict(module.state_dict())
        assert from_import.get_extra(extra_name) == extra_value

    @pytest.mark.parametrize(
        "extra_badval",
        (torch.rand(3, 3), nn.Parameter(torch.rand(3, 3), False), nn.Linear(784, 10)),
        ids=("type=torch.Tensor", "type=nn.Parameter", "type=nn.Module"),
    )
    def test_register_extra_invalid_type(
        self, module, extra_name, extra_badval, invalid_type_register_substr
    ):
        with pytest.raises(TypeError) as excinfo:
            module.register_extra(extra_name, extra_badval)
        assert invalid_type_register_substr in str(excinfo.value)

    def test_register_extra_duplicate(
        self, module, extra_name, extra_value, duplicate_register_substr
    ):
        module.register_buffer(extra_name, None)
        with pytest.raises(KeyError) as excinfo:
            module.register_extra(extra_name, extra_value)
        assert duplicate_register_substr in str(excinfo.value)


class TestDimensionalModule:
    @pytest.fixture(scope="class")
    def constraints_5d(self):
        # implies shape of (_, _, 3, ..., 2, _)
        return ((2, 3), (-2, 2))

    @pytest.fixture(scope="class")
    def constraints_5d_repr(self):
        # implies shape of (_, _, 3, ..., 2, _)
        return "(_, _, 3, ..., 2, _)"

    @pytest.fixture(scope="function")
    def constrained_5d(self):
        return torch.rand(
            random.randint(1, 9), random.randint(1, 9), 3, 2, random.randint(1, 9)
        )

    @pytest.fixture(scope="class")
    def constraints_6d(self):
        # implies shape of (_, _, 3, 5, ..., 2, _)
        return ((2, 3), (3, 5), (-2, 2))

    @pytest.fixture(scope="class")
    def constrained_buffer_name(self):
        return "test_buffer_constrained"

    @pytest.fixture(scope="class")
    def constrained_parameter_name(self):
        return "test_param_constrained"

    @pytest.mark.parametrize(
        "constrained,compatible",
        (
            (torch.rand(1, 1, 3, 2, 1), True),
            (torch.rand(1, 1, 3, 3, 1), False),
            (torch.rand(1, 1, 3, 2), False),
        ),
        ids=("minimal", "inconsistent", "unconstrainable"),
    )
    def test_compatible_internal(self, constraints_5d, constrained, compatible):
        module = DimensionalModule(*constraints_5d)
        assert module.compatible(constrained) == compatible

    @pytest.mark.parametrize(
        "constrained,compatible",
        (
            (torch.rand(1, 1, 3, 2, 1), True),
            (torch.rand(1, 1, 3, 3, 1), False),
            (torch.rand(1, 1, 3, 2), False),
        ),
        ids=("minimal", "inconsistent", "unconstrainable"),
    )
    def test_compatible_external(self, constraints_5d, constrained, compatible):
        if compatible:
            module = DimensionalModule((0, 2))
        else:
            module = DimensionalModule((0, 1))
        assert (
            module.compatible(constrained, constraints=dict(constraints_5d))
            == compatible
        )

    def test_constraints(self, constraints_5d):
        module = DimensionalModule(*constraints_5d)
        assert tuple(module.constraints.items()) == constraints_5d

    def test_constraints_repr(self, constraints_5d, constraints_5d_repr):
        module = DimensionalModule(*constraints_5d)
        assert module.constraints_repr == constraints_5d_repr

    def test_reconstrain_empty_add(self, constraints_5d, constraints_6d):
        module = DimensionalModule(*constraints_5d)
        module.reconstrain(*constraints_6d[1])
        assert tuple(module.constraints.items()) == constraints_6d

    def test_reconstrain_empty_remove(self, constraints_5d, constraints_6d):
        module = DimensionalModule(*constraints_6d)
        module.reconstrain(constraints_6d[1][0], None)
        assert tuple(module.constraints.items()) == constraints_5d

    def test_reconstrain_empty_alter(self, constraints_5d):
        module = DimensionalModule(*constraints_5d)

        new_constraints_5d = dict(constraints_5d)
        new_constraints_5d[-2] = 4
        new_constraints_5d = tuple(new_constraints_5d.items())

        module.reconstrain(*new_constraints_5d[1])
        assert tuple(module.constraints.items()) == new_constraints_5d

    def test_reconstrain_add(
        self,
        constraints_5d,
        constrained_5d,
        constrained_buffer_name,
        constrained_parameter_name,
    ):
        module = DimensionalModule(*constraints_5d)
        module.register_buffer(constrained_buffer_name, constrained_5d.clone().detach())
        module.register_constrained(constrained_buffer_name)
        module.register_parameter(
            constrained_parameter_name,
            nn.Parameter(constrained_5d.clone().detach(), False),
        )
        module.register_constrained(constrained_parameter_name)

        module.reconstrain(1, constrained_5d.shape[1])

        assert tuple(module.get_buffer(constrained_buffer_name).shape) == tuple(constrained_5d.shape)
        assert tuple(module.get_parameter(constrained_parameter_name).shape) == tuple(constrained_5d.shape)

    def test_reconstrain_remove(
        self,
        constraints_5d,
        constrained_5d,
        constrained_buffer_name,
        constrained_parameter_name,
    ):
        module = DimensionalModule(*constraints_5d)
        module.register_buffer(constrained_buffer_name, constrained_5d.clone().detach())
        module.register_buffer(
            constrained_parameter_name,
            nn.Parameter(constrained_5d.clone().detach(), False),
        )

        new_constraints_5d = dict(constraints_5d)
        new_constraints_5d[-2] = None
        new_constraints_5d = tuple(new_constraints_5d.items())

        module.reconstrain(*new_constraints_5d[1])

        assert tuple(module.get_buffer(constrained_buffer_name).shape) == tuple(constrained_5d.shape)
        assert tuple(module.get_parameter(constrained_parameter_name).shape) == tuple(constrained_5d.shape)

    def test_reconstrain_alter(
        self,
        constraints_5d,
        constrained_5d,
        constrained_buffer_name,
        constrained_parameter_name,
    ):
        module = DimensionalModule(*constraints_5d)
        module.register_buffer(constrained_buffer_name, constrained_5d.clone().detach())
        module.register_constrained(constrained_buffer_name)
        module.register_parameter(
            constrained_parameter_name,
            nn.Parameter(constrained_5d.clone().detach(), False),
        )
        module.register_constrained(constrained_parameter_name)

        new_constraints_5d = dict(constraints_5d)
        new_constraints_5d[-2] = new_constraints_5d[-2] * 2

        target_shape = list(constrained_5d.shape)
        target_shape[len(target_shape) - 2] = new_constraints_5d[-2]
        target_shape = tuple(target_shape)

        new_constraints_5d = tuple(new_constraints_5d.items())

        module.reconstrain(*new_constraints_5d[1])

        assert tuple(module.get_buffer(constrained_buffer_name).shape) == target_shape
        assert tuple(module.get_parameter(constrained_parameter_name).shape) == target_shape
