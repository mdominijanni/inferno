from functools import reduce
from itertools import chain
import pytest
import random
import torch
import torch.nn as nn
from typing import Any
import uuid

from inferno import Module, DimensionalModule, RecordModule


def rgetattr(obj: object, attr: str, *default: Any) -> Any:
    try:
        return reduce(getattr, [obj] + attr.split("."))

    except AttributeError:
        if default:
            return default[0]
        else:
            raise


class TestModule:
    @staticmethod
    def random_identifier():
        return f"testvar_{uuid.uuid4().hex}"

    @staticmethod
    def random_integer():
        return uuid.uuid4().int

    @staticmethod
    def random_tensor():
        return torch.rand(random.randint(1, 9), random.randint(1, 9))

    @staticmethod
    def random_parameter():
        return nn.Parameter(
            torch.rand(random.randint(1, 9), random.randint(1, 9)), False
        )

    @staticmethod
    def random_linear():
        return nn.Linear(random.randint(1, 9), random.randint(1, 9))

    @staticmethod
    def sentinel():
        return object()

    @pytest.mark.parametrize(
        "depth",
        (0, 1, 2, 3),
        ids=("depth=0", "depth=1", "depth=2", "depth=3"),
    )
    def test_register_get_extra(self, depth):
        names = [self.random_identifier() for _ in range(depth + 1)]
        modules = [Module()]

        for n in names[:-1]:
            m = Module()
            modules[-1].add_module(n, m)
            modules.append(m)

        sentinel = self.sentinel()
        modules[-1].register_extra(names[-1], sentinel)

        assert id(sentinel) == id(rgetattr(modules[0], ".".join(names)))
        assert id(sentinel) == id(modules[0].get_extra(".".join(names)))

    def test_set_extra(self):
        name = self.random_identifier()
        module = Module()

        module.register_extra(name, self.sentinel())

        sentinel = self.sentinel()
        setattr(module, name, sentinel)

        assert id(sentinel) == id(module.get_extra(name))

    def test_del_extra(self):
        name = self.random_identifier()
        module = Module()
        module.register_extra(name, self.sentinel())

        delattr(module, name)

        assert not hasattr(module, name)
        assert name not in module.get_extra_state()

    @pytest.mark.parametrize(
        "constructor",
        ("random_tensor", "random_parameter", "random_linear"),
        ids=("type=torch.Tensor", "type=nn.Parameter", "type=nn.Module"),
    )
    def test_register_extra_invalid_type(self, constructor):
        name = self.random_identifier()
        value = getattr(self, constructor)()
        module = Module()

        with pytest.raises(TypeError) as excinfo:
            module.register_extra(name, value)

        assert f"cannot assign '{type(value).__name__}' object to '{name}'" in str(
            excinfo.value
        )
        assert not hasattr(module, name)

    def test_register_extra_assigned_name(self):
        name = self.random_identifier()
        module = Module()

        sentinel = torch.rand(3, 3)
        module.register_buffer(name, sentinel)

        with pytest.raises(KeyError) as excinfo:
            module.register_extra(name, self.random_integer())

        assert f"attribute '{name}' already exists" in str(excinfo.value)
        assert id(sentinel) == id(getattr(module, name))

    @pytest.mark.parametrize(
        "depth",
        (0, 1, 2, 3),
        ids=("depth=0", "depth=1", "depth=2", "depth=3"),
    )
    def test_state_dict_save_load(self, depth):
        module_names = [self.random_identifier() for _ in range(depth)]

        saved_modules = [Module()]
        for n in module_names:
            m = Module()
            saved_modules[-1].add_module(n, m)
            saved_modules.append(m)

        extra_names, extra_values = [], []
        for m in saved_modules:
            extra_names.append(self.random_identifier())
            extra_values.append(self.random_integer())
            m.register_extra(extra_names[-1], extra_values[-1])

        loaded_modules = [Module()]
        for n in module_names:
            m = Module()
            loaded_modules[-1].add_module(n, m)
            loaded_modules.append(m)

        loaded_modules[0].load_state_dict(saved_modules[0].state_dict())

        for attr, value in map(
            lambda p, n, v: (".".join(p + [n]), v),
            (
                module_names[:ts]
                for ts in chain(range(-len(module_names), 0, 1), (None,))
            ),
            extra_names,
            extra_values,
        ):
            assert value == rgetattr(loaded_modules[0], attr)

    def test_submodule_property(self):
        name = self.random_identifier()
        value = self.random_linear()
        module = Module()

        module.add_module(name, value)
        prop_name = self.random_identifier()

        module.__class__ = type(
            f"Testing{type(module).__name__}",
            (type(module),),
            {
                prop_name: property(
                    lambda obj, attr=name: getattr(obj, attr).weight * 2,
                    lambda obj, value, attr=name: setattr(obj, attr, value),
                    lambda obj, attr=name: getattr(obj, attr).weight.fill_(0),
                )
            },
        )

        assert torch.all(getattr(module, name).weight * 2 == getattr(module, prop_name))

        sentinel = self.random_linear()
        setattr(module, prop_name, sentinel)
        assert id(sentinel) == id(getattr(module, name))

        with torch.no_grad():
            delattr(module, prop_name)
        assert torch.all(
            torch.zeros_like(sentinel.weight) == getattr(module, name).weight
        )

    def test_parameter_property(self):
        name = self.random_identifier()
        value = self.random_parameter()
        module = Module()

        module.register_parameter(name, value)
        prop_name = self.random_identifier()

        module.__class__ = type(
            f"Testing{type(module).__name__}",
            (type(module),),
            {
                prop_name: property(
                    lambda obj, attr=name: getattr(obj, attr) * 2,
                    lambda obj, value, attr=name: setattr(obj, attr, value),
                    lambda obj, attr=name: getattr(obj, attr).fill_(0),
                )
            },
        )

        assert torch.all(getattr(module, name) * 2 == getattr(module, prop_name))

        sentinel = self.random_parameter()
        setattr(module, prop_name, sentinel)
        assert id(sentinel) == id(getattr(module, name))

        with torch.no_grad():
            delattr(module, prop_name)
        assert torch.all(torch.zeros_like(sentinel) == getattr(module, name))


class TestDimensionalModule:

    @staticmethod
    def random_identifier():
        return f"testvar_{uuid.uuid4().hex}"

    @staticmethod
    def random_init_final_module():
        return DimensionalModule((0, random.randint(1, 9)), (-1, random.randint(1, 9)))

    @staticmethod
    def random_valid_tensor(module, nadddim=0):
        dims = [
            random.randint(1, 9)
            for _ in range(
                max(max(module._constraints) + 1, 0)
                - min(min(module._constraints), 0)
                + nadddim
            )
        ]

        for d, s in module._constraints.items():
            dims[d] = s

        return torch.rand(*dims)

    @staticmethod
    def random_invalid_tensor(module, nadddim=0):
        dims = [
            random.randint(1, 9)
            for _ in range(
                max(max(module._constraints) + 1, 0)
                - min(min(module._constraints), 0)
                + nadddim
            )
        ]

        for d, s in module._constraints.items():
            dims[d] = (s % 9) + 1

        return torch.rand(*dims)

    @staticmethod
    def random_valid_parameter(module, nadddim=0):
        return nn.Parameter(
            TestDimensionalModule.random_valid_tensor(module, nadddim), False
        )

    @staticmethod
    def random_invalid_parameter(module, nadddim=0):
        return nn.Parameter(
            TestDimensionalModule.random_invalid_tensor(module, nadddim), False
        )

    @pytest.mark.parametrize(
        "sentinel",
        (torch.tensor(random.random()), torch.rand(0), None),
        ids=("scalar", "empty", "None"),
    )
    def test_constrained_buffer_assignment_ignorelive(self, sentinel):
        name = self.random_identifier()
        module = self.random_init_final_module()
        module.liveconstrain = True
        value = self.random_valid_tensor(module)

        module.register_buffer(name, value)
        module.register_constrained(name)

        setattr(module, name, sentinel)
        assert id(sentinel) == id(getattr(module, name))

    def test_constrained_buffer_assignment_nolive(self):
        name = self.random_identifier()
        module = self.random_init_final_module()
        module.liveconstrain = False
        value = self.random_invalid_tensor(module)

        module.register_buffer(name, self.random_valid_tensor(module))
        module.register_constrained(name)

        setattr(module, name, value)
        assert id(value) == id(getattr(module, name))

    def test_constrained_buffer_assignment_live(self):
        name = self.random_identifier()
        module = self.random_init_final_module()
        module.liveconstrain = True
        value = self.random_invalid_tensor(module)

        sentinel = self.random_valid_tensor(module)

        module.register_buffer(name, sentinel)
        module.register_constrained(name)

        with pytest.raises(RuntimeError) as excinfo:
            setattr(module, name, value)

        assert (
            f"tensor of shape {tuple(value.shape)} being assigned to '{name}' is "
            "not compatible with constraints" in str(excinfo.value)
        )
        assert id(sentinel) == id(getattr(module, name))

    @pytest.mark.parametrize(
        "sentinel",
        (
            nn.Parameter(torch.tensor(random.random()), False),
            nn.Parameter(torch.rand(0), False),
        ),
        ids=("scalar", "empty"),
    )
    def test_constrained_parameter_assignment_ignorelive(self, sentinel):
        name = self.random_identifier()
        module = self.random_init_final_module()
        module.liveconstrain = True
        value = self.random_valid_parameter(module)

        module.register_parameter(name, value)
        module.register_constrained(name)

        setattr(module, name, sentinel)
        assert id(sentinel) == id(getattr(module, name))

    def test_constrained_parameter_assignment_nolive(self):
        name = self.random_identifier()
        module = self.random_init_final_module()
        module.liveconstrain = False
        value = self.random_invalid_parameter(module)

        module.register_parameter(name, self.random_valid_parameter(module))
        module.register_constrained(name)

        setattr(module, name, value)
        assert id(value) == id(getattr(module, name))

    def test_constrained_parameter_assignment_live(self):
        name = self.random_identifier()
        module = self.random_init_final_module()
        module.liveconstrain = True
        value = self.random_invalid_parameter(module)

        sentinel = self.random_valid_parameter(module)

        module.register_parameter(name, sentinel)
        module.register_constrained(name)

        with pytest.raises(RuntimeError) as excinfo:
            setattr(module, name, value)

        assert (
            f"tensor of shape {tuple(value.shape)} being assigned to '{name}' is "
            "not compatible with constraints" in str(excinfo.value)
        )
        assert id(sentinel) == id(getattr(module, name))

    def test_register_validate_trim(self):
        buffer_name = self.random_identifier()
        param_name = self.random_identifier()
        module = self.random_init_final_module()

        module.register_buffer(buffer_name, self.random_valid_tensor(module))
        module.register_constrained(buffer_name)

        module.register_parameter(param_name, self.random_valid_parameter(module))
        module.register_constrained(param_name)

        assert buffer_name in module._constrained_buffers
        assert param_name in module._constrained_params

        delattr(module, buffer_name)
        delattr(module, param_name)
        module.validate()

        assert buffer_name not in module._constrained_buffers
        assert param_name not in module._constrained_params

    def test_register_validate_compatible(self):
        module = self.random_init_final_module()

        for _ in range(0, random.randint(10, 25)):
            name = self.random_identifier()
            value = self.random_valid_tensor(module, random.randint(0, 4))
            module.register_buffer(name, value)
            module.register_constrained(name)

        for _ in range(0, random.randint(10, 25)):
            name = self.random_identifier()
            value = self.random_valid_parameter(module, random.randint(0, 4))
            module.register_parameter(name, value)
            module.register_constrained(name)

        module.validate()

    def test_register_validate_incompatible_buffer(self):
        name = self.random_identifier()
        module = self.random_init_final_module()
        module.liveconstrain = False

        module.register_buffer(name, self.random_valid_tensor(module))
        module.register_constrained(name)
        setattr(module, name, self.random_invalid_tensor(module))

        with pytest.raises(RuntimeError) as excinfo:
            module.validate()
        assert f"constrained buffer '{name}' is invalid" in str(excinfo.value)

    def test_register_validate_incompatible_parameter(self):
        name = self.random_identifier()
        module = self.random_init_final_module()
        module.liveconstrain = False

        module.register_parameter(name, self.random_valid_parameter(module))
        module.register_constrained(name)
        setattr(module, name, self.random_invalid_parameter(module))

        with pytest.raises(RuntimeError) as excinfo:
            module.validate()
        assert f"constrained parameter '{name}' is invalid" in str(excinfo.value)

    def test_register_deregister(self):
        buffer_name = self.random_identifier()
        param_name = self.random_identifier()
        module = self.random_init_final_module()

        module.register_buffer(buffer_name, self.random_valid_tensor(module))
        module.register_constrained(buffer_name)

        module.register_parameter(param_name, self.random_valid_parameter(module))
        module.register_constrained(param_name)

        assert buffer_name in module._constrained_buffers
        assert param_name in module._constrained_params

        module.deregister_constrained(buffer_name)
        module.deregister_constrained(param_name)

        assert buffer_name not in module._constrained_buffers
        assert param_name not in module._constrained_params
        assert buffer_name in dict(module.named_buffers())
        assert param_name in dict(module.named_parameters())

    def test_reconstrain_remove(self):
        module = self.random_init_final_module()

        for _ in range(0, random.randint(10, 25)):
            name = self.random_identifier()
            value = self.random_valid_tensor(module, random.randint(0, 4))
            module.register_buffer(name, value)
            module.register_constrained(name)

        for _ in range(0, random.randint(10, 25)):
            name = self.random_identifier()
            value = self.random_valid_parameter(module, random.randint(0, 4))
            module.register_parameter(name, value)
            module.register_constrained(name)

        dim = [*module._constraints.keys()][
            random.randint(0, len(module._constraints) - 1)
        ]

        module.reconstrain(dim, None)

        assert dim not in module._constraints
        module.validate()

    def test_reconstrain_create_invalid_buffer(self):
        name = self.random_identifier()
        module = self.random_init_final_module()
        value = self.random_valid_tensor(module)

        module.register_buffer(name, value)
        module.register_constrained(name)

        dim = max(filter(lambda x: x >= 0, module._constraints.keys())) + 1
        if dim < value.ndim:
            size = value.size(dim) + 1
        else:
            size = 1

        with pytest.raises(RuntimeError) as excinfo:
            module.reconstrain(dim, size)
        assert (
            f"constrained buffer '{name}' would be invalidated by the "
            f"addition of constraint {(dim, size)}"
        ) in str(excinfo.value)

    def test_reconstrain_create_invalid_parameter(self):
        name = self.random_identifier()
        module = self.random_init_final_module()
        value = self.random_valid_parameter(module)

        module.register_parameter(name, value)
        module.register_constrained(name)

        dim = max(filter(lambda x: x >= 0, module._constraints.keys())) + 1
        if dim < value.ndim:
            size = value.size(dim) + 1
        else:
            size = 1

        with pytest.raises(RuntimeError) as excinfo:
            module.reconstrain(dim, size)
        assert (
            f"constrained parameter '{name}' would be invalidated by the "
            f"addition of constraint {(dim, size)}"
        ) in str(excinfo.value)

    def test_reconstrain_create_valid(self):
        buffer_name = self.random_identifier()
        param_name = self.random_identifier()
        module = self.random_init_final_module()
        module.liveconstrain = False

        module.register_buffer(buffer_name, self.random_valid_tensor(module))
        module.register_constrained(buffer_name)
        module.register_parameter(param_name, self.random_valid_parameter(module))
        module.register_constrained(param_name)

        dim = max(filter(lambda x: x >= 0, module._constraints.keys())) + 1
        size = random.randint(1, 9)

        shape_module = DimensionalModule(*module._constraints.items(), (dim, size))
        setattr(module, buffer_name, self.random_valid_tensor(shape_module))
        setattr(module, param_name, self.random_valid_parameter(shape_module))

        module.reconstrain(dim, size)
        assert shape_module._constraints == module._constraints

    @pytest.mark.parametrize(
        "sentinel",
        (torch.tensor(random.random()), torch.rand(0), None),
        ids=("scalar", "empty", "None"),
    )
    def test_reconstrain_create_ignore_buffer(self, sentinel):
        name = self.random_identifier()
        module = self.random_init_final_module()
        module.liveconstrain = False

        module.register_buffer(name, sentinel)

        module.reconstrain(
            max(filter(lambda x: x >= 0, module._constraints.keys())) + 1,
            random.randint(1, 9),
        )
        assert id(sentinel) == id(getattr(module, name))

    @pytest.mark.parametrize(
        "sentinel",
        (
            nn.Parameter(torch.tensor(random.random()), False),
            nn.Parameter(torch.rand(0), False),
        ),
        ids=("scalar", "empty"),
    )
    def test_reconstrain_create_ignore_parameter(self, sentinel):
        name = self.random_identifier()
        module = self.random_init_final_module()
        module.liveconstrain = False

        module.register_parameter(name, sentinel)

        module.reconstrain(
            max(filter(lambda x: x >= 0, module._constraints.keys())) + 1,
            random.randint(1, 9),
        )

        assert id(sentinel) == id(getattr(module, name))

    def test_reconstrain_alter_valid(self):
        buffer_name = self.random_identifier()
        param_name = self.random_identifier()
        module = self.random_init_final_module()
        module.liveconstrain = False

        module.register_buffer(buffer_name, self.random_valid_tensor(module))
        module.register_constrained(buffer_name)
        module.register_parameter(param_name, self.random_valid_parameter(module))
        module.register_constrained(param_name)

        dim, size = [*module._constraints.items()][
            random.randint(0, len(module._constraints) - 1)
        ]
        while size == module._constraints[dim]:
            size = random.randint(1, 9)

        revised_constraints = dict(module._constraints.items())
        revised_constraints[dim] = size
        shape_module = DimensionalModule(*revised_constraints.items())

        setattr(module, buffer_name, self.random_valid_tensor(shape_module))
        setattr(module, param_name, self.random_valid_parameter(shape_module))

        buffer_data = getattr(module, buffer_name).clone().detach()
        param_data = getattr(module, param_name).data.clone().detach()

        module.reconstrain(dim, size)

        assert revised_constraints == module._constraints
        assert torch.all(buffer_data == getattr(module, buffer_name))
        assert torch.all(param_data == getattr(module, param_name).data)

    def test_reconstrain_alter_invalid(self):
        buffer_name = self.random_identifier()
        param_name = self.random_identifier()
        module = self.random_init_final_module()
        module.liveconstrain = False

        module.register_buffer(buffer_name, self.random_valid_tensor(module))
        module.register_constrained(buffer_name)
        module.register_parameter(param_name, self.random_valid_parameter(module))
        module.register_constrained(param_name)

        dim, size = [*module._constraints.items()][
            random.randint(0, len(module._constraints) - 1)
        ]
        while size == module._constraints[dim]:
            size = random.randint(1, 9)
        module.reconstrain(dim, size)

        assert size == module._constraints[dim]
        assert size == getattr(module, buffer_name).size(dim)
        assert size == getattr(module, param_name).size(dim)
        assert torch.all(0 == getattr(module, buffer_name))
        assert torch.all(0 == getattr(module, param_name))


class TestRecordModule:
    @staticmethod
    def random_identifier():
        return f"testvar_{uuid.uuid4().hex}"

    def test_set_step_time_without_reconstrain(self):
        name = self.random_identifier()
        dt, duration = 1.0, 3.3
        module = RecordModule(dt, duration)

        assert duration == module.duration

        leading_dims = (random.randint(1, 9) for _ in range(random.randint(1, 9) + 1))
        module.register_buffer(name, torch.rand(*leading_dims, module.recordsz))
        module.register_constrained(name)
        data = getattr(module, name).clone().detach()

        module.dt = 1.09

        assert 1.09 == module.dt
        assert 5 == module.recordsz
        assert torch.all(data == getattr(module, name))

        module.dt = 0.9

        assert 0.9 == module.dt
        assert 5 == module.recordsz
        assert torch.all(data == getattr(module, name))

    def test_set_step_time_with_reconstrain(self):
        name = self.random_identifier()
        dt, duration = 1.0, 3.3
        module = RecordModule(dt, duration)

        assert duration == module.duration

        leading_dims = (random.randint(1, 9) for _ in range(random.randint(1, 9) + 1))
        module.register_buffer(name, torch.rand(*leading_dims, module.recordsz))
        module.register_constrained(name)

        module.dt = 1.1

        assert 1.1 == module.dt
        assert 4 == module.recordsz
        assert torch.all(0 == getattr(module, name))

    def test_set_duration_without_reconstrain(self):
        name = self.random_identifier()
        dt, duration = 1.0, 3.3
        module = RecordModule(dt, duration)

        assert duration == module.duration

        leading_dims = (random.randint(1, 9) for _ in range(random.randint(1, 9) + 1))
        module.register_buffer(name, torch.rand(*leading_dims, module.recordsz))
        module.register_constrained(name)
        data = getattr(module, name).clone().detach()

        module.duration = 4.0

        assert 4.0 == module.duration
        assert 5 == module.recordsz
        assert torch.all(data == getattr(module, name))

        module.duration = 3.1

        assert 3.1 == module.duration
        assert 5 == module.recordsz
        assert torch.all(data == getattr(module, name))

    def test_set_duration_with_reconstrain(self):
        name = self.random_identifier()
        dt, duration = 1.0, 3.3
        module = RecordModule(dt, duration)

        assert duration == module.duration

        leading_dims = (random.randint(1, 9) for _ in range(random.randint(1, 9) + 1))
        module.register_buffer(name, torch.rand(*leading_dims, module.recordsz))
        module.register_constrained(name)

        module.duration = 4.1

        assert 4.1 == module.duration
        assert 6 == module.recordsz
        assert torch.all(0 == getattr(module, name))

    def test_reconstrain_valid_dim(self):
        module = RecordModule(1.0, 0.0)

        dim, size = random.randint(1, 9), random.randint(1, 9)
        module.reconstrain(dim, size)

        assert len(module.constraints) == 2
        assert module.constraints[-1] == 1
        assert module.constraints[dim] == size

    def test_reconstrain_invalid_dim(self):
        module = RecordModule(1.0, 0.0)

        size, orig_size = module.recordsz, module.recordsz
        while size == orig_size:
            size = random.randint(1, 9)

        with pytest.raises(RuntimeError) as excinfo:
            module.reconstrain(-1, size)

        assert (
            f"{type(module).__name__}(RecordModule) cannot reconstrain the record dimension (-1)"
            in str(excinfo.value)
        )
        assert orig_size == module.constraints[-1]

    def test_record_latest(self):
        name = self.random_identifier()
        dt, duration = 1.0, 5.5
        module = RecordModule(dt, duration)

        leading_dims = [random.randint(1, 9) for _ in range(random.randint(1, 9) + 1)]
        module.register_buffer(name, torch.rand(*leading_dims, module.recordsz))
        module.register_constrained(name)

        data = torch.rand(*leading_dims)
        module.record(name, data)

        for _ in range(random.randint(15, 25)):
            assert torch.all(data == module.latest(name))
            newdata = torch.rand(*leading_dims)
            module.record(name, newdata)
            assert torch.all(data == module.latest(name, 2))
            data = newdata

    @pytest.mark.parametrize(
        "indexing",
        (
            "scalar",
            "tensorsingle",
            "tensormultiple",
        ),
        ids=("index=scalar", "index=oneslice", "index=manyslices"),
    )
    @pytest.mark.parametrize(
        "offset",
        (
            1,
            2,
        ),
        ids=("offset=default", "offset=2"),
    )
    @pytest.mark.parametrize(
        "tolerance",
        (
            0,
            0.1,
        ),
        ids=("tolerance=0", "tolerance=0.1"),
    )
    def test_tensor_select_interp_bypass(self, indexing, offset, tolerance):
        def bad_interp(prev_data, next_data, sample_at, step_time):
            return torch.rand_like(prev_data)

        name = self.random_identifier()
        dt, duration = 1.1, 5.5
        module = RecordModule(dt, duration)

        adjf = offset - 1

        leading_dims = [random.randint(1, 5) for _ in range(random.randint(1, 4) + 1)]
        module.register_buffer(name, torch.rand(*leading_dims, module.recordsz))
        module.register_constrained(name)

        data = torch.rand(*leading_dims, module.recordsz)
        for t in range(module.recordsz):
            module.record(name, data[..., t])

        match indexing:
            case "scalar":
                index = random.randint(0, module.recordsz - 1 - adjf)
                res = module.select(
                    name,
                    index * dt,
                    bad_interp,
                    tolerance=(0.1 + tolerance),
                    offset=offset,
                )
                assert leading_dims == [*res.shape]
                index = torch.full(leading_dims + [1], index) + adjf
                res = res.unsqueeze(-1)

            case "tensorsingle":
                index = torch.randint(0, module.recordsz - adjf, leading_dims)
                res = module.select(
                    name,
                    index * dt - tolerance / 2,
                    bad_interp,
                    tolerance=tolerance,
                    offset=offset,
                )
                assert leading_dims == [*res.shape]
                index = index.unsqueeze(-1) + adjf
                res = res.unsqueeze(-1)

            case "tensormultiple":
                nget = random.randint(2, 9)
                index = torch.randint(0, module.recordsz - adjf, leading_dims + [nget])
                res = module.select(
                    name,
                    index * dt + tolerance / 2,
                    bad_interp,
                    tolerance=tolerance,
                    offset=offset,
                )
                assert leading_dims + [nget] == [*res.shape]
                index = index + adjf

        idxs = [()]
        for d in leading_dims:
            tempidx = []
            for n in range(d):
                tempidx.extend(ix + (n,) for ix in idxs)
            idxs = [tuple(ix) for ix in tempidx]

        truth = torch.zeros_like(res)
        for prefix in idxs:
            for suffix in range(index.size(-1)):
                truth[*prefix, suffix] = data[
                    *prefix, module.recordsz - 1 - int(index[*prefix, suffix])
                ]

        assert torch.all(truth == res)

    @pytest.mark.parametrize(
        "indexing",
        (
            "scalar",
            "tensorsingle",
            "tensormultiple",
        ),
        ids=("index=scalar", "index=oneslice", "index=manyslices"),
    )
    @pytest.mark.parametrize(
        "offset",
        (
            1,
            2,
        ),
        ids=("offset=default", "offset=2"),
    )
    def test_tensor_select_interp(self, indexing, offset):
        def linear_interp(prev_data, next_data, sample_at, step_time):
            slope = (next_data - prev_data) / step_time
            return prev_data + slope * sample_at

        name = self.random_identifier()
        dt, duration = 1.1, 5.5
        module = RecordModule(dt, duration)

        adjf = offset - 1

        leading_dims = [random.randint(1, 5) for _ in range(random.randint(1, 4) + 1)]
        module.register_buffer(name, torch.rand(*leading_dims, module.recordsz))
        module.register_constrained(name)

        data = torch.rand(*leading_dims, module.recordsz)
        for t in range(module.recordsz):
            module.record(name, data[..., t])

        match indexing:
            case "scalar":
                index = random.random() * (module.recordsz - 1 - adjf)
                res = module.select(
                    name, index * dt, linear_interp, tolerance=0, offset=offset
                )
                assert leading_dims == [*res.shape]
                index = torch.full(leading_dims + [1], index) + adjf
                res = res.unsqueeze(-1)

            case "tensorsingle":
                index = torch.rand(leading_dims) * (module.recordsz - 1 - adjf)
                res = module.select(
                    name, index * dt, linear_interp, tolerance=0, offset=offset
                )
                assert leading_dims == [*res.shape]
                index = index.unsqueeze(-1) + adjf
                res = res.unsqueeze(-1)

            case "tensormultiple":
                nget = random.randint(2, 9)
                index = torch.rand(leading_dims + [nget]) * (module.recordsz - 1 - adjf)
                res = module.select(
                    name, index * dt, linear_interp, tolerance=0, offset=offset
                )
                assert leading_dims + [nget] == [*res.shape]
                index = index + adjf

        prev_index = module.recordsz - 1 - torch.ceil(index).long()
        next_index = module.recordsz - 1 - torch.floor(index).long()

        assert torch.all(
            linear_interp(
                torch.gather(data, -1, prev_index),
                torch.gather(data, -1, next_index),
                dt * (index % 1),
                dt,
            )
            - res
            < 1e-6
        )

    @pytest.mark.parametrize(
        "flip",
        (
            True,
            False,
        ),
        ids=("flip=True", "flip=False"),
    )
    def test_aligned(self, flip):
        name = self.random_identifier()
        dt, duration = 1.1, 5.5
        module = RecordModule(dt, duration)

        leading_dims = [random.randint(1, 9) for _ in range(random.randint(1, 9))]
        module.register_buffer(name, torch.rand(*leading_dims, module.recordsz))
        module.register_constrained(name)

        data = torch.rand(*leading_dims, module.recordsz)
        for t in range(module.recordsz):
            module.record(name, data[..., t])

        assert torch.all(
            (data.flip(-1) if flip else data) == module.aligned(name, latest_first=flip)
        )
