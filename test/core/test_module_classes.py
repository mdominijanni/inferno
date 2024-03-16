from functools import reduce
from itertools import chain
import pytest
import random
import torch
import torch.nn as nn
from typing import Any
import uuid

from inferno import Module, DimensionalModule


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
            None,
        ),
        ids=("scalar", "empty", "None"),
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
