import pytest
import random
import torch
import weakref

from inferno import Module
from inferno.neural import Clamping, Normalization


class MockModule(Module):

    def __init__(self, data=None):
        Module.__init__(self)
        self.data = data

    def forward(self, inputs=None):
        if inputs is not None:
            self.data = inputs
        return self.data.clone().detach()


class TestClamping:

    @staticmethod
    def random_shape(mindims=1, maxdims=9, minsize=1, maxsize=9):
        return tuple(
            random.randint(mindims, maxdims)
            for _ in range(random.randint(minsize, maxsize))
        )

    @pytest.mark.parametrize(
        "prehook",
        (True, False),
        ids=("prehook", "posthook"),
    )
    def test_finalizer_unregistered(self, prehook):
        module = MockModule()
        clamp = Clamping(module, "data", 0, 1, as_prehook=prehook)

        clamp.register()
        clamp.deregister()

        hookref = weakref.ref(clamp)
        del clamp

        assert hookref() is None
        if prehook:
            assert len(module._forward_pre_hooks) == 0
        else:
            assert len(module._forward_hooks) == 0

    @pytest.mark.parametrize(
        "prehook",
        (True, False),
        ids=("prehook", "posthook"),
    )
    def test_finalizer_registered(self, prehook):
        module = MockModule()
        clamp = Clamping(module, "data", 0, 1, as_prehook=prehook)

        clamp.register()

        if prehook:
            assert len(module._forward_pre_hooks) == 1
        else:
            assert len(module._forward_hooks) == 1

        hookref = weakref.ref(clamp)
        del clamp

        assert hookref() is None
        if prehook:
            assert len(module._forward_pre_hooks) == 0
        else:
            assert len(module._forward_hooks) == 0

    @pytest.mark.parametrize(
        "prehook",
        (True, False),
        ids=("prehook", "posthook"),
    )
    def test_activation(self, prehook):
        cmin, cmax = 0.25, 0.75
        data = torch.rand(self.random_shape(3, 7, 3, 9))
        module = MockModule(data)
        clamp = Clamping(module, "data", cmin, cmax, as_prehook=prehook)
        clamp.register()

        res = module()
        assert torch.all(module.data == torch.clamp(data, cmin, cmax))
        if prehook:
            assert torch.all(res == torch.clamp(data, cmin, cmax))
        else:
            assert torch.all(res == data)


class TestNormalization:

    @staticmethod
    def random_shape(mindims=1, maxdims=9, minsize=1, maxsize=9):
        return tuple(
            random.randint(mindims, maxdims)
            for _ in range(random.randint(minsize, maxsize))
        )

    @pytest.fixture
    def eq_tol(self):
        return 1e-7

    @pytest.mark.parametrize(
        "prehook",
        (True, False),
        ids=("prehook", "posthook"),
    )
    def test_finalizer_unregistered(self, prehook):
        module = MockModule()
        normalize = Normalization(module, "data", 1, 1, -1, as_prehook=prehook)

        normalize.register()
        normalize.deregister()

        hookref = weakref.ref(normalize)
        del normalize

        assert hookref() is None
        if prehook:
            assert len(module._forward_pre_hooks) == 0
        else:
            assert len(module._forward_hooks) == 0

    @pytest.mark.parametrize(
        "prehook",
        (True, False),
        ids=("prehook", "posthook"),
    )
    def test_finalizer_registered(self, prehook):
        module = MockModule()
        normalize = Normalization(module, "data", 1, 1, -1, as_prehook=prehook)

        normalize.register()

        if prehook:
            assert len(module._forward_pre_hooks) == 1
        else:
            assert len(module._forward_hooks) == 1

        hookref = weakref.ref(normalize)
        del normalize

        assert hookref() is None
        if prehook:
            assert len(module._forward_pre_hooks) == 0
        else:
            assert len(module._forward_hooks) == 0

    @pytest.mark.parametrize(
        "prehook",
        (True, False),
        ids=("prehook", "posthook"),
    )
    @pytest.mark.parametrize(
        "ndims",
        (0, 1, 2, 3),
        ids=("ndims=0", "ndims=1", "ndims=2", "ndims=3"),
    )
    def test_activation(self, prehook, ndims, eq_tol):
        shape = self.random_shape(ndims + 2, ndims + 4, 3, 5)
        data = torch.rand(shape)

        if ndims == 0:
            dims = None
        elif ndims == 1:
            dims = random.randint(0, len(shape) - 1)
        else:
            dims = set()
            while len(dims) < ndims:
                dims.add(random.randint(0, len(shape) - 1))
            dims = tuple(dims)

        order = random.uniform(1.0, 3.0)
        scale = random.uniform(0.5, 1.5)

        module = MockModule(data)
        normalize = Normalization(
            module, "data", order, scale, dims, as_prehook=prehook
        )
        normalize.register()

        normed = torch.linalg.vector_norm(data, order, dim=dims, keepdim=True)
        normed = (data / normed) * scale

        res = module()

        assert torch.all((module.data - normed).abs() < eq_tol)
        if prehook:
            assert torch.all((res - normed).abs() < eq_tol)
        else:
            assert torch.all(res == data)
