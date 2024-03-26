from functools import reduce
import pytest
import random
import torch
from typing import Any

from inferno import Module, Hook, StateHook


def rgetattr(obj: object, attr: str, *default: Any) -> Any:
    try:
        return reduce(getattr, [obj] + attr.split("."))

    except AttributeError:
        if default:
            return default[0]
        else:
            raise


class TestHook:

    class MockModule(Module):

        def __init__(self, data):
            Module.__init__(self)
            self.register_buffer("data", data)

        def forward(self, inputs, fold=None):
            if not fold:
                fold = lambda a, b: b  # noqa:E731;
            self.data = fold(self.data, inputs)
            return random.random()

    class MockPrehook:

        def __init__(self, attr):
            self._attr = attr
            self.last_module = None
            self.last_tensor = None
            self.last_inputs = None
            self.last_kwargs = None

        def __call__(self, module, inputs, kwargs):
            self.last_module = module
            self.last_tensor = getattr(module, self._attr).clone().detach()
            self.last_inputs = inputs
            self.last_kwargs = kwargs

    class MockPosthook:

        def __init__(self, attr):
            self._attr = attr
            self.last_module = None
            self.last_tensor = None
            self.last_inputs = None
            self.last_kwargs = None
            self.last_output = None

        def __call__(self, module, inputs, kwargs, output):
            self.last_module = module
            self.last_tensor = getattr(module, self._attr).clone().detach()
            self.last_inputs = inputs
            self.last_kwargs = kwargs
            self.last_output = output

    @staticmethod
    def random_shape(mindims=1, maxdims=9, minsize=1, maxsize=9):
        return tuple(
            random.randint(mindims, maxdims)
            for _ in range(random.randint(minsize, maxsize))
        )

    def test_register_exec(self):
        shape = self.random_shape(maxdims=4)
        module = self.MockModule(torch.zeros(shape))
        prehook, posthook = self.MockPrehook("data"), self.MockPosthook("data")

        hook = Hook(
            prehook,
            posthook,
            prehook_kwargs={"with_kwargs": True},
            posthook_kwargs={"with_kwargs": True},
            train_update=True,
            eval_update=True,
        )

        assert not hook.registered
        hook.register(module)
        assert hook.registered

        data = [torch.rand(shape), torch.rand(shape)]
        fold = lambda a, b: a + b  # noqa:E731;

        for d in data:
            res = module(d, fold=fold)

        assert id(prehook.last_module) == id(module)
        assert torch.all(prehook.last_module.data == sum(data))
        assert torch.all(prehook.last_tensor == sum(data[:-1]))

        assert len(prehook.last_inputs) == 1
        assert torch.all(prehook.last_inputs[0] == data[-1])

        assert prehook.last_kwargs is not None and "fold" in prehook.last_kwargs
        assert id(prehook.last_kwargs["fold"]) == id(fold)

        assert id(posthook.last_module) == id(module)
        assert torch.all(posthook.last_module.data == sum(data))
        assert torch.all(posthook.last_tensor == sum(data))

        assert len(posthook.last_inputs) == 1
        assert torch.all(posthook.last_inputs[0] == data[-1])

        assert posthook.last_kwargs is not None and "fold" in posthook.last_kwargs
        assert id(posthook.last_kwargs["fold"]) == id(fold)

        assert posthook.last_output == res

    def test_deregister_exec(self):
        shape = self.random_shape(maxdims=4)
        module = self.MockModule(torch.zeros(shape))
        prehook, posthook = self.MockPrehook("data"), self.MockPosthook("data")

        hook = Hook(
            prehook,
            posthook,
            prehook_kwargs={"with_kwargs": True},
            posthook_kwargs={"with_kwargs": True},
            train_update=True,
            eval_update=True,
        )

        assert not hook.registered
        hook.register(module)
        assert hook.registered

        data = [torch.rand(shape), torch.rand(shape)]
        rfold = lambda a, b: a + b  # noqa:E731;

        for d in data:
            res = module(d, fold=rfold)

        hook.deregister()
        assert not hook.registered

        ufold = lambda a, b: a + b**2  # noqa:E731;

        for d in [torch.rand(shape), torch.rand(shape)]:
            _ = module(d, fold=ufold)

        assert id(prehook.last_module) == id(module)
        assert not torch.all(prehook.last_module.data == sum(data))
        assert torch.all(prehook.last_tensor == sum(data[:-1]))

        assert len(prehook.last_inputs) == 1
        assert torch.all(prehook.last_inputs[0] == data[-1])

        assert prehook.last_kwargs is not None and "fold" in prehook.last_kwargs
        assert id(prehook.last_kwargs["fold"]) == id(rfold)

        assert id(posthook.last_module) == id(module)
        assert not torch.all(posthook.last_module.data == sum(data))
        assert torch.all(posthook.last_tensor == sum(data))

        assert len(posthook.last_inputs) == 1
        assert torch.all(posthook.last_inputs[0] == data[-1])

        assert posthook.last_kwargs is not None and "fold" in posthook.last_kwargs
        assert id(posthook.last_kwargs["fold"]) == id(rfold)

        assert posthook.last_output == res

    @pytest.mark.parametrize(
        "execwhen",
        ("train", "eval"),
        ids=("execwhen=train", "execwhen=eval"),
    )
    def test_exec_blocking(self, execwhen):
        shape = self.random_shape(maxdims=4)
        module = self.MockModule(torch.zeros(shape))
        prehook, posthook = self.MockPrehook("data"), self.MockPosthook("data")

        trainexec = execwhen == "train"
        evalexec = execwhen == "eval"

        hook = Hook(
            prehook,
            posthook,
            prehook_kwargs={"with_kwargs": True},
            posthook_kwargs={"with_kwargs": True},
            train_update=trainexec,
            eval_update=evalexec,
        )

        hook.register(module)

        assert hook.trainexec == trainexec
        assert hook.evalexec == evalexec

        hook.trainexec = not trainexec
        hook.evalexec = not evalexec

        assert hook.trainexec != trainexec
        assert hook.evalexec != evalexec

        hook.trainexec = trainexec
        hook.evalexec = evalexec

        assert hook.trainexec == trainexec
        assert hook.evalexec == evalexec

        module.train()

        tdata = [torch.rand(shape), torch.rand(shape)]
        tfold = lambda a, b: a + b  # noqa:E731;

        for d in tdata:
            tres = module(d, fold=tfold)

        module.eval()

        edata = [torch.rand(shape), torch.rand(shape)]
        efold = lambda a, b: a + b  # noqa:E731;

        for d in edata:
            eres = module(d, fold=efold)

        checktensor = sum(tdata) if trainexec else sum([*tdata, *edata])
        checkpretensor = sum(tdata[:-1]) if trainexec else sum([*tdata, *edata[:-1]])
        checkinputs = tdata[-1] if trainexec else edata[-1]
        checkkwargs = id(tfold) if trainexec else id(efold)
        checkres = tres if trainexec else eres

        assert id(prehook.last_module) == id(module)
        assert torch.all(prehook.last_tensor == checkpretensor)

        assert len(prehook.last_inputs) == 1
        assert torch.all(prehook.last_inputs[0] == checkinputs)

        assert prehook.last_kwargs is not None and "fold" in prehook.last_kwargs
        assert id(prehook.last_kwargs["fold"]) == checkkwargs

        assert id(posthook.last_module) == id(module)
        assert torch.all(posthook.last_tensor == checktensor)

        assert len(posthook.last_inputs) == 1
        assert torch.all(posthook.last_inputs[0] == checkinputs)

        assert posthook.last_kwargs is not None and "fold" in posthook.last_kwargs
        assert id(posthook.last_kwargs["fold"]) == checkkwargs

        assert posthook.last_output == checkres


class TestStateHook:

    class MockModule(Module):

        def __init__(self, data):
            Module.__init__(self)
            self.register_buffer("data", data)

        def forward(self, inputs):
            self.data = self.data + inputs

    class MockHook(StateHook):

        def __init__(self, attr, module, train_update, eval_update, as_prehook):
            StateHook.__init__(
                self, module, train_update, eval_update, as_prehook=as_prehook
            )
            self._attr = attr
            self.last_state = None

        def hook(self, module):
            self.last_state = getattr(module, self._attr).clone().detach()

    @staticmethod
    def random_shape(mindims=1, maxdims=9, minsize=1, maxsize=9):
        return tuple(
            random.randint(mindims, maxdims)
            for _ in range(random.randint(minsize, maxsize))
        )

    def test_register_exec(self):
        shape = self.random_shape(maxdims=4)
        module = self.MockModule(torch.zeros(shape))

        prehook = self.MockHook(
            "data",
            module,
            train_update=True,
            eval_update=True,
            as_prehook=True,
        )

        posthook = self.MockHook(
            "data",
            module,
            train_update=True,
            eval_update=True,
            as_prehook=False,
        )

        assert not prehook.registered
        assert not posthook.registered

        prehook.register()
        posthook.register()

        assert prehook.registered
        assert posthook.registered

        data = [torch.rand(shape) for _ in range(random.randint(2, 9))]

        for d in data:
            _ = module(d)

        assert torch.all(prehook.last_state == sum(data[:-1]))
        assert torch.all(posthook.last_state == sum(data))

    def test_deregister_exec(self):
        shape = self.random_shape(maxdims=4)
        module = self.MockModule(torch.zeros(shape))

        prehook = self.MockHook(
            "data",
            module,
            train_update=True,
            eval_update=True,
            as_prehook=True,
        )

        posthook = self.MockHook(
            "data",
            module,
            train_update=True,
            eval_update=True,
            as_prehook=False,
        )

        assert not prehook.registered
        assert not posthook.registered

        prehook.register()
        posthook.register()

        assert prehook.registered
        assert posthook.registered

        data = [torch.rand(shape) for _ in range(random.randint(2, 9))]

        for d in data:
            _ = module(d)

        prehook.deregister()
        posthook.deregister()

        assert not prehook.registered
        assert not posthook.registered

        assert torch.all(prehook.last_state == sum(data[:-1]))
        assert torch.all(posthook.last_state == sum(data))

    @pytest.mark.parametrize(
        "execwhen",
        ("train", "eval"),
        ids=("execwhen=train", "execwhen=eval"),
    )
    def test_exec_blocking(self, execwhen):
        shape = self.random_shape(maxdims=4)
        module = self.MockModule(torch.zeros(shape))

        trainexec = execwhen == "train"
        evalexec = execwhen == "eval"

        prehook = self.MockHook(
            "data",
            module,
            train_update=trainexec,
            eval_update=evalexec,
            as_prehook=True,
        )

        posthook = self.MockHook(
            "data",
            module,
            train_update=trainexec,
            eval_update=evalexec,
            as_prehook=False,
        )

        prehook.register()
        posthook.register()

        for hook in (prehook, posthook):
            assert hook.trainexec == trainexec
            assert hook.evalexec == evalexec

            hook.trainexec = not trainexec
            hook.evalexec = not evalexec

            assert hook.trainexec != trainexec
            assert hook.evalexec != evalexec

            hook.trainexec = trainexec
            hook.evalexec = evalexec

            assert hook.trainexec == trainexec
            assert hook.evalexec == evalexec

        module.train()

        tdata = [torch.rand(shape) for _ in range(random.randint(2, 9))]

        for d in tdata:
            _ = module(d)

        module.eval()

        edata = [torch.rand(shape) for _ in range(random.randint(2, 9))]

        for d in edata:
            _ = module(d)

        if trainexec:
            assert torch.all(prehook.last_state == sum(tdata[:-1]))
            assert torch.all(posthook.last_state == sum(tdata))

        if evalexec:
            assert torch.all(prehook.last_state == sum([*tdata, *edata[:-1]]))
            assert torch.all(posthook.last_state == sum([*tdata, *edata]))

    @pytest.mark.parametrize(
        "forced",
        (False, True),
        ids=("forced=False", "forced=True"),
    )
    def test_forward_exec_unregistered(self, forced):
        shape = self.random_shape(maxdims=4)
        module = self.MockModule(torch.zeros(shape))

        prehook = self.MockHook(
            "data",
            module,
            train_update=True,
            eval_update=True,
            as_prehook=True,
        )

        posthook = self.MockHook(
            "data",
            module,
            train_update=True,
            eval_update=True,
            as_prehook=False,
        )

        prehook(forced)
        posthook(forced)

        if forced:
            assert torch.all(prehook.last_state == 0)
            assert torch.all(posthook.last_state == 0)
        else:
            assert prehook.last_state is None
            assert posthook.last_state is None

    @pytest.mark.parametrize(
        "overridden",
        (False, True),
        ids=("overridden=False", "overridden=True"),
    )
    @pytest.mark.parametrize(
        "execwhen",
        ("train", "eval"),
        ids=("execwhen=train", "execwhen=eval"),
    )
    def test_forward_exec_block_override(self, overridden, execwhen):
        shape = self.random_shape(maxdims=4)
        module = self.MockModule(torch.zeros(shape))

        trainexec = execwhen == "train"
        evalexec = execwhen == "eval"

        for training in (True, False):

            module.train(mode=training)

            prehook = self.MockHook(
                "data",
                module,
                train_update=trainexec,
                eval_update=evalexec,
                as_prehook=True,
            )

            posthook = self.MockHook(
                "data",
                module,
                train_update=trainexec,
                eval_update=evalexec,
                as_prehook=False,
            )

            prehook.register()
            posthook.register()

            prehook(ignore_mode=overridden)
            posthook(ignore_mode=overridden)

            if (trainexec == training) or overridden:
                assert torch.all(prehook.last_state == 0)
                assert torch.all(posthook.last_state == 0)
            else:
                assert prehook.last_state is None
                assert posthook.last_state is None
