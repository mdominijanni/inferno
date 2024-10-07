import pytest
import random
import torch

import inferno
from inferno.extra import ExactNeuron
from inferno.neural import DeltaCurrent, LinearDense, Serial

from inferno.functional import exp_stdp_post_kernel, exp_stdp_pre_kernel
from inferno.learn import (
    KernelSTDP,
    DelayAdjustedSTDP,
    DelayAdjustedSTDPD,
    DelayAdjustedKernelSTDP,
    DelayAdjustedKernelSTDPD,
)


def aaeq(t0, t1, eps=5e-6) -> bool:
    try:
        return bool(t0 == t1)
    except RuntimeError:
        if torch.all(t0 == t1):
            return True
        else:
            return torch.all((t0 - t1).abs() < eps)


def randshape(mindims=1, maxdims=9, minsize=1, maxsize=9):
    return tuple(
        random.randint(mindims, maxdims)
        for _ in range(random.randint(minsize, maxsize))
    )


def mocklayer(shape, batchsz, dt, delay):
    neuron = ExactNeuron(shape, dt, rest_v=-60.0, thresh_v=-45.0, batch_size=batchsz)
    connection = LinearDense(
        shape,
        shape,
        dt,
        synapse=DeltaCurrent.partialconstructor(1.0),
        delay=delay,
        batch_size=batchsz,
    )
    connection.updater = connection.defaultupdater()
    return Serial(connection, neuron)


class TestKernelSTDP:

    def test_default_override(self):
        shape = randshape(1, 3, 3, 5)
        batchsz = random.randint(1, 5)
        dt = random.uniform(0.7, 1.4)
        delay = random.randint(0, 2) * dt

        base_kernel_post = lambda x: x**2  # noqa:E731;
        base_kernel_pre = lambda x: x  # noqa:E731;
        base_kernel_post_kwargs = {"a": 1, "b": torch.rand(2, 2)}
        base_kernel_pre_kwargs = {"a": 1, "b": torch.rand(2, 2)}
        base_delayed = delay == 0
        base_interp_tolerance = random.uniform(1e-7, 1e-5)
        base_batch_reduction = torch.amax
        base_inplace = True

        override_kernel_post = lambda x: x  # noqa:E731;
        override_kernel_pre = lambda x: x**2  # noqa:E731;
        override_kernel_post_kwargs = {"a": 2, "b": torch.rand(2, 2)}
        override_kernel_pre_kwargs = {"a": 2, "b": torch.rand(2, 2)}
        override_delayed = delay == 0
        override_interp_tolerance = random.uniform(1e-7, 1e-5)
        override_batch_reduction = torch.amin
        override_inplace = False

        layer = mocklayer(shape, batchsz, dt, delay)
        trainer = KernelSTDP(
            kernel_post=base_kernel_post,
            kernel_pre=base_kernel_pre,
            kernel_post_kwargs=base_kernel_post_kwargs,
            kernel_pre_kwargs=base_kernel_pre_kwargs,
            delayed=base_delayed,
            interp_tolerance=base_interp_tolerance,
            batch_reduction=base_batch_reduction,
            inplace=base_inplace,
        )

        unit = trainer.register_cell(
            "onlyone",
            layer.cell,
            kernel_post=override_kernel_post,
            kernel_pre=override_kernel_pre,
            kernel_post_kwargs=override_kernel_post_kwargs,
            kernel_pre_kwargs=override_kernel_pre_kwargs,
            delayed=override_delayed,
            interp_tolerance=override_interp_tolerance,
            batch_reduction=override_batch_reduction,
            inplace=override_inplace,
        )

        assert override_kernel_post == unit.state.kernel_post
        assert override_kernel_pre == unit.state.kernel_pre
        assert {"a": override_kernel_post_kwargs["a"]} == unit.state.kernel_post_kwargs
        assert "b" in {
            k: v for k, v in unit.state.kernel_post_tensor_kwargs.named_buffers()
        }
        assert (
            len({k: v for k, v in unit.state.kernel_post_tensor_kwargs.named_buffers()})
            == 1
        )
        assert torch.all(
            override_kernel_post_kwargs["b"] == unit.state.kernel_post_tensor_kwargs.b
        )
        assert {"a": override_kernel_pre_kwargs["a"]} == unit.state.kernel_pre_kwargs
        assert "b" in {
            k: v for k, v in unit.state.kernel_pre_tensor_kwargs.named_buffers()
        }
        assert (
            len({k: v for k, v in unit.state.kernel_pre_tensor_kwargs.named_buffers()})
            == 1
        )
        assert torch.all(
            override_kernel_pre_kwargs["b"] == unit.state.kernel_pre_tensor_kwargs.b
        )
        assert override_delayed == unit.state.delayed
        assert override_interp_tolerance == unit.state.tolerance
        assert override_batch_reduction == unit.state.batchreduce
        assert override_inplace == unit.state.inplace

    def test_default_passthrough(self):
        shape = randshape(1, 3, 3, 5)
        batchsz = random.randint(1, 5)
        dt = random.uniform(0.7, 1.4)
        delay = random.randint(0, 2) * dt

        base_kernel_post = lambda x: x**2  # noqa:E731;
        base_kernel_pre = lambda x: x  # noqa:E731;
        base_kernel_post_kwargs = {"a": 1, "b": torch.rand(2, 2)}
        base_kernel_pre_kwargs = {"a": 1, "b": torch.rand(2, 2)}
        base_delayed = delay == 0
        base_interp_tolerance = random.uniform(1e-7, 1e-5)
        base_batch_reduction = torch.amax
        base_inplace = True

        layer = mocklayer(shape, batchsz, dt, delay)
        trainer = KernelSTDP(
            kernel_post=base_kernel_post,
            kernel_pre=base_kernel_pre,
            kernel_post_kwargs=base_kernel_post_kwargs,
            kernel_pre_kwargs=base_kernel_pre_kwargs,
            delayed=base_delayed,
            interp_tolerance=base_interp_tolerance,
            batch_reduction=base_batch_reduction,
            inplace=base_inplace,
        )

        unit = trainer.register_cell("onlyone", layer.cell)

        assert base_kernel_post == unit.state.kernel_post
        assert base_kernel_pre == unit.state.kernel_pre
        assert {"a": base_kernel_post_kwargs["a"]} == unit.state.kernel_post_kwargs
        assert "b" in {
            k: v for k, v in unit.state.kernel_post_tensor_kwargs.named_buffers()
        }
        assert (
            len({k: v for k, v in unit.state.kernel_post_tensor_kwargs.named_buffers()})
            == 1
        )
        assert torch.all(
            base_kernel_post_kwargs["b"] == unit.state.kernel_post_tensor_kwargs.b
        )
        assert {"a": base_kernel_pre_kwargs["a"]} == unit.state.kernel_pre_kwargs
        assert "b" in {
            k: v for k, v in unit.state.kernel_pre_tensor_kwargs.named_buffers()
        }
        assert (
            len({k: v for k, v in unit.state.kernel_pre_tensor_kwargs.named_buffers()})
            == 1
        )
        assert torch.all(
            base_kernel_pre_kwargs["b"] == unit.state.kernel_pre_tensor_kwargs.b
        )
        assert base_delayed == unit.state.delayed
        assert base_interp_tolerance == unit.state.tolerance
        assert base_batch_reduction == unit.state.batchreduce
        assert base_inplace == unit.state.inplace

    @pytest.mark.parametrize(
        "mode",
        ("hebb", "anti", "ltp", "ltd"),
    )
    @pytest.mark.parametrize("inplace", (True, False), ids=("inplace", "normal"))
    @pytest.mark.parametrize("delayed", (True, False))
    @torch.no_grad
    def test_undelayed_partial_update(self, mode, inplace, delayed):
        shape = randshape(1, 3, 3, 5)
        shape = (2, 2)
        batchsz = random.randint(1, 5)
        dt = random.uniform(0.7, 1.4)
        delay = random.uniform(2 * dt, 12.9)

        layer = mocklayer(shape, batchsz, dt, delay)
        layer.connection.weight = inferno.uniform(layer.connection.delay)
        layer.connection.delay = inferno.zeros(layer.connection.delay)

        match mode:
            case "hebb":
                lr_pos = random.uniform(0.0, 1.0)
                lr_neg = random.uniform(-1.0, 0.0)
            case "anti":
                lr_pos = random.uniform(-1.0, 0.0)
                lr_neg = random.uniform(0.0, 1.0)
            case "ltp":
                lr_pos = random.uniform(0.0, 1.0)
                lr_neg = random.uniform(0.0, 1.0)
            case "ltd":
                lr_pos = random.uniform(-1.0, 0.0)
                lr_neg = random.uniform(-1.0, 0.0)
            case _:
                assert False

        tc_pos = random.uniform(15.0, 30.0)
        tc_neg = random.uniform(15.0, 30.0)
        interp_tolerance = random.uniform(1e-7, 1e-5)
        batch_reduction = torch.sum

        kstdp = KernelSTDP(
            kernel_post=exp_stdp_post_kernel,
            kernel_pre=exp_stdp_pre_kernel,
            kernel_post_kwargs={"learning_rate": lr_pos, "time_constant": tc_pos},
            kernel_pre_kwargs={"learning_rate": lr_neg, "time_constant": tc_neg},
            delayed=delayed,
            interp_tolerance=interp_tolerance,
            batch_reduction=batch_reduction,
            inplace=inplace,
        )
        _ = kstdp.register_cell("onlyone", layer.cell)

        dastdp = DelayAdjustedSTDP(
            lr_neg=lr_neg,
            lr_pos=lr_pos,
            tc_pos=tc_pos,
            tc_neg=tc_neg,
            interp_tolerance=interp_tolerance,
            batch_reduction=batch_reduction,
            inplace=inplace,
        )
        _ = dastdp.register_cell("onlyone", layer.cell)

        for _ in range(25):
            # inputs and outputs
            post, pre = (
                torch.rand(batchsz, *shape) < 0.3,
                torch.rand(batchsz, *shape) < 0.3,
            )

            # pass forward in layer
            _ = layer(pre, neuron_kwargs={"override": post})

            # get values from kernel stdp
            kstdp()
            kpos = layer.updater.weight.pos
            kpos = None if kpos is None else kpos.clone().detach()
            kneg = layer.updater.weight.neg
            kneg = None if kneg is None else kneg.clone().detach()
            del layer.updater.weight

            dastdp()
            dpos = layer.updater.weight.pos
            dpos = inferno.zeros(kpos) if dpos is None else dpos.clone().detach()
            dneg = layer.updater.weight.neg
            dneg = inferno.zeros(kneg) if dneg is None else dneg.clone().detach()
            del layer.updater.weight

            assert aaeq(kpos, dpos)
            assert aaeq(kneg, dneg)


class TestDelayAdjustedKernelSTDP:

    def test_default_override(self):
        shape = randshape(1, 3, 3, 5)
        batchsz = random.randint(1, 5)
        dt = random.uniform(0.7, 1.4)
        delay = random.randint(0, 10) * dt

        base_kernel_post = lambda x: x**2  # noqa:E731;
        base_kernel_pre = lambda x: x  # noqa:E731;
        base_kernel_post_kwargs = {"a": 1, "b": torch.rand(2, 2)}
        base_kernel_pre_kwargs = {"a": 1, "b": torch.rand(2, 2)}
        base_batch_reduction = torch.amax
        base_inplace = True

        override_kernel_post = lambda x: x  # noqa:E731;
        override_kernel_pre = lambda x: x**2  # noqa:E731;
        override_kernel_post_kwargs = {"a": 2, "b": torch.rand(2, 2)}
        override_kernel_pre_kwargs = {"a": 2, "b": torch.rand(2, 2)}
        override_batch_reduction = torch.amin
        override_inplace = False

        layer = mocklayer(shape, batchsz, dt, delay)
        trainer = DelayAdjustedKernelSTDP(
            kernel_post=base_kernel_post,
            kernel_pre=base_kernel_pre,
            kernel_post_kwargs=base_kernel_post_kwargs,
            kernel_pre_kwargs=base_kernel_pre_kwargs,
            batch_reduction=base_batch_reduction,
            inplace=base_inplace,
        )

        unit = trainer.register_cell(
            "onlyone",
            layer.cell,
            kernel_post=override_kernel_post,
            kernel_pre=override_kernel_pre,
            kernel_post_kwargs=override_kernel_post_kwargs,
            kernel_pre_kwargs=override_kernel_pre_kwargs,
            batch_reduction=override_batch_reduction,
            inplace=override_inplace,
        )

        assert override_kernel_post == unit.state.kernel_post
        assert override_kernel_pre == unit.state.kernel_pre
        assert {"a": override_kernel_post_kwargs["a"]} == unit.state.kernel_post_kwargs
        assert "b" in {
            k: v for k, v in unit.state.kernel_post_tensor_kwargs.named_buffers()
        }
        assert (
            len({k: v for k, v in unit.state.kernel_post_tensor_kwargs.named_buffers()})
            == 1
        )
        assert torch.all(
            override_kernel_post_kwargs["b"] == unit.state.kernel_post_tensor_kwargs.b
        )
        assert {"a": override_kernel_pre_kwargs["a"]} == unit.state.kernel_pre_kwargs
        assert "b" in {
            k: v for k, v in unit.state.kernel_pre_tensor_kwargs.named_buffers()
        }
        assert (
            len({k: v for k, v in unit.state.kernel_pre_tensor_kwargs.named_buffers()})
            == 1
        )
        assert torch.all(
            override_kernel_pre_kwargs["b"] == unit.state.kernel_pre_tensor_kwargs.b
        )
        assert override_batch_reduction == unit.state.batchreduce
        assert override_inplace == unit.state.inplace

    def test_default_passthrough(self):
        shape = randshape(1, 3, 3, 5)
        batchsz = random.randint(1, 5)
        dt = random.uniform(0.7, 1.4)
        delay = random.randint(0, 10) * dt

        base_kernel_post = lambda x: x**2  # noqa:E731;
        base_kernel_pre = lambda x: x  # noqa:E731;
        base_kernel_post_kwargs = {"a": 1, "b": torch.rand(2, 2)}
        base_kernel_pre_kwargs = {"a": 1, "b": torch.rand(2, 2)}
        base_batch_reduction = torch.amax
        base_inplace = True

        layer = mocklayer(shape, batchsz, dt, delay)
        trainer = KernelSTDP(
            kernel_post=base_kernel_post,
            kernel_pre=base_kernel_pre,
            kernel_post_kwargs=base_kernel_post_kwargs,
            kernel_pre_kwargs=base_kernel_pre_kwargs,
            batch_reduction=base_batch_reduction,
            inplace=base_inplace,
        )

        unit = trainer.register_cell("onlyone", layer.cell)

        assert base_kernel_post == unit.state.kernel_post
        assert base_kernel_pre == unit.state.kernel_pre
        assert {"a": base_kernel_post_kwargs["a"]} == unit.state.kernel_post_kwargs
        assert "b" in {
            k: v for k, v in unit.state.kernel_post_tensor_kwargs.named_buffers()
        }
        assert (
            len({k: v for k, v in unit.state.kernel_post_tensor_kwargs.named_buffers()})
            == 1
        )
        assert torch.all(
            base_kernel_post_kwargs["b"] == unit.state.kernel_post_tensor_kwargs.b
        )
        assert {"a": base_kernel_pre_kwargs["a"]} == unit.state.kernel_pre_kwargs
        assert "b" in {
            k: v for k, v in unit.state.kernel_pre_tensor_kwargs.named_buffers()
        }
        assert (
            len({k: v for k, v in unit.state.kernel_pre_tensor_kwargs.named_buffers()})
            == 1
        )
        assert torch.all(
            base_kernel_pre_kwargs["b"] == unit.state.kernel_pre_tensor_kwargs.b
        )
        assert base_batch_reduction == unit.state.batchreduce
        assert base_inplace == unit.state.inplace

    @pytest.mark.parametrize(
        "mode",
        ("hebb", "anti", "ltp", "ltd"),
    )
    @pytest.mark.parametrize("inplace", (True, False), ids=("inplace", "normal"))
    @torch.no_grad
    def test_undelayed_partial_update(self, mode, inplace):
        shape = randshape(1, 3, 3, 5)
        shape = (2, 2)
        batchsz = random.randint(1, 5)
        dt = random.uniform(0.7, 1.4)
        delay = random.uniform(2 * dt, 12.9)

        layer = mocklayer(shape, batchsz, dt, delay)
        layer.connection.weight = inferno.uniform(layer.connection.delay)
        layer.connection.delay = inferno.rescale(
            inferno.uniform(layer.connection.delay), resmin=0.0, resmax=delay
        )

        match mode:
            case "hebb":
                lr_pos = random.uniform(0.0, 1.0)
                lr_neg = random.uniform(-1.0, 0.0)
            case "anti":
                lr_pos = random.uniform(-1.0, 0.0)
                lr_neg = random.uniform(0.0, 1.0)
            case "ltp":
                lr_pos = random.uniform(0.0, 1.0)
                lr_neg = random.uniform(0.0, 1.0)
            case "ltd":
                lr_pos = random.uniform(-1.0, 0.0)
                lr_neg = random.uniform(-1.0, 0.0)
            case _:
                assert False

        tc_pos = random.uniform(15.0, 30.0)
        tc_neg = random.uniform(15.0, 30.0)
        batch_reduction = torch.sum

        dakstdp = DelayAdjustedKernelSTDP(
            kernel_post=exp_stdp_post_kernel,
            kernel_pre=exp_stdp_pre_kernel,
            kernel_post_kwargs={"learning_rate": lr_pos, "time_constant": tc_pos},
            kernel_pre_kwargs={"learning_rate": lr_neg, "time_constant": tc_neg},
            batch_reduction=batch_reduction,
            inplace=inplace,
        )
        _ = dakstdp.register_cell("onlyone", layer.cell)

        dastdp = DelayAdjustedSTDP(
            lr_neg=lr_neg,
            lr_pos=lr_pos,
            tc_pos=tc_pos,
            tc_neg=tc_neg,
            batch_reduction=batch_reduction,
            inplace=inplace,
        )
        _ = dastdp.register_cell("onlyone", layer.cell)

        for _ in range(30):
            # inputs and outputs
            post, pre = (
                torch.rand(batchsz, *shape) < 0.3,
                torch.rand(batchsz, *shape) < 0.3,
            )

            # pass forward in layer
            _ = layer(pre, neuron_kwargs={"override": post})

            # get values from kernel stdp
            dakstdp()
            kpos = layer.updater.weight.pos
            kpos = None if kpos is None else kpos.clone().detach()
            kneg = layer.updater.weight.neg
            kneg = None if kneg is None else kneg.clone().detach()
            del layer.updater.weight

            dastdp()
            dpos = layer.updater.weight.pos
            dpos = inferno.zeros(kpos) if dpos is None else dpos.clone().detach()
            dneg = layer.updater.weight.neg
            dneg = inferno.zeros(kneg) if dneg is None else dneg.clone().detach()
            del layer.updater.weight

            assert aaeq(kpos, dpos)
            assert aaeq(kneg, dneg)


class TestDelayAdjustedKernelSTDPD:

    def test_default_override(self):
        shape = randshape(1, 3, 3, 5)
        batchsz = random.randint(1, 5)
        dt = random.uniform(0.7, 1.4)
        delay = random.randint(0, 10) * dt

        base_kernel_post = lambda x: x**2  # noqa:E731;
        base_kernel_pre = lambda x: x  # noqa:E731;
        base_kernel_post_kwargs = {"a": 1, "b": torch.rand(2, 2)}
        base_kernel_pre_kwargs = {"a": 1, "b": torch.rand(2, 2)}
        base_batch_reduction = torch.amax
        base_inplace = True

        override_kernel_post = lambda x: x  # noqa:E731;
        override_kernel_pre = lambda x: x**2  # noqa:E731;
        override_kernel_post_kwargs = {"a": 2, "b": torch.rand(2, 2)}
        override_kernel_pre_kwargs = {"a": 2, "b": torch.rand(2, 2)}
        override_batch_reduction = torch.amin
        override_inplace = False

        layer = mocklayer(shape, batchsz, dt, delay)
        trainer = DelayAdjustedKernelSTDP(
            kernel_post=base_kernel_post,
            kernel_pre=base_kernel_pre,
            kernel_post_kwargs=base_kernel_post_kwargs,
            kernel_pre_kwargs=base_kernel_pre_kwargs,
            batch_reduction=base_batch_reduction,
            inplace=base_inplace,
        )

        unit = trainer.register_cell(
            "onlyone",
            layer.cell,
            kernel_post=override_kernel_post,
            kernel_pre=override_kernel_pre,
            kernel_post_kwargs=override_kernel_post_kwargs,
            kernel_pre_kwargs=override_kernel_pre_kwargs,
            batch_reduction=override_batch_reduction,
            inplace=override_inplace,
        )

        assert override_kernel_post == unit.state.kernel_post
        assert override_kernel_pre == unit.state.kernel_pre
        assert {"a": override_kernel_post_kwargs["a"]} == unit.state.kernel_post_kwargs
        assert "b" in {
            k: v for k, v in unit.state.kernel_post_tensor_kwargs.named_buffers()
        }
        assert (
            len({k: v for k, v in unit.state.kernel_post_tensor_kwargs.named_buffers()})
            == 1
        )
        assert torch.all(
            override_kernel_post_kwargs["b"] == unit.state.kernel_post_tensor_kwargs.b
        )
        assert {"a": override_kernel_pre_kwargs["a"]} == unit.state.kernel_pre_kwargs
        assert "b" in {
            k: v for k, v in unit.state.kernel_pre_tensor_kwargs.named_buffers()
        }
        assert (
            len({k: v for k, v in unit.state.kernel_pre_tensor_kwargs.named_buffers()})
            == 1
        )
        assert torch.all(
            override_kernel_pre_kwargs["b"] == unit.state.kernel_pre_tensor_kwargs.b
        )
        assert override_batch_reduction == unit.state.batchreduce
        assert override_inplace == unit.state.inplace

    def test_default_passthrough(self):
        shape = randshape(1, 3, 3, 5)
        batchsz = random.randint(1, 5)
        dt = random.uniform(0.7, 1.4)
        delay = random.randint(0, 10) * dt

        base_kernel_post = lambda x: x**2  # noqa:E731;
        base_kernel_pre = lambda x: x  # noqa:E731;
        base_kernel_post_kwargs = {"a": 1, "b": torch.rand(2, 2)}
        base_kernel_pre_kwargs = {"a": 1, "b": torch.rand(2, 2)}
        base_batch_reduction = torch.amax
        base_inplace = True

        layer = mocklayer(shape, batchsz, dt, delay)
        trainer = KernelSTDP(
            kernel_post=base_kernel_post,
            kernel_pre=base_kernel_pre,
            kernel_post_kwargs=base_kernel_post_kwargs,
            kernel_pre_kwargs=base_kernel_pre_kwargs,
            batch_reduction=base_batch_reduction,
            inplace=base_inplace,
        )

        unit = trainer.register_cell("onlyone", layer.cell)

        assert base_kernel_post == unit.state.kernel_post
        assert base_kernel_pre == unit.state.kernel_pre
        assert {"a": base_kernel_post_kwargs["a"]} == unit.state.kernel_post_kwargs
        assert "b" in {
            k: v for k, v in unit.state.kernel_post_tensor_kwargs.named_buffers()
        }
        assert (
            len({k: v for k, v in unit.state.kernel_post_tensor_kwargs.named_buffers()})
            == 1
        )
        assert torch.all(
            base_kernel_post_kwargs["b"] == unit.state.kernel_post_tensor_kwargs.b
        )
        assert {"a": base_kernel_pre_kwargs["a"]} == unit.state.kernel_pre_kwargs
        assert "b" in {
            k: v for k, v in unit.state.kernel_pre_tensor_kwargs.named_buffers()
        }
        assert (
            len({k: v for k, v in unit.state.kernel_pre_tensor_kwargs.named_buffers()})
            == 1
        )
        assert torch.all(
            base_kernel_pre_kwargs["b"] == unit.state.kernel_pre_tensor_kwargs.b
        )
        assert base_batch_reduction == unit.state.batchreduce
        assert base_inplace == unit.state.inplace

    @pytest.mark.parametrize(
        "mode",
        ("csync", "acsync", "sync", "desync"),
    )
    @pytest.mark.parametrize("inplace", (True, False), ids=("inplace", "normal"))
    @torch.no_grad
    def test_undelayed_partial_update(self, mode, inplace):
        shape = randshape(1, 3, 3, 5)
        shape = (2, 2)
        batchsz = random.randint(1, 5)
        dt = random.uniform(0.7, 1.4)
        delay = random.uniform(2 * dt, 12.9)

        layer = mocklayer(shape, batchsz, dt, delay)
        layer.connection.weight = inferno.uniform(layer.connection.delay)
        layer.connection.delay = inferno.rescale(
            inferno.uniform(layer.connection.delay), resmin=0.0, resmax=delay
        )

        match mode:
            case "csync":
                lr_pos = random.uniform(0.0, 1.0)
                lr_neg = random.uniform(-1.0, 0.0)
            case "acsync":
                lr_pos = random.uniform(-1.0, 0.0)
                lr_neg = random.uniform(0.0, 1.0)
            case "desync":
                lr_pos = random.uniform(0.0, 1.0)
                lr_neg = random.uniform(0.0, 1.0)
            case "sync":
                lr_pos = random.uniform(-1.0, 0.0)
                lr_neg = random.uniform(-1.0, 0.0)
            case _:
                assert False

        tc_pos = random.uniform(15.0, 30.0)
        tc_neg = random.uniform(15.0, 30.0)
        batch_reduction = torch.sum

        dakstdpd = DelayAdjustedKernelSTDPD(
            kernel_post=exp_stdp_post_kernel,
            kernel_pre=exp_stdp_pre_kernel,
            kernel_post_kwargs={"learning_rate": lr_neg, "time_constant": tc_neg},
            kernel_pre_kwargs={"learning_rate": lr_pos, "time_constant": tc_pos},
            batch_reduction=batch_reduction,
            inplace=inplace,
        )
        _ = dakstdpd.register_cell("onlyone", layer.cell)

        dastdpd = DelayAdjustedSTDPD(
            lr_neg=lr_neg,
            lr_pos=lr_pos,
            tc_pos=tc_pos,
            tc_neg=tc_neg,
            batch_reduction=batch_reduction,
            inplace=inplace,
        )
        _ = dastdpd.register_cell("onlyone", layer.cell)

        for _ in range(30):
            # inputs and outputs
            post, pre = (
                torch.rand(batchsz, *shape) < 0.3,
                torch.rand(batchsz, *shape) < 0.3,
            )

            # pass forward in layer
            _ = layer(pre, neuron_kwargs={"override": post})

            # get values from kernel stdp
            dakstdpd()
            kpos = layer.updater.delay.pos
            kpos = None if kpos is None else kpos.clone().detach()
            kneg = layer.updater.delay.neg
            kneg = None if kneg is None else kneg.clone().detach()
            del layer.updater.delay

            dastdpd()
            dpos = layer.updater.delay.pos
            dpos = inferno.zeros(kpos) if dpos is None else dpos.clone().detach()
            dneg = layer.updater.delay.neg
            dneg = inferno.zeros(kneg) if dneg is None else dneg.clone().detach()
            del layer.updater.delay

            assert aaeq(kpos, dpos)
            assert aaeq(kneg, dneg)
