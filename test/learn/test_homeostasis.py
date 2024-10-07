import pytest
import random
import torch

import inferno
from inferno.extra import ExactNeuron
from inferno.neural import DeltaCurrent, LinearDense, Serial

from inferno.learn import LinearHomeostasis


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


def mocklayer(shape, batchsz, dt, delay, biased, exclude_weight=False):
    neuron = ExactNeuron(shape, dt, rest_v=-60.0, thresh_v=-45.0, batch_size=batchsz)
    connection = LinearDense(
        shape,
        shape,
        dt,
        synapse=DeltaCurrent.partialconstructor(1.0),
        delay=delay,
        bias=biased,
        batch_size=batchsz,
    )

    connection.updater = connection.defaultupdater(exclude_weight=exclude_weight)
    return Serial(connection, neuron)


class TestLinearHomeostasis:

    @pytest.mark.parametrize(
        "param",
        ("weight", "bias", "delay"),
    )
    def test_register_bad_param(self, param):
        biased = param.lower() != "bias"
        delayed = param.lower() != "delay"

        shape = randshape(1, 3, 3, 5)
        batchsz = random.randint(1, 5)
        dt = random.uniform(0.7, 1.4)
        delay = (random.randint(0, 2) * dt) if delayed else None

        plasticity = 0.1
        target = None
        param = param.lower()
        batch_reduction = None

        layer = mocklayer(
            shape,
            batchsz,
            dt,
            delay,
            biased,
            exclude_weight=(param.lower() == "weight"),
        )
        trainer = LinearHomeostasis(
            plasticity=plasticity,
            target=target,
            param=param,
            batch_reduction=batch_reduction,
        )
        with pytest.raises(RuntimeError) as excinfo:
            _ = trainer.register_cell("onlyone", layer.cell)

        assert f"'cell' does not contain required parameter '{param}'" in str(
            excinfo.value
        )

    def test_default_override(self):
        shape = randshape(1, 3, 3, 5)
        batchsz = random.randint(1, 5)
        dt = random.uniform(0.7, 1.4)
        delay = random.randint(0, 2) * dt

        base_plasticity = random.uniform(0.05, 0.15)
        base_target = random.uniform(0.05, 0.15)
        base_param = "weight"
        base_batch_reduction = torch.amax

        override_plasticity = random.uniform(0.05, 0.15)
        override_target = random.uniform(0.05, 0.15)
        override_param = "delay"
        override_batch_reduction = torch.amin

        layer = mocklayer(shape, batchsz, dt, delay, True)
        trainer = LinearHomeostasis(
            plasticity=base_plasticity,
            target=base_target,
            param=base_param,
            batch_reduction=base_batch_reduction,
        )

        unit = trainer.register_cell(
            "onlyone",
            layer.cell,
            plasticity=override_plasticity,
            target=override_target,
            param=override_param,
            batch_reduction=override_batch_reduction,
        )

        override_plasticity = unit.state.plasticity
        override_target = unit.state.target
        override_param = unit.state.param
        override_batch_reduction = unit.state.batchreduce

    def test_default_passthrough(self):
        shape = randshape(1, 3, 3, 5)
        batchsz = random.randint(1, 5)
        dt = random.uniform(0.7, 1.4)
        delay = random.randint(0, 2) * dt

        base_plasticity = random.uniform(0.05, 0.15)
        base_target = random.uniform(0.05, 0.15)
        base_param = "weight"
        base_batch_reduction = torch.amax

        layer = mocklayer(shape, batchsz, dt, delay, True)
        trainer = LinearHomeostasis(
            plasticity=base_plasticity,
            target=base_target,
            param=base_param,
            batch_reduction=base_batch_reduction,
        )

        unit = trainer.register_cell("onlyone", layer.cell)

        base_plasticity = unit.state.plasticity
        base_target = unit.state.target
        base_param = unit.state.param
        base_batch_reduction = unit.state.batchreduce

    @pytest.mark.parametrize(
        "param",
        ("weight", "bias", "delay"),
    )
    @pytest.mark.parametrize("sign", (1.0, -1.0), ids=("pos", "neg"))
    @torch.no_grad
    def test_partial_update(self, param, sign):
        shape = randshape(1, 3, 3, 5)
        batchsz = random.randint(1, 5)
        dt = random.uniform(0.7, 1.4)
        delay = random.randint(0, 5) * dt

        plasticity = random.uniform(0.05, 0.15) * sign
        target = inferno.rescale(torch.rand(batchsz, *shape), resmin=0.05, resmax=0.15)
        param = param.lower()
        batch_reduction = torch.mean

        layer = mocklayer(shape, batchsz, dt, delay, True)
        trainer = LinearHomeostasis(
            plasticity=plasticity,
            target=None,
            param=param,
            batch_reduction=batch_reduction,
        )

        _ = trainer.register_cell("onlyone", layer.cell, target=target)

        # flip plasticity sign if delay mode
        if param == "delay":
            lr = plasticity * -1.0
        else:
            lr = plasticity

        psum = torch.zeros(batchsz, *shape)

        for count in range(1, 41):
            # inputs and outputs
            post, pre = (
                torch.rand(batchsz, *shape) < 0.3,
                torch.rand(batchsz, *shape) < 0.3,
            )

            # update term
            psum += post.float()
            update = (
                layer.connection.postsyn_receptive(
                    (target - (psum / count)) / target
                ).mean(dim=-1)
                * lr
            )

            # test on layer
            del layer.updater.weight
            del layer.updater.bias
            del layer.updater.delay
            _ = layer(pre, neuron_kwargs={"override": post})
            trainer()

            match param.lower():
                case "weight":
                    assert aaeq(
                        batch_reduction(update.clamp_min(0.0), 0),
                        layer.updater.weight.pos,
                    )
                    assert aaeq(
                        batch_reduction(update.clamp_max(0.0), 0),
                        layer.updater.weight.neg,
                    )
                case "bias":
                    assert aaeq(
                        layer.connection.like_bias(
                            batch_reduction(update.clamp_min(0.0), 0)
                        ),
                        layer.updater.bias.pos,
                    )
                    assert aaeq(
                        layer.connection.like_bias(
                            batch_reduction(update.clamp_max(0.0), 0)
                        ),
                        layer.updater.bias.neg,
                    )
                case "delay":
                    assert aaeq(
                        batch_reduction(update.clamp_min(0.0), 0),
                        layer.updater.delay.pos,
                    )
                    assert aaeq(
                        batch_reduction(update.clamp_max(0.0), 0),
                        layer.updater.delay.neg,
                    )
                case _:
                    raise RuntimeError(f"test 'param' ('{param}') not valid")
