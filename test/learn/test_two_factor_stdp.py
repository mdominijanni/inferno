import einops as ein
import pytest
import math
import random
import torch

import inferno
from inferno.extra import ExactNeuron
from inferno.learn import STDP, TripletSTDP
from inferno.neural import DeltaCurrent, LinearDirect, LinearDense, Neuron, Serial


def aaeq(t0, t1, eps=5e-6) -> bool:
    try:
        return bool(t0 == t1)
    except RuntimeError:
        if torch.all(t0 == t1):
            return True
        else:
            return torch.all((t0 - t1).abs() < eps)


class MockNeuron(Neuron):

    def __init__(
        self,
        shape: tuple[int, ...],
        batchsz: int,
        dt: float,
        thresh: float,
    ):
        Neuron.__init__(self)
        if not hasattr(shape, "__iter__"):
            self.__shape = (int(shape),)
        else:
            self.__shape = tuple(int(s) for s in shape)
        self.__batchsz = int(batchsz)
        self.__dt = float(dt)
        self.__thresh = float(thresh)
        self.__voltages = torch.zeros(self.__shape).float()
        self.__refracs = torch.zeros(self.__shape).float()
        self.__spikes = torch.zeros(self.__shape).bool()

    @property
    def shape(self) -> tuple[int, ...]:
        return self.__shape

    @property
    def count(self) -> int:
        return math.prod(self.__shape)

    @property
    def batchsz(self) -> int:
        return self.__batchsz

    @batchsz.setter
    def batchsz(self, value: int) -> None:
        self.__batchsz = int(value)

    @property
    def batchedshape(self) -> tuple[int, ...]:
        return (self.__batchsz, *self.__shape)

    @property
    def dt(self) -> float:
        return self.__dt

    @dt.setter
    def dt(self, value: float):
        self.__dt = float(value)

    @property
    def voltage(self) -> torch.Tensor:
        return self.__voltages

    @voltage.setter
    def voltage(self, value: torch.Tensor):
        self.__voltages = value

    @property
    def refrac(self) -> torch.Tensor:
        return self.__refracs

    @refrac.setter
    def refrac(self, value: torch.Tensor) -> torch.Tensor:
        self.__refracs = value

    @property
    def spike(self) -> torch.Tensor:
        return self.__spikes

    def clear(self, **kwargs):
        self.__voltages.fill_(0)
        self.__refracs.fill_(0)
        self.__spikes.fill_(0)

    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        self.__voltages = inputs
        self.__spikes = self.__voltages > self.__thresh
        return self.__spikes


def randshape(mindims=1, maxdims=9, minsize=1, maxsize=9):
    return tuple(
        random.randint(mindims, maxdims)
        for _ in range(random.randint(minsize, maxsize))
    )


def mocklayer(shape, batchsz, dt, delay):
    neuron = MockNeuron(shape, batchsz, dt, 0.5)
    connection = LinearDirect(
        shape,
        dt,
        synapse=DeltaCurrent.partialconstructor(1.0),
        delay=delay,
        batch_size=batchsz,
    )
    connection.updater = connection.defaultupdater()
    return Serial(connection, neuron)


def mocklayer2(shape, batchsz, dt, delay):
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


class TestSTDP:

    def test_default_override(self):
        shape = randshape(1, 3, 3, 5)
        batchsz = random.randint(1, 5)
        dt = random.uniform(0.7, 1.4)
        delay = random.randint(0, 2) * dt

        base_lr_post = random.uniform(-1.0, 1.0)
        base_lr_pre = random.uniform(-1.0, 1.0)
        base_tc_post = random.uniform(15.0, 30.0)
        base_tc_pre = random.uniform(15.0, 30.0)
        base_delayed = delay == 0
        base_interp_tolerance = random.uniform(1e-7, 1e-5)
        base_trace_mode = "nearest"
        base_batch_reduction = torch.amax

        override_lr_post = random.uniform(-1.0, 1.0)
        override_lr_pre = random.uniform(-1.0, 1.0)
        override_tc_post = random.uniform(15.0, 30.0)
        override_tc_pre = random.uniform(15.0, 30.0)
        override_delayed = delay != 0
        override_interp_tolerance = random.uniform(1e-7, 1e-5)
        override_trace_mode = "cumulative"
        override_batch_reduction = torch.amin

        layer = mocklayer(shape, batchsz, dt, delay)
        updater = STDP(
            lr_post=base_lr_post,
            lr_pre=base_lr_pre,
            tc_post=base_tc_post,
            tc_pre=base_tc_pre,
            delayed=base_delayed,
            interp_tolerance=base_interp_tolerance,
            trace_mode=base_trace_mode,
            batch_reduction=base_batch_reduction,
        )

        unit = updater.register_cell(
            "onlyone",
            layer.cell,
            lr_post=override_lr_post,
            lr_pre=override_lr_pre,
            tc_post=override_tc_post,
            tc_pre=override_tc_pre,
            delayed=override_delayed,
            interp_tolerance=override_interp_tolerance,
            trace_mode=override_trace_mode,
            batch_reduction=override_batch_reduction,
        )

        assert override_lr_post == unit.state.lr_post
        assert override_lr_pre == unit.state.lr_pre
        assert override_tc_post == unit.state.tc_post
        assert override_tc_pre == unit.state.tc_pre
        assert override_delayed == unit.state.delayed
        assert override_interp_tolerance == unit.state.tolerance
        assert override_trace_mode == unit.state.tracemode
        assert override_batch_reduction == unit.state.batchreduce

    def test_default_passthrough(self):
        shape = randshape(1, 3, 3, 5)
        batchsz = random.randint(1, 5)
        dt = random.uniform(0.7, 1.4)
        delay = random.randint(0, 2) * dt

        base_lr_post = random.uniform(-1.0, 1.0)
        base_lr_pre = random.uniform(-1.0, 1.0)
        base_tc_post = random.uniform(15.0, 30.0)
        base_tc_pre = random.uniform(15.0, 30.0)
        base_delayed = delay == 0
        base_interp_tolerance = random.uniform(1e-7, 1e-5)
        base_trace_mode = "nearest"
        base_batch_reduction = torch.amax

        layer = mocklayer(shape, batchsz, dt, delay)
        updater = STDP(
            lr_post=base_lr_post,
            lr_pre=base_lr_pre,
            tc_post=base_tc_post,
            tc_pre=base_tc_pre,
            delayed=base_delayed,
            interp_tolerance=base_interp_tolerance,
            trace_mode=base_trace_mode,
            batch_reduction=base_batch_reduction,
        )

        unit = updater.register_cell("onlyone", layer.cell)

        assert base_lr_post == unit.state.lr_post
        assert base_lr_pre == unit.state.lr_pre
        assert base_tc_post == unit.state.tc_post
        assert base_tc_pre == unit.state.tc_pre
        assert base_delayed == unit.state.delayed
        assert base_interp_tolerance == unit.state.tolerance
        assert base_trace_mode == unit.state.tracemode
        assert base_batch_reduction == unit.state.batchreduce

    @pytest.mark.parametrize(
        "mode",
        ("hebb", "anti", "ltp", "ltd"),
    )
    def test_partial_update(self, mode):
        shape = randshape(1, 3, 3, 5)
        batchsz = random.randint(1, 5)
        dt = random.uniform(0.7, 1.4)
        delay = 0

        layer = mocklayer(shape, batchsz, dt, delay)
        layer.connection.weight.fill_(1)

        match mode:
            case "hebb":
                lr_post_dir = 1.0
                lr_pre_dir = -1.0
            case "anti":
                lr_post_dir = -1.0
                lr_pre_dir = 1.0
            case "ltp":
                lr_post_dir = 1.0
                lr_pre_dir = 1.0
            case "ltd":
                lr_post_dir = -1.0
                lr_pre_dir = -1.0

        lr_post = random.uniform(0.0, 1.0) * lr_post_dir
        lr_pre = random.uniform(0.0, 1.0) * lr_pre_dir
        tc_post = random.uniform(15.0, 30.0)
        tc_pre = random.uniform(15.0, 30.0)
        delayed = delay != 0
        interp_tolerance = random.uniform(1e-7, 1e-5)
        trace_mode = "cumulative"
        batch_reduction = torch.sum
        field_reduction = torch.sum

        updater = STDP(
            lr_post=lr_post,
            lr_pre=lr_pre,
            tc_post=tc_post,
            tc_pre=tc_pre,
            delayed=delayed,
            interp_tolerance=interp_tolerance,
            trace_mode=trace_mode,
            batch_reduction=batch_reduction,
        )
        unit = updater.register_cell("onlyone", layer.cell)

        _ = layer(torch.rand(batchsz, *shape))
        updater()

        match mode:
            case "hebb":
                assert torch.all(
                    (
                        batch_reduction(
                            field_reduction(
                                layer.connection.postsyn_receptive(
                                    unit.monitors["spike_post"].peek()
                                )
                                * layer.connection.presyn_receptive(
                                    unit.monitors["trace_pre"].peek()
                                ),
                                -1,
                            ),
                            0,
                        )
                        - layer.updater.weight.pos
                    ).abs()
                    < 1e-5
                )
                assert torch.all(
                    (
                        batch_reduction(
                            field_reduction(
                                layer.connection.presyn_receptive(
                                    unit.monitors["spike_pre"].peek()
                                )
                                * layer.connection.postsyn_receptive(
                                    unit.monitors["trace_post"].peek()
                                ),
                                -1,
                            ),
                            0,
                        )
                        - layer.updater.weight.neg
                    ).abs()
                    < 1e-5
                )
            case "anti":
                assert torch.all(
                    (
                        batch_reduction(
                            field_reduction(
                                layer.connection.postsyn_receptive(
                                    unit.monitors["spike_post"].peek()
                                )
                                * layer.connection.presyn_receptive(
                                    unit.monitors["trace_pre"].peek()
                                ),
                                -1,
                            ),
                            0,
                        )
                        - layer.updater.weight.neg
                    ).abs()
                    < 1e-5
                )
                assert torch.all(
                    (
                        batch_reduction(
                            field_reduction(
                                layer.connection.presyn_receptive(
                                    unit.monitors["spike_pre"].peek()
                                )
                                * layer.connection.postsyn_receptive(
                                    unit.monitors["trace_post"].peek()
                                ),
                                -1,
                            ),
                            0,
                        )
                        - layer.updater.weight.pos
                    ).abs()
                    < 1e-5
                )
            case "ltp":
                assert torch.all(
                    (
                        (
                            batch_reduction(
                                field_reduction(
                                    layer.connection.postsyn_receptive(
                                        unit.monitors["spike_post"].peek()
                                    )
                                    * layer.connection.presyn_receptive(
                                        unit.monitors["trace_pre"].peek()
                                    ),
                                    -1,
                                ),
                                0,
                            )
                            + batch_reduction(
                                field_reduction(
                                    layer.connection.presyn_receptive(
                                        unit.monitors["spike_pre"].peek()
                                    )
                                    * layer.connection.postsyn_receptive(
                                        unit.monitors["trace_post"].peek()
                                    ),
                                    -1,
                                ),
                                0,
                            )
                        )
                        - layer.updater.weight.pos
                    ).abs()
                    < 1e-5
                )
                assert layer.cell.updater.weight.neg is None
            case "ltd":
                assert torch.all(
                    (
                        (
                            batch_reduction(
                                field_reduction(
                                    layer.connection.postsyn_receptive(
                                        unit.monitors["spike_post"].peek()
                                    )
                                    * layer.connection.presyn_receptive(
                                        unit.monitors["trace_pre"].peek()
                                    ),
                                    -1,
                                ),
                                0,
                            )
                            + batch_reduction(
                                field_reduction(
                                    layer.connection.presyn_receptive(
                                        unit.monitors["spike_pre"].peek()
                                    )
                                    * layer.connection.postsyn_receptive(
                                        unit.monitors["trace_post"].peek()
                                    ),
                                    -1,
                                ),
                                0,
                            )
                        )
                        - layer.updater.weight.neg
                    ).abs()
                    < 1e-5
                )
                assert layer.cell.updater.weight.pos is None

    def test_delayed_update(self):
        delaysteps = 3
        shape = randshape(1, 3, 3, 5)
        batchsz = 1
        dt = random.uniform(0.7, 1.4)
        delay = dt * delaysteps

        layer = mocklayer(shape, batchsz, dt, delay)
        layer.connection.weight.fill_(1)
        delaybase = torch.randint(0, delaysteps + 1, layer.connection.delay.shape)
        layer.connection.delay = delaybase.float() * dt

        lr_post = random.uniform(0.0, 1.0)
        lr_pre = -random.uniform(0.0, 1.0)
        tc_post = random.uniform(15.0, 30.0)
        tc_pre = random.uniform(15.0, 30.0)
        delayed = True
        interp_tolerance = 1e-3
        trace_mode = "cumulative"
        batch_reduction = torch.sum

        updater = STDP(
            lr_post=lr_post,
            lr_pre=lr_pre,
            tc_post=tc_post,
            tc_pre=tc_pre,
            delayed=delayed,
            interp_tolerance=interp_tolerance,
            trace_mode=trace_mode,
            batch_reduction=batch_reduction,
        )
        _ = updater.register_cell("onlyone", layer.cell)

        for d in range(delaysteps + 1):
            _ = layer(torch.ones(batchsz, *shape).float())
            updater()
            assert torch.all(layer.cell.updater.weight._pos[-1][delaybase <= d] > 0)
            assert torch.all(layer.cell.updater.weight._pos[-1][delaybase > d] == 0)
            assert torch.all(layer.cell.updater.weight._neg[-1][delaybase <= d] > 0)
            assert torch.all(layer.cell.updater.weight._neg[-1][delaybase > d] == 0)


class TestTripletSTDP:

    def test_default_override(self):
        shape = randshape(1, 3, 3, 5)
        batchsz = random.randint(1, 5)
        dt = random.uniform(0.7, 1.4)
        delay = random.randint(0, 2) * dt

        base_lr_post_pair = random.uniform(-1.0, 1.0)
        base_lr_post_triplet = random.uniform(0.0, 1.0)
        base_lr_pre_pair = random.uniform(-1.0, 1.0)
        base_lr_pre_triplet = random.uniform(0.0, 1.0)
        base_tc_post_fast = random.uniform(15.0, 30.0)
        base_tc_post_slow = base_tc_post_fast + random.uniform(1.0, 3.0)
        base_tc_pre_fast = random.uniform(15.0, 30.0)
        base_tc_pre_slow = base_tc_pre_fast + random.uniform(1.0, 3.0)
        base_delayed = delay == 0
        base_interp_tolerance = random.uniform(1e-7, 1e-5)
        base_trace_mode = "nearest"
        base_batch_reduction = torch.amax
        base_inplace = True

        override_lr_post_pair = random.uniform(-1.0, 1.0)
        override_lr_post_triplet = random.uniform(0.0, 1.0)
        override_lr_pre_pair = random.uniform(-1.0, 1.0)
        override_lr_pre_triplet = random.uniform(0.0, 1.0)
        override_tc_post_fast = random.uniform(15.0, 30.0)
        override_tc_post_slow = override_tc_post_fast + random.uniform(1.0, 3.0)
        override_tc_pre_fast = random.uniform(15.0, 30.0)
        override_tc_pre_slow = override_tc_pre_fast + random.uniform(1.0, 3.0)
        override_delayed = delay != 0
        override_interp_tolerance = random.uniform(1e-7, 1e-5)
        override_trace_mode = "cumulative"
        override_batch_reduction = torch.amin
        override_inplace = False

        layer = mocklayer(shape, batchsz, dt, delay)
        updater = TripletSTDP(
            lr_post_pair=base_lr_post_pair,
            lr_post_triplet=base_lr_post_triplet,
            lr_pre_pair=base_lr_pre_pair,
            lr_pre_triplet=base_lr_pre_triplet,
            tc_post_fast=base_tc_post_fast,
            tc_post_slow=base_tc_post_slow,
            tc_pre_fast=base_tc_pre_fast,
            tc_pre_slow=base_tc_pre_slow,
            delayed=base_delayed,
            interp_tolerance=base_interp_tolerance,
            trace_mode=base_trace_mode,
            batch_reduction=base_batch_reduction,
            inplace=base_inplace,
        )

        unit = updater.register_cell(
            "onlyone",
            layer.cell,
            lr_post_pair=override_lr_post_pair,
            lr_post_triplet=override_lr_post_triplet,
            lr_pre_pair=override_lr_pre_pair,
            lr_pre_triplet=override_lr_pre_triplet,
            tc_post_fast=override_tc_post_fast,
            tc_post_slow=override_tc_post_slow,
            tc_pre_fast=override_tc_pre_fast,
            tc_pre_slow=override_tc_pre_slow,
            delayed=override_delayed,
            interp_tolerance=override_interp_tolerance,
            trace_mode=override_trace_mode,
            batch_reduction=override_batch_reduction,
            inplace=override_inplace,
        )

        assert override_lr_post_pair == unit.state.lr_post_pair
        assert override_lr_post_triplet == unit.state.lr_post_triplet
        assert override_lr_pre_pair == unit.state.lr_pre_pair
        assert override_lr_pre_triplet == unit.state.lr_pre_triplet
        assert override_tc_post_fast == unit.state.tc_post_fast
        assert override_tc_post_slow == unit.state.tc_post_slow
        assert override_tc_pre_fast == unit.state.tc_pre_fast
        assert override_tc_pre_slow == unit.state.tc_pre_slow
        assert override_delayed == unit.state.delayed
        assert override_interp_tolerance == unit.state.tolerance
        assert override_trace_mode == unit.state.tracemode
        assert override_batch_reduction == unit.state.batchreduce
        assert override_inplace == unit.state.inplace

    def test_default_passthrough(self):
        shape = randshape(1, 3, 3, 5)
        batchsz = random.randint(1, 5)
        dt = random.uniform(0.7, 1.4)
        delay = random.randint(0, 2) * dt

        base_lr_post_pair = random.uniform(-1.0, 1.0)
        base_lr_post_triplet = random.uniform(0.0, 1.0)
        base_lr_pre_pair = random.uniform(-1.0, 1.0)
        base_lr_pre_triplet = random.uniform(0.0, 1.0)
        base_tc_post_fast = random.uniform(15.0, 30.0)
        base_tc_post_slow = base_tc_post_fast + random.uniform(1.0, 3.0)
        base_tc_pre_fast = random.uniform(15.0, 30.0)
        base_tc_pre_slow = base_tc_pre_fast + random.uniform(1.0, 3.0)
        base_delayed = delay == 0
        base_interp_tolerance = random.uniform(1e-7, 1e-5)
        base_trace_mode = "nearest"
        base_batch_reduction = torch.amax
        base_inplace = True

        layer = mocklayer(shape, batchsz, dt, delay)
        updater = TripletSTDP(
            lr_post_pair=base_lr_post_pair,
            lr_post_triplet=base_lr_post_triplet,
            lr_pre_pair=base_lr_pre_pair,
            lr_pre_triplet=base_lr_pre_triplet,
            tc_post_fast=base_tc_post_fast,
            tc_post_slow=base_tc_post_slow,
            tc_pre_fast=base_tc_pre_fast,
            tc_pre_slow=base_tc_pre_slow,
            delayed=base_delayed,
            interp_tolerance=base_interp_tolerance,
            trace_mode=base_trace_mode,
            batch_reduction=base_batch_reduction,
            inplace=base_inplace,
        )

        unit = updater.register_cell("onlyone", layer.cell)

        assert base_lr_post_pair == unit.state.lr_post_pair
        assert base_lr_post_triplet == unit.state.lr_post_triplet
        assert base_lr_pre_pair == unit.state.lr_pre_pair
        assert base_lr_pre_triplet == unit.state.lr_pre_triplet
        assert base_tc_post_fast == unit.state.tc_post_fast
        assert base_tc_post_slow == unit.state.tc_post_slow
        assert base_tc_pre_fast == unit.state.tc_pre_fast
        assert base_tc_pre_slow == unit.state.tc_pre_slow
        assert base_delayed == unit.state.delayed
        assert base_interp_tolerance == unit.state.tolerance
        assert base_trace_mode == unit.state.tracemode
        assert base_batch_reduction == unit.state.batchreduce
        assert base_inplace == unit.state.inplace

    @pytest.mark.parametrize(
        "mode",
        ("hebb", "anti", "ltp", "ltd"),
    )
    @pytest.mark.parametrize("trace", ("cumulative", "nearest"))
    @pytest.mark.parametrize("inplace", (True, False), ids=("inplace", "normal"))
    @torch.no_grad
    def test_partial_update(self, mode, trace, inplace):
        shape = randshape(1, 3, 3, 5)
        batchsz = random.randint(1, 5)
        dt = random.uniform(0.7, 1.4)
        delay = 0

        layer = mocklayer2(shape, batchsz, dt, delay)
        layer.connection.weight.fill_(1)

        match mode:
            case "hebb":
                lr_post_dir = 1.0
                lr_pre_dir = -1.0
            case "anti":
                lr_post_dir = -1.0
                lr_pre_dir = 1.0
            case "ltp":
                lr_post_dir = 1.0
                lr_pre_dir = 1.0
            case "ltd":
                lr_post_dir = -1.0
                lr_pre_dir = -1.0

        lr_post_pair = random.uniform(0.0, 1.0)
        lr_post_triplet = random.uniform(0.0, 1.0)
        lr_pre_pair = random.uniform(0.0, 1.0)
        lr_pre_triplet = random.uniform(0.0, 1.0)
        tc_post_fast = random.uniform(15.0, 30.0)
        tc_post_slow = tc_post_fast + random.uniform(1.0, 3.0)
        tc_pre_fast = random.uniform(15.0, 30.0)
        tc_pre_slow = tc_pre_fast + random.uniform(1.0, 3.0)
        delayed = delay == 0
        interp_tolerance = random.uniform(1e-7, 1e-5)
        trace_mode = trace
        batch_reduction = torch.sum

        trainer = TripletSTDP(
            lr_post_pair=lr_post_pair * lr_post_dir,
            lr_post_triplet=lr_post_triplet,
            lr_pre_pair=lr_pre_pair * lr_pre_dir,
            lr_pre_triplet=lr_pre_triplet,
            tc_post_fast=tc_post_fast,
            tc_post_slow=tc_post_slow,
            tc_pre_fast=tc_pre_fast,
            tc_pre_slow=tc_pre_slow,
            delayed=delayed,
            interp_tolerance=interp_tolerance,
            trace_mode=trace_mode,
            batch_reduction=batch_reduction,
            inplace=inplace,
        )
        _ = trainer.register_cell("onlyone", layer.cell)

        match trace_mode.lower():
            case "nearest":
                tracef = inferno.exp_trace_nearest
            case "cumulative":
                tracef = inferno.exp_trace_cumulative
            case _:
                raise RuntimeError("test has invalid trace mode")

        xa = torch.zeros(batchsz, *shape)
        xb = torch.zeros(batchsz, *shape)
        ya = torch.zeros(batchsz, *shape)
        yb = torch.zeros(batchsz, *shape)

        postsynr = layer.connection.postsyn_receptive
        presynr = layer.connection.presyn_receptive

        for _ in range(20):
            # inputs and outputs
            post, pre = (
                torch.rand(batchsz, *shape) < 0.3,
                torch.rand(batchsz, *shape) < 0.3,
            )

            # fast trace updates
            xa = tracef(
                pre,
                xa,
                step_time=dt,
                time_constant=tc_pre_fast,
                amplitude=1.0,
                target=True,
            )
            ya = tracef(
                post,
                ya,
                step_time=dt,
                time_constant=tc_post_fast,
                amplitude=1.0,
                target=True,
            )

            # compute update terms
            dpost = batch_reduction(
                ein.einsum(
                    postsynr(
                        (lr_post_pair + yb * lr_post_triplet) * post.to(dtype=yb.dtype)
                    ),
                    presynr(layer.connection.like_synaptic(xa)),
                    "b ... r, b ... r -> b ...",
                ),
                0,
            )
            dpre = batch_reduction(
                ein.einsum(
                    postsynr(ya),
                    presynr(
                        layer.connection.like_synaptic(
                            (lr_pre_pair + xb * lr_pre_triplet) * pre.to(dtype=xb.dtype)
                        )
                    ),
                    "b ... r, b ... r -> b ...",
                ),
                0,
            )

            # test on layer
            del layer.updater.weight
            _ = layer(pre, neuron_kwargs={"override": post})
            trainer()

            match mode:
                case "hebb":
                    assert aaeq(dpost, layer.updater.weight.pos, 5e-5)
                    assert aaeq(dpre, layer.updater.weight.neg, 5e-5)
                case "anti":
                    assert aaeq(dpost, layer.updater.weight.neg, 5e-5)
                    assert aaeq(dpre, layer.updater.weight.pos, 5e-5)
                case "ltp":
                    assert aaeq(dpost + dpre, layer.updater.weight.pos, 5e-5)
                case "ltd":
                    assert aaeq(dpost + dpre, layer.updater.weight.neg, 5e-5)

            # slow trace updates
            xb = tracef(
                pre,
                xb,
                step_time=dt,
                time_constant=tc_pre_slow,
                amplitude=1.0,
                target=True,
            )
            yb = tracef(
                post,
                yb,
                step_time=dt,
                time_constant=tc_post_slow,
                amplitude=1.0,
                target=True,
            )
