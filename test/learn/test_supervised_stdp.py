import pytest
import math
import random
import torch

from inferno.learn import MSTDPET, MSTDP
from inferno.neural import DeltaCurrent, LinearDirect, Neuron, Serial


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


class MockProbNeuron(Neuron):

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
        self.__voltages = inputs * 1.0
        self.__spikes = torch.rand_like(self.__voltages) < self.__thresh
        return self.__spikes


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
        random.randint(minsize, maxsize)
        for _ in range(random.randint(mindims, maxdims))
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


def mockproblayer(shape, batchsz, dt, delay, prob):
    neuron = MockProbNeuron(shape, batchsz, dt, prob)
    connection = LinearDirect(
        shape,
        dt,
        synapse=DeltaCurrent.partialconstructor(1.0),
        delay=delay,
        batch_size=batchsz,
    )
    connection.updater = connection.defaultupdater()
    return Serial(connection, neuron)


class TestMSTDPET:

    def test_default_override(self):
        shape = randshape(1, 3, 3, 5)
        batchsz = random.randint(1, 5)
        dt = random.uniform(0.7, 1.4)
        delay = random.randint(0, 2) * dt

        layer = mocklayer(shape, batchsz, dt, delay)

        base_step_time = random.uniform(0.7, 1.4)
        base_lr_post = random.uniform(-1.0, 1.0)
        base_lr_pre = random.uniform(-1.0, 1.0)
        base_tc_post = random.uniform(15.0, 30.0)
        base_tc_pre = random.uniform(15.0, 30.0)
        base_tc_eligibility = random.uniform(15.0, 30.0)
        base_interp_tolerance = random.uniform(1e-7, 1e-5)
        base_trace_mode = "nearest"
        base_batch_reduction = torch.amax

        override_step_time = random.uniform(0.7, 1.4)
        override_lr_post = random.uniform(-1.0, 1.0)
        override_lr_pre = random.uniform(-1.0, 1.0)
        override_tc_post = random.uniform(15.0, 30.0)
        override_tc_pre = random.uniform(15.0, 30.0)
        override_tc_eligibility = random.uniform(15.0, 30.0)
        override_interp_tolerance = random.uniform(1e-7, 1e-5)
        override_trace_mode = "cumulative"
        override_batch_reduction = torch.amin

        updater = MSTDPET(
            step_time=base_step_time,
            lr_post=base_lr_post,
            lr_pre=base_lr_pre,
            tc_post=base_tc_post,
            tc_pre=base_tc_pre,
            tc_eligibility=base_tc_eligibility,
            interp_tolerance=base_interp_tolerance,
            trace_mode=base_trace_mode,
            batch_reduction=base_batch_reduction,
        )

        unit = updater.register_cell(
            "onlyone",
            layer.cell,
            step_time=override_step_time,
            lr_post=override_lr_post,
            lr_pre=override_lr_pre,
            tc_post=override_tc_post,
            tc_pre=override_tc_pre,
            tc_eligibility=override_tc_eligibility,
            interp_tolerance=override_interp_tolerance,
            trace_mode=override_trace_mode,
            batch_reduction=override_batch_reduction,
        )

        assert override_step_time == unit.state.step_time
        assert override_lr_post == unit.state.lr_post
        assert override_lr_pre == unit.state.lr_pre
        assert override_tc_post == unit.state.tc_post
        assert override_tc_pre == unit.state.tc_pre
        assert override_tc_eligibility == unit.state.tc_eligibility
        assert override_interp_tolerance == unit.state.tolerance
        assert override_trace_mode == unit.state.tracemode
        assert override_batch_reduction == unit.state.batchreduce

    def test_default_passthrough(self):
        shape = randshape(1, 3, 3, 5)
        batchsz = random.randint(1, 5)
        dt = random.uniform(0.7, 1.4)
        delay = random.randint(0, 2) * dt

        layer = mocklayer(shape, batchsz, dt, delay)

        base_step_time = random.uniform(0.7, 1.4)
        base_lr_post = random.uniform(-1.0, 1.0)
        base_lr_pre = random.uniform(-1.0, 1.0)
        base_tc_post = random.uniform(15.0, 30.0)
        base_tc_pre = random.uniform(15.0, 30.0)
        base_tc_eligibility = random.uniform(15.0, 30.0)
        base_interp_tolerance = random.uniform(1e-7, 1e-5)
        base_trace_mode = "nearest"
        base_batch_reduction = torch.amax

        updater = MSTDPET(
            step_time=base_step_time,
            lr_post=base_lr_post,
            lr_pre=base_lr_pre,
            tc_post=base_tc_post,
            tc_pre=base_tc_pre,
            tc_eligibility=base_tc_eligibility,
            interp_tolerance=base_interp_tolerance,
            trace_mode=base_trace_mode,
            batch_reduction=base_batch_reduction,
        )

        unit = updater.register_cell("onlyone", layer.cell)

        assert base_step_time == unit.state.step_time
        assert base_lr_post == unit.state.lr_post
        assert base_lr_pre == unit.state.lr_pre
        assert base_tc_post == unit.state.tc_post
        assert base_tc_pre == unit.state.tc_pre
        assert base_tc_eligibility == unit.state.tc_eligibility
        assert base_interp_tolerance == unit.state.tolerance
        assert base_batch_reduction == unit.state.batchreduce
        assert base_batch_reduction == unit.state.batchreduce

    @pytest.mark.parametrize(
        "kind",
        ("hebb", "anti", "ltp", "ltp"),
    )
    @pytest.mark.parametrize(
        "mode",
        ("tensor", "negscalar", "posscalar"),
    )
    def test_partial_update(self, kind, mode):
        # shape = randshape(1, 3, 3, 5)
        shape = (2, 2)
        batchsz = random.randint(2, 9)
        dt = random.uniform(0.7, 1.4)
        delay = random.randint(0, 2) * dt

        layer = mockproblayer(shape, batchsz, dt, delay, 0.2)
        layer.connection.weight.fill_(1)

        match kind:
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

        step_time = random.uniform(0.7, 1.4)
        lr_post = random.uniform(0.1, 1.0) * lr_post_dir
        lr_pre = random.uniform(0.1, 1.0) * lr_pre_dir
        tc_post = random.uniform(15.0, 30.0)
        tc_pre = random.uniform(15.0, 30.0)
        tc_eligibility = random.uniform(15.0, 30.0)
        interp_tolerance = random.uniform(1e-7, 1e-5)
        batch_reduction = torch.sum

        updater = MSTDPET(
            step_time=step_time,
            lr_post=lr_post,
            lr_pre=lr_pre,
            tc_post=tc_post,
            tc_pre=tc_pre,
            tc_eligibility=tc_eligibility,
            interp_tolerance=interp_tolerance,
            batch_reduction=batch_reduction,
        )

        _ = updater.register_cell("onlyone", layer.cell)

        xpost = torch.zeros(batchsz, *shape)
        xpre = torch.zeros(batchsz, *shape)
        epost = torch.zeros(batchsz, *shape)
        epre = torch.zeros(batchsz, *shape)
        ecomb = torch.zeros(batchsz, *shape)

        for k in range(100):
            inputs = torch.rand(batchsz, *shape) < 0.2
            outputs = layer(inputs.float())

            xpost = xpost * math.exp(-step_time / tc_post) + abs(lr_pre) * outputs
            xpre = xpre * math.exp(-step_time / tc_pre) + abs(lr_post) * inputs

            epost = (
                epost * math.exp(-step_time / tc_eligibility)
                + (xpre / tc_eligibility) * outputs
            )
            epre = (
                epre * math.exp(-step_time / tc_eligibility)
                + (xpost / tc_eligibility) * inputs
            )
            ecomb = (
                ecomb * math.exp(-step_time / tc_eligibility)
                + ((xpost * lr_pre_dir * inputs) + (xpre * lr_post_dir * outputs))
                / tc_eligibility
            )

            match mode:
                case "tensor":
                    remreward = torch.tensor(
                        [1 - 2 * (k % 2) for k in range(batchsz)]
                    ) * torch.rand(batchsz)
                    reward = remreward.clone().detach()

                case "negscalar":
                    remreward = random.uniform(0.1, 1.0) * -1.0
                    reward = torch.ones(batchsz) * remreward
                case "posscalar":
                    remreward = random.uniform(0.1, 1.0)
                    reward = torch.ones(batchsz) * remreward

            reward_pos = torch.argwhere(reward >= 0).view(-1)
            reward_neg = torch.argwhere(reward < 0).view(-1)

            updater(remreward)

            upost = epost.view(batchsz, -1) * reward.view(-1, 1)
            upre = epre.view(batchsz, -1) * reward.view(-1, 1)
            ucomb = ecomb.view(batchsz, -1) * reward.view(-1, 1)

            upost_reg, upost_inv = upost.abs()[reward_pos], upost.abs()[reward_neg]
            upre_reg, upre_inv = upre.abs()[reward_pos], upre.abs()[reward_neg]

            match kind:
                case "ltd":
                    upos = torch.cat((upost_inv, upre_inv), 0)
                    uneg = torch.cat((upost_reg, upre_reg), 0)
                case "anti":
                    upos = torch.cat((upost_inv, upre_reg), 0)
                    uneg = torch.cat((upost_reg, upre_inv), 0)
                case "hebb":
                    upos = torch.cat((upost_reg, upre_inv), 0)
                    uneg = torch.cat((upost_inv, upre_reg), 0)
                case "ltp":
                    upos = torch.cat((upost_reg, upre_reg), 0)
                    uneg = torch.cat((upost_inv, upre_inv), 0)

            assert aaeq(upost * lr_post_dir + upre * lr_pre_dir, ucomb)
            assert aaeq(
                batch_reduction(upos, 0) if upos.numel() else None,
                layer.updater.weight.pos,
            )
            assert aaeq(
                batch_reduction(uneg, 0) if uneg.numel() else None,
                layer.updater.weight.neg,
            )

            if upos.numel() and uneg.numel():
                aaeq(layer.updater.weight.pos - layer.updater.weight.neg, ucomb)
            elif upos.numel():
                aaeq(layer.updater.weight.pos, ucomb)
            elif uneg.numel():
                aaeq(-layer.updater.weight.neg, ucomb)

            assert aaeq(layer.updater.weight.update(None), batch_reduction(ucomb, 0))

            del layer.updater.weight


class TestMSTDP:

    def test_default_override(self):
        shape = randshape(1, 3, 3, 5)
        batchsz = random.randint(1, 5)
        dt = random.uniform(0.7, 1.4)
        delay = random.randint(0, 2) * dt

        base_step_time = random.uniform(0.7, 1.4)
        base_lr_post = random.uniform(-1.0, 1.0)
        base_lr_pre = random.uniform(-1.0, 1.0)
        base_tc_post = random.uniform(15.0, 30.0)
        base_tc_pre = random.uniform(15.0, 30.0)
        base_delayed = delay == 0
        base_interp_tolerance = random.uniform(1e-7, 1e-5)
        base_trace_mode = "nearest"
        base_batch_reduction = torch.amax

        override_step_time = random.uniform(0.7, 1.4)
        override_lr_post = random.uniform(-1.0, 1.0)
        override_lr_pre = random.uniform(-1.0, 1.0)
        override_tc_post = random.uniform(15.0, 30.0)
        override_tc_pre = random.uniform(15.0, 30.0)
        override_delayed = delay != 0
        override_interp_tolerance = random.uniform(1e-7, 1e-5)
        override_trace_mode = "cumulative"
        override_batch_reduction = torch.amin

        layer = mocklayer(shape, batchsz, dt, delay)
        updater = MSTDP(
            step_time=base_step_time,
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
            step_time=override_step_time,
            lr_post=override_lr_post,
            lr_pre=override_lr_pre,
            tc_post=override_tc_post,
            tc_pre=override_tc_pre,
            delayed=override_delayed,
            interp_tolerance=override_interp_tolerance,
            trace_mode=override_trace_mode,
            batch_reduction=override_batch_reduction,
        )

        assert override_step_time == unit.state.step_time
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

        base_step_time = random.uniform(0.7, 1.4)
        base_lr_post = random.uniform(-1.0, 1.0)
        base_lr_pre = random.uniform(-1.0, 1.0)
        base_tc_post = random.uniform(15.0, 30.0)
        base_tc_pre = random.uniform(15.0, 30.0)
        base_delayed = delay == 0
        base_interp_tolerance = random.uniform(1e-7, 1e-5)
        base_trace_mode = "nearest"
        base_batch_reduction = torch.amax

        layer = mocklayer(shape, batchsz, dt, delay)
        updater = MSTDP(
            step_time=base_step_time,
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

        assert base_step_time == unit.state.step_time
        assert base_lr_post == unit.state.lr_post
        assert base_lr_pre == unit.state.lr_pre
        assert base_tc_post == unit.state.tc_post
        assert base_tc_pre == unit.state.tc_pre
        assert base_delayed == unit.state.delayed
        assert base_interp_tolerance == unit.state.tolerance
        assert base_trace_mode == unit.state.tracemode
        assert base_batch_reduction == unit.state.batchreduce

    @pytest.mark.parametrize(
        "kind",
        ("hebb", "anti", "ltp", "ltp"),
    )
    @pytest.mark.parametrize(
        "mode",
        ("tensor", "negscalar", "posscalar"),
    )
    def test_partial_update(self, kind, mode):
        # shape = randshape(1, 3, 3, 5)
        shape = (2, 2)
        batchsz = random.randint(2, 9)
        dt = random.uniform(0.7, 1.4)
        delay = random.randint(0, 2) * dt

        layer = mockproblayer(shape, batchsz, dt, delay, 0.2)
        layer.connection.weight.fill_(1)

        match kind:
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

        step_time = random.uniform(0.7, 1.4)
        lr_post = random.uniform(0.1, 1.0) * lr_post_dir
        lr_pre = random.uniform(0.1, 1.0) * lr_pre_dir
        tc_post = random.uniform(15.0, 30.0)
        tc_pre = random.uniform(15.0, 30.0)
        interp_tolerance = random.uniform(1e-7, 1e-5)
        trace_mode = "cumulative"
        batch_reduction = torch.sum

        updater = MSTDP(
            step_time=step_time,
            lr_post=lr_post,
            lr_pre=lr_pre,
            tc_post=tc_post,
            tc_pre=tc_pre,
            interp_tolerance=interp_tolerance,
            trace_mode=trace_mode,
            batch_reduction=batch_reduction,
        )

        _ = updater.register_cell("onlyone", layer.cell)

        xpost = torch.zeros(batchsz, *shape)
        xpre = torch.zeros(batchsz, *shape)

        for k in range(100):
            inputs = torch.rand(batchsz, *shape) < 0.2
            outputs = layer(inputs.float())

            xpost = xpost * math.exp(-step_time / tc_post) + abs(lr_pre) * outputs
            xpre = xpre * math.exp(-step_time / tc_pre) + abs(lr_post) * inputs

            match mode:
                case "tensor":
                    remreward = torch.tensor(
                        [1 - 2 * (k % 2) for k in range(batchsz)]
                    ) * torch.rand(batchsz)
                    reward = remreward.clone().detach()

                case "negscalar":
                    remreward = random.uniform(0.1, 1.0) * -1.0
                    reward = torch.ones(batchsz) * remreward
                case "posscalar":
                    remreward = random.uniform(0.1, 1.0)
                    reward = torch.ones(batchsz) * remreward

            reward_pos = torch.argwhere(reward >= 0).view(-1)
            reward_neg = torch.argwhere(reward < 0).view(-1)

            updater(remreward)

            dpost = xpre * outputs * reward.unsqueeze(-1).unsqueeze(-1)
            dpre = xpost * inputs * reward.unsqueeze(-1).unsqueeze(-1)

            dpost_reg, dpost_inv = dpost.abs()[reward_pos], dpost.abs()[reward_neg]
            dpre_reg, dpre_inv = dpre.abs()[reward_pos], dpre.abs()[reward_neg]

            match kind:
                case "ltd":
                    dpos = torch.cat((dpost_inv, dpre_inv), 0)
                    dneg = torch.cat((dpost_reg, dpre_reg), 0)
                case "anti":
                    dpos = torch.cat((dpost_inv, dpre_reg), 0)
                    dneg = torch.cat((dpost_reg, dpre_inv), 0)
                case "hebb":
                    dpos = torch.cat((dpost_reg, dpre_inv), 0)
                    dneg = torch.cat((dpost_inv, dpre_reg), 0)
                case "ltp":
                    dpos = torch.cat((dpost_reg, dpre_reg), 0)
                    dneg = torch.cat((dpost_inv, dpre_inv), 0)

            assert aaeq(
                batch_reduction(dpos, 0).view(-1) if dpos.numel() else None,
                layer.updater.weight.pos,
            )
            assert aaeq(
                batch_reduction(dneg, 0).view(-1) if dneg.numel() else None,
                layer.updater.weight.neg,
            )

            del layer.updater.weight
