import math
import random
import torch

from inferno.learn import MSTDPET
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
        base_batch_reduction = torch.amax
        base_field_reduction = torch.mean

        override_step_time = random.uniform(0.7, 1.4)
        override_lr_post = random.uniform(-1.0, 1.0)
        override_lr_pre = random.uniform(-1.0, 1.0)
        override_tc_post = random.uniform(15.0, 30.0)
        override_tc_pre = random.uniform(15.0, 30.0)
        override_tc_eligibility = random.uniform(15.0, 30.0)
        override_interp_tolerance = random.uniform(1e-7, 1e-5)
        override_batch_reduction = torch.amin
        override_field_reduction = torch.sum

        updater = MSTDPET(
            step_time=base_step_time,
            lr_post=base_lr_post,
            lr_pre=base_lr_pre,
            tc_post=base_tc_post,
            tc_pre=base_tc_pre,
            tc_eligibility=base_tc_eligibility,
            interp_tolerance=base_interp_tolerance,
            batch_reduction=base_batch_reduction,
            field_reduction=base_field_reduction,
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
            batch_reduction=override_batch_reduction,
            field_reduction=override_field_reduction,
        )

        assert override_step_time == unit.state.step_time
        assert override_lr_post == unit.state.lr_post
        assert override_lr_pre == unit.state.lr_pre
        assert override_tc_post == unit.state.tc_post
        assert override_tc_pre == unit.state.tc_pre
        assert override_tc_eligibility == unit.state.tc_eligibility
        assert override_interp_tolerance == unit.state.tolerance
        assert override_batch_reduction == unit.state.batchreduce
        assert override_field_reduction == unit.state.fieldreduce

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
        base_batch_reduction = torch.amax
        base_field_reduction = torch.mean

        updater = MSTDPET(
            step_time=base_step_time,
            lr_post=base_lr_post,
            lr_pre=base_lr_pre,
            tc_post=base_tc_post,
            tc_pre=base_tc_pre,
            tc_eligibility=base_tc_eligibility,
            interp_tolerance=base_interp_tolerance,
            batch_reduction=base_batch_reduction,
            field_reduction=base_field_reduction,
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
        assert base_field_reduction == unit.state.fieldreduce
