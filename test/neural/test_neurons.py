from functools import reduce
import pytest
import random
import torch

from inferno.neural import Neuron, LIF, ALIF, GLIF1, GLIF2, QIF, Izhikevich, EIF, AdEx


def repeat(fn, times):
    if times == 1:
        return fn()
    else:
        return tuple(fn() for _ in range(times))


def validate_batchsz(neuron: Neuron, batchsz: int) -> None:
    assert neuron.batchsz == batchsz


def validate_shape(neuron: Neuron, shape: tuple[int, ...]) -> None:
    assert neuron.shape == shape


def validate_count(neuron: Neuron, shape: tuple[int, ...]) -> None:
    assert neuron.count == reduce(lambda a, b: a * b, shape)


def validate_batchedshape(neuron: Neuron, batchsz: int, shape: tuple[int, ...]) -> None:
    assert neuron.batchedshape == (batchsz,) + shape


def validate_spikes_voltagedriven(neuron: Neuron, set_voltage: float) -> None:
    mask = torch.rand(neuron.batchedshape) > 0.5

    neuron.voltage[mask] = set_voltage
    res = neuron(torch.zeros(neuron.batchedshape))

    assert torch.all(neuron.spike == mask)
    assert torch.all(res == mask)


def validate_refrac_voltagedriven(
    neuron: Neuron, set_voltage: float, dt: float, absrefrac: float
) -> None:
    mask = torch.rand(neuron.batchedshape) > 0.5

    neuron.voltage[mask] = set_voltage
    _ = neuron(torch.zeros(neuron.batchedshape))

    assert torch.all(neuron.refrac[mask] == (mask * absrefrac)[mask])

    _ = neuron(torch.zeros(neuron.batchedshape))

    assert torch.all(neuron.refrac[mask] == (mask * (absrefrac - dt))[mask])


def validate_refrac_voltagedriven_eps(
    neuron: Neuron, set_voltage: float, dt: float, absrefrac: float, eps: float
) -> None:
    mask = torch.rand(neuron.batchedshape) > 0.5

    neuron.voltage[mask] = set_voltage
    _ = neuron(torch.zeros(neuron.batchedshape))

    assert torch.all(neuron.refrac[mask] == (mask * absrefrac)[mask])

    _ = neuron(torch.zeros(neuron.batchedshape))
    assert torch.all(
        (neuron.refrac[mask] - (mask * (absrefrac - dt))[mask]).abs() < eps
    )


def validate_voltage_lock_voltagedriven(
    neuron: Neuron,
    set_supra_voltage: float,
    set_sub_voltage: float,
    absrefrac: float,
) -> None:
    mask = torch.rand(neuron.batchedshape) > 0.5

    neuron.voltage[mask] = set_supra_voltage
    neuron.voltage[~mask] = set_sub_voltage
    _ = neuron(torch.zeros(neuron.batchedshape))

    neuron.refrac[mask] = absrefrac
    voltages = neuron.voltage.clone().detach()
    _ = neuron(torch.zeros(neuron.batchedshape))

    assert torch.all(neuron.voltage[mask] == voltages[mask])
    assert torch.all(neuron.voltage[~mask] < voltages[~mask])


def validate_voltage_lock_voltagedriven_neq(
    neuron: Neuron,
    set_supra_voltage: float,
    set_sub_voltage: float,
    absrefrac: float,
) -> None:
    mask = torch.rand(neuron.batchedshape) > 0.5

    neuron.voltage[mask] = set_supra_voltage
    neuron.voltage[~mask] = set_sub_voltage
    _ = neuron(torch.zeros(neuron.batchedshape))

    neuron.refrac[mask] = absrefrac
    voltages = neuron.voltage.clone().detach()
    _ = neuron(torch.zeros(neuron.batchedshape))

    assert torch.all(neuron.voltage[mask] == voltages[mask])
    assert torch.all(neuron.voltage[~mask] != voltages[~mask])


class TestLIF:

    @staticmethod
    def random_shape(mindims=1, maxdims=9, minsize=1, maxsize=9):
        return tuple(
            random.randint(mindims, maxdims)
            for _ in range(random.randint(minsize, maxsize))
        )

    @staticmethod
    def random_hyper():
        hyper = {}
        hyper["step_time"] = random.uniform(0.6, 1.8)
        hyper["rest_v"] = random.uniform(-70, -60)
        hyper["reset_v"] = random.uniform(-75, hyper["rest_v"])
        hyper["thresh_v"] = random.uniform(hyper["rest_v"] + 5, -50)
        hyper["refrac_t"] = random.uniform(2, 4) * hyper["step_time"]
        hyper["time_constant"] = random.uniform(5, 20)
        hyper["resistance"] = random.uniform(1, 1.5)
        hyper["batch_size"] = random.randint(1, 9)
        return hyper

    def test_batchsz(self):
        shape = self.random_shape()
        hyper = self.random_hyper()
        neuron = LIF(shape, **hyper)

        validate_batchsz(neuron, hyper["batch_size"])

    def test_shape(self):
        shape = self.random_shape()
        hyper = self.random_hyper()
        neuron = LIF(shape, **hyper)

        validate_shape(neuron, shape)

    def test_count(self):
        shape = self.random_shape()
        hyper = self.random_hyper()
        neuron = LIF(shape, **hyper)

        validate_count(neuron, shape)

    def test_batchedshape(self):
        shape = self.random_shape()
        hyper = self.random_hyper()
        neuron = LIF(shape, **hyper)

        validate_batchedshape(neuron, hyper["batch_size"], shape)

    def test_spikes(self):
        shape = self.random_shape()
        hyper = self.random_hyper()
        neuron = LIF(shape, **hyper)

        validate_spikes_voltagedriven(neuron, hyper["thresh_v"] + 5)

    def test_refrac(self):
        shape = self.random_shape()
        hyper = self.random_hyper()
        neuron = LIF(shape, **hyper)

        validate_refrac_voltagedriven(
            neuron, hyper["thresh_v"] + 5, hyper["step_time"], hyper["refrac_t"]
        )

    def test_voltage_lock(self):
        shape = self.random_shape()
        hyper = self.random_hyper()
        neuron = LIF(shape, **hyper)

        validate_voltage_lock_voltagedriven(
            neuron,
            hyper["thresh_v"] + 5,
            hyper["rest_v"] + 3.0,
            hyper["refrac_t"],
        )


class TestALIF:

    @staticmethod
    def random_shape(mindims=1, maxdims=9, minsize=1, maxsize=9):
        return tuple(
            random.randint(mindims, maxdims)
            for _ in range(random.randint(minsize, maxsize))
        )

    @staticmethod
    def random_hyper(nadapts=1):
        hyper = {}
        hyper["step_time"] = random.uniform(0.6, 1.8)
        hyper["rest_v"] = random.uniform(-70, -60)
        hyper["reset_v"] = random.uniform(-75, hyper["rest_v"])
        hyper["thresh_eq_v"] = random.uniform(hyper["rest_v"] + 5, -50)
        hyper["refrac_t"] = random.uniform(2, 4) * hyper["step_time"]
        hyper["tc_membrane"] = random.uniform(5, 20)
        hyper["tc_adaptation"] = repeat(lambda: random.uniform(1e3, 1e4), nadapts)
        hyper["spike_increment"] = repeat(lambda: random.uniform(0.1, 1.1), nadapts)
        hyper["resistance"] = random.uniform(1, 1.5)
        hyper["batch_size"] = random.randint(1, 9)
        hyper["batch_reduction"] = torch.mean
        return hyper

    def test_batchsz(self):
        shape = self.random_shape()
        hyper = self.random_hyper()
        neuron = ALIF(shape, **hyper)

        validate_batchsz(neuron, hyper["batch_size"])

    def test_shape(self):
        shape = self.random_shape()
        hyper = self.random_hyper()
        neuron = ALIF(shape, **hyper)

        validate_shape(neuron, shape)

    def test_count(self):
        shape = self.random_shape()
        hyper = self.random_hyper()
        neuron = ALIF(shape, **hyper)

        validate_count(neuron, shape)

    def test_batchedshape(self):
        shape = self.random_shape()
        hyper = self.random_hyper()
        neuron = ALIF(shape, **hyper)

        validate_batchedshape(neuron, hyper["batch_size"], shape)

    @pytest.mark.parametrize(
        "nadapts",
        (1, 10),
        ids=("nadapts=1", "nadapts=10"),
    )
    def test_spikes(self, nadapts):
        shape = self.random_shape()
        hyper = self.random_hyper(nadapts)
        neuron = ALIF(shape, **hyper)

        validate_spikes_voltagedriven(neuron, hyper["thresh_eq_v"] + 5)

    @pytest.mark.parametrize(
        "nadapts",
        (1, 10),
        ids=("nadapts=1", "nadapts=10"),
    )
    def test_refrac(self, nadapts):
        shape = self.random_shape()
        hyper = self.random_hyper(nadapts)
        neuron = ALIF(shape, **hyper)

        validate_refrac_voltagedriven_eps(
            neuron,
            hyper["thresh_eq_v"] + 5,
            hyper["step_time"],
            hyper["refrac_t"],
            5e-7,
        )

    @pytest.mark.parametrize(
        "nadapts",
        (1, 10),
        ids=("nadapts=1", "nadapts=10"),
    )
    def test_voltage_lock(self, nadapts):
        shape = self.random_shape()
        hyper = self.random_hyper(nadapts)
        neuron = ALIF(shape, **hyper)

        validate_voltage_lock_voltagedriven(
            neuron,
            hyper["thresh_eq_v"] + 5,
            hyper["rest_v"] + 3.0,
            hyper["refrac_t"],
        )


class TestGLIF1:

    @staticmethod
    def random_shape(mindims=1, maxdims=9, minsize=1, maxsize=9):
        return tuple(
            random.randint(mindims, maxdims)
            for _ in range(random.randint(minsize, maxsize))
        )

    @staticmethod
    def random_hyper():
        hyper = {}
        hyper["step_time"] = random.uniform(0.6, 1.8)
        hyper["rest_v"] = random.uniform(-70, -60)
        hyper["reset_v"] = random.uniform(-75, hyper["rest_v"])
        hyper["thresh_v"] = random.uniform(hyper["rest_v"] + 5, -50)
        hyper["refrac_t"] = random.uniform(2, 4) * hyper["step_time"]
        hyper["time_constant"] = random.uniform(5, 20)
        hyper["resistance"] = random.uniform(1, 1.5)
        hyper["batch_size"] = random.randint(1, 9)
        return hyper

    def test_batchsz(self):
        shape = self.random_shape()
        hyper = self.random_hyper()
        neuron = GLIF1(shape, **hyper)

        validate_batchsz(neuron, hyper["batch_size"])

    def test_shape(self):
        shape = self.random_shape()
        hyper = self.random_hyper()
        neuron = GLIF1(shape, **hyper)

        validate_shape(neuron, shape)

    def test_count(self):
        shape = self.random_shape()
        hyper = self.random_hyper()
        neuron = GLIF1(shape, **hyper)

        validate_count(neuron, shape)

    def test_batchedshape(self):
        shape = self.random_shape()
        hyper = self.random_hyper()
        neuron = GLIF1(shape, **hyper)

        validate_batchedshape(neuron, hyper["batch_size"], shape)

    def test_spikes(self):
        shape = self.random_shape()
        hyper = self.random_hyper()
        neuron = GLIF1(shape, **hyper)

        validate_spikes_voltagedriven(neuron, hyper["thresh_v"] + 5)

    def test_refrac(self):
        shape = self.random_shape()
        hyper = self.random_hyper()
        neuron = GLIF1(shape, **hyper)

        validate_refrac_voltagedriven(
            neuron, hyper["thresh_v"] + 5, hyper["step_time"], hyper["refrac_t"]
        )

    def test_voltage_lock(self):
        shape = self.random_shape()
        hyper = self.random_hyper()
        neuron = GLIF1(shape, **hyper)

        validate_voltage_lock_voltagedriven(
            neuron,
            hyper["thresh_v"] + 5,
            hyper["rest_v"] + 3.0,
            hyper["refrac_t"],
        )


class TestGLIF2:

    @staticmethod
    def random_shape(mindims=1, maxdims=9, minsize=1, maxsize=9):
        return tuple(
            random.randint(mindims, maxdims)
            for _ in range(random.randint(minsize, maxsize))
        )

    @staticmethod
    def random_hyper(nadapts=1):
        hyper = {}
        hyper["step_time"] = random.uniform(0.6, 1.8)
        hyper["rest_v"] = random.uniform(-70, -60)
        hyper["reset_v_add"] = random.uniform(5, 10)
        hyper["reset_v_mul"] = random.uniform(-0.15, -0.05)
        hyper["thresh_eq_v"] = random.uniform(hyper["rest_v"] + 5, -50)
        hyper["refrac_t"] = random.uniform(2, 4) * hyper["step_time"]
        hyper["tc_membrane"] = random.uniform(5, 20)
        hyper["rc_adaptation"] = repeat(
            lambda: random.uniform(1 / 1e4, 1 / 1e3), nadapts
        )
        hyper["spike_increment"] = repeat(lambda: random.uniform(0.1, 1.1), nadapts)
        hyper["resistance"] = random.uniform(1, 1.5)
        hyper["batch_size"] = random.randint(1, 9)
        hyper["batch_reduction"] = torch.mean
        return hyper

    def test_batchsz(self):
        shape = self.random_shape()
        hyper = self.random_hyper()
        neuron = GLIF2(shape, **hyper)

        validate_batchsz(neuron, hyper["batch_size"])

    def test_shape(self):
        shape = self.random_shape()
        hyper = self.random_hyper()
        neuron = GLIF2(shape, **hyper)

        validate_shape(neuron, shape)

    def test_count(self):
        shape = self.random_shape()
        hyper = self.random_hyper()
        neuron = GLIF2(shape, **hyper)

        validate_count(neuron, shape)

    def test_batchedshape(self):
        shape = self.random_shape()
        hyper = self.random_hyper()
        neuron = GLIF2(shape, **hyper)

        validate_batchedshape(neuron, hyper["batch_size"], shape)

    @pytest.mark.parametrize(
        "nadapts",
        (1, 10),
        ids=("nadapts=1", "nadapts=10"),
    )
    def test_spikes(self, nadapts):
        shape = self.random_shape()
        hyper = self.random_hyper(nadapts)
        neuron = GLIF2(shape, **hyper)

        validate_spikes_voltagedriven(neuron, hyper["thresh_eq_v"] + 5)

    @pytest.mark.parametrize(
        "nadapts",
        (1, 10),
        ids=("nadapts=1", "nadapts=10"),
    )
    def test_refrac(self, nadapts):
        shape = self.random_shape()
        hyper = self.random_hyper(nadapts)
        neuron = GLIF2(shape, **hyper)

        validate_refrac_voltagedriven_eps(
            neuron,
            hyper["thresh_eq_v"] + 5,
            hyper["step_time"],
            hyper["refrac_t"],
            5e-7,
        )

    @pytest.mark.parametrize(
        "nadapts",
        (1, 10),
        ids=("nadapts=1", "nadapts=10"),
    )
    def test_voltage_lock(self, nadapts):
        shape = self.random_shape()
        hyper = self.random_hyper(nadapts)
        neuron = GLIF2(shape, **hyper)

        validate_voltage_lock_voltagedriven(
            neuron,
            hyper["thresh_eq_v"] + 5,
            hyper["rest_v"] + 3.0,
            hyper["refrac_t"],
        )


class TestQIF:

    @staticmethod
    def random_shape(mindims=1, maxdims=9, minsize=1, maxsize=9):
        return tuple(
            random.randint(mindims, maxdims)
            for _ in range(random.randint(minsize, maxsize))
        )

    @staticmethod
    def random_hyper():
        hyper = {}
        hyper["step_time"] = random.uniform(0.6, 1.8)
        hyper["rest_v"] = random.uniform(-70, -65)
        hyper["crit_v"] = random.uniform(-55, -50)
        hyper["affinity"] = random.uniform(0.75, 1.25)
        hyper["reset_v"] = random.uniform(-75, hyper["rest_v"])
        hyper["thresh_v"] = random.uniform(2, 5) + hyper["crit_v"]
        hyper["refrac_t"] = random.uniform(2, 4) * hyper["step_time"]
        hyper["time_constant"] = random.uniform(5, 20)
        hyper["resistance"] = random.uniform(1, 1.5)
        hyper["batch_size"] = random.randint(1, 9)
        return hyper

    def test_batchsz(self):
        shape = self.random_shape()
        hyper = self.random_hyper()
        neuron = QIF(shape, **hyper)

        validate_batchsz(neuron, hyper["batch_size"])

    def test_shape(self):
        shape = self.random_shape()
        hyper = self.random_hyper()
        neuron = QIF(shape, **hyper)

        validate_shape(neuron, shape)

    def test_count(self):
        shape = self.random_shape()
        hyper = self.random_hyper()
        neuron = QIF(shape, **hyper)

        validate_count(neuron, shape)

    def test_batchedshape(self):
        shape = self.random_shape()
        hyper = self.random_hyper()
        neuron = QIF(shape, **hyper)

        validate_batchedshape(neuron, hyper["batch_size"], shape)

    def test_spikes(self):
        shape = self.random_shape()
        hyper = self.random_hyper()
        neuron = QIF(shape, **hyper)

        validate_spikes_voltagedriven(neuron, hyper["thresh_v"] + 5)

    def test_refrac(self):
        shape = self.random_shape()
        hyper = self.random_hyper()
        neuron = QIF(shape, **hyper)

        validate_refrac_voltagedriven(
            neuron, hyper["thresh_v"] + 5, hyper["step_time"], hyper["refrac_t"]
        )

    def test_voltage_lock(self):
        shape = self.random_shape()
        hyper = self.random_hyper()
        neuron = QIF(shape, **hyper)

        validate_voltage_lock_voltagedriven_neq(
            neuron,
            hyper["thresh_v"] + 5,
            hyper["rest_v"] + 3.0,
            hyper["refrac_t"],
        )


class TestIzhikevich:

    @staticmethod
    def random_shape(mindims=1, maxdims=9, minsize=1, maxsize=9):
        return tuple(
            random.randint(mindims, maxdims)
            for _ in range(random.randint(minsize, maxsize))
        )

    @staticmethod
    def random_hyper(nadapts=1):
        hyper = {}
        hyper["step_time"] = random.uniform(0.6, 1.8)
        hyper["rest_v"] = random.uniform(-70, -65)
        hyper["crit_v"] = random.uniform(-55, -50)
        hyper["affinity"] = random.uniform(0.75, 1.25)
        hyper["reset_v"] = random.uniform(-75, hyper["rest_v"])
        hyper["thresh_v"] = random.uniform(2, 5) + hyper["crit_v"]
        hyper["refrac_t"] = random.uniform(2, 4) * hyper["step_time"]
        hyper["tc_membrane"] = random.uniform(5, 20)
        hyper["tc_adaptation"] = repeat(lambda: random.uniform(5, 20), nadapts)
        hyper["voltage_coupling"] = repeat(lambda: random.uniform(0.1, 0.2), nadapts)
        hyper["spike_increment"] = repeat(lambda: random.uniform(0.1, 1.1), nadapts)
        hyper["resistance"] = random.uniform(1, 1.5)
        hyper["batch_size"] = random.randint(1, 9)
        return hyper

    def test_batchsz(self):
        shape = self.random_shape()
        hyper = self.random_hyper()
        neuron = Izhikevich(shape, **hyper)

        validate_batchsz(neuron, hyper["batch_size"])

    def test_shape(self):
        shape = self.random_shape()
        hyper = self.random_hyper()
        neuron = Izhikevich(shape, **hyper)

        validate_shape(neuron, shape)

    def test_count(self):
        shape = self.random_shape()
        hyper = self.random_hyper()
        neuron = Izhikevich(shape, **hyper)

        validate_count(neuron, shape)

    def test_batchedshape(self):
        shape = self.random_shape()
        hyper = self.random_hyper()
        neuron = Izhikevich(shape, **hyper)

        validate_batchedshape(neuron, hyper["batch_size"], shape)

    @pytest.mark.parametrize(
        "nadapts",
        (1, 10),
        ids=("nadapts=1", "nadapts=10"),
    )
    def test_spikes(self, nadapts):
        shape = self.random_shape()
        hyper = self.random_hyper()
        neuron = Izhikevich(shape, **hyper)

        validate_spikes_voltagedriven(neuron, hyper["thresh_v"] + 5)

    @pytest.mark.parametrize(
        "nadapts",
        (1, 10),
        ids=("nadapts=1", "nadapts=10"),
    )
    def test_refrac(self, nadapts):
        shape = self.random_shape()
        hyper = self.random_hyper()
        neuron = Izhikevich(shape, **hyper)

        validate_refrac_voltagedriven_eps(
            neuron, hyper["thresh_v"] + 5, hyper["step_time"], hyper["refrac_t"], 5e-7
        )

    @pytest.mark.parametrize(
        "nadapts",
        (1, 10),
        ids=("nadapts=1", "nadapts=10"),
    )
    def test_voltage_lock(self, nadapts):
        shape = self.random_shape()
        hyper = self.random_hyper()
        neuron = Izhikevich(shape, **hyper)

        validate_voltage_lock_voltagedriven_neq(
            neuron,
            hyper["thresh_v"] + 5,
            hyper["rest_v"] + 3.0,
            hyper["refrac_t"],
        )


class TestEIF:

    @staticmethod
    def random_shape(mindims=1, maxdims=5, minsize=1, maxsize=9):
        return tuple(
            random.randint(mindims, maxdims)
            for _ in range(random.randint(minsize, maxsize))
        )

    @staticmethod
    def random_hyper():
        hyper = {}
        hyper["step_time"] = random.uniform(0.6, 1.8)
        hyper["rest_v"] = random.uniform(-70, -65)
        hyper["rheobase_v"] = random.uniform(-55, -50)
        hyper["sharpness"] = random.uniform(0.75, 1.25)
        hyper["reset_v"] = random.uniform(-75, hyper["rest_v"])
        hyper["thresh_v"] = random.uniform(5, 10) + hyper["rheobase_v"]
        hyper["refrac_t"] = random.uniform(2, 4) * hyper["step_time"]
        hyper["time_constant"] = random.uniform(5, 20)
        hyper["resistance"] = random.uniform(1, 1.5)
        hyper["batch_size"] = random.randint(1, 9)
        return hyper

    def test_batchsz(self):
        shape = self.random_shape()
        hyper = self.random_hyper()
        neuron = EIF(shape, **hyper)

        validate_batchsz(neuron, hyper["batch_size"])

    def test_shape(self):
        shape = self.random_shape()
        hyper = self.random_hyper()
        neuron = EIF(shape, **hyper)

        validate_shape(neuron, shape)

    def test_count(self):
        shape = self.random_shape()
        hyper = self.random_hyper()
        neuron = EIF(shape, **hyper)

        validate_count(neuron, shape)

    def test_batchedshape(self):
        shape = self.random_shape()
        hyper = self.random_hyper()
        neuron = EIF(shape, **hyper)

        validate_batchedshape(neuron, hyper["batch_size"], shape)

    def test_spikes(self):
        shape = self.random_shape()
        hyper = self.random_hyper()
        neuron = EIF(shape, **hyper)

        validate_spikes_voltagedriven(neuron, hyper["thresh_v"] + 5)

    def test_refrac(self):
        shape = self.random_shape()
        hyper = self.random_hyper()
        neuron = EIF(shape, **hyper)

        validate_refrac_voltagedriven(
            neuron, hyper["thresh_v"] + 5, hyper["step_time"], hyper["refrac_t"]
        )

    def test_voltage_lock(self):
        shape = self.random_shape()
        hyper = self.random_hyper()
        neuron = EIF(shape, **hyper)

        validate_voltage_lock_voltagedriven(
            neuron,
            hyper["thresh_v"] + 5,
            hyper["rest_v"] + 3.0,
            hyper["refrac_t"],
        )


class TestAdEx:

    @staticmethod
    def random_shape(mindims=1, maxdims=5, minsize=1, maxsize=9):
        return tuple(
            random.randint(mindims, maxdims)
            for _ in range(random.randint(minsize, maxsize))
        )

    @staticmethod
    def random_hyper(nadapts=1):
        hyper = {}
        hyper["step_time"] = random.uniform(0.6, 1.8)
        hyper["rest_v"] = random.uniform(-70, -65)
        hyper["rheobase_v"] = random.uniform(-55, -50)
        hyper["sharpness"] = random.uniform(0.75, 1.25)
        hyper["reset_v"] = random.uniform(-75, hyper["rest_v"])
        hyper["thresh_v"] = random.uniform(5, 10) + hyper["rheobase_v"]
        hyper["refrac_t"] = random.uniform(2, 4) * hyper["step_time"]
        hyper["tc_membrane"] = random.uniform(5, 20)
        hyper["tc_adaptation"] = repeat(lambda: random.uniform(5, 20), nadapts)
        hyper["voltage_coupling"] = repeat(lambda: random.uniform(0.1, 0.2), nadapts)
        hyper["spike_increment"] = repeat(lambda: random.uniform(0.1, 1.1), nadapts)
        hyper["resistance"] = random.uniform(1, 1.5)
        hyper["batch_size"] = random.randint(1, 9)
        return hyper

    def test_batchsz(self):
        shape = self.random_shape()
        hyper = self.random_hyper()
        neuron = AdEx(shape, **hyper)

        validate_batchsz(neuron, hyper["batch_size"])

    def test_shape(self):
        shape = self.random_shape()
        hyper = self.random_hyper()
        neuron = AdEx(shape, **hyper)

        validate_shape(neuron, shape)

    def test_count(self):
        shape = self.random_shape()
        hyper = self.random_hyper()
        neuron = AdEx(shape, **hyper)

        validate_count(neuron, shape)

    def test_batchedshape(self):
        shape = self.random_shape()
        hyper = self.random_hyper()
        neuron = AdEx(shape, **hyper)

        validate_batchedshape(neuron, hyper["batch_size"], shape)

    @pytest.mark.parametrize(
        "nadapts",
        (1, 10),
        ids=("nadapts=1", "nadapts=10"),
    )
    def test_spikes(self, nadapts):
        shape = self.random_shape()
        hyper = self.random_hyper(nadapts)
        neuron = AdEx(shape, **hyper)

        validate_spikes_voltagedriven(neuron, hyper["thresh_v"] + 5)

    @pytest.mark.parametrize(
        "nadapts",
        (1, 10),
        ids=("nadapts=1", "nadapts=10"),
    )
    def test_refrac(self, nadapts):
        shape = self.random_shape()
        hyper = self.random_hyper(nadapts)
        neuron = AdEx(shape, **hyper)

        validate_refrac_voltagedriven_eps(
            neuron, hyper["thresh_v"] + 5, hyper["step_time"], hyper["refrac_t"], 5e-7
        )

    @pytest.mark.parametrize(
        "nadapts",
        (1, 10),
        ids=("nadapts=1", "nadapts=10"),
    )
    def test_voltage_lock(self, nadapts):
        shape = self.random_shape()
        hyper = self.random_hyper(nadapts)
        neuron = AdEx(shape, **hyper)

        validate_voltage_lock_voltagedriven(
            neuron,
            hyper["thresh_v"] + 5,
            hyper["rest_v"] + 3.0,
            hyper["refrac_t"],
        )
