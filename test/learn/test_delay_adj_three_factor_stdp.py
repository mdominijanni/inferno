import pytest
import random
import torch

import inferno
from inferno.extra import ExactNeuron
from inferno.neural import DeltaCurrent, LinearDense, Serial

from inferno.learn import DelayAdjustedMSTDP, DelayAdjustedMSTDPD


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


class TestDelayAdjustedMSTDP:

    def test_default_override(self):
        shape = randshape(1, 3, 3, 5)
        batchsz = random.randint(1, 5)
        dt = random.uniform(0.7, 1.4)
        delay = random.randint(0, 2) * dt

        base_lr_pos = random.uniform(0.0, 1.0)
        base_lr_neg = random.uniform(-1.0, 1.0)
        base_tc_pos = random.uniform(15.0, 30.0)
        base_tc_neg = random.uniform(15.0, 30.0)
        base_interp_tolerance = random.uniform(1e-7, 1e-5)
        base_batch_reduction = torch.amax
        base_inplace = True

        override_lr_pos = random.uniform(0.0, 1.0)
        override_lr_neg = random.uniform(-1.0, 1.0)
        override_tc_pos = random.uniform(15.0, 30.0)
        override_tc_neg = random.uniform(15.0, 30.0)
        override_interp_tolerance = random.uniform(1e-7, 1e-5)
        override_batch_reduction = torch.amin
        override_inplace = False

        layer = mocklayer(shape, batchsz, dt, delay)
        trainer = DelayAdjustedMSTDP(
            lr_pos=base_lr_pos,
            lr_neg=base_lr_neg,
            tc_pos=base_tc_pos,
            tc_neg=base_tc_neg,
            interp_tolerance=base_interp_tolerance,
            batch_reduction=base_batch_reduction,
            inplace=base_inplace,
        )

        unit = trainer.register_cell(
            "onlyone",
            layer.cell,
            lr_pos=override_lr_pos,
            lr_neg=override_lr_neg,
            tc_pos=override_tc_pos,
            tc_neg=override_tc_neg,
            interp_tolerance=override_interp_tolerance,
            batch_reduction=override_batch_reduction,
            inplace=override_inplace,
        )

        assert override_lr_neg == unit.state.lr_neg
        assert override_lr_pos == unit.state.lr_pos
        assert override_tc_pos == unit.state.tc_pos
        assert override_tc_neg == unit.state.tc_neg
        assert override_interp_tolerance == unit.state.tolerance
        assert override_batch_reduction == unit.state.batchreduce
        assert override_inplace == unit.state.inplace

    def test_default_passthrough(self):
        shape = randshape(1, 3, 3, 5)
        batchsz = random.randint(1, 5)
        dt = random.uniform(0.7, 1.4)
        delay = random.randint(0, 2) * dt

        base_lr_neg = random.uniform(-1.0, 1.0)
        base_lr_pos = random.uniform(0.0, 1.0)
        base_tc_pos = random.uniform(15.0, 30.0)
        base_tc_neg = random.uniform(15.0, 30.0)
        base_interp_tolerance = random.uniform(1e-7, 1e-5)
        base_batch_reduction = torch.amax
        base_inplace = True

        layer = mocklayer(shape, batchsz, dt, delay)
        trainer = DelayAdjustedMSTDP(
            lr_pos=base_lr_pos,
            lr_neg=base_lr_neg,
            tc_pos=base_tc_pos,
            tc_neg=base_tc_neg,
            interp_tolerance=base_interp_tolerance,
            batch_reduction=base_batch_reduction,
            inplace=base_inplace,
        )

        unit = trainer.register_cell("onlyone", layer.cell)

        assert base_lr_neg == unit.state.lr_neg
        assert base_lr_pos == unit.state.lr_pos
        assert base_tc_pos == unit.state.tc_pos
        assert base_tc_neg == unit.state.tc_neg
        assert base_interp_tolerance == unit.state.tolerance
        assert base_batch_reduction == unit.state.batchreduce
        assert base_inplace == unit.state.inplace

    @pytest.mark.parametrize(
        "mode",
        ("hebb", "anti", "ltp", "ltd"),
    )
    @pytest.mark.parametrize(
        "sigmode",
        ("tensor", "negscalar", "posscalar"),
    )
    @pytest.mark.parametrize("inplace", (True, False), ids=("inplace", "normal"))
    @torch.no_grad
    def test_partial_update(self, mode, sigmode, inplace):
        shape = randshape(1, 3, 3, 5)
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
        interp_tolerance = random.uniform(1e-7, 1e-5)
        batch_reduction = torch.sum

        trainer = DelayAdjustedMSTDP(
            lr_neg=lr_neg,
            lr_pos=lr_pos,
            tc_pos=tc_pos,
            tc_neg=tc_neg,
            interp_tolerance=interp_tolerance,
            batch_reduction=batch_reduction,
            inplace=inplace,
        )

        _ = trainer.register_cell("onlyone", layer.cell)

        tpost = layer.connection.postsyn_receptive(
            torch.full((batchsz, *shape), float("inf"))
        )
        tpre = layer.connection.presyn_receptive(
            layer.connection.like_synaptic(torch.full((batchsz, *shape), float("inf")))
        )

        for _ in range(int(delay // dt) + 5):
            # modulation term
            match sigmode:
                case "tensor":
                    remsignal = torch.tensor(
                        [1 - 2 * (k % 2) for k in range(batchsz)]
                    ) * torch.rand(batchsz)
                    signal = remsignal.clone().detach()

                case "negscalar":
                    remsignal = random.uniform(0.1, 1.0) * -1.0
                    signal = torch.ones(batchsz) * remsignal
                case "posscalar":
                    remsignal = random.uniform(0.1, 1.0)
                    signal = torch.ones(batchsz) * remsignal
                case _:
                    assert False

            scale = random.uniform(0.5, 1.2)

            # inputs and outputs
            post, pre = (
                torch.rand(batchsz, *shape) < 0.3,
                torch.rand(batchsz, *shape) < 0.3,
            )

            # time of prior spike
            tpre = torch.where(
                layer.connection.presyn_receptive(layer.connection.like_synaptic(pre)),
                0.0,
                tpre + dt,
            )
            tpost = torch.where(
                layer.connection.postsyn_receptive(post),
                0.0,
                tpost + dt,
            )

            # adjusted time difference
            t_delta = (
                tpre.nan_to_num(posinf=float("nan"))
                - tpost.nan_to_num(posinf=float("nan"))
                - layer.connection.delay.unsqueeze(-1)
            )
            t_delta_abs = t_delta.abs()

            # partial updates
            dpost = torch.nansum(
                torch.exp(-t_delta_abs / tc_pos)
                * (abs(lr_pos) * (t_delta >= 0).to(dtype=t_delta_abs.dtype)),
                -1,
            )

            dpre = torch.nansum(
                torch.exp(-t_delta_abs / tc_neg)
                * (abs(lr_neg) * (t_delta < 0).to(dtype=t_delta_abs.dtype)),
                -1,
            )

            # test on layer
            del layer.updater.weight
            _ = layer(pre, neuron_kwargs={"override": post})
            trainer(remsignal, scale)

            if layer.updater.weight.pos is not None:
                assert torch.isnan(layer.updater.weight.pos).sum() == 0
            if layer.updater.weight.neg is not None:
                assert torch.isnan(layer.updater.weight.neg).sum() == 0

            signal_pos = torch.argwhere(signal >= 0).view(-1)
            signal_neg = torch.argwhere(signal < 0).view(-1)

            dpost = dpost.view(batchsz, -1) * signal.view(-1, 1) * scale
            dpre = dpre.view(batchsz, -1) * signal.view(-1, 1) * scale

            dpost_reg, dpost_inv = dpost.abs()[signal_pos], dpost.abs()[signal_neg]
            dpre_reg, dpre_inv = dpre.abs()[signal_pos], dpre.abs()[signal_neg]

            match mode:
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
                case _:
                    assert False

            assert aaeq(
                (
                    batch_reduction(dpos, 0).view(layer.updater.weight.pos.shape)
                    if dpos.numel()
                    else None
                ),
                layer.updater.weight.pos,
            )
            assert aaeq(
                (
                    batch_reduction(dneg, 0).view(layer.updater.weight.neg.shape)
                    if dneg.numel()
                    else None
                ),
                layer.updater.weight.neg,
            )

            del layer.updater.weight


class TestDelayAdjustedMSTDPD:

    def test_default_override(self):
        shape = randshape(1, 3, 3, 5)
        batchsz = random.randint(1, 5)
        dt = random.uniform(0.7, 1.4)
        delay = random.randint(0, 2) * dt

        base_lr_pos = random.uniform(0.0, 1.0)
        base_lr_neg = random.uniform(-1.0, 1.0)
        base_tc_pos = random.uniform(15.0, 30.0)
        base_tc_neg = random.uniform(15.0, 30.0)
        base_interp_tolerance = random.uniform(1e-7, 1e-5)
        base_batch_reduction = torch.amax
        base_inplace = True

        override_lr_pos = random.uniform(0.0, 1.0)
        override_lr_neg = random.uniform(-1.0, 1.0)
        override_tc_pos = random.uniform(15.0, 30.0)
        override_tc_neg = random.uniform(15.0, 30.0)
        override_interp_tolerance = random.uniform(1e-7, 1e-5)
        override_batch_reduction = torch.amin
        override_inplace = False

        layer = mocklayer(shape, batchsz, dt, delay)
        trainer = DelayAdjustedMSTDPD(
            lr_pos=base_lr_pos,
            lr_neg=base_lr_neg,
            tc_pos=base_tc_pos,
            tc_neg=base_tc_neg,
            interp_tolerance=base_interp_tolerance,
            batch_reduction=base_batch_reduction,
            inplace=base_inplace,
        )

        unit = trainer.register_cell(
            "onlyone",
            layer.cell,
            lr_pos=override_lr_pos,
            lr_neg=override_lr_neg,
            tc_pos=override_tc_pos,
            tc_neg=override_tc_neg,
            interp_tolerance=override_interp_tolerance,
            batch_reduction=override_batch_reduction,
            inplace=override_inplace,
        )

        assert override_lr_neg == unit.state.lr_neg
        assert override_lr_pos == unit.state.lr_pos
        assert override_tc_pos == unit.state.tc_pos
        assert override_tc_neg == unit.state.tc_neg
        assert override_interp_tolerance == unit.state.tolerance
        assert override_batch_reduction == unit.state.batchreduce
        assert override_inplace == unit.state.inplace

    def test_default_passthrough(self):
        shape = randshape(1, 3, 3, 5)
        batchsz = random.randint(1, 5)
        dt = random.uniform(0.7, 1.4)
        delay = random.randint(0, 2) * dt

        base_lr_neg = random.uniform(-1.0, 1.0)
        base_lr_pos = random.uniform(0.0, 1.0)
        base_tc_pos = random.uniform(15.0, 30.0)
        base_tc_neg = random.uniform(15.0, 30.0)
        base_interp_tolerance = random.uniform(1e-7, 1e-5)
        base_batch_reduction = torch.amax
        base_inplace = True

        layer = mocklayer(shape, batchsz, dt, delay)
        trainer = DelayAdjustedMSTDPD(
            lr_pos=base_lr_pos,
            lr_neg=base_lr_neg,
            tc_pos=base_tc_pos,
            tc_neg=base_tc_neg,
            interp_tolerance=base_interp_tolerance,
            batch_reduction=base_batch_reduction,
            inplace=base_inplace,
        )

        unit = trainer.register_cell("onlyone", layer.cell)

        assert base_lr_neg == unit.state.lr_neg
        assert base_lr_pos == unit.state.lr_pos
        assert base_tc_pos == unit.state.tc_pos
        assert base_tc_neg == unit.state.tc_neg
        assert base_interp_tolerance == unit.state.tolerance
        assert base_batch_reduction == unit.state.batchreduce
        assert base_inplace == unit.state.inplace

    @pytest.mark.parametrize(
        "mode",
        ("hebb", "anti", "ltp", "ltd"),
    )
    @pytest.mark.parametrize(
        "sigmode",
        ("tensor", "negscalar", "posscalar"),
    )
    @pytest.mark.parametrize("inplace", (True, False), ids=("inplace", "normal"))
    @torch.no_grad
    def test_partial_update(self, mode, sigmode, inplace):
        shape = randshape(1, 3, 3, 5)
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
        interp_tolerance = random.uniform(1e-7, 1e-5)
        batch_reduction = torch.sum

        trainer = DelayAdjustedMSTDPD(
            lr_neg=lr_neg,
            lr_pos=lr_pos,
            tc_pos=tc_pos,
            tc_neg=tc_neg,
            interp_tolerance=interp_tolerance,
            batch_reduction=batch_reduction,
            inplace=inplace,
        )

        _ = trainer.register_cell("onlyone", layer.cell)

        tpost = layer.connection.postsyn_receptive(
            torch.full((batchsz, *shape), float("inf"))
        )
        tpre = layer.connection.presyn_receptive(
            layer.connection.like_synaptic(torch.full((batchsz, *shape), float("inf")))
        )

        for _ in range(int(delay // dt) + 5):
            # modulation term
            match sigmode:
                case "tensor":
                    remsignal = torch.tensor(
                        [1 - 2 * (k % 2) for k in range(batchsz)]
                    ) * torch.rand(batchsz)
                    signal = remsignal.clone().detach()

                case "negscalar":
                    remsignal = random.uniform(0.1, 1.0) * -1.0
                    signal = torch.ones(batchsz) * remsignal
                case "posscalar":
                    remsignal = random.uniform(0.1, 1.0)
                    signal = torch.ones(batchsz) * remsignal
                case _:
                    assert False

            scale = random.uniform(0.5, 1.2)

            # inputs and outputs
            post, pre = (
                torch.rand(batchsz, *shape) < 0.3,
                torch.rand(batchsz, *shape) < 0.3,
            )

            # time of prior spike
            tpre = torch.where(
                layer.connection.presyn_receptive(layer.connection.like_synaptic(pre)),
                0.0,
                tpre + dt,
            )
            tpost = torch.where(
                layer.connection.postsyn_receptive(post),
                0.0,
                tpost + dt,
            )

            # adjusted time difference
            t_delta = (
                tpre.nan_to_num(posinf=float("nan"))
                - tpost.nan_to_num(posinf=float("nan"))
                - layer.connection.delay.unsqueeze(-1)
            )
            t_delta_abs = t_delta.abs()

            # partial updates
            dpost = torch.nansum(
                torch.exp(-t_delta_abs / tc_neg)
                * (abs(lr_neg) * (t_delta >= 0).to(dtype=t_delta_abs.dtype)),
                -1,
            )

            dpre = torch.nansum(
                torch.exp(-t_delta_abs / tc_pos)
                * (abs(lr_pos) * (t_delta < 0).to(dtype=t_delta_abs.dtype)),
                -1,
            )

            # test on layer
            del layer.updater.delay
            _ = layer(pre, neuron_kwargs={"override": post})
            trainer(remsignal, scale)

            if layer.updater.delay.pos is not None:
                assert torch.isnan(layer.updater.delay.pos).sum() == 0
            if layer.updater.delay.neg is not None:
                assert torch.isnan(layer.updater.delay.neg).sum() == 0

            signal_pos = torch.argwhere(signal >= 0).view(-1)
            signal_neg = torch.argwhere(signal < 0).view(-1)

            dpost = dpost.view(batchsz, -1) * signal.view(-1, 1) * scale
            dpre = dpre.view(batchsz, -1) * signal.view(-1, 1) * scale

            dpost_reg, dpost_inv = dpost.abs()[signal_pos], dpost.abs()[signal_neg]
            dpre_reg, dpre_inv = dpre.abs()[signal_pos], dpre.abs()[signal_neg]

            match mode:
                case "ltd":
                    dpos = torch.cat((dpost_inv, dpre_inv), 0)
                    dneg = torch.cat((dpost_reg, dpre_reg), 0)
                case "anti":
                    dpos = torch.cat((dpost_reg, dpre_inv), 0)
                    dneg = torch.cat((dpost_inv, dpre_reg), 0)
                case "hebb":
                    dpos = torch.cat((dpost_inv, dpre_reg), 0)
                    dneg = torch.cat((dpost_reg, dpre_inv), 0)
                case "ltp":
                    dpos = torch.cat((dpost_reg, dpre_reg), 0)
                    dneg = torch.cat((dpost_inv, dpre_inv), 0)
                case _:
                    assert False

            assert aaeq(
                (
                    batch_reduction(dpos, 0).view(layer.updater.delay.pos.shape)
                    if dpos.numel()
                    else None
                ),
                layer.updater.delay.pos,
            )
            assert aaeq(
                (
                    batch_reduction(dneg, 0).view(layer.updater.delay.neg.shape)
                    if dneg.numel()
                    else None
                ),
                layer.updater.delay.neg,
            )

            del layer.updater.delay
