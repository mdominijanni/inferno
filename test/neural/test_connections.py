import math
import pytest
import random
import torch
import torch.nn as nn

from inferno.neural import DeltaCurrent
from inferno.neural import LinearDense, LinearDirect, LinearLateral, Conv2D


def randshape(mindims=1, maxdims=9, minsize=1, maxsize=9):
    return tuple(
        random.randint(mindims, maxdims)
        for _ in range(random.randint(minsize, maxsize))
    )


def aaeq(t0, t1, eps=1e-6) -> bool:
    if torch.all(t0 == t1):
        return True
    else:
        return torch.all((t0 - t1).abs() < eps)


class TestLinearDense:

    @staticmethod
    def makeconn(
        in_shape: tuple[int, ...] | int,
        out_shape: tuple[int, ...] | int,
        step_time: float,
        batch_size=1,
        bias=False,
        delay=None,
        *,
        weight_init=None,
        bias_init=None,
        delay_init=None,
    ) -> LinearDense:
        return LinearDense(
            in_shape,
            out_shape,
            step_time,
            synapse=DeltaCurrent.partialconstructor(1.0, "previous", 1e-7),
            bias=bias,
            delay=delay,
            batch_size=batch_size,
            weight_init=weight_init,
            bias_init=bias_init,
            delay_init=delay_init,
        )

    @staticmethod
    def load_synapse(conn: LinearDense, p=0.1) -> None:
        conn.synapse.spike_.value = torch.rand(conn.synapse.spike_.value.shape) < p

    @pytest.mark.parametrize("biased", (True, False), ids=("biased", "unbiased"))
    def test_forward_undelayed(self, biased):
        inshape = randshape(1, 3, 2, 5)
        outshape = randshape(1, 3, 2, 5)
        batchsz = random.randint(1, 9)
        conn = self.makeconn(
            inshape,
            outshape,
            1.0,
            batchsz,
            bias=biased,
            weight_init=(lambda x: torch.rand_like(x)),
        )

        inputs = torch.rand(batchsz, *inshape) < 0.5
        outputs = conn(inputs)

        if biased:
            res = (
                torch.matmul(inputs.float().view(batchsz, -1), conn.weight.t())
                + conn.bias.view(1, -1)
            ).view(batchsz, *outshape)
        else:
            res = torch.matmul(inputs.float().view(batchsz, -1), conn.weight.t()).view(
                batchsz, *outshape
            )

        assert tuple(outputs.shape) == (batchsz, *outshape)
        assert aaeq(outputs, res, 2e-6)

    @pytest.mark.parametrize("biased", (True, False), ids=("biased", "unbiased"))
    def test_forward_delayed(self, biased):
        inshape = randshape(1, 3, 2, 5)
        outshape = randshape(1, 3, 2, 5)
        batchsz = random.randint(1, 9)
        delay = 4
        conn = self.makeconn(
            inshape,
            outshape,
            1.0,
            batchsz,
            bias=biased,
            delay=float(delay),
            weight_init=(lambda x: torch.rand_like(x)),
            delay_init=(lambda x, d=delay: torch.ones_like(x) * d),
        )

        self.load_synapse(conn, p=0.3)
        conn.synapse.spike_.align(0)
        inputs = conn.synapse.spike_.value.clone().detach()

        for k in range(1, delay + 1):
            outputs = conn(torch.zeros(batchsz, *inshape))

            if biased:
                res = (
                    torch.matmul(
                        inputs[k, ...].view(batchsz, -1).float(), conn.weight.t()
                    )
                    + conn.bias.view(1, -1)
                ).view(batchsz, *outshape)
            else:
                res = torch.matmul(
                    inputs[k, ...].view(batchsz, -1).float(), conn.weight.t()
                ).view(batchsz, *outshape)

            assert aaeq(outputs, res, 1e-6)

    def test_like_input(self):
        inshape = randshape(1, 3, 2, 5)
        outshape = randshape(1, 3, 2, 5)
        batchsz = random.randint(1, 9)
        conn = self.makeconn(
            inshape,
            outshape,
            1.0,
            batchsz,
            weight_init=(lambda x: torch.rand_like(x)),
        )

        assert tuple(conn.like_input(conn.synapse.spike).shape) == (batchsz, *inshape)

    def test_like_bias(self):
        inshape = randshape(1, 3, 2, 5)
        outshape = randshape(1, 3, 2, 5)
        batchsz = random.randint(1, 9)
        conn = self.makeconn(
            inshape,
            outshape,
            1.0,
            batchsz,
            bias=True,
            weight_init=(lambda x: torch.rand_like(x)),
        )

        assert (
            conn.like_bias(
                conn.postsyn_receptive(torch.rand(batchsz, *outshape)).sum(-1).sum(0)
            ).shape
            == conn.bias.shape
        )

    def test_like_synaptic(self):
        inshape = randshape(1, 3, 2, 5)
        outshape = randshape(1, 3, 2, 5)
        batchsz = random.randint(1, 9)
        conn = self.makeconn(
            inshape,
            outshape,
            1.0,
            batchsz,
            weight_init=(lambda x: torch.rand_like(x)),
        )

        assert (
            tuple(conn.like_synaptic(torch.rand(batchsz, *inshape)).shape)
            == conn.synapse.spike.shape
        )

    @pytest.mark.parametrize("hasoutdim", (True, False), ids=("outdim", "inonly"))
    def test_presyn_receptive(self, hasoutdim):
        inshape = randshape(1, 3, 2, 5)
        outshape = randshape(1, 3, 2, 5)
        batchsz = random.randint(1, 9)
        conn = self.makeconn(
            inshape,
            outshape,
            1.0,
            batchsz,
            weight_init=(lambda x: torch.rand_like(x)),
        )

        assert tuple(
            conn.presyn_receptive(
                torch.rand(batchsz, math.prod(inshape), math.prod(outshape))
                if hasoutdim
                else torch.rand(batchsz, math.prod(inshape))
            ).shape
        ) == (
            batchsz,
            math.prod(outshape) if hasoutdim else 1,
            math.prod(inshape),
            1,
        )

    def test_postsyn_receptive(self):
        inshape = randshape(1, 3, 2, 5)
        outshape = randshape(1, 3, 2, 5)
        batchsz = random.randint(1, 9)
        conn = self.makeconn(
            inshape,
            outshape,
            1.0,
            batchsz,
            weight_init=(lambda x: torch.rand_like(x)),
        )

        outputs = conn(torch.rand(batchsz, *inshape))

        assert tuple(conn.postsyn_receptive(outputs).shape) == (
            batchsz,
            math.prod(outshape),
            1,
            1,
        )


class TestLinearDirect:

    @staticmethod
    def makeconn(
        shape: tuple[int, ...] | int,
        step_time: float,
        batch_size=1,
        bias=False,
        delay=None,
        *,
        weight_init=None,
        bias_init=None,
        delay_init=None,
    ) -> LinearDirect:
        return LinearDirect(
            shape,
            step_time,
            synapse=DeltaCurrent.partialconstructor(1.0, "previous", 1e-7),
            bias=bias,
            delay=delay,
            batch_size=batch_size,
            weight_init=weight_init,
            bias_init=bias_init,
            delay_init=delay_init,
        )

    @staticmethod
    def load_synapse(conn: LinearDirect, p=0.1) -> None:
        conn.synapse.spike_.value = torch.rand(conn.synapse.spike_.value.shape) < p

    @pytest.mark.parametrize("biased", (True, False), ids=("biased", "unbiased"))
    def test_forward_undelayed(self, biased):
        shape = randshape(1, 3, 2, 5)
        batchsz = random.randint(1, 9)
        conn = self.makeconn(
            shape,
            1.0,
            batchsz,
            bias=biased,
            weight_init=(lambda x: torch.rand_like(x)),
        )

        inputs = torch.rand(batchsz, *shape) < 0.5
        outputs = conn(inputs)

        if biased:
            res = (
                inputs.float().view(batchsz, -1) * conn.weight + conn.bias.view(1, -1)
            ).view(batchsz, *shape)
        else:
            res = (inputs.float().view(batchsz, -1) * conn.weight).view(batchsz, *shape)

        assert tuple(outputs.shape) == (batchsz, *shape)
        assert aaeq(outputs, res)

    @pytest.mark.parametrize("biased", (True, False), ids=("biased", "unbiased"))
    def test_forward_delayed(self, biased):
        shape = randshape(1, 3, 2, 5)
        batchsz = random.randint(1, 9)
        delay = 4
        conn = self.makeconn(
            shape,
            1.0,
            batchsz,
            bias=biased,
            delay=float(delay),
            weight_init=(lambda x: torch.rand_like(x)),
            delay_init=(lambda x, d=delay: torch.ones_like(x) * d),
        )

        self.load_synapse(conn, p=0.3)
        conn.synapse.spike_.align(0)
        inputs = conn.synapse.spike_.value.clone().detach()

        for k in range(1, delay + 1):
            outputs = conn(torch.zeros(batchsz, *shape))

            if biased:
                res = (
                    inputs[k, ...].view(batchsz, -1).float() * conn.weight
                    + conn.bias.view(1, -1)
                ).view(batchsz, *shape)
            else:
                res = (inputs[k, ...].view(batchsz, -1).float() * conn.weight).view(
                    batchsz, *shape
                )

            assert aaeq(outputs, res)

    def test_like_input(self):
        shape = randshape(1, 3, 2, 5)
        batchsz = random.randint(1, 9)
        conn = self.makeconn(
            shape,
            1.0,
            batchsz,
            weight_init=(lambda x: torch.rand_like(x)),
        )

        assert tuple(conn.like_input(conn.synapse.spike).shape) == (batchsz, *shape)

    def test_like_bias(self):
        shape = randshape(1, 3, 2, 5)
        batchsz = random.randint(1, 9)
        conn = self.makeconn(
            shape,
            1.0,
            batchsz,
            bias=True,
            weight_init=(lambda x: torch.rand_like(x)),
        )

        assert (
            conn.like_bias(
                conn.postsyn_receptive(torch.rand(batchsz, *shape)).sum(-1).sum(0)
            ).shape
            == conn.bias.shape
        )

    def test_like_synaptic(self):
        shape = randshape(1, 3, 2, 5)
        batchsz = random.randint(1, 9)
        conn = self.makeconn(
            shape,
            1.0,
            batchsz,
            weight_init=(lambda x: torch.rand_like(x)),
        )

        assert (
            tuple(conn.like_synaptic(torch.rand(batchsz, *shape)).shape)
            == conn.synapse.spike.shape
        )

    @pytest.mark.parametrize("hasoutdim", (True, False), ids=("outdim", "inonly"))
    def test_presyn_receptive(self, hasoutdim):
        shape = randshape(1, 3, 2, 5)
        batchsz = random.randint(1, 9)
        conn = self.makeconn(
            shape,
            1.0,
            batchsz,
            weight_init=(lambda x: torch.rand_like(x)),
        )

        assert tuple(
            conn.presyn_receptive(
                torch.rand(batchsz, math.prod(shape))
                if hasoutdim
                else torch.rand(batchsz, math.prod(shape), 1)
            ).shape
        ) == (
            batchsz,
            math.prod(shape),
            1,
        )

    def test_postsyn_receptive(self):
        shape = randshape(1, 3, 2, 5)
        batchsz = random.randint(1, 9)
        conn = self.makeconn(
            shape,
            1.0,
            batchsz,
            weight_init=(lambda x: torch.rand_like(x)),
        )

        outputs = conn(torch.rand(batchsz, *shape))

        assert tuple(conn.postsyn_receptive(outputs).shape) == (
            batchsz,
            math.prod(shape),
            1,
        )


class TestLinearLateral:

    @staticmethod
    def makeconn(
        shape: tuple[int, ...] | int,
        step_time: float,
        batch_size=1,
        bias=False,
        delay=None,
        *,
        weight_init=None,
        bias_init=None,
        delay_init=None,
    ) -> LinearLateral:
        return LinearLateral(
            shape,
            step_time,
            synapse=DeltaCurrent.partialconstructor(1.0, "previous", 1e-7),
            bias=bias,
            delay=delay,
            batch_size=batch_size,
            weight_init=weight_init,
            bias_init=bias_init,
            delay_init=delay_init,
        )

    @staticmethod
    def load_synapse(conn: LinearLateral, p=0.1) -> None:
        conn.synapse.spike_.value = torch.rand(conn.synapse.spike_.value.shape) < p

    @staticmethod
    def mask(shape) -> torch.Tensor:
        return 1 - torch.eye(math.prod(shape))

    def test_like_bias(self):
        shape = randshape(1, 3, 2, 5)
        batchsz = random.randint(1, 9)
        conn = self.makeconn(
            shape,
            1.0,
            batchsz,
            bias=True,
            weight_init=(lambda x: torch.rand_like(x)),
        )

        assert (
            conn.like_bias(
                conn.postsyn_receptive(torch.rand(batchsz, *shape)).sum(-1).sum(0)
            ).shape
            == conn.bias.shape
        )

    @pytest.mark.parametrize("biased", (True, False), ids=("biased", "unbiased"))
    def test_forward_undelayed(self, biased):
        shape = randshape(1, 3, 2, 5)
        batchsz = random.randint(1, 9)
        conn = self.makeconn(
            shape,
            1.0,
            batchsz,
            bias=biased,
            weight_init=(lambda x: torch.rand_like(x)),
        )

        inputs = torch.rand(batchsz, *shape) < 0.5
        outputs = conn(inputs)

        if biased:
            res = (
                torch.matmul(inputs.float().view(batchsz, -1), conn.weight.t())
                + conn.bias.view(1, -1)
            ).view(batchsz, *shape)
            maskres = (
                torch.matmul(
                    inputs.float().view(batchsz, -1),
                    (conn.weight * self.mask(shape)).t(),
                )
                + conn.bias.view(1, -1)
            ).view(batchsz, *shape)
        else:
            res = torch.matmul(inputs.float().view(batchsz, -1), conn.weight.t()).view(
                batchsz, *shape
            )
            maskres = torch.matmul(
                inputs.float().view(batchsz, -1), (conn.weight * self.mask(shape)).t()
            ).view(batchsz, *shape)

        assert tuple(outputs.shape) == (batchsz, *shape)
        assert aaeq(outputs, res)
        assert aaeq(outputs, maskres)

    @pytest.mark.parametrize("biased", (True, False), ids=("biased", "unbiased"))
    def test_forward_delayed(self, biased):
        shape = randshape(1, 3, 2, 5)
        batchsz = random.randint(1, 9)
        shape, batchsz = (2, 2), 1
        delay = 4
        conn = self.makeconn(
            shape,
            1.0,
            batchsz,
            bias=biased,
            delay=float(delay),
            weight_init=(lambda x: torch.rand_like(x)),
            delay_init=(lambda x, d=delay: torch.ones_like(x) * d),
        )

        self.load_synapse(conn, p=0.3)
        conn.synapse.spike_.align(0)
        inputs = conn.synapse.spike_.value.clone().detach()

        for k in range(1, delay + 1):
            outputs = conn(torch.zeros(batchsz, *shape))

            if biased:
                res = (
                    torch.matmul(
                        inputs[k, ...].view(batchsz, -1).float(), conn.weight.t()
                    )
                    + conn.bias.view(1, -1)
                ).view(batchsz, *shape)
                maskres = (
                    torch.matmul(
                        inputs[k, ...].view(batchsz, -1).float(),
                        (conn.weight * self.mask(shape)).t(),
                    )
                    + conn.bias.view(1, -1)
                ).view(batchsz, *shape)
            else:
                res = torch.matmul(
                    inputs[k, ...].view(batchsz, -1).float(), conn.weight.t()
                ).view(batchsz, *shape)
                maskres = torch.matmul(
                    inputs[k, ...].view(batchsz, -1).float(),
                    (conn.weight * self.mask(shape)).t(),
                ).view(batchsz, *shape)

            assert aaeq(outputs, res)
            assert aaeq(outputs, maskres)


class TestConv2D:

    @staticmethod
    def makeconn(
        height: int,
        width: int,
        channels: int,
        filters: int,
        step_time: float,
        kernel: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        batch_size=1,
        bias=False,
        delay=None,
        *,
        weight_init=None,
        bias_init=None,
        delay_init=None,
    ) -> Conv2D:
        return Conv2D(
            height=height,
            width=width,
            channels=channels,
            filters=filters,
            kernel=kernel,
            step_time=step_time,
            stride=stride,
            padding=padding,
            dilation=dilation,
            synapse=DeltaCurrent.partialconstructor(1.0, "previous", 1e-7),
            bias=bias,
            delay=delay,
            batch_size=batch_size,
            weight_init=weight_init,
            bias_init=bias_init,
            delay_init=delay_init,
        )

    @staticmethod
    def load_synapse(conn: Conv2D, p=0.1) -> None:
        conn.synapse.spike_.value = torch.rand(conn.synapse.spike_.value.shape) < p

    def test_input_to_synaptic(self):
        length = random.randint(28, 64)
        channels = random.randint(3, 7)
        filters = random.randint(8, 18)
        kernel = random.randint(2, 5)
        batchsz = random.randint(1, 9)
        conn = self.makeconn(
            length,
            length,
            channels,
            filters,
            1.0,
            kernel,
            batch_size=batchsz,
        )

        img = torch.rand(batchsz, channels, length, length)

        assert aaeq(img, conn.like_input(conn.like_synaptic(img)), 1e-6)

    def test_like_bias(self):
        length = random.randint(28, 64)
        channels = random.randint(3, 7)
        filters = random.randint(8, 18)
        kernel = random.randint(2, 5)
        batchsz = random.randint(1, 9)
        conn = self.makeconn(
            length,
            length,
            channels,
            filters,
            1.0,
            kernel,
            bias=True,
            weight_init=(lambda x: torch.rand_like(x)),
            batch_size=batchsz,
        )

        assert (
            conn.like_bias(
                conn.postsyn_receptive(torch.rand(batchsz, *conn.outshape))
                .sum(-1)
                .sum(0)
            ).shape
            == conn.bias.shape
        )

    @pytest.mark.parametrize("biased", (True, False), ids=("biased", "unbiased"))
    def test_forward_undelayed(self, biased):
        height = random.randint(14, 53)
        width = random.randint(14, 53)
        channels = random.randint(3, 7)
        filters = random.randint(8, 18)
        kernel = random.randint(2, 5)
        batchsz = random.randint(1, 9)

        conn = self.makeconn(
            height=height,
            width=width,
            channels=channels,
            filters=filters,
            step_time=1.0,
            kernel=kernel,
            batch_size=batchsz,
            bias=biased,
            delay=None,
        )

        refconn = nn.Conv2d(channels, filters, kernel, bias=biased)

        conn.weight = refconn.weight
        if biased:
            conn.bias = refconn.bias

        for _ in range(10):
            data = torch.rand(batchsz, channels, height, width) < 0.3
            assert aaeq(conn(data), refconn(data.float()), 1e-6)

    @pytest.mark.parametrize("biased", (True, False), ids=("biased", "unbiased"))
    def test_forward_delayed(self, biased):
        height = random.randint(14, 53)
        width = random.randint(14, 53)
        channels = random.randint(3, 7)
        filters = random.randint(8, 18)
        kernel = random.randint(2, 5)
        batchsz = random.randint(1, 9)
        delay = float(random.randint(3, 7))

        conn = self.makeconn(
            height=height,
            width=width,
            channels=channels,
            filters=filters,
            step_time=1.0,
            kernel=kernel,
            batch_size=batchsz,
            bias=biased,
            delay=float(delay),
        )

        refconn = nn.Conv2d(channels, filters, kernel, bias=biased)

        conn.weight = refconn.weight
        if biased:
            conn.bias = refconn.bias

        _ = conn.delay.fill_(delay)

        data = [
            torch.rand(batchsz, channels, height, width) < 0.3
            for _ in range(2 * int(delay) + 1)
        ]
        res = []
        for n, d in enumerate(data):
            r = conn(d)
            if n >= delay:
                res.append(r)

        for s, r in zip(data, res):
            assert aaeq(r, refconn(s.float()), 1e-6)
