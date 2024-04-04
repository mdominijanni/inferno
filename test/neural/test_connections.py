import math
import pytest
import random
import torch

from inferno.neural import DeltaCurrent
from inferno.neural import LinearDense, LinearDirect, LinearLateral


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

    def test_forward_undelayed(self):
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

        inputs = torch.rand(batchsz, *inshape)
        outputs = conn(inputs)

        assert tuple(outputs.shape) == (batchsz, *outshape)
        assert aaeq(
            outputs,
            torch.matmul(inputs.view(batchsz, -1), conn.weight.t()).view(
                batchsz, *outshape
            ),
        )

    def test_forward_delayed(self):
        inshape = randshape(1, 3, 2, 5)
        outshape = randshape(1, 3, 2, 5)
        batchsz = random.randint(1, 9)
        delay = 4
        conn = self.makeconn(
            inshape,
            outshape,
            1.0,
            batchsz,
            delay=float(delay),
            weight_init=(lambda x: torch.rand_like(x)),
            delay_init=(lambda x, d=delay: torch.ones_like(x) * d),
        )

        self.load_synapse(conn, p=0.3)
        conn.synapse.spike_.align(0)
        inputs = conn.synapse.spike_.value.clone().detach()
        for k in range(1, delay + 1):
            outputs = conn(torch.zeros(batchsz, *inshape))
            assert aaeq(
                outputs,
                torch.matmul(
                    inputs[..., k].view(batchsz, -1).float(), conn.weight.t()
                ).view(batchsz, *outshape),
            )

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

    def test_forward_undelayed(self):
        shape = randshape(1, 3, 2, 5)
        batchsz = random.randint(1, 9)
        conn = self.makeconn(
            shape,
            1.0,
            batchsz,
            weight_init=(lambda x: torch.rand_like(x)),
        )

        inputs = torch.rand(batchsz, *shape)
        outputs = conn(inputs)

        assert tuple(outputs.shape) == (batchsz, *shape)
        assert aaeq(
            outputs,
            (inputs.view(batchsz, -1) * conn.weight).view(batchsz, *shape),
        )

    def test_forward_delayed(self):
        shape = randshape(1, 3, 2, 5)
        batchsz = random.randint(1, 9)
        delay = 4
        conn = self.makeconn(
            shape,
            1.0,
            batchsz,
            delay=float(delay),
            weight_init=(lambda x: torch.rand_like(x)),
            delay_init=(lambda x, d=delay: torch.ones_like(x) * d),
        )

        self.load_synapse(conn, p=0.3)
        conn.synapse.spike_.align(0)
        inputs = conn.synapse.spike_.value.clone().detach()
        for k in range(1, delay + 1):
            outputs = conn(torch.zeros(batchsz, *shape))
            assert aaeq(
                outputs,
                (inputs[..., k].view(batchsz, -1).float() * conn.weight).view(
                    batchsz, *shape
                ),
            )

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

    def test_forward_undelayed(self):
        shape = randshape(1, 3, 2, 5)
        batchsz = random.randint(1, 9)
        conn = self.makeconn(
            shape,
            1.0,
            batchsz,
            weight_init=(lambda x: torch.rand_like(x)),
        )

        inputs = torch.rand(batchsz, *shape)
        outputs = conn(inputs)

        assert tuple(outputs.shape) == (batchsz, *shape)
        assert aaeq(
            outputs,
            (inputs.view(batchsz, -1) * conn.weight).view(batchsz, *shape),
        )
        assert aaeq(
            outputs,
            torch.matmul(
                inputs.view(batchsz, -1), (conn.weight * self.mask(shape)).t()
            ).view(batchsz, *shape),
        )

    def test_forward_delayed(self):
        shape = randshape(1, 3, 2, 5)
        batchsz = random.randint(1, 9)
        delay = 4
        conn = self.makeconn(
            shape,
            1.0,
            batchsz,
            delay=float(delay),
            weight_init=(lambda x: torch.rand_like(x)),
            delay_init=(lambda x, d=delay: torch.ones_like(x) * d),
        )

        self.load_synapse(conn, p=0.3)
        conn.synapse.spike_.align(0)
        inputs = conn.synapse.spike_.value.clone().detach()
        for k in range(1, delay + 1):
            outputs = conn(torch.zeros(batchsz, *shape))
            assert aaeq(
                outputs,
                torch.sum(
                    inputs[..., k]
                    .view(batchsz, -1, 1)
                    .expand(-1, -1, math.prod(shape))
                    .float()
                    * (conn.weight * self.mask(shape)).t(),
                    -1,
                ).view(batchsz, *shape),
            )