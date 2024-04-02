import pytest
import random
import torch

from inferno.neural import DeltaCurrent
from inferno.neural import LinearDense


def randshape(mindims=1, maxdims=9, minsize=1, maxsize=9):
    return tuple(
        random.randint(mindims, maxdims)
        for _ in range(random.randint(minsize, maxsize))
    )


def aaeq(t0, t1, eps=1e-7) -> bool:
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
            synapse=DeltaCurrent.partialconstructor(step_time, "previous", 1e-7),
            bias=bias,
            delay=delay,
            batch_size=batch_size,
            weight_init=weight_init,
            bias_init=bias_init,
            delay_init=delay_init,
        )

    @staticmethod
    def load_synapse(conn: LinearDense, p=0.1) -> LinearDense:
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
        delay = 5
        conn = self.makeconn(
            inshape,
            outshape,
            1.0,
            batchsz,
            delay=float(delay),
            weight_init=(lambda x: torch.rand_like(x)),
            delay_init=(lambda x: torch.ones_like(x)),
        )

        self.load_synapse(conn, p=0.3)
        conn.synapse.spike_.align(0)
        inputs = conn.synapse.spike_.value.clone().detach()

        for k in range(1, delay):
            outputs = conn(torch.zeros(batchsz, *inshape))
            assert aaeq(
                outputs,
                torch.matmul(
                    inputs[..., k].view(batchsz, -1).float(), conn.weight.t()
                ).view(batchsz, *outshape),
            )
