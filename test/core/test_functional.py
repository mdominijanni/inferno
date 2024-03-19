from collections.abc import Iterable
import einops as ein
from itertools import product
import pytest
import random
import torch

from inferno import isi


def allindices(shape: Iterable[int]) -> list[tuple[int, ...]]:
    return [*product(*(range(sz) for sz in shape))]


class TestISI:

    @staticmethod
    def random_shape(mindims=1, maxdims=9, minsize=1, maxsize=9):
        return tuple(
            random.randint(mindims, maxdims)
            for _ in range(random.randint(minsize, maxsize))
        )

    @pytest.mark.parametrize(
        "time_first",
        (True, False),
        ids=("time_first=True", "time_first=False"),
    )
    def test_isi_indices(self, time_first):
        shape = self.random_shape(mindims=2, maxdims=4, minsize=3)
        steps = random.randint(125, 375)

        spikes = torch.rand(steps, *shape) < (0.05 + torch.rand(1, *shape) * 0.3)
        spikes[:, *(0 for _ in spikes.shape[1:])] = 0
        spikes[:, *(-1 for _ in spikes.shape[1:])] = 1

        numisi = torch.amax(torch.sum(spikes, dim=0)) - 1

        if time_first:
            libisi = isi(spikes, time_first=True)
            assert (numisi, *shape) == tuple(libisi.shape)
        else:
            libisi = isi(ein.rearrange(spikes, "t ... -> ... t"), time_first=False)
            assert (*shape, numisi) == tuple(libisi.shape)
            libisi = ein.rearrange(libisi, "... t -> t ...")

        for idx in allindices(shape):
            upisi = libisi[:, *idx][torch.logical_not(torch.isnan(libisi[:, *idx]))]
            icisi = torch.diff(torch.nonzero(spikes[:, *idx]).squeeze())
            assert icisi.numel() == upisi.numel()
            assert torch.all(icisi == upisi)

    @pytest.mark.parametrize(
        "time_first",
        (True, False),
        ids=("time_first=True", "time_first=False"),
    )
    def test_isi_times(self, time_first):
        shape = self.random_shape(mindims=2, maxdims=4, minsize=3)
        steps = random.randint(125, 375)
        dt = 0.5 + random.random()

        spikes = torch.rand(steps, *shape) < (0.05 + torch.rand(1, *shape) * 0.3)
        spikes[:, *(0 for _ in spikes.shape[1:])] = 0
        spikes[:, *(-1 for _ in spikes.shape[1:])] = 1

        numisi = torch.amax(torch.sum(spikes, dim=0)) - 1

        if time_first:
            libisi = isi(spikes, dt, time_first=True)
            assert (numisi, *shape) == tuple(libisi.shape)
        else:
            libisi = isi(ein.rearrange(spikes, "t ... -> ... t"), dt, time_first=False)
            assert (*shape, numisi) == tuple(libisi.shape)
            libisi = ein.rearrange(libisi, "... t -> t ...")

        for idx in allindices(shape):
            upisi = libisi[:, *idx][torch.logical_not(torch.isnan(libisi[:, *idx]))]
            icisi = torch.diff(torch.nonzero(spikes[:, *idx]).squeeze() * dt)
            assert icisi.numel() == upisi.numel()
            assert torch.all((icisi - upisi).abs() < 5e-7)

    @pytest.mark.parametrize(
        "time_first",
        (True, False),
        ids=("time_first=True", "time_first=False"),
    )
    def test_isi_nospikes(self, time_first):
        shape = self.random_shape(mindims=2, maxdims=4, minsize=3)
        steps = random.randint(125, 375)

        fullshape = (steps, *shape) if time_first else (*shape, steps)
        zeroshape = (0, *shape) if time_first else (*shape, 0)

        spikes = torch.zeros(*fullshape).bool()
        libisi = isi(spikes, time_first=time_first)

        assert zeroshape == tuple(libisi.shape)
