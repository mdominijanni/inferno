from inferno import Module
from inferno._internal import numeric_limit
import torch


class IntervalPoisson(Module):
    def __init__(
        self,
        step_time: float,
        steps: int,
        maxfreq: float,
        generator: torch.Generator | None = None,
    ):
        # call superclass constructor
        Module.__init__(self)

        # encoder attributes
        self.step_time = numeric_limit("`step_time`", step_time, 0, "gt", float)
        self.num_steps = numeric_limit("`steps`", steps, 0, "gt", int)
        self.maxfreq = numeric_limit("`maxfreq`", maxfreq, 0, "gte", float)
        self.rng = generator

    @property
    def dt(self) -> float:
        return self.step_time

    @dt.setter
    def dt(self, value: float) -> None:
        self.step_time = numeric_limit("`value`", value, 0, "gt", float)

    @property
    def generator(self) -> torch.Generator | None:
        return self.rng

    @generator.setter
    def generator(self, value: torch.Generator | None) -> None:
        self.rng = value

    @property
    def steps(self) -> float:
        return self.num_steps

    @steps.setter
    def steps(self, value: float) -> None:
        self.num_steps = numeric_limit("`value`", value, 0, "gt", int)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs as 0-1 range, multiplied by maxfreq
        return interval_poisson(
            self.maxfreq * inputs,
            step_time=self.dt,
            steps=self.steps,
            generator=self.generator,
        )


def interval_poisson(
    inputs: torch.Tensor,
    step_time: float,
    steps: int,
    generator: torch.Generator | None = None,
):
    r"""Generates a tensor of spikes using a Poisson distribution.

    Args:
        inputs (torch.Tensor): expected spike frequencies, in :math:`\text{Hz}`.
        step_time (float): length of time between outputs, in :math:`\text{ms}`.
        steps (int | None, optional): number of spikes to generate.
        generator (torch.Generator | None, optional): pseudorandom number generator
            for sampling. Defaults to None.

    .. admonition:: Shape
        :class: tensorshape

        ``inputs``:

        :math:`B \times N_0 \times \cdots`

        ``return``:

        :math:`T \times B \times N_0 \times \cdots`

        Where:
            * :math:`B` is the batch size.
            * :math:`N_0, \ldots` are the dimensions of the spikes being generated.
            * :math:`T` is the number of time steps generated for, ``steps``.

    Important:
        All elements of ``inputs`` must be nonnegative.
    """
    # disable gradient computation
    with torch.no_grad():
        # convert frequencies to rates (Poisson parameter)
        res = torch.nan_to_num(1 / inputs, posinf=0) * (1000 / step_time)

        # convert rates into intervals via sampling
        res = torch.poisson(res.expand(steps + 1, *res.shape), generator=generator)

        # increment zero-intervals at nonzero-rates to avoid collisions
        res[:, inputs != 0] += res[:, inputs != 0] == 0

        # convert intervals to steps via cumumlative summation
        res = res.cumsum(dim=0)

        # set steps exceeding simulation steps to zero
        res[res > steps] = 0

        # convert steps to spikes via scattering
        res = torch.zeros_like(res).scatter_(0, res.long(), 1.0)

        # trim zeroth step (will be all ones since it was the "no spike" condition)
        res = res[1:]

        # cast to boolean (datatype used for spikes)
        res = res.bool()

    return res
