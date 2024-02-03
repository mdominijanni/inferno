from inferno import Module
from inferno._internal import numeric_limit
import torch


class IntervalPoisson(Module):
    def __init__(
        self,
        step_time: float,
        steps: int,
        intensity: float,
        refrac: float,
        generator: torch.Generator | None = None,
    ):
        # call superclass constructor
        Module.__init__(self)

        # encoder attributes
        self.step_time, e = numeric_limit("step_time", step_time, 0, "gt", float)
        if e:
            raise e
        self.num_steps, e = numeric_limit("steps", steps, 0, "gt", int)
        if e:
            raise e
        self.maxfreq, e = numeric_limit("maxfreq", maxfreq, 0, "gte", float)
        if e:
            raise e
        self.rng = generator

    @property
    def dt(self) -> float:
        return self.step_time

    @dt.setter
    def dt(self, value: float) -> None:
        self.step_time, e = numeric_limit("value", value, 0, "gt", float)
        if e:
            raise e

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
        self.num_steps, e = numeric_limit("value", value, 0, "gt", int)
        if e:
            raise e

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs as 0-1 range, multiplied by maxfreq
        return interval_poisson(
            self.maxfreq * inputs,
            step_time=self.dt,
            steps=self.steps,
            generator=self.generator,
        )


def interval_poisson_v1(
    inputs: torch.Tensor,
    step_time: float,
    steps: int,
    generator: torch.Generator | None = None,
):
    r"""Generates a tensor of spikes using a Poisson distribution.

    Args:
        inputs (torch.Tensor): expected spike frequencies, in :math:`\text{Hz}`.
        step_time (float): length of time between outputs, in :math:`\text{ms}`.
        steps (int): number of spikes to generate.
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
        # convert frequencies to Poisson rate parameter
        res = torch.nan_to_num(1 / inputs, posinf=0) * (1000 / step_time)

        # convert rates into intervals via sampling
        res = torch.poisson(res.expand(steps + 1, *res.shape), generator=generator)

        # increment zero-intervals at nonzero-rates to avoid collisions
        mask = inputs != 0
        res[:, mask] += res[:, mask] == 0

        # convert intervals to steps via cumumlative summation
        res = res.cumsum(dim=0)

        # limit steps to the maximum given
        res.clamp_max_(steps)

        # convert steps to spikes via scattering
        res = torch.zeros_like(res).scatter_(0, res.long(), 1.0)

        # trim last "fake" step and cast to boolean (spike datatype)
        res = res[:-1].bool()

    return res


def interval_poisson(
    inputs: torch.Tensor,
    step_time: float,
    duration: float,
    refrac: float = 0.0,
    generator: torch.Generator | None = None,
):
    r"""Generates a tensor of spikes using a Poisson distribution.

    This method samples randomly from an exponential distribution (the interval
    between samples in a Poisson point process).

    Args:
        inputs (torch.Tensor): expected spike frequencies, :math:`f`,
            in :math:`\text{Hz}`.
        step_time (float): length of time between outputs, :math:`\Delta t`,
            in :math:`\text{ms}`.
        duration (float): duration of the spike train, :math:`T`,
            in :math:`\text{ms}`.
        refrac (float, optional): minimum interal between spikes, in :math:`\text{ms}`.
            Defaults to 0.0.
        generator (torch.Generator | None, optional): pseudorandom number generator
            for sampling. Defaults to None.

    .. admonition:: Shape
        :class: tensorshape

        ``inputs``:

        :math:`B \times N_0 \times \cdots`

        ``return``:

        :math:`\left\lfloor\frac{T}{\Delta t}\right\rfloor \times B \times N_0 \times \cdots`

        Where:
            * :math:`B` is the batch size.
            * :math:`N_0, \ldots` are the dimensions of the spikes being generated.
            * :math:`T` is the duration of the spike train, ``duration``.
            * :math:`\Delta t` is the length of each simulation step ``step_time``.

    Important:
        All elements of ``inputs`` must be nonnegative.
    """
    # disable gradient computation
    with torch.no_grad():
        # get number of steps, convert refrac from ms to #dt, integer refrac for bounds
        steps, refrac = int(duration // step_time), refrac / step_time

        # convert frequencies (in Hz) to expected time between spikes in #dt
        res = (1 / inputs) * (1000.0 / step_time)

        # compensate scale parameter with refractory length
        res = res - refrac

        # maximum possible spikes would be one per (non-refrac) time step
        nbins = steps // int(refrac + 1)

        # sample from exponential distribution
        res = (
            res.new_empty(nbins, *inputs.shape).exponential_(1.0, generator=generator)
            * res
            + refrac
        )

        # convert intervals into times through cumsum, shift by refrac back to start
        res = res.cumsum(dim=0) - refrac

        # round to int, clamp to range, and cast for index use
        res = res.round_().clamp_max_(steps).long()

        # scatter into a zero tensor
        res = res.new_zeros(steps + 1, *inputs.shape, dtype=torch.bool).scatter_(
            0, res, 1
        )[:-1]

    return res
