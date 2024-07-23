import einops as ein
import torch
from typing import Iterator


def poisson_interval(
    inputs: torch.Tensor,
    steps: int,
    step_time: float,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    r"""Generates a tensor of spikes with Poisson determined intervals.

    This is included to replicate BindsNET's Poisson spike generation. The intervals
    between spikes follow the Poisson distribution parameterized with the inverse of
    the expected rate (i.e. the scale is given as the rate).

    Args:
        inputs (torch.Tensor): expected spike frequencies, in :math:`\text{Hz}`.
        steps (int): number of steps for which to generate spikes, :math:`S`.
        step_time (float): length of time between outputs, in :math:`\text{ms}`.
        generator (torch.Generator | None, optional): pseudorandom number generator
            for sampling. Defaults to ``None``.

    Returns:
        torch.Tensor: the generated spike train, time first.

    .. admonition:: Shape
        :class: tensorshape

        ``inputs``:

        :math:`B \times N_0 \times \cdots`

        ``return``:

        :math:`S \times B \times N_0 \times \cdots`

        Where:
            * :math:`B` is the batch size.
            * :math:`N_0, \ldots` are the dimensions of the spikes being generated.
            * :math:`S` is the number of steps for which to generate spikes, ``steps``.

    Important:
        All elements of ``inputs`` must be nonnegative.
    """
    # disable gradient computation
    with torch.no_grad():
        # convert refrac from ms to dt
        steps = int(steps)

        # create valid input mask
        mask = inputs > 0

        # convert frequencies (in Hz) to expected time between spikes in dt
        inputs = (1 / inputs) * (1000.0 / step_time)

        # mask inputs
        inputs[~mask] = 0

        # convert rates into intervals via sampling
        res = torch.poisson(
            inputs.expand(steps + 2, *inputs.shape), generator=generator
        )

        # increment zero-intervals at nonzero-rates to avoid collisions
        res[:, mask] += res[:, mask] == 0

        # convert intervals to steps via cumulative summation
        res = res.cumsum(dim=0)

        # limit steps to the maximum given
        res = res.clamp_max_(steps).long()

        # convert steps to spikes via scattering
        res = torch.zeros_like(res, dtype=torch.bool).scatter_(0, res, 1)

        # remove bad values
        res = res[1:-1]

    return res


def poisson_interval_online(
    inputs: torch.Tensor,
    steps: int,
    step_time: float,
    *,
    generator: torch.Generator | None = None,
) -> Iterator[torch.Tensor]:
    r"""Yields a generator for tensor slices of a spike train with Poisson-sampled intervals.

    This is included to replicate BindsNET's Poisson spike generation. The intervals
    between spikes follow the Poisson distribution parameterized with the inverse of
    the expected rate (i.e. the scale is given as the rate).

    Args:
        inputs (torch.Tensor): expected spike frequencies, in :math:`\text{Hz}`.
        steps (int): number of steps for which to generate spikes, :math:`S`.
        step_time (float): length of time between outputs, in :math:`\text{ms}`.
        generator (torch.Generator | None, optional): pseudorandom number generator
            for sampling. Defaults to ``None``.

    Yields:
        torch.Tensor: time slices of the generated spike train.

    .. admonition:: Shape
        :class: tensorshape

        ``inputs``:

        :math:`B \times N_0 \times \cdots`

        ``yield``:

        :math:`B \times N_0 \times \cdots`

        Where:
            * :math:`B` is the batch size.
            * :math:`N_0, \ldots` are the dimensions of the spikes being generated.

    Important:
        All elements of ``inputs`` must be nonnegative.
    """
    # disable gradient computation
    with torch.no_grad():
        # convert refrac from ms to dt
        steps = int(steps)

        # create valid input mask
        mask = inputs > 0

        # convert frequencies (in Hz) to expected time between spikes in dt
        inputs = (1 / inputs) * (1000.0 / step_time)

        # mask inputs
        inputs[~mask] = 0

        # convert rates into intervals via sampling
        intervals = torch.poisson(inputs, generator=generator)

        # main loop
        for _ in range(steps):
            # decrement intervals
            intervals -= 1

            # generate spikes
            spikes = torch.logical_and(intervals < 1, mask)

            # calculate new intervals for fired neurons
            intervals[spikes] = torch.poisson(inputs[spikes], generator=generator)

            # yields the current spikes
            yield spikes


def homogeneous_poisson_exp_interval(
    inputs: torch.Tensor,
    steps: int,
    step_time: float,
    *,
    refrac: float | None = None,
    compensate: bool = True,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    r"""Generates a tensor of spikes using a Poisson distribution.

    This method samples randomly from an exponential distribution (the interval
    between samples in a Poisson point process), adding an additional refractory
    period and compensating the rate.

    Args:
        inputs (torch.Tensor): expected spike frequencies, :math:`f`,
            in :math:`\text{Hz}`.
        steps (int): number of steps for which to generate spikes, :math:`S`.
        step_time (float): length of time between outputs, :math:`\Delta t`,
            in :math:`\text{ms}`.
        refrac (float | None, optional): minimum interval between spikes set to the step
            time if ``None``, in :math:`\text{ms}`. Defaults to ``None``.
        compensate (bool, optional): if the spike generation rate should be compensate
            for the refractory period. Defaults to ``True``.
        generator (torch.Generator | None, optional): pseudorandom number generator
            for sampling. Defaults to ``None``.

    Returns:
        torch.Tensor: the generated spike train, time first.

    .. admonition:: Shape
        :class: tensorshape

        ``inputs``:

        :math:`B \times N_0 \times \cdots`

        ``return``:

        :math:`S \times B \times N_0 \times \cdots`

        Where:
            * :math:`B` is the batch size.
            * :math:`N_0, \ldots` are the dimensions of the spikes being generated.
            * :math:`S` is the number of steps for which to generate spikes, ``steps``.

    Caution:
        If the refractory period is greater than or equal to the expected intervals between
        spikes, output will be nonsensical. The expected intervals are equal to
        :math:`1000 \frac{1}{f} \text{ ms}`.

    Important:
        All elements of ``inputs`` must be nonnegative.

    Note:
        ``refrac`` at its default still allows for a spike to be generated at every
        step (since the distance between is :math:`\Delta t`). To get behavior where
        at most every :math:`n^\text{th}` step is a spike, the refractory period needs
        to be set to :math:`n \Delta t`.
    """
    # disable gradient computation
    with torch.no_grad():
        # assume refrac is dt if unspecified
        refrac = step_time if refrac is None else step_time

        # get number of steps, convert refrac from ms to dt
        steps, refrac = int(steps), refrac / step_time

        # convert frequencies (in Hz) to expected time between spikes in dt
        res = (1 / inputs) * (1000.0 / step_time)

        # compensate scale parameter with refractory length
        if compensate:
            res = res - refrac

        # maximum possible spikes would be one per (non-refrac) time step
        nbins = int(steps // max(refrac, 1))

        # sample from exponential distribution
        res = (
            res.new_empty(nbins, *inputs.shape).exponential_(1.0, generator=generator)
            * res
            + refrac
        )

        # convert intervals into times through cumsum
        res = res.cumsum(dim=0)

        # clamp to range, and cast for index use
        res = res.clamp_max_(steps).long()

        # scatter into a zero tensor
        res = res.new_zeros(steps + 1, *inputs.shape, dtype=torch.bool).scatter_(
            0, res, 1
        )

        # remove bad values
        res = res[:-1]

    return res


def homogeneous_poisson_exp_interval_online(
    inputs: torch.Tensor,
    steps: int,
    step_time: float,
    *,
    refrac: float | None = None,
    compensate: bool = True,
    generator: torch.Generator | None = None,
) -> Iterator[torch.Tensor]:
    r"""Yields a generator for tensor slices of a Poisson spike train.

    This method samples randomly from an exponential distribution (the interval
    between samples in a Poisson point process), adding an additional refractory
    period and compensating the rate.

    Args:
        inputs (torch.Tensor): expected spike frequencies, :math:`f`,
            in :math:`\text{Hz}`.
        steps (int): number of steps for which to generate spikes, :math:`S`.
        step_time (float): length of time between outputs, :math:`\Delta t`,
            in :math:`\text{ms}`.
        refrac (float | None, optional): minimum interval between spikes set to the step
            time if ``None``, in :math:`\text{ms}`. Defaults to ``None``.
        compensate (bool, optional): if the spike generation rate should be compensate
            for the refractory period. Defaults to ``True``.
        generator (torch.Generator | None, optional): pseudorandom number generator
            for sampling. Defaults to ``None``.

    Yields:
        torch.Tensor: time slices of the generated spike train.

    .. admonition:: Shape
        :class: tensorshape

        ``inputs``:

        :math:`B \times N_0 \times \cdots`

        ``yield``:

        :math:`B \times N_0 \times \cdots`

        Where:
            * :math:`B` is the batch size.
            * :math:`N_0, \ldots` are the dimensions of the spikes being generated.

    Caution:
        If the refractory period is greater than or equal to the expected intervals between
        spikes, output will be nonsensical. The expected intervals are equal to
        :math:`1000 \frac{1}{f} \text{ ms}`.

    Important:
        All elements of ``inputs`` must be nonnegative.

    Note:
        ``refrac`` at its default still allows for a spike to be generated at every
        step (since the distance between is :math:`\Delta t`). To get behavior where
        at most every :math:`n^\text{th}` step is a spike, the refractory period needs
        to be set to :math:`n \Delta t`.
    """
    # disable gradient computation
    with torch.no_grad():
        # assume refrac is dt if unspecified
        refrac = step_time if refrac is None else step_time

        # get number of steps, convert refrac from ms to dt
        steps, refrac = int(steps), refrac / step_time

        # convert frequencies (in Hz) to expected time between spikes in #dt
        inputs = (1 / inputs) * (1000.0 / step_time)

        # compensate scale parameter with refractory length
        if compensate:
            inputs = inputs - refrac

        # calculate initial intervals
        intervals = (
            torch.empty_like(inputs).exponential_(1.0, generator=generator) * inputs
            + refrac
        )

        # main loop
        for _ in range(steps):
            # decrement intervals
            intervals -= 1

            # generate spikes
            spikes = intervals < 1

            # calculate new intervals for fired neurons
            intervals[spikes] = (
                torch.empty_like(intervals[spikes]).exponential_(
                    1.0, generator=generator
                )
                * inputs
                + refrac
            )

            # yields the current spikes
            yield spikes


def homogenous_poisson_bernoulli_approx(
    inputs: torch.Tensor,
    steps: int,
    step_time: float,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    r"""Generates a tensor of spikes approximating a homogeneous Poisson distribution.

    This method takes in a tensor of expected frequencies, converts them to an expected
    probability, performs a non-reallocating expansion along the time dimension, and
    samples randomly from a Bernoulli distribution.

    Args:
        inputs (torch.Tensor): expected spike frequencies, :math:`f`,
            in :math:`\text{Hz}`.
        steps (int): number of steps for which to generate spikes, :math:`S`.
        step_time (float): length of time between outputs, :math:`\Delta t`,
            in :math:`\text{ms}`.
        generator (torch.Generator | None, optional): pseudorandom number generator
            for sampling. Defaults to ``None``.

    Returns:
        torch.Tensor: the generated spike train, time first.

    .. admonition:: Shape
        :class: tensorshape

        ``inputs``:

        :math:`B \times N_0 \times \cdots`

        ``return``:

        :math:`S \times B \times N_0 \times \cdots`

        Where:
            * :math:`S` is the number of steps for which to generate spikes.
            * :math:`B` is the batch size.
            * :math:`N_0, \ldots` are the dimensions of the spikes being generated.

    Important:
        All elements of ``inputs`` must be nonnegative. Inputs will also be clamped if
        they exceed the maximum (i.e. if the expected value for the number of spikes
        in a time step is greater than 1).
    """
    # disable gradient computation
    with torch.no_grad():
        # convert frequencies (in Hz) to expected spike probabilities (EV spikes per dt)
        res = (inputs / 1000.0) * step_time

        # sample directly from Bernoulli distribution, clamping max probability
        return torch.bernoulli(
            ein.repeat(res.clamp_max_(1.0), "... -> t ...", t=int(steps)),
            generator=generator,
        ).bool()


def homogenous_poisson_bernoulli_approx_online(
    inputs: torch.Tensor,
    steps: int,
    step_time: float,
    *,
    generator: torch.Generator | None = None,
) -> Iterator[torch.Tensor]:
    r"""Yields a generator for tensor slices approximating a homogeneous Poisson distribution.

    This method takes in a tensor of expected frequencies, converts them to an expected
    probability, and samples randomly from a Bernoulli distribution.

    Args:
        inputs (torch.Tensor): expected spike frequencies, :math:`f`,
            in :math:`\text{Hz}`.
        steps (int): number of steps for which to generate spikes, :math:`S`.
        step_time (float): length of time between outputs, :math:`\Delta t`,
            in :math:`\text{ms}`.
        generator (torch.Generator | None, optional): pseudorandom number generator
            for sampling. Defaults to ``None``.

    Returns:
        torch.Tensor: the generated spike train, time first.

    .. admonition:: Shape
        :class: tensorshape

        ``inputs``:

        :math:`B \times N_0 \times \cdots`

        ``yield``:

        :math:`B \times N_0 \times \cdots`

        Where:
            * :math:`B` is the batch size.
            * :math:`N_0, \ldots` are the dimensions of the spikes being generated.

    Important:
        All elements of ``inputs`` must be nonnegative. Inputs will also be clamped if
        they exceed the maximum (i.e. if the expected value for the number of spikes
        in a time step is greater than 1).
    """
    # disable gradient computation
    with torch.no_grad():
        # convert frequencies (in Hz) to expected spike probabilities (EV spikes per dt)
        res = ((inputs / 1000.0) * step_time).clamp_max_(1.0)

        # main loop
        for _ in range(steps):
            # sample directly from Bernoulli distribution
            yield torch.bernoulli(res, generator=generator).bool()


def inhomogeneous_poisson_bernoulli_approx(
    inputs: torch.Tensor,
    step_time: float,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    r"""Generates a tensor of spikes approximating an inhomogeneous Poisson distribution.

    This method takes in a tensor of frequencies over a number of time steps.

    Args:
        inputs (torch.Tensor): expected spike frequencies, :math:`f`,
            in :math:`\text{Hz}`.
        step_time (float): length of time between outputs, :math:`\Delta t`,
            in :math:`\text{ms}`.
        generator (torch.Generator | None, optional): pseudorandom number generator
            for sampling. Defaults to ``None``.

    Returns:
        torch.Tensor: the generated spike train, time first.

    .. admonition:: Shape
        :class: tensorshape

        ``inputs``:

        :math:`S \times B \times N_0 \times \cdots`

        ``return``:

        :math:`S \times B \times N_0 \times \cdots`

        Where:
            * :math:`S` is the number of steps for which to generate spikes.
            * :math:`B` is the batch size.
            * :math:`N_0, \ldots` are the dimensions of the spikes being generated.

    Important:
        All elements of ``inputs`` must be nonnegative. Inputs will also be clamped if
        they exceed the maximum (i.e. if the expected value for the number of spikes
        in a time step is greater than 1).
    """
    # disable gradient computation
    with torch.no_grad():
        # convert frequencies (in Hz) to expected spike probabilities (EV spikes per dt)
        res = (inputs / 1000.0) * step_time

        # sample directly from Bernoulli distribution, clamping max probability
        return torch.bernoulli(res.clamp_max_(1.0), generator=generator).bool()
