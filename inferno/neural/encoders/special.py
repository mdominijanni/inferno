from .mixins import GeneratorMixin, StepMixin
from .. import functional as nf
from ... import Module
from ..._internal import argtest
import torch
from typing import Iterator


class PoissonIntervalEncoder(GeneratorMixin, StepMixin, Module):
    r"""Encoder to generate spike trains with intervals sampled from a Poisson distribution.

    This is included to replicate BindsNET's Poisson spike generation. The intervals
    between spikes follow the Poisson distribution parameterized with the inverse of
    the expected rate (i.e. the scale is given as the rate).

    Args:
        steps (int): number of steps for which to generate spikes, :math:`S`.
        step_time (float): length of time between outputs, :math:`\Delta t`,
            in :math:`\text{ms}`.
        frequency (float): maximum spike frequency (associated with an input of 1),
            :math:`f`, in :math:`\text{Hz}`.
        generator (torch.Generator | None, optional): pseudorandom number generator
            for sampling. Defaults to ``None``.
    """

    def __init__(
        self,
        steps: int,
        step_time: float,
        frequency: float,
        *,
        generator: torch.Generator | None = None,
    ):
        # call superclass constructor
        Module.__init__(self)

        # set encoder attributes
        self.__frequency_scale = argtest.gte("frequency", frequency, 0, float)

        # call mixin constructors
        StepMixin.__init__(self, steps=steps, step_time=step_time)
        GeneratorMixin.__init__(self, generator=generator)

    @property
    def frequency(self) -> float:
        r"""Expected frequency of spikes by which inputs are scaled, in hertz.

        Args:
            value (float): new frequency scale for inputs.

        Returns:
            float: present frequency scale for inputs.
        """
        return self.__frequency_scale

    @frequency.setter
    def frequency(self, value: float) -> None:
        self.__frequency_scale = argtest.gte("frequency", value, 0, float)

    def forward(
        self, inputs: torch.Tensor, online: bool = False
    ) -> torch.Tensor | Iterator[torch.Tensor]:
        r"""Generates a spike train from inputs.

        The spike trains are generated with frequencies scaled linearly by the input,
        with a maximum frequency equal to the hyperparameter defined on initialization.

        Args:
            inputs (torch.Tensor): intensities, scaled :math:`[0, 1]`,
                for spike frequencies.
            online (bool, optional): if spike generation should be computed separately
                at each time step. Defaults to ``False``.

        Returns:
            torch.Tensor | Iterator[torch.Tensor]: tensor spike train (if not online)
            otherwise a generator which yields time slices of the spike train.

        .. admonition:: Shape
            :class: tensorshape

            ``inputs``:

            :math:`B \times N_0 \times \cdots`

            ``return (online=False)``:

            :math:`S \times B \times N_0 \times \cdots`

            ``yield (online=True)``:

            :math:`B \times N_0 \times \cdots`


            Where:
                * :math:`B` is the batch size.
                * :math:`N_0, \ldots` are the dimensions of the spikes being generated.
                * :math:`S` is the number of steps for which to generate spikes, ``steps``.
        """
        if online:
            return nf.poisson_interval_online(
                self.frequency * inputs,
                steps=self.steps,
                step_time=self.dt,
                generator=self.generator,
            )
        else:
            return nf.poisson_interval(
                self.frequency * inputs,
                steps=self.steps,
                step_time=self.dt,
                generator=self.generator,
            )
