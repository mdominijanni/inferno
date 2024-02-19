from .. import functional as nf
from .mixins import GeneratorMixin, RefractoryStepMixin
from ... import Module
from inferno._internal import numeric_limit
import torch
from typing import Iterator


class HomogeneousPoissonEncoder(GeneratorMixin, RefractoryStepMixin, Module):
    r"""Encoder to generate spike trains sampled from a Poisson distribution.

    This method samples randomly from an exponential distribution (the interval
    between samples in a Poisson point process), adding an additional refractory
    period and compensating the rate.

    Args:
        step_time (float): length of time between outputs, :math:`\Delta t`,
            in :math:`\text{ms}`.
        steps (int): number of steps for which to generate spikes, :math:`S`.
        frequency (float): maximum spike frequency (associated with an input of 1),
            :math:`f`, in :math:`\text{Hz}`.
        refrac (float | None, optional): minimum interal between spikes set to the step
            time if None, in :math:`\text{ms}`. Defaults to None.
        compensate (bool, optional): if the spike generation rate should be compensate
            for the refractory period. Defaults to True.
        generator (torch.Generator | None, optional): pseudorandom number generator
            for sampling. Defaults to None.

    Note:
        ``refrac`` at its default still allows for a spike to be generated at every
        step (since the distance between is :math:`\Delta t`). To get behavior where
        at most every :math:`n^\text{th}` step is a spike, the refractory period needs
        to be set to :math:`n \Delta t`.
    """

    def __init__(
        self,
        step_time: float,
        steps: int,
        frequency: float,
        *,
        refrac: float | None = None,
        compensate: bool = True,
        generator: torch.Generator | None = None,
    ):
        # call superclass constructor
        Module.__init__(self)

        # set encoder attributes
        self.freqscale, e = numeric_limit("frequency", frequency, 0, "gte", float)
        if e:
            raise e

        self.refrac_compensated = bool(compensate)

        # call mixin constructors
        RefractoryStepMixin.__init__(
            self, step_time=step_time, steps=steps, refrac=refrac
        )
        GeneratorMixin.__init__(self, generator=generator)

    @property
    def compensated(self) -> bool:
        r"""If the spike frequency compensates for the refractory period.

        Args:
            value (bool): if the spike frequency compensates for the refractory period.

        Returns:
            bool: if the spike frequency compensates for the refractory period.
        """
        return self.refrac_compensated

    @compensated.setter
    def compensated(self, value: bool) -> None:
        # refrac-frequency compatibility test
        if value:
            _, e = numeric_limit(
                "frequency * refrac", self.frequency * self.refrac, 1000, "lt", float
            )
            if e:
                raise e

        self.refrac_compensated = bool(value)

    @property
    def frequency(self) -> float:
        r"""Expected frequency of spikes by which inputs are scaled, in hertz.

        Args:
            value (float): new frequency scale for inputs.

        Returns:
            float: present frequency scale for inputs.
        """
        return self.freqscale

    @frequency.setter
    def frequency(self, value: float) -> None:
        # refrac-frequency compatibility test
        if self.compensated:
            _, e = numeric_limit(
                "frequency * refrac", value * self.refrac, 1000, "lt", float
            )
            if e:
                raise e

        self.freqscale, e = numeric_limit("frequency", value, 0, "gte", float)
        if e:
            raise e

    @property
    def refrac(self) -> float:
        r"""Length of the refractory period, in milliseconds.

        Args:
            value (float | None): new refractory period length,
                pins to the step time if None.

        Returns:
            float: present refractory period length.
        """
        return RefractoryStepMixin.refrac.fget(self)

    @refrac.setter
    def refrac(self, value: float | None) -> None:
        # refrac-frequency compatibility test
        if self.compensated:
            newrefrac = self.dt if value is None else value
            _, e = numeric_limit(
                "refrac * frequency ", newrefrac * self.frequency, 1000, "lt", float
            )
            if e:
                raise e

        return RefractoryStepMixin.refrac.fset(self, value)

    def forward(
        self, inputs: torch.Tensor, online: bool = False
    ) -> torch.Tensor | Iterator[torch.Tensor]:
        r"""Generates a spike train from inputs.

        Args:
            inputs (torch.Tensor): intensities, scaled :math:`[0, 1]`,
                for spike frequencies.
            online (bool, optional): if spike generation should be computed separately
                at each time step. Defaults to False.

        Returns:
            torch.Tensor | Iterator[torch.Tensor]: tensor spike train (if not online)
            otherwise a generator which yields time slices of the spike train.

        Note:
            Values in ``inputs`` should be on the interval :math:`[0, 1]`. Where the
            inputs are ``0``, no spikes will be generated. Where the inputs are ``1``,
            spikes will be generated with a frequency of :py:attr:`frequency`.

        Note:
            In general, setting ``online`` to ``False`` will be faster but more
            more memory-intensive.

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
            return nf.enc_homogeneous_poisson_exp_interval_online(
                self.frequency * inputs,
                step_time=self.dt,
                steps=self.steps,
                refrac=self.refrac,
                compensate=self.compensated,
                generator=self.generator,
            )
        else:
            return nf.enc_homogeneous_poisson_exp_interval(
                self.frequency * inputs,
                step_time=self.dt,
                steps=self.steps,
                refrac=self.refrac,
                compensate=self.compensated,
                generator=self.generator,
            )
