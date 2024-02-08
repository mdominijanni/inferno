from ..base import Encoder
from .. import functional as nf
from .mixins import GeneratorMixin, RefractoryMixin
from inferno._internal import numeric_limit
import torch
from typing import Iterator


class HomogeneousPoissonEncoder(GeneratorMixin, RefractoryMixin, Encoder):
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
        Encoder.__init__(self, step_time=step_time, steps=steps)

        # set encoder attributes
        self.freqscale, e = numeric_limit("frequency", frequency, 0, "gte", float)
        if e:
            raise e

        self.refrac_compensated = compensate

        # call mixin constructors
        RefractoryMixin.__init__(self, refrac)
        GeneratorMixin.__init__(self, generator=generator)

        # check for invalid configuration
        e = self.valid
        if e:
            raise e

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
        self.refrac_compensated = value

        # test that the state is still valid
        e = self.valid
        if e:
            raise e

    @property
    def frequency(self) -> float:
        r"""Expected frequency of spikes by which inputs are scaled, in hertz.

        Args:
            value (float): new frequency scale for inputs.

        Returns:
            float: present frequency scale for inputs.
        """
        return self.frequency

    @frequency.setter
    def frequency(self, value: float) -> None:
        self.frequency, e = numeric_limit("value", value, 0, "gte", float)
        if e:
            raise e

        # test that the state is still valid
        e = self.valid
        if e:
            raise e

    @property
    def refracvalid(self) -> None | Exception:
        r"""If the current refractory period is valid.

        An exception will be returned if the refractory period is greater than
        or equal to the maximum expected time between spikes, and refractory periods
        are compensated for.

        Returns:
            None | Exception: None if the refractory period is valid, otherwise an
            exception to raise.
        """
        if self.compensated:
            if self.refrac >= 1000 / self.frequency:
                raise RuntimeError(
                    f"in {type(self).__name__}, if the refrac period is compensated, "
                    "refractory period must be strictly less than the "
                    f"expected interspike interval of {1000 / self.frequency}."
                )

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
            return nf.encode_interval_poisson_online(
                self.frequency * inputs,
                step_time=self.dt,
                steps=self.steps,
                refrac=self.refrac,
                compensate=self.compensated,
                generator=self.generator,
            )
        else:
            return nf.encode_interval_poisson(
                self.frequency * inputs,
                step_time=self.dt,
                steps=self.steps,
                refrac=self.refrac,
                compensate=self.compensated,
                generator=self.generator,
            )


class PoissonIntervalEncoder(GeneratorMixin, Encoder):
    r"""Encoder to generate spike trains with intervals sampled from a Poisson distribution.

    This is included to replicate BindsNET's Poisson spike generation. The intervals
    between spikes follow the Poisson distribution parameterized with the inverse of
    the expected rate (i.e. the scale is given as the rate).

    Args:
        step_time (float): length of time between outputs, :math:`\Delta t`,
            in :math:`\text{ms}`.
        steps (int): number of steps for which to generate spikes, :math:`S`.
        frequency (float): maximum spike frequency (associated with an input of 1),
            :math:`f`, in :math:`\text{Hz}`.
        generator (torch.Generator | None, optional): pseudorandom number generator
            for sampling. Defaults to None.
    """

    def __init__(
        self,
        step_time: float,
        steps: int,
        frequency: float,
        *,
        generator: torch.Generator | None = None,
    ):
        # call superclass constructor
        Encoder.__init__(self, step_time=step_time, steps=steps)

        # set encoder attributes
        self.freqscale, e = numeric_limit("frequency", frequency, 0, "gte", float)
        if e:
            raise e

        # call mixin constructor
        GeneratorMixin.__init__(self, generator=generator)

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
        self.freqscale, e = numeric_limit("value", value, 0, "gte", float)
        if e:
            raise e

        # test that the state is still valid
        e = self.valid
        if e:
            raise e

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
            return nf.encode_poisson_spaced_online(
                self.frequency * inputs,
                step_time=self.dt,
                steps=self.steps,
                generator=self.generator,
            )
        else:
            return nf.encode_poisson_spaced(
                self.frequency * inputs,
                step_time=self.dt,
                steps=self.steps,
                generator=self.generator,
            )
