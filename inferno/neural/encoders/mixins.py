from ..._internal import argtest
import torch


class StepTimeMixin:
    r"""Mixin for encoders with a base step time.

    Args:
        step_time (float): length of a simulation time step, in :math:`ms`.
    """

    def __init__(self, step_time: float):
        self.__step_time = argtest.gt("step_time", step_time, 0, float)

    @property
    def dt(self) -> float:
        r"""Length of the simulation time step, in milliseconds.

        Args:
            value (float): new simulation time step length.

        Returns:
            float: present simulation time step length.
        """
        return self.__step_time

    @dt.setter
    def dt(self, value: float) -> None:
        self.__step_time = argtest.gt("dt", value, 0, float)


class StepMixin(StepTimeMixin):
    r"""Mixin for encoders with a globally meaningful number of steps.

    Args:
        steps (int): number of steps over which to generate a spike train.
    """

    def __init__(self, steps: int, step_time: float):
        StepTimeMixin.__init__(self, step_time)
        self.__num_steps = argtest.gt("steps", steps, 0, int)

    @property
    def steps(self) -> int:
        r"""Number of steps for which a spike train should be generated.

        Args:
            value (int): new number of steps over which to generate.

        Returns:
            int: present number of steps over which to generate.
        """
        return self.__num_steps

    @steps.setter
    def steps(self, value: int) -> None:
        self.__num_steps = argtest.gt("steps", value, 0, int)

    @property
    def duration(self) -> float:
        r"""Length of simulated time for which to generate a spike train, in milliseconds.

        Returns:
            float: length of simulation time for which to generate a spike train.
        """
        return self.__num_steps * self.dt


class RefractoryStepMixin(StepMixin):
    r"""Mixin for encoders with a refractory period and a notion of global step.

    Args:
        steps (int): number of steps over which to generate a spike train.
        step_time (float): length of a simulation time step, in :math:`ms`.
        refrac (float): refractory period, in :math:`\text{ms}`.
    """

    def __init__(self, steps: int, step_time: float, refrac: float | None):
        # call superclass mixin constructor
        StepMixin.__init__(self, steps, step_time)

        # encoder attributes
        if refrac is None:
            self.__derive_refrac = True
            self.__refrac_time = self.dt
        else:
            self.__derive_refrac = False
            self.__refrac_time = argtest.gte("refrac", refrac, 0, float)

    @property
    def dt(self) -> float:
        r"""Length of the simulation time step, in milliseconds.

        Args:
            value (float): new simulation time step length.

        Returns:
            float: present simulation time step length.
        """
        return StepMixin.dt.fget(self)

    @dt.setter
    def dt(self, value: float) -> None:
        StepMixin.dt.fset(self, value)
        if self.__derive_refrac:
            self.__refrac_time = StepMixin.dt.fget(self)

    @property
    def refrac(self) -> float:
        r"""Length of the refractory period, in milliseconds.

        Args:
            value (float | None): new refractory period length, pins to the
                step time if ``None``.

        Returns:
            float: present refractory period length.
        """
        return self.__refrac_time

    @refrac.setter
    def refrac(self, value: float | None) -> None:
        if value is None:
            self.__derive_refrac = True
            self.__refrac_time = self.__step_time
        else:
            self.__derive_refrac = False
            self.__refrac_time = argtest.gte("refrac", value, 0, float)


class GeneratorMixin:
    r"""Mixin for encoders with a random number generator.

    Args:
        generator (torch.Generator | None): random number generator to use.
    """

    def __init__(self, generator: torch.Generator | None):
        self.__rng = generator

    @property
    def generator(self) -> torch.Generator | None:
        r"""PyTorch random number generator.

        Args:
            value (torch.Generator | None): new random number generator.

        Returns:
            float: present random number generator.
        """
        return self.__rng

    @generator.setter
    def generator(self, value: torch.Generator | None) -> None:
        self.__rng = value
