from ... import scalar
from ..._internal import argtest
import torch


class StepTimeMixin:
    r"""Mixin for encoders with a base step time.

    Args:
        step_time (float): length of a simulation time step, in :math:`ms`.

    Note:
        This creates the property :py:attr:`dt` for managing step time using Python
        built-ins and the attribute ``step_time`` for access as a :py:class`~torch.Tensor`.
    """

    def __init__(self, step_time: float):
        # encoder attributes
        self.register_buffer(
            "step_time",
            torch.tensor(argtest.gt("step_time", step_time, 0, float)),
            persistent=False,
        )

    @property
    def dt(self) -> float:
        r"""Length of the simulation time step, in milliseconds.

        Args:
            value (float): new simulation time step length.

        Returns:
            float: present simulation time step length.
        """
        return float(self.step_time)

    @dt.setter
    def dt(self, value: float) -> None:
        self.step_time = scalar(argtest.gt("dt", value, 0, float), self.step_time)


class StepMixin(StepTimeMixin):
    r"""Mixin for encoders with a globally meaningful number of steps.

    Args:
        step_time (float): length of a simulation time step, in :math:`ms`.
        steps (int): number of steps over which to generate a spike train.

    Note:
        This creates the property :py:attr:`dt` for managing step time using Python
        built-ins and the attribute ``step_time`` for access as a :py:class`~torch.Tensor`.
    """

    def __init__(self, step_time: float, steps: int):
        # call superclass mixin constructor
        StepTimeMixin.__init__(self, step_time)

        # encoder attributes
        self.num_steps = argtest.gt("steps", steps, 0, int)

    @property
    def steps(self) -> int:
        r"""Number of steps for which a spike train should be generated.

        Args:
            value (int): new number of steps over which to generate.

        Returns:
            int: present number of steps over which to generate.
        """
        return self.num_steps

    @steps.setter
    def steps(self, value: int) -> None:
        self.num_steps = argtest.gt("steps", value, 0, int)

    @property
    def duration(self) -> float:
        r"""Length of simulated time for which to generate a spike train, in milliseconds.

        Returns:
            float: length of simulation time for which to generate a spike train.
        """
        return float(self.steps * self.dt)


class RefractoryStepMixin(StepMixin):
    r"""Mixin for encoders with a refractory period and a notion of global step.

    Args:
        step_time (float): length of a simulation time step, in :math:`ms`.
        steps (int): number of steps over which to generate a spike train.
        refrac (float): refractory period, in :math:`\text{ms}`.

    Note:
        This creates the properties :py:attr:`dt` and :py:attr:`refrac` for managing
        step time and refractory period using Python built-ins and the attributes
        ``step_time`` and ``interval_min`` for access as a :py:class`~torch.Tensor`.

    """

    def __init__(self, step_time: float, steps: int, refrac: float | None):
        # call superclass mixin constructor
        StepMixin.__init__(self, step_time, steps)

        # encoder attributes
        if refrac is None:
            self.autorefrac = True
            self.register_buffer(
                "interval_min",
                torch.tensor(self.dt),
                persistent=False,
            )
        else:
            self.autorefrac = False
            self.register_buffer(
                "interval_min",
                torch.tensor(argtest.gte("refrac", refrac, 0, float)),
                persistent=False,
            )

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
        if self.autorefrac:
            self.interval_min = scalar(self.dt, self.interval_min)

    @property
    def refrac(self) -> float:
        r"""Length of the refractory period, in milliseconds.

        Args:
            value (float | None): new refractory period length,
                pins to the step time if None.

        Returns:
            float: present refractory period length.
        """
        return float(self.interval_min)

    @refrac.setter
    def refrac(self, value: float | None) -> None:
        if value is None:
            self.autorefrac = True
            self.interval_min = scalar(self.dt, self.interval_min)
        else:
            self.autorefrac = False
            self.interval_min = scalar(
                argtest.gte("refrac", value, 0, float), self.interval_min
            )


class GeneratorMixin:
    r"""Mixin for encoders with a random number generator.

    Args:
        generator (torch.Generator | None): random number generator to use.
    """

    def __init__(self, generator: torch.Generator | None):
        self.rng = generator

    @property
    def generator(self) -> torch.Generator | None:
        r"""PyTorch random number generator.

        Args:
            value (torch.Generator | None): new random number generator.

        Returns:
            float: present random number generator.
        """
        return self.rng

    @generator.setter
    def generator(self, value: torch.Generator | None) -> None:
        self.rng = value
