from inferno._internal import numeric_limit
import torch


class StepTimeMixin:
    r"""Mixin for encoders with a base step time.

    Args:
        step_time (float): length of a simulation time step, in :math:`ms`.
    """

    def __init__(self, step_time: float):
        # encoder attributes
        self.step_time, e = numeric_limit("step_time", step_time, 0, "gt", float)
        if e:
            raise e

    @property
    def dt(self) -> float:
        r"""Length of the simulation time step, in milliseconds.

        Args:
            value (float): new simulation time step length.

        Returns:
            float: present simulation time step length.
        """
        return self.step_time

    @dt.setter
    def dt(self, value: float) -> None:
        self.step_time, e = numeric_limit("dt", value, 0, "gt", float)
        if e:
            raise e


class StepMixin(StepTimeMixin):
    r"""Mixin for encoders with a globally meaningful number of steps.

    Args:
        step_time (float): length of a simulation time step, in :math:`ms`.
        steps (int): number of steps over which to generate a spike train.
    """

    def __init__(self, step_time: float, steps: int):
        # call superclass mixin constructor
        StepTimeMixin.__init__(self, step_time)

        # encoder attributes
        self.num_steps, e = numeric_limit("steps", steps, 0, "gt", int)
        if e:
            raise e

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
        self.num_steps, e = numeric_limit("steps", value, 0, "gt", int)
        if e:
            raise e

    @property
    def duration(self) -> float:
        r"""Length of simulated time for which to generate a spike train, in milliseconds.

        Returns:
            float: length of simulation time for which to generate a spike train.
        """
        return self.steps * self.dt


class RefractoryStepMixin(StepMixin):
    r"""Mixin for encoders with a refractory period and a notion of global step.

    Args:
        step_time (float): length of a simulation time step, in :math:`ms`.
        steps (int): number of steps over which to generate a spike train.
        refrac (float): refractory period, in :math:`\text{ms}`.
    """

    def __init__(self, step_time: float, steps: int, refrac: float | None):
        # call superclass mixin constructor
        StepMixin.__init__(self, step_time, steps)

        # encoder attributes
        if refrac is None:
            self.autorefrac = True
            self.interval_min = self.dt
        else:
            self.autorefrac = False
            self.interval_min, e = numeric_limit("refrac", refrac, 0, "gte", float)
            if e:
                raise e

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
            self.interval_min = self.dt

    @property
    def refrac(self) -> float:
        r"""Length of the refractory period, in milliseconds.

        Args:
            value (float | None): new refractory period length,
                pins to the step time if None.

        Returns:
            float: present refractory period length.
        """
        return self.interval_min

    @refrac.setter
    def refrac(self, value: float | None) -> None:
        if value is None:
            self.autorefrac = True
            self.interval_min = self.dt
        else:
            self.autorefrac = False
            self.interval_min, e = numeric_limit("refrac", value, 0, "gte", float)
            if e:
                raise e


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
