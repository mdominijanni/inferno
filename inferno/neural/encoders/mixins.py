from ..base import Encoder
from inferno._internal import numeric_limit, instance_of
import torch


class RefractoryMixin:
    r"""Mixin for encoders with a refractory period.

    Args:
        refrac (float): refractory period, in :math:`\text{ms}`.

    Caution:
        This must be added to a class which inherits from
        :py:class:`Encoder`, and the constructor for this
        mixin must be called after the module constructor.
    """

    def __init__(self, refrac: float | None):
        # check for correct class
        e = instance_of("self", self, Encoder)
        if e:
            raise e

        # set refractory period
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
        return Encoder.dt.fget(self)

    @dt.setter
    def dt(self, value: float) -> None:
        Encoder.dt.fset(self, value)
        if self.autorefrac:
            self.interval_min = self.dt

        # test that the state is still valid
        e = self.refracvalid
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

        # test that the state is still valid
        e = self.refracvalid
        if e:
            raise e

    @property
    def refracvalid(self) -> None | Exception:
        r"""If the current refractory period is valid.

        Returns:
            None | Exception: None if the refractory period is valid, otherwise an
            exception to raise.

        Important:
            This will always return None. If a limitation should exist, the subclass
            should implement this.
        """
        return None


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
