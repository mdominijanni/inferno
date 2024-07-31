from .. import ShapedTensor, RecordTensor
from .._internal import argtest
import math


class BatchMixin:
    r"""Mixin for modules with batch-size dependent parameters or buffers.

    Attributes which have are registered as constrained this way will have a constraint
    on their 0th dimension equal to the batch size placed.

    Args:
        batch_size (int): initial batch size.
    """

    def __init__(self, batch_size: int):
        self.__batch_size = argtest.gt("batch_size", batch_size, 0, int)
        self.__constrained = set()

    def add_batched(self, *attr: str) -> None:
        r"""Add batch-dependent attributes.

        Each attribute must specify the name of a :py:class:`~inferno.ShapedTensor`.

        Args:
            *attr (str): names of the attributes to set as batched.
        """
        for a in attr:
            if not hasattr(self, a):
                raise RuntimeError(f"no attribute '{a}' exists")
            elif not isinstance(getattr(self, a), ShapedTensor):
                raise TypeError(
                    f"attribute '{a}' specifies a {type(getattr(self, a).__name__)}, not a ShapedTensor"
                )
            else:
                getattr(self, a).reconstrain(0, self.__batch_size)
                self.__constrained.add(a)

    @property
    def batchsz(self) -> int:
        r"""Batch size of the module.

        Args:
            value (int): new batch size.

        Returns:
            int: present batch size.
        """
        return self.__batch_size

    @batchsz.setter
    def batchsz(self, value: int) -> None:
        value = argtest.gt("batchsz", value, 0, int)
        if value != self.__batch_size:
            for cstr in self.__constrained:
                getattr(self, cstr).reconstrain(0, value)
            self.__batch_size = value


class ShapeMixin(BatchMixin):
    r"""Mixin for modules with a concept of shape.

    This mixin does not provide options for altering shape, only storing the given
    shape and offering related properties.

    Args:
        shape (tuple[int, ...] | int): shape of the group being represented,
            excluding batch size.
    """

    def __init__(
        self,
        shape: tuple[int, ...] | int,
    ):
        # validate and set shape
        try:
            self.__shape = (argtest.gt("shape", shape, 0, int),)
        except TypeError:
            self.__shape = argtest.ofsequence("shape", shape, argtest.gt, 0, int)

    @property
    def shape(self) -> tuple[int, ...]:
        r"""Shape of the module.

        Returns:
            tuple[int, ...]: Shape of the module.
        """
        return self.__shape

    @property
    def count(self) -> int:
        r"""Number of elements in the module, excluding replication along the batch dim.

        Returns:
            int: number of elements in the module.
        """
        return math.prod(self.__shape)


class BatchShapeMixin(ShapeMixin, BatchMixin):
    r"""Mixin for modules with a concept of shape and with batch-size dependencies.

    Args:
        shape (tuple[int, ...] | int): shape of the group being represented,
            excluding batch size.
        batch_size (int): initial batch size.
    """

    def __init__(self, shape: tuple[int, ...] | int, batch_size: int):
        # call superclass constructors
        ShapeMixin.__init__(self, shape)
        BatchMixin.__init__(self, batch_size)

    @property
    def batchedshape(self) -> tuple[int, ...]:
        r"""Batch shape of the module

        Returns:
            tuple[int, ...]: Shape of the module, including the batch dimension.
        """
        return (self.batchsz,) + self.shape


class DelayedMixin:
    r"""Mixin for modules with delay-record tensors with shared step time and duration.

    Attributes which have are registered as constrained this way will have a constraint
    on their final dimension equal to the computed record size.

    Args:
        step_time (float): length of time between stored values in the record.
        delay (float): length of time over which prior values are stored.

    Caution:
        :py:class:`~inferno.RecordTensor` attributes must be added as attributes prior to
        initialization.
    """

    def __init__(self, step_time: float, delay: float):
        self.__step_time = argtest.gt("step_time", step_time, 0, float)
        self.__delay = argtest.gte("delay", delay, 0, float)
        self.__constrained = set()

    def add_delayed(self, *attr: str) -> None:
        r"""Add delay-dependent attributes.

        Each attribute must specify the name of a :py:class:`~inferno.RecordTensor`.

        Args:
            *attr (str): names of the attributes to set as batched.
        """
        for a in attr:
            if not hasattr(self, a):
                raise RuntimeError(f"no attribute '{a}' exists")
            elif not isinstance(getattr(self, a), RecordTensor):
                raise TypeError(
                    f"attribute '{a}' specifies a {type(getattr(self, a).__name__)}, not a RecordTensor"
                )
            else:
                getattr(self, a).dt = self.__step_time
                getattr(self, a).duration = self.__delay
                getattr(self, a).inclusive = True
                self.__constrained.add(a)

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
        value = argtest.gt("dt", value, 0, float)
        if value != self.__step_time:
            for cstr in self.__constrained:
                getattr(self, cstr).dt = value
            self.__step_time = value

    @property
    def delay(self) -> float:
        r"""Maximum supported delay, in milliseconds.

        Returns:
            float: maximum supported delay.
        """
        return self.__delay

    @delay.setter
    def delay(self, value: float) -> None:
        value = argtest.gte("delay", value, 0, float)
        if value != self.__delay:
            for cstr in self.__constrained:
                getattr(self, cstr).duration = value + self.__step_time
            self.__delay = value
