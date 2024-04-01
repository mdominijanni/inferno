from .. import ShapedTensor, RecordTensor
from .._internal import argtest
import math


class BatchMixin:
    """Mixin for modules with batch-size dependent parameters or buffers.

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

        Each attribute must specify the name of a :py:class:`ShapedTensor`.

        Args:
            *attr (str): names of the attribute to set as batched.
        """
        for a in attr:
            if not hasattr(self, a):
                raise RuntimeError(f"no attribute '{a}' exists")
            elif not isinstance(getattr(self, a), ShapedTensor):
                raise TypeError(
                    f"attribute '{a}' specifies a {type(getattr(self, a).__name__)}, not a ShapedTensor"
                )
            else:
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
    """Mixin for modules with a concept of shape.

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
    """Mixin for modules with a concept of shape and with batch-size dependencies.

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


class RecordMixin:
    """Mixin for modules with one or more records with shared step time and duration.

    Args:
        step_time (float): length of time between stored values in the record.
        duration (float): length of time over which prior values are stored.
        *constrained (str): names of :py:class:`RecordTensor` attributes which are
            batch-size constrained.

    Caution:
        :py:class:`RecordTensor` attributes must be added as attributes prior to
        initialization.
    """

    def __init__(self, step_time: float, duration: float, *constrained: str):
        # validate and set step time and duration
        self.__step_time = argtest.gt("step_time", step_time, 0, float)
        self.__duration = argtest.gte("duration", duration, 0, float)

        # validate and set constrained tensors
        for attr in constrained:
            if not isinstance(getattr(self, attr, None), RecordTensor):
                raise RuntimeError(
                    f"attribute name '{attr}' in 'constrained' is either not an "
                    "attribute of self or is not a RecordTensor"
                )
        self.__constrained = constrained
