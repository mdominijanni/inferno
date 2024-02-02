from functools import cached_property
from inferno import DimensionalModule
from inferno._internal import instance_of, numeric_limit, multiple_numeric_limit
import math


class BatchMixin:
    """Mixin for modules with batch-size dependent parameters or buffers.

    Args:
        batch_size (int): initial batch size.

    Caution:
        This must be added to a class which inherits from
        :py:class:`DimensionalModule`, and the constructor for this
        mixin must be called after the module constructor.

    Caution:
        Module which inherit from this mixin cannot be constrained on ``dim=0``.
        Do not manually change this constraint, it is managed through :py:attr:`bsize`.
    """

    def __init__(
        self,
        batch_size: int,
    ):
        e = instance_of("self", self, DimensionalModule)
        if e:
            raise e
        batch_size, e = numeric_limit("batch_size", batch_size, 0, "gt", int)
        if e:
            raise e

        # check that a conflicting constraint doesn't exist
        if 0 in self.constraints:
            raise RuntimeError(
                f"{type(self).__name__} `self` already contains"
                f"constraint size={self.constraints[0]} at dim={0}."
            )

        # register new constraint
        self.reconstrain(dim=0, size=batch_size)

    @property
    def bsize(self) -> int:
        r"""Batch size of the module.

        Args:
            value (int): new batch size.

        Returns:
            int: present batch size.
        """
        return self.constraints.get(0)

    @bsize.setter
    def bsize(self, value: int) -> None:
        value, e = numeric_limit("bsize", value, 0, "gt", int)
        if e:
            raise e
        if value != self.bsize:
            self.reconstrain(0, value)


class ShapeMixin(BatchMixin):
    """Mixin for modules with a concept of shape and with batch-size dependencies.

    Args:
        shape (tuple[int, ...] | int): shape of the group being represented,
            excluding batch size.
        batch_size (int): initial batch size.

    Caution:
        This must be added to a class which inherits from
        :py:class:`DimensionalModule`, and the constructor for this
        mixin must be called after the module constructor.

    Caution:
        Module which inherit from this mixin cannot be constrained on ``dim=0``.
        Do not manually change this constraint, it is managed through :py:attr:`bsize`.

    Note:
        This sets an attribute ``_shape`` which is managed internally.
    """

    def __init__(self, shape: tuple[int, ...] | int, batch_size: int):
        # call superclass constructor
        BatchMixin.__init__(self, batch_size)

        # validate and set shape
        try:
            self._shape, e = numeric_limit("shape", shape, 0, "gt", int)
            if e:
                raise e
            self._shape = (self._shape,)
        except TypeError:
            self._shape, e = multiple_numeric_limit("shape", shape, 0, "gt", int, False)
            if e:
                raise e

    @property
    def bsize(self) -> int:
        r"""Batch size of the module.

        Args:
            value (int): new batch size.

        Returns:
            int: present batch size.
        """
        return BatchMixin.bsize.fget(self)

    @bsize.setter
    def bsize(self, value: int) -> None:
        BatchMixin.bsize.fset(self, value)
        if hasattr(self, "bshape"):
            del self.bshape

    @property
    def shape(self) -> tuple[int, ...]:
        r"""Shape of the module.

        Returns:
            tuple[int, ...]: Shape of the module.
        """
        return self._shape

    @cached_property
    def bshape(self) -> tuple[int, ...]:
        r"""Batch shape of the module

        Returns:
            tuple[int, ...]: Shape of the module, including the batch dimension.
        """
        return (self.bsize,) + self.shape

    @cached_property
    def count(self) -> int:
        r"""Number of elements in the module, excluding replication along the batch dim.

        Returns:
            int: number of elements in the module.
        """
        return math.prod(self._shape)
