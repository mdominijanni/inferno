from functools import cached_property
from inferno import DimensionalModule
from inferno._internal import instance_of, numeric_limit
import math


class BatchMixin:
    """Mixin for modules with batch-size dependent parameters or buffers.

    Args:
        batch_size (int): initial batch size.

    Note:
        This must be added to a class which inherits from
        :py:class:`DimensionalModule`, and the constructor for this
        mixin must be called after the module constructor.

    Note:
        Module which inherit from this mixin cannot be constrained on ``dim=0``.
        Do not manually change this constraint, it is managed through :py:attr:`bsize`.
    """

    def __init__(
        self,
        batch_size: int,
    ):
        instance_of("`self`", self, DimensionalModule)
        batch_size = numeric_limit("`batch_size`", batch_size, 0, "gt", int)

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
        value = numeric_limit("`bsize`", value, 0, "gt", int)
        if value != self.bsize:
            self.reconstrain(0, value)


class ShapeMixin(BatchMixin):
    """Mixin for modules with a concept of shape and with batch-size dependencies.

    Args:
        shape (tuple[int, ...] | int): shape of the group being represented,
            excluding batch size.
        batch_size (int): initial batch size.

    Note:
        This must be added to a class which inherits from
        :py:class:`DimensionalModule`, and the constructor for this
        mixin must be called after the module constructor.

    Note:
        Module which inherit from this mixin cannot be constrained on ``dim=0``.
        Do not manually change this constraint, it is managed through :py:attr:`bsize`.

    Note:
        This sets an attribute ``_shape`` and is managed internally.
    """

    def __init__(self, shape: tuple[int, ...] | int, batch_size: int):
        # call superclass constructor
        BatchMixin.__init__(self, batch_size)

        # register extras
        try:
            self.register_extra("_shape", (int(shape),))
        except TypeError:
            self.register_extra("_shape", tuple(int(s) for s in shape))

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
        r"""Number of elements in the module, excluding replication along the batch axis.

        Returns:
            int: number of elements in the module.
        """
        return math.prod(self._shape)
