from __future__ import annotations
from functools import cached_property
import math
import torch


class VoltageMixin:
    @property
    def voltage(self) -> torch.Tensor:
        r"""Membrane voltages of the neurons, in millivolts.

        Args:
            value (torch.Tensor): new membrane voltages.

        Returns:
            torch.Tensor: current membrane voltages.
        """
        return self.voltages.data

    @voltage.setter
    def voltage(self, value: torch.Tensor):
        self.voltages.data[:] = value


class RefractoryMixin:
    @property
    def spike(self) -> torch.Tensor:
        r"""Which neurons generated an action potential on the last simulation step.

        Returns:
            torch.Tensor: if the correspond neuron generated an action potential last step.
        """
        return self.refracs.data == self.refrac_t

    @property
    def refrac(self) -> torch.Tensor:
        r"""Remaining refractory periods of the neurons, in milliseconds.

        Returns:
            torch.Tensor: current remaining refractory periods.
        """
        return self.refracs.data


class WeightBiasDelayMixin:
    @property
    def weight(self) -> torch.Tensor:
        r"""Learnable weights of the connection.

        Args:
            value (torch.Tensor): new weights.

        Returns:
            torch.Tensor: current weights.
        """
        return self.weights.data

    @weight.setter
    def weight(self, value: torch.Tensor):
        self.weights.data = value

    @property
    def bias(self) -> torch.Tensor | None:
        r"""Learnable biases of the connection.

        Args:
            value (torch.Tensor): new biases.

        Returns:
            torch.Tensor | None: current biases, if the connection has any.

        Raises:
            RuntimeError: ``bias`` cannot be set on a connection without learnable biases.
        """
        if self.biases is not None:
            return self.biases.data

    @bias.setter
    def bias(self, value: torch.Tensor):
        if self.biases is not None:
            self.biases.data = value
        else:
            raise RuntimeError(
                f"cannot set `bias` on a {type(self).__name__} without trainable biases"
            )

    @property
    def delay(self) -> torch.Tensor | None:
        r"""Learnable delays of the connection.

        Args:
            value (torch.Tensor): new delays.

        Returns:
            torch.Tensor | None: current delays, if the connection has any.

        Raises:
            RuntimeError: ``delay`` cannot be set on a connection without learnable delays.
        """
        if self.delays is not None:
            return self.delays.data

    @delay.setter
    def delay(self, value: torch.Tensor):
        if self.delays is not None:
            self.delays.data = value
        else:
            raise RuntimeError(
                f"cannot set `delay` on a {type(self).__name__} without trainable delays"
            )


class BatchMixin:
    """Mixin for modules with batch-size dependent parameters or buffers.

    Args:
        batch_size (int): initial batch size.

    Raises:
        RuntimeError: batch size must be positive.
        RuntimeError: object cannot already have a constraint on the zeroth dimension.

    Note:
        This must be added to a class which inherits from :py:class:`DimensionalModule`.

    Note:
        The mixin constructor must be called after the constructor for the class
        which calls the :py:class:`~inferno.DimensionalModule` constructor is.
    """
    def __init__(
        self,
        batch_size: int,
    ):
        # cast batch_size as float
        batch_size = float(batch_size)

        # check that batch size is valid
        if batch_size < 1:
            raise RuntimeError(f"batch size must be positive, received {batch_size}")

        # check that a conflicting constraint doesn't exist
        if 0 in self.constraints:
            raise RuntimeError(f"{type(self).__name__} object already contains"
                               f"constraint size={self.constraints[0]} at dim={0}.")

        # register new constraint
        self.reconstrain(dim=0, size=batch_size)

    @property
    def bsize(self) -> int:
        return self.constraints.get(0)

    @bsize.setter
    def bsize(self, value: int) -> None:
        # cast value as float
        value = int(value)

        # ensure valid batch size
        if value < 1:
            raise RuntimeError(f"batch size must be positive, received {value}")

        # reconstrain if required
        if value != self.bsize:
            self.reconstrain(0, value)


class ShapeMixin(BatchMixin):
    """Mixin for modules a concept of shape and with batch-size dependent parameters or buffers.

    Args:
        shape (tuple[int, ...] | int): shape of the group being represented, excluding batch size.
        batch_size (int): initial batch size.

    Raises:
        RuntimeError: batch size must be positive.

    Note:
        This must be added to a class which inherits from :py:class:`DimensionalModule`.

    Note:
        The mixin constructor must be called after the constructor for the class
        which calls the :py:class:`~inferno.DimensionalModule` constructor is.
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
        return (self.bsize,) + self._shape

    @cached_property
    def count(self) -> int:
        r"""Number of elements in the module, excluding batch.

        Returns:
            int: number of elements in the module.
        """
        return math.prod(self._shape)
