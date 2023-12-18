from __future__ import annotations
from functools import cached_property
import math
import torch
import torch.nn as nn


class AdaptationMixin:
    r"""Mixin for neurons with membrane adaptations.

    Args:
        adaptation (torch.Tensor): initial membrane adaptations.
        requires_grad (bool, optional): if the parameters created require gradients. Defaults to False.

    Note:
        This must be added to a class which inherits from :py:class:`DimensionalModule`.

    Note:
        The mixin constructor must be called after the :py:class:`~inferno.DimensionalModule` constructor.
    """

    def __init__(self, adaptation, requires_grad=False):
        self.register_parameter("adaptation_", nn.Parameter(adaptation, requires_grad))
        self.register_constrained("adaptation_")

    @property
    def adaptation(self) -> torch.Tensor:
        r"""Membrane adaptations.

        Args:
            value (torch.Tensor): membrane adaptations.

        Returns:
            torch.Tensor: membrane adaptations.
        """
        return self.adaptation_.data

    @adaptation.setter
    def adaptation(self, value: torch.Tensor):
        self.adaptation_.data = value


class CurrentMixin:
    def __init__(self, data, requires_grad=False):
        self.register_parameter("current_", nn.Parameter(data, requires_grad))
        self.register_constrained("current_")

    @property
    def current(self) -> torch.Tensor:
        r"""Membrane currents in nanoamperes.

        Args:
            value (torch.Tensor): new membrane currents.

        Returns:
            torch.Tensor: current membrane currents.
        """
        return self.current_.data

    @current.setter
    def current(self, value: torch.Tensor):
        self.current_.data = value


class VoltageMixin:
    def __init__(self, data, requires_grad=False):
        self.register_parameter("voltage_", nn.Parameter(data, requires_grad))
        self.register_constrained("voltage_")

    @property
    def voltage(self) -> torch.Tensor:
        r"""Membrane voltages in millivolts.

        Args:
            value (torch.Tensor): new membrane voltages.

        Returns:
            torch.Tensor: current membrane voltages.
        """
        return self.voltage_.data

    @voltage.setter
    def voltage(self, value: torch.Tensor):
        self.voltage_.data = value


class RefractoryMixin:
    r"""Mixin for neurons with refractory periods.

    Args:
        refrac (torch.Tensor): initial remaining refractory periods, in :math:`\mathrm{ms}`.
        requires_grad (bool, optional): if the parameters created require gradients. Defaults to False.

    Note:
        This must be added to a class which inherits from :py:class:`DimensionalModule`.

    Note:
        The mixin constructor must be called after the :py:class:`~inferno.DimensionalModule` constructor.
    """

    def __init__(self, refrac, requires_grad=False):
        self.register_parameter("refrac_", nn.Parameter(refrac, requires_grad))
        self.register_constrained("refrac_")

    @property
    def refrac(self) -> torch.Tensor:
        r"""Remaining refractory periods, in milliseconds.

        Args:
            value (torch.Tensor): remaining refractory periods, in :math:`\mathrm{ms}`.

        Returns:
            torch.Tensor: remaining refractory periods, in :math:`\mathrm{ms}`.
        """
        return self.refrac_.data

    @refrac.setter
    def refrac(self, value: torch.Tensor) -> None:
        self.refrac_.data = value


class SpikeRefractoryMixin(RefractoryMixin):
    r"""Mixin for neurons with refractory periods and spikes based off of them.

    Args:
        refrac (torch.Tensor): initial remaining refractory periods, in :math:`\mathrm{ms}`.
        requires_grad (bool, optional): if the parameters created require gradients. Defaults to False.

    Note:
        This must be added to a class which inherits from :py:class:`DimensionalModule`.

    Note:
        This must be added to a class with instances containing a variable ``refrac_t``,
        representing the length of the absolute refractory period in milliseconds.

    Note:
        The mixin constructor must be called after the :py:class:`~inferno.DimensionalModule` constructor.
    """

    def __init__(self, refrac, requires_grad=False):
        RefractoryMixin.__init__(self, refrac, requires_grad)

    @property
    def spike(self) -> torch.Tensor:
        r"""Action potentials last generated.

        Returns:
            torch.Tensor: if the corresponding neuron generated an action potential last step.
        """
        return self.refracs == self.refrac_t


class WeightBiasMixin:
    def __init__(
        self, weight: torch.Tensor, bias: torch.Tensor | None, requires_grad=False
    ):
        self.register_parameter("weight_", nn.Parameter(weight, requires_grad))
        self.register_parameter("bias_", nn.Parameter(bias, requires_grad))

    @property
    def weight(self) -> torch.Tensor:
        r"""Learnable weights of the connection.

        Args:
            value (torch.Tensor): new weights.

        Returns:
            torch.Tensor: current weights.
        """
        return self.weight_.data

    @weight.setter
    def weight(self, value: torch.Tensor):
        self.weight_.data = value

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
        if self.bias_ is not None:
            return self.bias_.data

    @bias.setter
    def bias(self, value: torch.Tensor):
        if self.bias_ is not None:
            self.bias_.data = value
        else:
            raise RuntimeError(
                f"cannot set `bias` on a {type(self).__name__} without trainable biases"
            )


class WeightBiasDelayMixin(WeightBiasMixin):
    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        delay: torch.Tensor | None,
        requires_grad=False,
    ):
        WeightBiasMixin.__init__(self, weight, bias, requires_grad)
        self.register_parameter("delay_", nn.Parameter(delay, requires_grad))

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
        if self.delay_ is not None:
            return self.delay_.data

    @delay.setter
    def delay(self, value: torch.Tensor):
        if self.delay_ is not None:
            self.delay_.data = value
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
            raise RuntimeError(
                f"{type(self).__name__} object already contains"
                f"constraint size={self.constraints[0]} at dim={0}."
            )

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
