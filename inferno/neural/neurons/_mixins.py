from __future__ import annotations
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
