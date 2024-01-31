from inferno import DimensionalModule, Module
from inferno._internal import attr_members, instance_of
import torch
import torch.nn as nn


class AdaptationMixin:
    r"""Mixin for neurons with membrane adaptations.

    Args:
        data (torch.Tensor): initial membrane adaptations.
        requires_grad (bool, optional): if the parameters created require gradients.
            Defaults to False.

    Caution:
        This must be added to a class which inherits from
        :py:class:`Module`, and the constructor for this
        mixin must be called after the module constructor.

    Note:
        This registers a parameter ``adaptation_`` and sets it as constrained.
    """

    def __init__(self, data: torch.Tensor, requires_grad=False):
        instance_of("`self`", self, Module)
        self.register_parameter("adaptation_", nn.Parameter(data, requires_grad))

    @property
    def adaptation(self) -> torch.Tensor:
        r"""Membrane adaptations.

        Args:
            value (torch.Tensor): new membrane adaptations.

        Returns:
            torch.Tensor: present membrane adaptations.
        """
        return self.adaptation_.data

    @adaptation.setter
    def adaptation(self, value: torch.Tensor):
        self.adaptation_.data = value


class CurrentMixin:
    r"""Mixin for neurons with membrane currents.

    Args:
        data (torch.Tensor): initial currents, in :math:`\text{nA}`.
        requires_grad (bool, optional): if the parameters created require gradients.
            Defaults to False.

    Caution:
        This must be added to a class which inherits from
        :py:class:`DimensionalModule`, and the constructor for this
        mixin must be called after the module constructor.

    Note:
        This registers a parameter ``current_`` and sets it as constrained.
    """

    def __init__(self, data: torch.Tensor, requires_grad=False):
        instance_of("`self`", self, DimensionalModule)
        self.register_parameter("current_", nn.Parameter(data, requires_grad))
        self.register_constrained("current_")

    @property
    def current(self) -> torch.Tensor:
        r"""Membrane current in nanoamperes.

        Args:
            value (torch.Tensor): new membrane currents.

        Returns:
            torch.Tensor: present membrane currents.
        """
        return self.current_.data

    @current.setter
    def current(self, value: torch.Tensor):
        self.current_.data = value


class VoltageMixin:
    r"""Mixin for neurons driven by membrane voltage.

    Args:
        data (torch.Tensor): initial membrane voltages, in :math:`\text{mV}`.
        requires_grad (bool, optional): if the parameters created require gradients.
            Defaults to False.

    Caution:
        This must be added to a class which inherits from
        :py:class:`DimensionalModule`, and the constructor for this
        mixin must be called after the module constructor.

    Note:
        This registers a parameter ``voltage_`` and sets it as constrained.
    """

    def __init__(self, data: torch.Tensor, requires_grad=False):
        instance_of("`self`", self, DimensionalModule)
        self.register_parameter("voltage_", nn.Parameter(data, requires_grad))
        self.register_constrained("voltage_")

    @property
    def voltage(self) -> torch.Tensor:
        r"""Membrane voltages in millivolts.

        Args:
            value (torch.Tensor): new membrane voltages.

        Returns:
            torch.Tensor: present membrane voltages.
        """
        return self.voltage_.data

    @voltage.setter
    def voltage(self, value: torch.Tensor):
        self.voltage_.data = value


class RefractoryMixin:
    r"""Mixin for neurons with refractory periods.

    Args:
        refrac (torch.Tensor): initial refractory periods, in :math:`\text{ms}`.
        requires_grad (bool, optional): if the parameters created require gradients.
            Defaults to False.

    Caution:
        This must be added to a class which inherits from
        :py:class:`DimensionalModule`, and the constructor for this
        mixin must be called after the module constructor.

    Note:
        This registers a parameter ``refrac_`` and sets it as constrained.
    """

    def __init__(self, refrac, requires_grad=False):
        instance_of("`self`", self, DimensionalModule)
        self.register_parameter("refrac_", nn.Parameter(refrac, requires_grad))
        self.register_constrained("refrac_")

    @property
    def refrac(self) -> torch.Tensor:
        r"""Remaining refractory periods, in milliseconds.

        Args:
            value (torch.Tensor): new remaining refractory periods.

        Returns:
            torch.Tensor: present remaining refractory periods.
        """
        return self.refrac_.data

    @refrac.setter
    def refrac(self, value: torch.Tensor) -> None:
        self.refrac_.data = value


class SpikeRefractoryMixin(RefractoryMixin):
    r"""Mixin for neurons with refractory periods with spikes based off of them.

    Args:
        refrac (torch.Tensor): initial refractory periods, in :math:`\text{ms}`.
        requires_grad (bool, optional): if the parameters created require gradients.
            Defaults to False.

    Caution:
        This must be added to a class which inherits from
        :py:class:`DimensionalModule`, and the constructor for this
        mixin must be called after the module constructor.

    Note:
        This registers a parameter ``refrac_`` and sets it as constrained.

    Important:
        This must be added to a class which has an attribute named ``refrac_t``, which
        represents the length of the absolute refractory period in :math:`\text{ms}`.
    """

    def __init__(self, refrac, requires_grad=False):
        attr_members("`self`", self, "refrac_t")
        RefractoryMixin.__init__(self, refrac, requires_grad)

    @property
    def spike(self) -> torch.Tensor:
        r"""Action potentials last generated.

        .. math::
            f(t) =
            \begin{cases}
                1, &t_\text{refrac}(t) = \text{ARP}
                0, &\text{otherwise}
            \end{cases}

        Where:
            * :math:`f_(t)` are the postsynaptic spikes.
            * :math:`t_\text{refrac}`(t) are the remaining refractory periods, in :math:`\text{ms}`.
            * :math:`\text{ARP}` is the absolute refractory period, in :math:`\text{ms}`.

        Returns:
            torch.Tensor: if the corresponding neuron generated an action potential
                during the prior step.
        """
        return self.refrac == self.refrac_t
