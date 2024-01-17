from inferno._internal import instance_of
import torch
import torch.nn as nn


class WeightMixin:
    """Mixin for connections with weights.

    Args:
        weight (torch.Tensor): initial connection weights.
        requires_grad (bool, optional): if the parameters created require gradients.
            Defaults to False.

    Caution:
        This must be added to a class which inherits from
        :py:class:`~torch.nn.Module`, and the constructor for this
        mixin must be called after the module constructor.

    Note:
        This registers a parameter ``weight_``.
    """

    def __init__(self, weight: torch.Tensor, requires_grad: bool = False):
        instance_of("`self`", self, nn.Module)
        self.register_parameter("weight_", nn.Parameter(weight, requires_grad))

    @property
    def weight(self) -> torch.Tensor:
        r"""Learnable connection weights.

        Args:
            value (torch.Tensor): new weights.

        Returns:
            torch.Tensor: present weights.
        """
        return self.weight_.data

    @weight.setter
    def weight(self, value: torch.Tensor):
        self.weight_.data = value


class WeightBiasMixin(WeightMixin):
    """Mixin for connections with weights and biases.

    Args:
        weight (torch.Tensor): initial connection weights.
        bias (torch.Tensor): initial connection biases, if any.
        requires_grad (bool, optional): if the parameters created require gradients.
            Defaults to False.

    Caution:
        This must be added to a class which inherits from
        :py:class:`~torch.nn.Module`, and the constructor for this
        mixin must be called after the module constructor.

    Note:
        This registers a parameter ``weight_``, and if ``bias`` is not None, a
        parameter ``bias_`` as well.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        requires_grad: bool = False,
    ):
        WeightMixin.__init__(self, weight, requires_grad)
        if bias is not None:
            self.register_parameter("bias_", nn.Parameter(bias, requires_grad))

    @property
    def bias(self) -> torch.Tensor | None:
        r"""Learnable connection biases.

        Args:
            value (torch.Tensor): new biases.

        Returns:
            torch.Tensor | None: present biases, if any.

        """
        if hasattr(self, "bias_"):
            return self.bias_.data

    @bias.setter
    def bias(self, value: torch.Tensor):
        if hasattr(self, "bias_"):
            self.bias_.data = value


class WeightBiasDelayMixin(WeightBiasMixin):
    """Mixin for connections with weights, biases, and delays.

    Args:
        weight (torch.Tensor): initial connection weights.
        bias (torch.Tensor): initial connection biases, if any.
        delay (torch.Tensor): initial connection delays, if any.
        requires_grad (bool, optional): if the parameters created require gradients.
            Defaults to False.

    Caution:
        This must be added to a class which inherits from
        :py:class:`~torch.nn.Module`, and the constructor for this
        mixin must be called after the module constructor.

    Note:
        This registers a parameter ``weight_``. If either ``bias`` or ``delay``
        is not None, parameters ``bias_`` and ``delay_`` are created respectively.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        delay: torch.Tensor | None,
        requires_grad=False,
    ):
        WeightBiasMixin.__init__(self, weight, bias, requires_grad)
        if delay is not None:
            self.register_parameter("delay_", nn.Parameter(delay, requires_grad))

    @property
    def delay(self) -> torch.Tensor | None:
        r"""Learnable connection delays.

        Args:
            value (torch.Tensor): new delays.

        Returns:
            torch.Tensor | None: present delays, if any.
        """
        if hasattr(self, "delay_"):
            return self.delay_.data

    @delay.setter
    def delay(self, value: torch.Tensor):
        if hasattr(self, "delay_"):
            self.delay_.data = value
