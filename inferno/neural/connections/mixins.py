from ..._internal import argtest
import torch
import torch.nn as nn


class WeightMixin:
    r"""Mixin for connections with weights.

    Args:
        weight (torch.Tensor): initial connection weights.
        requires_grad (bool, optional): if the parameters created require gradients.
            Defaults to ``False``.

    Caution:
        This must be added to a class which inherits from
        :py:class:`~torch.nn.Module`, and the constructor for this
        mixin must be called after the module constructor.

    Note:
        This registers a parameter ``weight_``.
    """

    def __init__(self, weight: torch.Tensor, requires_grad: bool = False):
        _ = argtest.instance("self", self, nn.Module)
        self.register_parameter("weight_", nn.Parameter(weight, requires_grad))

    @property
    def weight(self) -> nn.Parameter:
        r"""Learnable connection weights.

        Args:
            value (torch.Tensor | nn.Parameter): new weights.

        Returns:
            torch.Tensor: present weights.
        """
        return self.weight_

    @weight.setter
    def weight(self, value: torch.Tensor | nn.Parameter) -> None:
        self.weight_.data = value


class WeightBiasMixin(WeightMixin):
    r"""Mixin for connections with weights and biases.

    Args:
        weight (torch.Tensor): initial connection weights.
        bias (torch.Tensor): initial connection biases, if any.
        requires_grad (bool, optional): if the parameters created require gradients.
            Defaults to ``False``.

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
    def bias(self) -> nn.Parameter | None:
        r"""Learnable connection biases.

        Args:
            value (torch.Tensor | nn.Parameter): new biases.

        Returns:
            nn.Parameter | None: present biases, if any.

        """
        if hasattr(self, "bias_"):
            return self.bias_

    @bias.setter
    def bias(self, value: torch.Tensor | nn.Parameter) -> None:
        if hasattr(self, "bias_"):
            self.bias_.data = value


class WeightBiasDelayMixin(WeightBiasMixin):
    r"""Mixin for connections with weights, biases, and delays.

    Args:
        weight (torch.Tensor): initial connection weights.
        bias (torch.Tensor): initial connection biases, if any.
        delay (torch.Tensor): initial connection delays, if any.
        requires_grad (bool, optional): if the parameters created require gradients.
            Defaults to ``False``.

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
    def delay(self) -> nn.Parameter | None:
        r"""Learnable connection delays.

        Args:
            value (torch.Tensor | nn.Parameter): new delays.

        Returns:
            nn.Parameter | None: present delays, if any.
        """
        if hasattr(self, "delay_"):
            return self.delay_

    @delay.setter
    def delay(self, value: torch.Tensor | nn.Parameter) -> None:
        if hasattr(self, "delay_"):
            self.delay_.data = value
