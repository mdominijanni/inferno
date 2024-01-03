from __future__ import annotations
import torch
import torch.nn as nn


class WeightBiasMixin:
    def __init__(
        self, weight: torch.Tensor, bias: torch.Tensor | None, requires_grad=False
    ):
        self.register_parameter("weight_", nn.Parameter(weight, requires_grad))
        self.register_parameter(
            "bias_", None if bias is None else nn.Parameter(bias, requires_grad)
        )

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
        self.register_parameter(
            "delay_", None if delay is None else nn.Parameter(delay, requires_grad)
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
