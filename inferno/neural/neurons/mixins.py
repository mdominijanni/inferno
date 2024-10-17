from ... import ShapedTensor
from ..._internal import argtest
from ..base import InfernoNeuron
import torch
import torch.nn as nn
from typing import Callable


class AdaptiveThresholdMixin:
    r"""Mixin for neurons with adaptative thresholds.

    Args:
        data (torch.Tensor): initial threshold adaptations.
        batch_reduction (Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] | None):
            function to reduce adaptation updates over the batch dimension,
            :py:func:`torch.mean` when ``None``. Defaults to ``None``.

    Note:
        ``batch_reduction`` can be one of the functions in PyTorch including but not
        limited to :py:func:`torch.sum`, :py:func:`torch.mean`, and :py:func:`torch.amax`.
        A custom function with similar behavior can also be passed in. Like with the
        included function, it should not keep the original dimensions by default.
    """

    def __init__(
        self,
        data: torch.Tensor,
        batch_reduction: (
            Callable[[torch.Tensor, int | tuple[int, ...]], torch.Tensor] | None
        ) = None,
    ):
        _ = argtest.instance("self", self, nn.Module)
        self.register_buffer("threshold_adaptation_", data)
        self.__batchreduce = batch_reduction if batch_reduction else torch.mean

    @property
    def threshold_adaptation(self) -> torch.Tensor:
        r"""Threshold adaptations.

        If the value the setter attempts to assign has the same shape but with an
        additonal leading dimension, it will assume that is an unreduced batch dimension
        and reduce it.

        Args:
            value (torch.Tensor): new threshold adaptations.

        Returns:
            torch.Tensor: present threshold adaptations.
        """
        return self.threshold_adaptation_

    @threshold_adaptation.setter
    def threshold_adaptation(self, value: torch.Tensor) -> None:
        if value.shape[1:] == self.threshold_adaptation_.shape:
            self.threshold_adaptation_ = self.__batchreduce(value, 0)
        else:
            self.threshold_adaptation_ = value


class AdaptiveCurrentMixin:
    r"""Mixin for neurons with adaptative input currents.

    Args:
        data (torch.Tensor): initial input adaptations.
        batch_reduction (Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] | None):
            function to reduce adaptation updates over the batch dimension,
            :py:func:`torch.mean` when ``None``. Defaults to ``None``.

    Note:
        ``batch_reduction`` can be one of the functions in PyTorch including but not
        limited to :py:func:`torch.sum`, :py:func:`torch.mean`, and :py:func:`torch.amax`.
        A custom function with similar behavior can also be passed in. Like with the
        included function, it should not keep the original dimensions by default.
    """

    def __init__(
        self,
        data: torch.Tensor,
        batch_reduction: (
            Callable[[torch.Tensor, int | tuple[int, ...]], torch.Tensor] | None
        ) = None,
    ):
        _ = argtest.instance("self", self, nn.Module)
        self.register_buffer("current_adaptation_", data)
        self.__batchreduce = batch_reduction if batch_reduction else torch.mean

    @property
    def current_adaptation(self) -> torch.Tensor:
        r"""Input current adaptations.

        If the value the setter attempts to assign has the same shape but with an
        additional leading dimension, it will assume that is an unreduced batch dimension
        and reduce it.

        Args:
            value (torch.Tensor): new threshold adaptations.

        Returns:
            torch.Tensor: present threshold adaptations.
        """
        return self.current_adaptation_

    @current_adaptation.setter
    def current_adaptation(self, value: torch.Tensor) -> None:
        if value.shape[1:] == self.current_adaptation_.shape:
            self.current_adaptation_ = self.__batchreduce(value, 0)
        else:
            self.current_adaptation_ = value


class CurrentMixin:
    r"""Mixin for neurons with membrane currents.

    Args:
        data (torch.Tensor): initial currents, in :math:`\text{nA}`.
    """

    def __init__(self, data: torch.Tensor):
        _ = argtest.instance("self", self, InfernoNeuron)
        ShapedTensor.create(
            self,
            "current_",
            data,
            persist_data=True,
            persist_constraints=False,
            strict=True,
            live=False,
        )
        self.add_batched("current_")

    @property
    def current(self) -> torch.Tensor:
        r"""Membrane current in nanoamperes.

        Args:
            value (torch.Tensor): new membrane currents.

        Returns:
            torch.Tensor: present membrane currents.
        """
        return self.current_.value

    @current.setter
    def current(self, value: torch.Tensor) -> None:
        self.current_.value = value


class VoltageMixin:
    r"""Mixin for neurons driven by membrane voltage.

    Args:
        data (torch.Tensor): initial membrane voltages, in :math:`\text{mV}`.
    """

    def __init__(self, data: torch.Tensor):
        _ = argtest.instance("self", self, InfernoNeuron)
        ShapedTensor.create(
            self,
            "voltage_",
            data,
            persist_data=True,
            persist_constraints=False,
            strict=True,
            live=False,
        )
        self.add_batched("voltage_")

    @property
    def voltage(self) -> torch.Tensor:
        r"""Membrane voltages in millivolts.

        Args:
            value (torch.Tensor): new membrane voltages.

        Returns:
            torch.Tensor: present membrane voltages.
        """
        return self.voltage_.value

    @voltage.setter
    def voltage(self, value: torch.Tensor) -> None:
        self.voltage_.value = value


class RefractoryMixin:
    r"""Mixin for neurons with refractory periods.

    Args:
        data (torch.Tensor): initial refractory periods, in :math:`\text{ms}`.
    """

    def __init__(self, data: torch.Tensor):
        _ = argtest.instance("self", self, InfernoNeuron)
        ShapedTensor.create(
            self,
            "refrac_",
            data,
            persist_data=True,
            persist_constraints=False,
            strict=True,
            live=False,
        )
        self.add_batched("refrac_")

    @property
    def refrac(self) -> torch.Tensor:
        r"""Remaining refractory periods, in milliseconds.

        Args:
            value (torch.Tensor): new remaining refractory periods.

        Returns:
            torch.Tensor: present remaining refractory periods.
        """
        return self.refrac_.value

    @refrac.setter
    def refrac(self, value: torch.Tensor) -> None:
        self.refrac_.value = value


class SpikeRefractoryMixin(RefractoryMixin):
    r"""Mixin for neurons with refractory periods with spikes based off of them.

    Args:
        refrac (torch.Tensor): initial refractory periods, in :math:`\text{ms}`.
        absrefrac (str): attribute containing the absolute refractory period,
            in :math:`\text{ms}`.
    """

    def __init__(self, refrac: torch.Tensor, absrefrac: str):
        RefractoryMixin.__init__(self, refrac)
        self.__absrefrac_attr = absrefrac

    @property
    def absrefrac(self) -> float:
        r"""Absolute refractory period.

        Returns:
            float: absolute refractory period in :math:`\text{ms}`.
        """
        return getattr(self, self.__absrefrac_attr)

    @property
    def spike(self) -> torch.Tensor:
        r"""Action potentials last generated.

        .. math::
            f(t) =
            \begin{cases}
                1, &t_\text{refrac}(t) = \text{ARP} \\
                0, &\text{otherwise}
            \end{cases}

        Where:
            * :math:`f_(t)` are the postsynaptic spikes.
            * :math:`t_\text{refrac}(t)` are the remaining refractory periods, in :math:`\text{ms}`.
            * :math:`\text{ARP}` is the absolute refractory period, in :math:`\text{ms}`.

        Returns:
            torch.Tensor: if the corresponding neuron generated an action potential
                during the prior step.
        """
        return self.refrac == getattr(self, self.__absrefrac_attr)
