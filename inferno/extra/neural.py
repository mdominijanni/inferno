from collections.abc import Sequence
from .. import zeros, ShapedTensor, VirtualTensor
from .._internal import argtest
from ..neural import InfernoNeuron
import torch


class ExactNeuron(InfernoNeuron):
    r"""Simple neuron class useful for getting predictable outputs for visualization.

    An action potential will be generated if the input to :py:meth:`forward()` is
    positive, unless an ``override``boolean tensor is given, in which case that will be
    used instead. Membrane voltages will be set to ``thresh_v`` if a spike was generated,
    and otherwise will be set to ``rest_v``.

    Args:
        shape (Sequence[int]): shape of the group of neurons being simulated.
        step_time (float): length of a simulation time step,
            :math:`\Delta t`, in :math:`\text{ms}`.
        rest_v (float): membrane potential difference at equilibrium,
            :math:`V_\text{rest}`, in :math:`\text{mV}`.
        thresh_v (float): membrane voltage at which action potentials are generated,
            in :math:`\text{mV}`.
        batch_size (int, optional): size of input batches for simulation.
            Defaults to ``1``.

    Note:
        Unlike in an actual neuron model, ``rest_v`` and ``thresh_v`` don't control any
        spiking behavior—these just change the presentation of the membrane voltage.
    """

    def __init__(
        self,
        shape: Sequence[int],
        step_time: float,
        *,
        rest_v: float,
        thresh_v: float,
        batch_size: int = 1,
    ):
        # call superclass constructor
        InfernoNeuron.__init__(self, shape, batch_size)

        # dynamics attributes
        self.step_time = argtest.gt("step_time", step_time, 0, float)
        self.rest_v = argtest.lt("rest_v", rest_v, thresh_v, float, "thresh_v")
        self.thresh_v = float(thresh_v)

        # buffers, real and imaginary
        ShapedTensor.create(
            self,
            "spike_",
            torch.full(self.batchedshape, False, dtype=torch.bool),
            persist_data=True,
            persist_constraints=False,
            strict=True,
            live=False,
        )
        VirtualTensor.create(
            self,
            "voltage_",
            "_derived_voltage",
            persist=False,
        )
        VirtualTensor.create(
            self,
            "refrac_",
            "_derived_refrac",
            persist=False,
        )

    def _derived_voltage(
        self, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        return torch.where(self.spike_.value, self.thresh_v, self.rest_v).to(
            dtype=dtype, device=device
        )

    def _derived_refrac(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return zeros(self.spike_.value, dtype=dtype, device=device)

    @property
    def dt(self) -> float:
        r"""Length of the simulation time step, in milliseconds.

        Args:
            value (float): new simulation time step length.

        Returns:
            float: present simulation time step length.
        """
        return self.step_time

    @dt.setter
    def dt(self, value: float) -> None:
        self.step_time = argtest.gt("dt", value, 0, float)

    @property
    def voltage(self) -> torch.Tensor:
        r"""Membrane voltages in millivolts.

        Args:
            value (torch.Tensor): new membrane voltages.

        Returns:
            torch.Tensor: present membrane voltages.

        Note:
            :py:class:`ExactNeuron` derives membrane voltage from action
            potentials. Therefore the setter will do nothing.
        """
        return self.voltage_.value

    @voltage.setter
    def voltage(self, value: torch.Tensor) -> None:
        pass

    @property
    def spike(self) -> torch.Tensor:
        r"""Action potentials last generated.

        Returns:
            torch.Tensor: if the corresponding neuron generated an action potential
                during the prior step.
        """
        return self.spike_.value

    @property
    def refrac(self) -> torch.Tensor:
        r"""Remaining refractory periods, in milliseconds.

        Args:
            value (torch.Tensor): new remaining refractory periods.

        Returns:
            torch.Tensor: present remaining refractory periods.

        Note:
            :py:class:`ExactNeuron` doesn't support refractory periods. The
            getter will always return a tensor of zeros and the setter will do
            nothing.
        """
        return self.refrac_.value

    @refrac.setter
    def refrac(self, value: torch.Tensor) -> None:
        pass

    def clear(self, **kwargs) -> None:
        r"""Resets neurons to their resting state."""
        self.spike_.value = torch.full_like(self.spike, False)

    def forward(
        self, inputs: torch.Tensor, override: torch.Tensor | None = None, **kwargs
    ) -> torch.Tensor:
        r"""Runs a simulation step of the neuronal dynamics.

        Args:
            inputs (torch.Tensor): presynaptic currents, :math:`I(t)`,
                in :math:`\text{nA}`.
            override (optional, torch.Tensor | None): tensor of spikes to use for output
                if spiking output should not be based on inputs. Defaults to ``None``.

        Returns:
            torch.Tensor: if the corresponding neuron generated an action potential.
        """
        # set spikes based on threshold
        if override is None:
            self.spike_.value = (inputs > 0).to(
                device=self.spike_.value.device, dtype=self.spike_.value.dtype
            )

        # manual override of spikes
        else:
            self.spike_.value = override.view(self.batchedshape).to(
                device=self.spike_.value.device, dtype=self.spike_.value.dtype
            )

        # return spiking output
        return self.spike_.value
