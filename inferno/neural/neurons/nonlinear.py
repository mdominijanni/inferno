from .. import Neuron
from .. import functional as nf
from .mixins import VoltageMixin, SpikeRefractoryMixin
from inferno._internal import numeric_limit, numeric_relative
import torch


class QIF(VoltageMixin, SpikeRefractoryMixin, Neuron):
    r"""Simulation of quadratic integrate-and-fire (QIF) neuron dynamics.

    .. math::
        V_m(t + \Delta t) = \frac{\Delta t}{\tau_m}
        \left[ a \left(V_m(t) - V_\text{rest}\right)\left(V_m(t) - V_\text{crit}\right) + R_mI(t) \right]

    If a spike was generated at time :math:`t`, then.

    .. math::
        V_m(t) \leftarrow V_\text{reset}

    Args:
        shape (tuple[int, ...] | int): shape of the group of neurons being simulated.
        step_time (float): length of a simulation time step,
            :math:`\Delta t`, in :math:`\text{ms}`.
        rest_v (float): membrane potential difference at equilibrium,
            :math:`V_\text{rest}`, in :math:`\text{mV}`.
        crit_v (float): membrane potential difference at which potential
            naturally increases, :math:`V_\text{crit}`, in :math:`\text{mV}`.
        attraction (float): controls the strength of the membrane
            potential's drift towards :math:`V_\text{rest}` and away from
            :math:`V_\text{crit}`, :math:`a`, unitless.
        reset_v (float): membrane voltage after an action potential is generated,
            :math:`V_\text{reset}`, in :math:`\text{mV}`.
        thresh_v (float): membrane voltage at which action potentials are generated,
            in :math:`\text{mV}`.
        refrac_t (float): length the absolute refractory period, in :math:`\text{ms}`.
        time_constant (float): time constant of exponential decay for membrane voltage,
            :math:`\tau_m`, in :math:`\text{ms}`.
        resistance (float, optional): resistance across the cell membrane,
            :math:`R_m`, in :math:`\text{M}\Omega`. Defaults to 1.0.
        batch_size (int, optional): size of input batches for simualtion. Defaults to 1.

    See Also:
        For more details and references, visit
        :ref:`zoo/neurons-nonlinear:Quadratic Integrate-and-Fire (QIF)` in the zoo.
    """
    def __init__(
        self,
        shape: tuple[int, ...] | int,
        step_time: float,
        *,
        rest_v: float,
        crit_v: float,
        attraction: float,
        reset_v: float,
        thresh_v: float,
        refrac_t: float,
        time_constant: float,
        resistance: float = 1.0,
        batch_size: int = 1,
    ):
        # call superclass constructor
        Neuron.__init__(self, shape, batch_size)

        # dynamics attributes
        self.step_time, e = numeric_limit("step_time", step_time, 0, "gt", float)
        if e:
            raise e

        self.rest_v, self.crit_v, e = numeric_relative(
            "rest_v", rest_v, "crit_v", crit_v, "lt", float
        )
        if e:
            raise e

        _, _, e = numeric_relative(
            "rest_v", rest_v, "thresh_v", thresh_v, "lt", float
        )
        if e:
            raise e

        _, _, e = numeric_relative(
            "crit_v", crit_v, "thresh_v", thresh_v, "lt", float
        )
        if e:
            raise e

        self.attraction, e = numeric_limit("attraction", attraction, 0, "gt", float)
        if e:
            raise e

        self.reset_v, self.thresh_v, e = numeric_relative(
            "reset_v", reset_v, "thresh_v", thresh_v, "lt", float
        )
        if e:
            raise e

        self.refrac_t, e = numeric_limit("refrac_t", refrac_t, 0, "gte", float)
        if e:
            raise e

        self.time_constant, e = numeric_limit(
            "time_constant", time_constant, 0, "gt", float
        )
        if e:
            raise e

        self.resistance, e = numeric_limit("resistance", resistance, 0, "neq", float)
        if e:
            raise

        # call mixin constructors
        VoltageMixin.__init__(self, torch.full(self.bshape, self.rest_v), False)
        SpikeRefractoryMixin.__init__(self, torch.zeros(self.bshape), False)

    def _integrate_v(self, masked_inputs):
        r"""Internal, voltage function for :py:func:`~nf.voltage_thresholding`."""
        return nf.voltage_integration_quadratic(
            masked_inputs,
            self.voltage,
            step_time=self.step_time,
            rest_v=self.rest_v,
            crit_v=self.crit_v,
            attraction=self.attraction,
            time_constant=self.time_constant,
            resistance=self.resistance,
        )

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
    def dt(self, value: float):
        self.step_time, e = numeric_limit("dt", value, 0, "gt", float)
        if e:
            raise e

    def clear(self, **kwargs):
        r"""Resets neurons to their resting state."""
        self.voltage = torch.full_like(self.voltage, self.rest_v)
        self.refrac = torch.zeros_like(self.refrac)

    def forward(self, inputs: torch.Tensor, refrac_lock=True, **kwargs) -> torch.Tensor:
        r"""Runs a simulation step of the neuronal dynamics.

        Args:
            inputs (torch.Tensor): presynaptic currents,
                :math:`I(t)`, in :math:`\text{nA}`.
            refrac_lock (bool, optional): if membrane voltages should be fixed while
                in the refractory period. Defaults to True.

        Returns:
            torch.Tensor: if the corresponding neuron generated an action potential.
        """
        # use voltage thresholding function
        spikes, voltages, refracs = nf.voltage_thresholding(
            inputs=inputs,
            refracs=self.refrac,
            dynamics=self._integrate_v,
            step_time=self.step_time,
            reset_v=self.reset_v,
            thresh_v=self.thresh_v,
            refrac_t=self.refrac_t,
            voltages=(self.voltage if refrac_lock else None),
        )

        # update parameters
        self.voltage = voltages
        self.refrac = refracs

        # return spiking output
        return spikes
