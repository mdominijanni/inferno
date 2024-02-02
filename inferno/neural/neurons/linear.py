from .. import Neuron
from .. import functional as nf
from .mixins import AdaptationMixin, VoltageMixin, SpikeRefractoryMixin
from inferno._internal import numeric_limit, multiple_numeric_limit, regroup
import math
import torch


class LIF(VoltageMixin, SpikeRefractoryMixin, Neuron):
    r"""Simulation of leaky integrate-and-fire (LIF) neuron dynamics.

    .. math::
        V_m(t + \Delta t) = \left[V_m(t) - V_\text{rest}\right]
        \exp\left(-\frac{t}{\tau_m}\right) + V_\text{rest} + R_mI(t)

    If a spike was generated at time :math:`t`, then.

    .. math::
        V_m(t) \leftarrow V_\text{reset}

    Args:
        shape (tuple[int, ...] | int): shape of the group of neurons being simulated.
        step_time (float): length of a simulation time step,
            :math:`\Delta t`, in :math:`\text{ms}`.
        rest_v (float): membrane potential difference at equilibrium,
            :math:`V_\text{rest}`, in :math:`\text{mV}`.
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
        :ref:`zoo/neurons-linear:Leaky Integrate-and-Fire (LIF)` in the zoo.
    """

    def __init__(
        self,
        shape: tuple[int, ...] | int,
        step_time: float,
        *,
        rest_v: float,
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
        self.time_constant, e = numeric_limit(
            "time_constant", time_constant, 0, "gt", float
        )
        if e:
            raise e
        self.decay = math.exp(-self.step_time / self.time_constant)
        self.rest_v = float(rest_v)
        self.reset_v = float(reset_v)
        self.thresh_v = float(thresh_v)
        self.refrac_t, e = numeric_limit("refrac_t", refrac_t, 0, "gte", float)
        if e:
            raise e
        self.resistance, e = numeric_limit("resistance", resistance, 0, "neq", float)
        if e:
            raise

        # call mixin constructors
        VoltageMixin.__init__(self, torch.full(self.bshape, self.rest_v), False)
        SpikeRefractoryMixin.__init__(self, torch.zeros(self.bshape), False)

    def _integrate_v(self, masked_inputs):
        """Internal, voltage function for :py:func:`~nf.voltage_thresholding`."""
        return nf.voltage_integration_linear(
            masked_inputs,
            self.voltage,
            decay=self.decay,
            rest_v=self.rest_v,
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
        self.decay = math.exp(-self.step_time / self.time_constant)

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


class ALIF(AdaptationMixin, VoltageMixin, SpikeRefractoryMixin, Neuron):
    r"""Simulation of adaptive leaky integrate-and-fire (ALIF) neuron dynamics.

    ALIF is implemented as a step of leaky integrate-and-fire applying existing
    adaptations, using linear spike-dependent adaptive thresholds, then updating
    those adaptations for the next timestep.

    .. math::
        \begin{align*}
            V_m(t + \Delta t) &= \left[V_m(t) - V_\text{rest}\right]
            \exp\left(-\frac{t}{\tau_m}\right) + V_\text{rest} + R_mI(t) \\
            \Theta(t) &= \Theta_\infty + \sum_k \theta_k(t) \\
            \theta_k(t + \Delta t) &= \theta_k(t) \exp\left(-\frac{\Delta t}{\tau_k}\right)
        \end{align*}

    If a spike was generated at time :math:`t`, then.

    .. math::
        \begin{align*}
            V_m(t) &\leftarrow V_\text{reset} \\
            \theta_k(t) &\leftarrow \theta_k(t) + a_k
        \end{align*}

    Args:
        shape (tuple[int, ...] | int): shape of the group of neurons being simulated.
        step_time (float): length of a simulation time step,
            :math:`\Delta t`, in :math:`\text{ms}`.
        rest_v (float): membrane potential difference at equilibrium,
            :math:`V_\text{rest}`, in :math:`\text{mV}`.
        reset_v (float): membrane voltage after an action potential is generated,
            :math:`V_\text{reset}`, in :math:`\text{mV}`.
        thresh_eq_v (float): equilibrium of the firing threshold,
            :math:`\Theta_\infty$`, in :math:`\text{mV}`.
        refrac_t (float): length the absolute refractory period, in :math:`\text{ms}`.
        tc_membrane (float): time constant of exponential decay for membrane voltage,
            :math:`\tau_m`, in :math:`\text{ms}`.
        tc_adaptation (float | tuple[float]): time constant of exponential decay for
            threshold adaptations, :math:`\tau_k`, in :math:`\text{ms}`.
        spike_adapt_incr (float | tuple[float]): amount by which the adaptive
            threshold is increased after a spike, :math:`a_k`, in :math:`\text{mV}`.
        resistance (float, optional): resistance across the cell membrane,
            :math:`R_m`, in :math:`\text{M}\Omega`. Defaults to 1.0.
        batch_size (int, optional): size of input batches for simualtion. Defaults to 1.

    See Also:
        For more details and references, visit
        :ref:`zoo/neurons-linear:Adaptive Leaky Integrate-and-Fire (ALIF)` in the zoo.
    """

    def __init__(
        self,
        shape: tuple[int, ...] | int,
        step_time: float,
        *,
        rest_v: float,
        reset_v: float,
        thresh_eq_v: float,
        refrac_t: float,
        tc_membrane: float,
        tc_adaptation: float | tuple[float, ...],
        spike_adapt_incr: float | tuple[float, ...],
        resistance: float = 1.0,
        batch_size: int = 1,
    ):
        # call superclass constructor
        Neuron.__init__(self, shape, batch_size)

        # possible tuple/float combinations for tc_adaptation and spike_adapt_incr
        match (
            hasattr(tc_adaptation, "__iter__"),
            hasattr(spike_adapt_incr, "__iter__"),
        ):
            case (True, True):
                # test that tc_adaptation and spike_adapt_incr are of equal length
                if len(tc_adaptation) != len(spike_adapt_incr):
                    raise RuntimeError(
                        "'tc_adaptation' and 'spike_adapt_incr' must be of the same "
                        f"length, received with lengths of {len(tc_adaptation)} and "
                        f"{len(spike_adapt_incr)} respectively."
                    )

                # cast and test tc_adaptation values
                tc_adaptation, e = multiple_numeric_limit(
                    "tc_adaptation", tc_adaptation, 0, "gt", float, False
                )
                if e:
                    raise e

                # cast spike_adapt_incr values
                spike_adapt_incr = tuple(float(val) for val in spike_adapt_incr)

            case (True, False):
                # cast and test tc_adaptation values
                tc_adaptation = multiple_numeric_limit(
                    "tc_adaptation", tc_adaptation, 0, "gt", float, False
                )
                if e:
                    raise e

                # cast spike_adapt_incr and make it a tuple of matching length
                spike_adapt_incr = float(spike_adapt_incr)
                spike_adapt_incr = tuple(spike_adapt_incr for _ in tc_adaptation)

            case (False, True):
                # cast spike_adapt_incr values
                spike_adapt_incr = tuple(float(val) for val in spike_adapt_incr)
                if len(spike_adapt_incr == 0):
                    raise ValueError("'spike_adapt_incr' cannot be empty.")

                # cast and test tc_adaptation and make it a tuple of matching length
                tc_adaptation, e = regroup(
                    numeric_limit("tc_adaptation", tc_adaptation, 0, "gt", float),
                    ((0 for _ in spike_adapt_incr), 1),
                )
                if e:
                    raise e

            case (False, False):
                # cast and test tc_adaptation and make it a 1-tuple
                tc_adaptation, e = regroup(
                    numeric_limit("tc_adaptation", tc_adaptation, 0, "gt", float),
                    ((0,), 1),
                )
                if e:
                    raise e

                # cast spike_adapt_incr and make it a 1-tuple
                spike_adapt_incr = (float(spike_adapt_incr),)

        # dynamics attributes
        self.step_time, e = numeric_limit("step_time", step_time, 0, "gt", float)
        if e:
            raise e
        self.tc_membrane, e = numeric_limit(
            "tc_membrane", tc_membrane, 0, "gt", float
        )
        if e:
            raise e
        self.decay = math.exp(-self.step_time / self.tc_membrane)
        self.rest_v = float(rest_v)
        self.reset_v = float(reset_v)
        self.thresh_eq_v = float(thresh_eq_v)
        self.refrac_t, e = numeric_limit("refrac_t", refrac_t, 0, "gte", float)
        if e:
            raise e
        self.resistance, e = numeric_limit("resistance", resistance, 0, "neq", float)
        if e:
            raise
        self.tc_adaptation = tc_adaptation
        self.spike_adapt_incr = spike_adapt_incr

        # register adaptation attributes as buffers (for tensor ops and compatibility)
        self.register_buffer(
            "adapt_decay",
            torch.exp(-self.step_time / torch.tensor(self.tc_adaptation)),
            persistent=False,
        )
        self.register_buffer(
            "adapt_increment", torch.tensor(self.spike_adapt_incr), persistent=False
        )

        # call mixin constructors
        VoltageMixin.__init__(self, torch.full(self.bshape, self.rest_v), False)
        SpikeRefractoryMixin.__init__(self, torch.zeros(self.bshape), False)
        AdaptationMixin.__init__(
            self, torch.zeros(*self.shape, len(self.tc_adaptation)), False
        )

    def _integrate_v(self, masked_inputs):
        """Internal, voltage function for :py:func:`~nf.voltage_thresholding`."""
        return nf.voltage_integration_linear(
            masked_inputs,
            self.voltage,
            decay=self.decay,
            rest_v=self.rest_v,
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
        self.decay = math.exp(-self.step_time / self.time_constant)
        self.adapt_decay = torch.exp(
            -self.step_time
            / torch.tensor(
                self.tc_adaptation,
                dtype=self.adapt_decay.dtype,
                device=self.adapt_decay.device,
                requires_grad=self.adapt_decay.requires_grad,
            )
        )

    def clear(self, keep_adaptations=True, **kwargs):
        r"""Resets neurons to their resting state.

        Args:
            keep_adaptations (bool, optional): if learned adaptations should be
                preserved. Defaults to True.
        """
        self.voltage = torch.full_like(self.voltage, self.rest_v)
        self.refrac = torch.zeros_like(self.refrac)
        if not keep_adaptations:
            self.adaptation = torch.zeros_like(self.adaptation)

    def forward(
        self,
        inputs: torch.Tensor,
        adapt: bool | None = None,
        refrac_lock: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        r"""Runs a simulation step of the neuronal dynamics.

        Args:
            inputs (torch.Tensor): presynaptic currents,
                :math:`I(t)`, in :math:`\text{nA}`.
            adapt (bool | None, optional): if adaptations should be updated
                based on this step. Defaults to None.
            refrac_lock (bool, optional): if membrane voltages should be fixed
                while in the refractory period. Defaults to True.

        Returns:
            torch.Tensor: if the corresponding neuron generated an action potential.

        Note:
            When ``adapt`` is set to None, adaptations will be updated when the neuron
            is in training mode but not when it is in evaluation mode.
        """
        # use voltage thresholding function
        spikes, voltages, refracs = nf.voltage_thresholding(
            inputs=inputs,
            refracs=self.refrac,
            dynamics=self._integrate_v,
            step_time=self.step_time,
            reset_v=self.reset_v,
            thresh_v=nf.neuron_adaptation.apply_adaptive_thresholds(
                self.thresh_eq_v, self.adaptation
            ),
            refrac_t=self.refrac_t,
            voltages=(self.voltage if refrac_lock else None),
        )

        # update parameters
        self.voltage = voltages
        self.refrac = refracs

        # conditionally update adaptive thresholds
        if adapt or (adapt is None and self.training):
            # use adaptive thresholds update function
            adaptations = nf.adaptive_thresholds_linear_spike(
                adaptations=self.adaptation,
                spikes=spikes,
                decay=self.adapt_decay,
                spike_increment=self.adapt_increment,
                refracs=(self.refrac if refrac_lock else None),
            )
            # update parameter
            self.adaptation = torch.mean(adaptations, dim=0)

        # return spiking output
        return spikes


class GLIF1(VoltageMixin, SpikeRefractoryMixin, Neuron):
    r"""Simulation of generalized leaky integrate-and-fire 1 (GLIF\ :sub:`1`) neuron dynamics.

    Alias for :py:class:`~inferno.neural.LIF`.

    See Also:
        For more details and references, visit
        :ref:`zoo/neurons-linear:generalized leaky integrate-and-fire 1 (glif{sub}\`1\`)` in the zoo.
    """

    def __init__(
        self,
        shape: tuple[int, ...] | int,
        step_time: float,
        *,
        rest_v: float,
        reset_v: float,
        thresh_v: float,
        refrac_t: float,
        time_constant: float,
        resistance: float = 1.0,
        batch_size: int = 1,
    ):
        # call superclass constructor
        LIF.__init__(
            self,
            shape=shape,
            step_time=step_time,
            rest_v=rest_v,
            reset_v=reset_v,
            thresh_v=thresh_v,
            refrac_t=refrac_t,
            time_constant=time_constant,
            resistance=resistance,
            batch_size=batch_size,
        )

    def _integrate_v(self, masked_inputs):
        """Internal, voltage function for :py:func:`~nf.voltage_thresholding`."""
        return LIF._integrate_v(self, masked_inputs)

    @property
    def dt(self) -> float:
        r"""Length of the simulation time step, in milliseconds.

        Args:
            value (float): new simulation time step length.

        Returns:
            float: present simulation time step length.
        """
        return LIF.dt.fget(self)

    @dt.setter
    def dt(self, value: float):
        LIF.dt.fset(self, value)

    def clear(self, **kwargs):
        r"""Resets neurons to their resting state."""
        LIF.clear(self, **kwargs)

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
        return LIF.forward(inputs, refrac_lock=refrac_lock)


class GLIF2(AdaptationMixin, VoltageMixin, SpikeRefractoryMixin, Neuron):
    r"""Simulation of generalized leaky integrate-and-fire 2 (GLIF\ :sub:`2`) neuron dynamics.

    .. math::
        \begin{align*}
            V_m(t + \Delta t) &= \left[V_m(t) - V_\text{rest}\right]
            \exp\left(-\frac{t}{\tau_m}\right) + V_\text{rest} + R_mI(t) \\
            \Theta(t) &= \Theta_\infty + \sum_k \theta_k(t) \\
            \theta_k(t + \Delta t) &= \theta_k(t) \exp\left(-\frac{\Delta t}{\tau_k}\right)
        \end{align*}

    If a spike was generated at time :math:`t`, then.

    .. math::
        \begin{align*}
            V_m(t) &\leftarrow V_\text{rest} + m_v \left[ V_m(t) - V_\text{rest} \right] - b_v \\
            \theta_k(t) &\leftarrow \theta_k(t) + a_k
        \end{align*}

    Args:
        shape (tuple[int, ...] | int): shape of the group of neurons being simulated.
        step_time (float): length of a simulation time step,
            :math:`\Delta t`, in :math:`\text{ms}`.
        rest_v (float): membrane potential difference at equilibrium,
            :math:`V_\text{rest}`, in :math:`\text{mV}`.
        reset_v_add (float): additive parameter controlling reset voltage,
            :math:`b_v`, in :math:`\text{mV}`.
        reset_v_mul (float): multiplicative parameter controlling reset voltage,
            :math:`m_v`, unitless.
        thresh_eq_v (float): equilibrium of the firing threshold,
            :math:`\Theta_\infty`, in :math:`\text{mV}`.
        refrac_t (float): length the absolute refractory period, in :math:`\text{ms}`.
        tc_membrane (float): time constant of exponential decay for membrane voltage,
            :math:`\tau_m`, in :math:`\text{ms}`.
        tc_adaptation (float | tuple[float]): time constant of exponential decay for
            threshold adaptations, :math:`\tau_k`, in :math:`\text{ms}`.
        spike_adapt_incr (float | tuple[float]): amount by which the adaptive
            threshold is increased after a spike, :math:`a_k`, in :math:`\text{mV}`.
        resistance (float, optional): resistance across the cell membrane,
            :math:`R_m`, in :math:`\text{M}\Omega`. Defaults to 1.0.
        batch_size (int, optional): size of input batches for simualtion. Defaults to 1.

    See Also:
        For more details and references, visit
        :ref:`zoo/neurons-linear:generalized leaky integrate-and-fire 2 (glif{sub}\`2\`)` in the zoo.
    """

    def __init__(
        self,
        shape: tuple[int, ...] | int,
        step_time: float,
        *,
        rest_v: float,
        reset_v_add: float,
        reset_v_mul: float,
        thresh_eq_v: float,
        refrac_t: float,
        tc_membrane: float,
        tc_adaptation: float | tuple[float],
        spike_adapt_incr: float | tuple[float],
        resistance: float = 1.0,
        batch_size: int = 1,
    ):
        # call superclass constructor
        Neuron.__init__(self, shape, batch_size)

        # possible tuple/float combinations for tc_adaptation and spike_adapt_incr
        match (
            hasattr(tc_adaptation, "__iter__"),
            hasattr(spike_adapt_incr, "__iter__"),
        ):
            case (True, True):
                # test that tc_adaptation and spike_adapt_incr are of equal length
                if len(tc_adaptation) != len(spike_adapt_incr):
                    raise RuntimeError(
                        "'tc_adaptation' and 'spike_adapt_incr' must be of the same "
                        f"length, received with lengths of {len(tc_adaptation)} and "
                        f"{len(spike_adapt_incr)} respectively."
                    )

                # cast and test tc_adaptation values
                tc_adaptation, e = multiple_numeric_limit(
                    "tc_adaptation", tc_adaptation, 0, "gt", float, False
                )
                if e:
                    raise e

                # cast spike_adapt_incr values
                spike_adapt_incr = tuple(float(val) for val in spike_adapt_incr)

            case (True, False):
                # cast and test tc_adaptation values
                tc_adaptation = multiple_numeric_limit(
                    "tc_adaptation", tc_adaptation, 0, "gt", float, False
                )
                if e:
                    raise e

                # cast spike_adapt_incr and make it a tuple of matching length
                spike_adapt_incr = float(spike_adapt_incr)
                spike_adapt_incr = tuple(spike_adapt_incr for _ in tc_adaptation)

            case (False, True):
                # cast spike_adapt_incr values
                spike_adapt_incr = tuple(float(val) for val in spike_adapt_incr)
                if len(spike_adapt_incr == 0):
                    raise ValueError("'spike_adapt_incr' cannot be empty.")

                # cast and test tc_adaptation and make it a tuple of matching length
                tc_adaptation, e = regroup(
                    numeric_limit("tc_adaptation", tc_adaptation, 0, "gt", float),
                    ((0 for _ in spike_adapt_incr), 1),
                )
                if e:
                    raise e

            case (False, False):
                # cast and test tc_adaptation and make it a 1-tuple
                tc_adaptation, e = regroup(
                    numeric_limit("tc_adaptation", tc_adaptation, 0, "gt", float),
                    ((0,), 1),
                )
                if e:
                    raise e

                # cast spike_adapt_incr and make it a 1-tuple
                spike_adapt_incr = (float(spike_adapt_incr),)

        # dynamics attributes
        self.step_time, e = numeric_limit("step_time", step_time, 0, "gt", float)
        if e:
            raise e
        self.tc_membrane, e = numeric_limit(
            "tc_membrane", tc_membrane, 0, "gt", float
        )
        if e:
            raise e
        self.decay = math.exp(-self.step_time / self.tc_membrane)
        self.rest_v = float(rest_v)
        self.reset_v_add = float(reset_v_add)
        self.reset_v_mul = float(reset_v_mul)
        self.thresh_eq_v = float(thresh_eq_v)
        self.refrac_t, e = numeric_limit("refrac_t", refrac_t, 0, "gte", float)
        if e:
            raise e
        self.resistance, e = numeric_limit("resistance", resistance, 0, "neq", float)
        if e:
            raise
        self.tc_adaptation = tc_adaptation
        self.spike_adapt_incr = spike_adapt_incr

        # register adaptation attributes as buffers (for tensor ops and compatibility)
        self.register_buffer(
            "adapt_decay",
            torch.exp(-self.step_time / torch.tensor(self.tc_adaptation)),
            persistent=False,
        )
        self.register_buffer(
            "adapt_increment", torch.tensor(self.spike_adapt_incr), persistent=False
        )

        # call mixin constructors
        VoltageMixin.__init__(self, torch.full(self.bshape, self.rest_v), False)
        SpikeRefractoryMixin.__init__(self, torch.zeros(self.bshape), False)
        AdaptationMixin.__init__(
            self, torch.zeros(*self.shape, len(self.tc_adaptation)), False
        )

    def _integrate_v(self, masked_inputs):
        """Internal, voltage function for :py:func:`~nf.voltage_thresholding`."""
        return nf.voltage_integration_linear(
            masked_inputs,
            self.voltage,
            decay=self.decay,
            rest_v=self.rest_v,
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
        self.decay = math.exp(-self.step_time / self.time_constant)
        self.adapt_decay = torch.exp(
            -self.step_time
            / torch.tensor(
                self.tc_adaptation,
                dtype=self.adapt_decay.dtype,
                device=self.adapt_decay.device,
                requires_grad=self.adapt_decay.requires_grad,
            )
        )

    def clear(self, keep_adaptations=True, **kwargs):
        r"""Resets neurons to their resting state.

        Args:
            keep_adaptations (bool, optional): if learned adaptations should be
                preserved. Defaults to True.
        """
        self.voltage = torch.full_like(self.voltage, self.rest_v)
        self.refrac = torch.zeros_like(self.refrac)
        if not keep_adaptations:
            self.adaptation = torch.zeros_like(self.adaptation)

    def forward(
        self,
        inputs: torch.Tensor,
        adapt: bool | None = None,
        refrac_lock: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        r"""Runs a simulation step of the neuronal dynamics.

        Args:
            inputs (torch.Tensor): presynaptic currents,
                :math:`I(t)`, in :math:`\text{nA}`.
            adapt (bool | None, optional): if adaptations should be updated
                based on this step. Defaults to None.
            refrac_lock (bool, optional): if membrane voltages should be fixed
                while in the refractory period. Defaults to True.

        Returns:
            torch.Tensor: if the corresponding neuron generated an action potential.

        Note:
            When ``adapt`` is set to None, adaptations will be updated when the neuron
            is in training mode but not when it is in evaluation mode.
        """
        # use naturalistic voltage thresholding function
        spikes, voltages, refracs = nf.voltage_thresholding_slope_intercept(
            inputs=inputs,
            refracs=self.refrac,
            dynamics=self._integrate_v,
            step_time=self.step_time,
            rest_v=self.rest_v,
            v_slope=self.reset_v_mul,
            v_intercept=self.reset_v_add,
            thresh_v=nf.neuron_adaptation.apply_adaptive_thresholds(
                self.thresh_eq_v, self.adaptation
            ),
            refrac_t=self.refrac_t,
            voltages=(self.voltage if refrac_lock else None),
        )

        # update parameters
        self.voltage = voltages
        self.refrac = refracs

        # conditionally update adaptive thresholds
        if adapt or (adapt is None and self.training):
            # use adaptive thresholds update function
            adaptations = nf.adaptive_thresholds_linear_spike(
                adaptations=self.adaptation,
                spikes=spikes,
                decay=self.adapt_decay,
                spike_increment=self.adapt_increment,
                refracs=(self.refrac if refrac_lock else None),
            )
            # update parameter
            self.adaptation = torch.mean(adaptations, dim=0)

        # return spiking output
        return spikes
