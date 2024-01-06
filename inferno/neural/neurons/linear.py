from .. import Neuron
from .. import functional as nf
import math
import torch
from ._mixins import AdaptationMixin, VoltageMixin, SpikeRefractoryMixin


class LIF(VoltageMixin, SpikeRefractoryMixin, Neuron):
    r"""Simulation of leaky integrate-and-fire (LIF) neuron dynamics.

    .. math::
        V_m(t + \Delta t) = \left[V_m(t) - V_\mathrm{rest} - R_mI(t)\right]
        \exp\left(-\frac{t}{\tau_m}\right) + V_\mathrm{rest} + R_mI(t)

    If a spike was generated at time :math:`t`, then.

    .. math::
        V_m(t) \leftarrow V_\mathrm{reset}

    Args:
        shape (tuple[int, ...] | int): shape of the group of neurons being simulated.
        step_time (float): length of a simulation time step,
            :math:`\Delta t`, in :math:`\mathrm{ms}`.
        rest_v (float): membrane potential difference at equilibrium,
            :math:`V_\mathrm{rest}`, in :math:`\mathrm{mV}`.
        reset_v (float): membrane voltage after an action potential is generated,
            :math:`V_\mathrm{reset}`, in :math:`\mathrm{mV}`.
        thresh_v (float): membrane voltage at which action potentials are generated,
            in :math:`\mathrm{mV}`.
        refrac_t (float): length of time the absolute refractory period lasts,
            in :math:`\mathrm{ms}`.
        time_constant (float): time constant of exponential decay for membrane voltage,
            :math:`\tau_m`, in :math:`\mathrm{ms}`.
        resistance (float, optional): resistance across the cell membrane,
            :math:`R_m`, in :math:`\mathrm{M\Omega}`. Defaults to 1.0.
        batch_size (int, optional): size of input batches for simualtion. Defaults to 1.

    Raises:
        ValueError: ``step_time`` must be a positive real.

    Note:
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

        # check that the step time is valid
        if float(step_time) <= 0:
            raise ValueError(f"step time must be positive, received {float(step_time)}")

        # check that absolute refractory period is valid
        if float(refrac_t) <= 0:
            raise ValueError(f"refrac_t be positive, received {float(refrac_t)}")

        # register extras
        self.register_extra("step_time", float(step_time))
        self.register_extra("time_constant", float(time_constant))

        # register buffers
        self.register_buffer(
            "decay", torch.tensor(math.exp(-self.step_time / self.time_constant))
        )
        self.register_buffer("rest_v", torch.tensor(float(rest_v)))
        self.register_buffer("reset_v", torch.tensor(float(reset_v)))
        self.register_buffer("thresh_v", torch.tensor(float(thresh_v)))
        self.register_buffer("refrac_t", torch.tensor(float(refrac_t)))
        self.register_buffer("resistance", torch.tensor(float(resistance)))

        # call mixin constructors
        VoltageMixin.__init__(self, torch.full(self.bshape, self.rest_v), False)
        SpikeRefractoryMixin.__init__(self, torch.zeros(self.bshape), False)

    def _integrate_v(self, masked_inputs):
        return nf.voltage_integration_linear(
            masked_inputs,
            self.voltage,
            decay=self.decay,
            rest_v=self.rest_v,
            resistance=self.resistance,
        )

    def clear(self, **kwargs):
        r"""Resets neurons to their resting state."""
        self.voltage = torch.full_like(self.voltage, self.rest_v)
        self.refrac = torch.zeros_like(self.refrac)

    def forward(self, inputs: torch.Tensor, refrac_lock=True, **kwargs) -> torch.Tensor:
        r"""Runs a simulation step of the neuronal dynamics.

        Args:
            inputs (torch.Tensor): presynaptic currents,
                :math:`I(t)`, in :math:`\mathrm{nA}`.
            refrac_lock (bool, optional): if membrane voltages should be fixed while in the
                refractory period. Defaults to True.

        Returns:
            torch.Tensor: which neurons generated an action potential.
        """
        # use voltage thresholding function
        spikes, voltages, refracs = nf.voltage_thresholding(
            inputs=inputs,
            refracs=self.refrac,
            voltage_fn=self._integrate_v,
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

    @property
    def dt(self) -> float:
        r"""Length of the simulation time step, in milliseconds.

        Args:
            value (float): new simulation time step length.

        Returns:
            float: length of the simulation time step
        """
        return self.step_time

    @dt.setter
    def dt(self, value: float):
        if float(value) <= 0:
            raise ValueError(f"step time must be positive, received {float(value)}")
        self.step_time = float(value)
        self.decay.fill_(math.exp(-self.step_time / self.time_constant))


class ALIF(AdaptationMixin, VoltageMixin, SpikeRefractoryMixin, Neuron):
    r"""Simulation of adaptive leaky integrate-and-fire (ALIF) neuron dynamics.

    ALIF is implemented as a step of leaky integrate-and-fire applying existing adaptations,
    using linear spike-dependent adaptive thresholds, then updating those adaptations for the
    next timestep.

    .. math::
        \begin{align*}
            V_m(t + \Delta t) &= \left[V_m(t) - V_\mathrm{rest} - R_mI(t)\right]
            \exp\left(-\frac{t}{\tau_m}\right) + V_\mathrm{rest} + R_mI(t) \\
            \Theta(t) &= \Theta_\infty + \sum_k \theta_k(t) \\
            \theta_k(t + \Delta t) &= \theta_k(t) \exp\left(-\frac{\Delta t}{\tau_k}\right)
        \end{align*}

    If a spike was generated at time :math:`t`, then.

    .. math::
        \begin{align*}
            V_m(t) &\leftarrow V_\mathrm{reset} \\
            \theta_k(t) &\leftarrow \theta_k(t) + a_k
        \end{align*}

    Args:
        shape (tuple[int, ...] | int): shape of the group of neurons being simulated.
        step_time (float): length of a simulation time step,
            :math:`\Delta t`, in :math:`\mathrm{ms}`.
        rest_v (float): membrane potential difference at equilibrium,
            :math:`V_\mathrm{rest}`, in :math:`\mathrm{mV}`.
        reset_v (float): membrane voltage after an action potential is generated,
            :math:`V_\mathrm{reset}`, in :math:`\mathrm{mV}`.
        thresh_eq_v (float): equilibrium of the firing threshold,
            :math:`\Theta_\infty$`, in :math:`\mathrm{mV}`.
        refrac_t (float): length of time the absolute refractory period lasts,
            in :math:`\mathrm{ms}`.
        tc_membrane (float): time constant of exponential decay for membrane voltage,
            :math:`\tau_m`, in :math:`\mathrm{ms}`.
        tc_adaptation (float | tuple[float]): time constant of exponential decay for threshold adaptations,
            :math:`\tau_k`, in :math:`\mathrm{ms}`.
        spike_adapt_increment (float | tuple[float]): amount by which the adaptive threshold is increased
            after a spike, :math:`a_k`, in :math:`\mathrm{mV}`.
        resistance (float, optional): resistance across the cell membrane,
            :math:`R_m`, in :math:`\mathrm{M\Omega}`. Defaults to 1.0.
        batch_size (int, optional): size of input batches for simualtion. Defaults to 1.

    Raises:
        ValueError: ``step_time`` must be a positive real.
        RuntimeError: if ``tc_adaptation`` and ``spike_adapt_increment`` are tuples, they must
            be of equal length.

    Note:
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
        spike_adapt_increment: float | tuple[float, ...],
        resistance: float = 1.0,
        batch_size: int = 1,
    ):
        # call superclass constructor
        Neuron.__init__(self, shape, batch_size)

        # check that the step time is valid
        if float(step_time) <= 0:
            raise ValueError(f"step time must be positive, received {float(step_time)}")

        # check that absolute refractory period is valid
        if float(refrac_t) <= 0:
            raise ValueError(f"refrac_t be positive, received {float(refrac_t)}")

        # check that tc_adapt and incr_adapt are of equal length
        if isinstance(tc_adaptation, list | tuple):
            if isinstance(spike_adapt_increment, list | tuple):
                if len(tc_adaptation) != len(spike_adapt_increment):
                    raise RuntimeError(
                        "`tc_adaptation` and `spike_adapt_increment` "
                        "must be equal in length"
                    )
            else:
                tc_adaptation = tuple(float(tc) for tc in tc_adaptation)
                spike_adapt_increment = tuple(
                    float(spike_adapt_increment) for _ in tc_adaptation
                )
        else:
            if isinstance(spike_adapt_increment, list | tuple):
                tc_adaptation = tuple(
                    float(tc_adaptation) for _ in spike_adapt_increment
                )
                spike_adapt_increment = tuple(
                    float(sai) for sai in spike_adapt_increment
                )
            else:
                tc_adaptation = (float(tc_adaptation),)
                spike_adapt_increment = (float(spike_adapt_increment),)

        # register extras
        self.register_extra("step_time", float(step_time))
        self.register_extra("tc_membrane", float(tc_membrane))
        self.register_extra("tc_adaptation", tc_adaptation)

        # register buffers
        self.register_buffer(
            "decay_adaptation",
            torch.tensor([math.exp(-self.step_time / tc) for tc in self.tc_adaptation]),
        )
        self.register_buffer("incr_adaptation", torch.tensor(spike_adapt_increment))
        self.register_buffer(
            "decay_membrane",
            torch.tensor(math.exp(-self.step_time / self.tc_membrane)),
        )
        self.register_buffer("rest_v", torch.tensor(float(rest_v)))
        self.register_buffer("reset_v", torch.tensor(float(reset_v)))
        self.register_buffer("thresh_eq_v", torch.tensor(float(thresh_eq_v)))
        self.register_buffer("refrac_t", torch.tensor(float(refrac_t)))
        self.register_buffer("resistance", torch.tensor(float(resistance)))

        # call mixin constructors
        VoltageMixin.__init__(self, torch.full(self.bshape, self.rest_v), False)
        SpikeRefractoryMixin.__init__(self, torch.zeros(self.bshape), False)
        AdaptationMixin.__init__(
            self, torch.zeros(*self.shape, self.decay_adaptation.shape[-1]), False
        )

    def _integrate_v(self, masked_inputs):
        return nf.voltage_integration_linear(
            masked_inputs,
            self.voltage,
            decay=self.decay,
            rest_v=self.rest_v,
            resistance=self.resistance,
        )

    def clear(self, keep_adaptations=True, **kwargs):
        r"""Resets neurons to their resting state.

        Args:
            keep_adaptations (bool, optional): if learned adaptations should be preserved. Defaults to True.
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
                :math:`I(t)`, in :math:`\mathrm{nA}`.
            adapt (bool | None, optional): if adaptations should be updated based on this step.
                Defaults to None.
            refrac_lock (bool, optional): if membrane voltages should be fixed while in the
                refractory period. Defaults to True.

        Returns:
            torch.Tensor: which neurons generated an action potential.

        Note:
            When ``adapt`` is set to None, adaptations will be updated when the neuron is in
            training mode but not when it is in evaluation mode.
        """
        # use voltage thresholding function
        spikes, voltages, refracs = nf.voltage_thresholding(
            inputs=inputs,
            refracs=self.refrac,
            voltage_fn=self._integrate_v,
            step_time=self.step_time,
            reset_v=self.reset_v,
            thresh_v=nf.neuron_adaptation.apply_adaptive_thresholds(
                self.thresh_eq_v, self.adaptations
            ),
            refrac_t=self.refrac_t,
            voltages=(self.voltage if refrac_lock else None),
        )
        self.voltage = voltages
        self.refrac = refracs

        # use adaptive thresholds update function
        if adapt or (adapt is None and self.training):
            adaptations = nf.adaptive_thresholds_linear_spike(
                adaptations=self.adaptation,
                postsyn_spikes=spikes,
                decay=self.decay_adaptation,
                spike_increment=self.incr_adaptation,
                refracs=(self.refracs if refrac_lock else None),
            )
            self.adaptation = torch.mean(adaptations, dim=0)

        # return spiking output
        return spikes

    @property
    def dt(self) -> float:
        r"""Length of the simulation time step, in milliseconds.

        Args:
            value (float): new simulation time step length.

        Returns:
            float: length of the simulation time step
        """
        return self.step_time

    @dt.setter
    def dt(self, value: float):
        if float(value) <= 0:
            raise ValueError(f"step time must be positive, received {float(value)}")
        self.step_time = float(value)
        self.decay_membrane.fill_(math.exp(-self.step_time / self.time_constant))
        self.decay_adaptation[:] = torch.tensor(
            [math.exp(-self.step_time / tc) for tc in self.tc_adaptation]
        )


class GLIF1(VoltageMixin, SpikeRefractoryMixin, Neuron):
    r"""Simulation of generalized leaky integrate-and-fire 1 (GLIF\ :sub:`1`) neuron dynamics.

    Alias for :py:class:`~inferno.neural.LIF`.

    Note:
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
        return LIF._integrate_v(self, masked_inputs)

    def clear(self, **kwargs):
        r"""Resets neurons to their resting state."""
        LIF.clear(self, **kwargs)

    def forward(self, inputs: torch.Tensor, refrac_lock=True, **kwargs) -> torch.Tensor:
        r"""Runs a simulation step of the neuronal dynamics.

        Args:
            inputs (torch.Tensor): presynaptic currents,
                :math:`I(t)`, in :math:`\mathrm{nA}`.
            refrac_lock (bool, optional): if membrane voltages should be fixed while in the
                refractory period. Defaults to True.

        Returns:
            torch.Tensor: which neurons generated an action potential.
        """
        return LIF.forward(inputs, refrac_lock=refrac_lock)

    @property
    def dt(self) -> float:
        r"""Length of the simulation time step, in milliseconds.

        Args:
            value (float): new simulation time step length.

        Returns:
            float: length of the simulation time step
        """
        return LIF.dt.fget(self)

    @dt.setter
    def dt(self, value: float):
        LIF.dt.fset(self, value)


class GLIF2(AdaptationMixin, VoltageMixin, SpikeRefractoryMixin, Neuron):
    r"""Simulation of generalized leaky integrate-and-fire 2 (GLIF\ :sub:`2`) neuron dynamics.

    .. math::
        \begin{align*}
            V_m(t + \Delta t) &= \left[V_m(t) - V_\mathrm{rest} - R_mI(t)\right]
            \exp\left(-\frac{t}{\tau_m}\right) + V_\mathrm{rest} + R_mI(t) \\
            \Theta(t) &= \Theta_\infty + \sum_k \theta_k(t) \\
            \theta_k(t + \Delta t) &= \theta_k(t) \exp\left(-\frac{\Delta t}{\tau_k}\right)
        \end{align*}

    If a spike was generated at time :math:`t`, then.

    .. math::
        \begin{align*}
            V_m(t) &\leftarrow V_\mathrm{rest} + m_v \left[ V_m(t) - V_\mathrm{rest} \right] - b_v \\
            \theta_k(t) &\leftarrow \theta_k(t) + a_k
        \end{align*}

    Args:
        shape (tuple[int, ...] | int): shape of the group of neurons being simulated.
        step_time (float): length of a simulation time step,
            :math:`\Delta t`, in :math:`\mathrm{ms}`.
        rest_v (float): membrane potential difference at equilibrium,
            :math:`V_\mathrm{rest}`, in :math:`\mathrm{mV}`.
        reset_v_add (float): additive parameter controlling reset voltage,
            :math:`b_v`, in :math:`\mathrm{mV}`.
        reset_v_mul (float): multiplicative parameter controlling reset voltage,
            :math:`m_v`, unitless.
        thresh_eq_v (float): equilibrium of the firing threshold,
            :math:`\Theta_\infty$`, in :math:`\mathrm{mV}`.
        refrac_t (float): length of time the absolute refractory period lasts,
            in :math:`\mathrm{ms}`.
        tc_membrane (float): time constant of exponential decay for membrane voltage,
            :math:`\tau_m`, in :math:`\mathrm{ms}`.
        tc_adaptation (float | tuple[float]): time constant of exponential decay for threshold adaptations,
            :math:`\tau_k`, in :math:`\mathrm{ms}`.
        spike_adapt_increment (float | tuple[float]): amount by which the adaptive threshold is increased
            after a spike, :math:`a_k`, in :math:`\mathrm{mV}`.
        resistance (float, optional): resistance across the cell membrane,
            :math:`R_m`, in :math:`\mathrm{M\Omega}`. Defaults to 1.0.
        batch_size (int, optional): size of input batches for simualtion. Defaults to 1.

    Raises:
        ValueError: ``step_time`` must be a positive real.
        RuntimeError: if ``tc_adaptation`` and ``spike_adapt_increment`` are tuples, they must
            be of equal length.

    Note:
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
        spike_adapt_increment: float | tuple[float],
        resistance: float = 1.0,
        batch_size: int = 1,
    ):
        # call superclass constructor
        Neuron.__init__(self, shape, batch_size)

        # check that the step time is valid
        if float(step_time) <= 0:
            raise ValueError(f"step time must be positive, received {float(step_time)}")

        # check that absolute refractory period is valid
        if float(refrac_t) <= 0:
            raise ValueError(f"refrac_t be positive, received {float(refrac_t)}")

        # check that tc_adapt and incr_adapt are of equal length
        if isinstance(tc_adaptation, list | tuple):
            if isinstance(spike_adapt_increment, list | tuple):
                if len(tc_adaptation) != len(spike_adapt_increment):
                    raise RuntimeError(
                        "`tc_adaptation` and `spike_adapt_increment` "
                        "must be equal in length"
                    )
            else:
                tc_adaptation = tuple(float(tc) for tc in tc_adaptation)
                spike_adapt_increment = tuple(
                    float(spike_adapt_increment) for _ in tc_adaptation
                )
        else:
            if isinstance(spike_adapt_increment, list | tuple):
                tc_adaptation = tuple(
                    float(tc_adaptation) for _ in spike_adapt_increment
                )
                spike_adapt_increment = tuple(
                    float(sai) for sai in spike_adapt_increment
                )
            else:
                tc_adaptation = (float(tc_adaptation),)
                spike_adapt_increment = (float(spike_adapt_increment),)

        # register extras
        self.register_extra("step_time", float(step_time))
        self.register_extra("tc_membrane", float(tc_membrane))
        self.register_extra("tc_adaptation", tc_adaptation)

        # register buffers
        self.register_buffer(
            "decay_adaptation",
            torch.tensor([math.exp(-self.step_time / tc) for tc in self.tc_adaptation]),
        )
        self.register_buffer("incr_adaptation", torch.tensor(spike_adapt_increment))
        self.register_buffer(
            "decay_membrane", torch.tensor(math.exp(-self.step_time / self.tc_membrane))
        )
        self.register_buffer("rest_v", torch.tensor(float(rest_v)))
        self.register_buffer("reset_v_add", torch.tensor(float(reset_v_add)))
        self.register_buffer("reset_v_mul", torch.tensor(float(reset_v_mul)))
        self.register_buffer("thresh_eq_v", torch.tensor(float(thresh_eq_v)))
        self.register_buffer("refrac_t", torch.tensor(float(refrac_t)))
        self.register_buffer("resistance", torch.tensor(float(resistance)))

        # call mixin constructors
        VoltageMixin.__init__(self, torch.full(self.bshape, self.rest_v), False)
        SpikeRefractoryMixin.__init__(self, torch.zeros(self.bshape), False)
        AdaptationMixin.__init__(
            self, torch.zeros(*self.shape, self.decay_adaptation.shape[-1]), False
        )

    def _integrate_v(self, masked_inputs):
        return nf.voltage_integration_linear(
            masked_inputs,
            self.voltage,
            decay=self.decay,
            rest_v=self.rest_v,
            resistance=self.resistance,
        )

    def clear(self, keep_adaptations=True, **kwargs):
        r"""Resets neurons to their resting state.

        Args:
            keep_adaptations (bool, optional): if learned adaptations should be preserved. Defaults to True.
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
                :math:`I(t)`, in :math:`\mathrm{nA}`.
            adapt (bool | None, optional): if adaptations should be updated based on this step.
                Defaults to None.
            refrac_lock (bool, optional): if membrane voltages should be fixed while in the
                refractory period. Defaults to True.

        Returns:
            torch.Tensor: which neurons generated an action potential.

        Note:
            When ``adapt`` is set to None, adaptations will be updated when the neuron is in
            training mode but not when it is in evaluation mode.
        """
        # use naturalistic voltage thresholding function
        spikes, voltages, refracs = nf.voltage_thresholding_slope_intercept(
            inputs=inputs,
            refracs=self.refrac,
            voltage_fn=self._integrate_v,
            step_time=self.step_time,
            rest_v=self.rest_v,
            v_slope=self.reset_v_mul,
            v_intercept=self.reset_v_add,
            thresh_v=nf.neuron_adaptation.apply_adaptive_thresholds(
                self.thresh_eq_v, self.adaptations
            ),
            refrac_t=self.refrac_t,
            voltages=(self.voltage if refrac_lock else None),
        )
        self.voltage = voltages
        self.refrac = refracs

        # use adaptive thresholds update function
        if adapt or (adapt is None and self.training):
            adaptations = nf.adaptive_thresholds_linear_spike(
                adaptations=self.adaptation,
                postsyn_spikes=spikes,
                decay=self.decay_adaptation,
                spike_increment=self.incr_adaptation,
                refracs=(self.refracs if refrac_lock else None),
            )
            self.adaptation = torch.mean(adaptations, dim=0)

        # return spiking output
        return spikes

    @property
    def dt(self) -> float:
        r"""Length of the simulation time step, in milliseconds.

        Args:
            value (float): new simulation time step length.

        Returns:
            float: length of the simulation time step
        """
        return self.step_time

    @dt.setter
    def dt(self, value: float):
        if float(value) <= 0:
            raise ValueError(f"step time must be positive, received {float(value)}")
        self.step_time = float(value)
        self.decay_membrane.fill_(math.exp(-self.step_time / self.time_constant))
        self.decay_adaptation[:] = torch.tensor(
            [math.exp(-self.step_time / tc) for tc in self.tc_adaptation]
        )
