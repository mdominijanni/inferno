from .. import Neuron
from .. import functional as nf
from .mixins import AdaptationMixin, VoltageMixin, SpikeRefractoryMixin
from inferno._internal import numeric_limit, numeric_relative
from itertools import zip_longest
import torch
from typing import Callable


class LIF(VoltageMixin, SpikeRefractoryMixin, Neuron):
    r"""Simulation of leaky integrate-and-fire (LIF) neuron dynamics.

    .. math::
        V_m(t + \Delta t) = \left[ V_m(t) - V_\text{rest} - R_mI(t) \right]
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

        self.rest_v, self.thresh_v, e = numeric_relative(
            "rest_v", rest_v, "thresh_v", thresh_v, "lt", float
        )
        if e:
            raise e

        self.reset_v, _, e = numeric_relative(
            "reset_v", reset_v, "thresh_v", thresh_v, "lt", float
        )
        if e:
            raise e

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
        r"""Internal, voltage function for :py:func:`~nf.voltage_thresholding`."""
        return nf.voltage_integration_linear(
            masked_inputs,
            self.voltage,
            step_time=self.dt,
            time_constant=self.time_constant,
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
            V_m(t + \Delta t) &= \left[ V_m(t) - V_\text{rest} - R_mI(t) \right]
            \exp\left(-\frac{t}{\tau_m}\right) + V_\text{rest} + R_mI(t) \\
            \Theta(t) &= \Theta_\infty + \sum_k \theta_k(t) \\
            \theta_k(t + \Delta t) &= \theta_k(t) \exp\left(-\frac{\Delta t}{\tau_k}\right)
        \end{align*}

    If a spike was generated at time :math:`t`, then.

    .. math::
        \begin{align*}
            V_m(t) &\leftarrow V_\text{reset} \\
            \theta_k(t) &\leftarrow \theta_k(t) + d_k
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
        tc_adaptation (float | tuple[float, ...]): time constant of exponential decay
            for threshold adaptations, :math:`\tau_k`, in :math:`\text{ms}`.
        spike_increment (float | tuple[float, ...]): amount by which the adaptive
            threshold is increased after a spike, :math:`d_k`, in :math:`\text{mV}`.
        resistance (float, optional): resistance across the cell membrane,
            :math:`R_m`, in :math:`\text{M}\Omega`. Defaults to 1.0.
        batch_size (int, optional): size of input batches for simualtion. Defaults to 1.
        batch_reduction (Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] | None):
            function to reduce adaptation updates over the batch dimension,
            :py:func:`torch.mean` when None. Defaults to None.

    Note:
        ``batch_reduction`` can be one of the functions in PyTorch including but not
        limited to :py:func:`torch.sum`, :py:func:`torch.max` and :py:func:`torch.max`.
        A custom function with similar behavior can also be passed in. Like with the
        included function, it should not keep the original dimensions by default.

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
        spike_increment: float | tuple[float, ...],
        resistance: float = 1.0,
        batch_size: int = 1,
        batch_reduction: (
            Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] | None
        ) = None,
    ):
        # call superclass constructor
        Neuron.__init__(self, shape, batch_size)

        # process adapation attributes
        # tuple-wrap if singleton
        if not hasattr(tc_adaptation, "__iter__"):
            tc_adaptation = (tc_adaptation,)
        if not hasattr(spike_increment, "__iter__"):
            spike_increment = (spike_increment,)

        # prepare converted lists
        tcL, siL = [], []

        # test values
        for idx, (tcA, siA) in enumerate(zip_longest(tc_adaptation, spike_increment)):
            # time constant of adaptation
            if tcA is None:
                tcL.append(tcL[-1])
            else:
                v, e = numeric_limit(f"tc_adaptation[{idx}]", tcA, 0, "gt", float)
                if e:
                    raise e
                tcL.append(v)

            # threshold spike increment
            if siL is None:
                siL.append(siL[-1])
            else:
                siL.append(float(siA))

        # reassign
        tc_adaptation, spike_increment = tcL, siL

        # dynamics attributes
        self.step_time, e = numeric_limit("step_time", step_time, 0, "gt", float)
        if e:
            raise e

        self.tc_membrane, e = numeric_limit("tc_membrane", tc_membrane, 0, "gt", float)
        if e:
            raise e

        self.rest_v, self.thresh_eq_v, e = numeric_relative(
            "rest_v", rest_v, "thresh_eq_v", thresh_eq_v, "lt", float
        )
        if e:
            raise e

        self.reset_v, _, e = numeric_relative(
            "reset_v", reset_v, "thresh_eq_v", thresh_eq_v, "lt", float
        )
        if e:
            raise e

        self.refrac_t, e = numeric_limit("refrac_t", refrac_t, 0, "gte", float)
        if e:
            raise e

        self.resistance, e = numeric_limit("resistance", resistance, 0, "neq", float)
        if e:
            raise

        # register adaptation attributes as buffers (for tensor ops and compatibility)
        self.register_buffer(
            "tc_adaptation", torch.tensor(tc_adaptation), persistent=False
        )
        self.register_buffer(
            "adapt_increment", torch.tensor(spike_increment), persistent=False
        )

        # call mixin constructors
        VoltageMixin.__init__(self, torch.full(self.bshape, self.rest_v), False)
        SpikeRefractoryMixin.__init__(self, torch.zeros(self.bshape), False)
        AdaptationMixin.__init__(
            self,
            torch.zeros(*self.shape, self.tc_adaptation.numel()),
            False,
            batch_reduction,
        )

    def _integrate_v(self, masked_inputs):
        """Internal, voltage function for :py:func:`~nf.voltage_thresholding`."""
        return nf.voltage_integration_linear(
            masked_inputs,
            self.voltage,
            step_time=self.dt,
            time_constant=self.tc_membrane,
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
                step_time=self.dt,
                time_constant=self.tc_adaptation,
                spike_increment=self.adapt_increment,
                refracs=(self.refrac if refrac_lock else None),
            )
            # update parameter
            self.adaptation = adaptations

        # return spiking output
        return spikes


class GLIF1(VoltageMixin, SpikeRefractoryMixin, Neuron):
    r"""Simulation of generalized leaky integrate-and-fire 1 (GLIF\ :sub:`1`) neuron dynamics.

    Alias for :py:class:`~inferno.neural.LIF`.

    See Also:
        For more details and references, visit
        :ref:`zoo/neurons-linear:generalized leaky integrate-and-fire 1 (glif{sub}\`1\`)`
        in the zoo.
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
            V_m(t + \Delta t)&= \left[ V_m(t) - V_\text{rest} - R_mI(t) \right]
            \exp\left(-\frac{t}{\tau_m}\right) + V_\text{rest} + R_mI(t) \\
            \Theta(t) &= \Theta_\infty + \sum_k \theta_k(t) \\
            \theta_k(t + \Delta t) &= \theta_k(t) \exp\left(-\lambda_k \Delta t\right)
        \end{align*}

    If a spike was generated at time :math:`t`, then.

    .. math::
        \begin{align*}
            V_m(t) &\leftarrow V_\text{rest} + m_v \left[ V_m(t) - V_\text{rest} \right] - b_v \\
            \theta_k(t) &\leftarrow \theta_k(t) + d_k
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
        rc_adaptation (float | tuple[float, ...]): rate constant of exponential decay
            for threshold adaptations, :math:`\lambda_k`, in :math:`\text{ms^{-1}}`.
        spike_increment (float | tuple[float, ...]): amount by which the adaptive
            threshold is increased after a spike, :math:`d_k`, in :math:`\text{mV}`.
        resistance (float, optional): resistance across the cell membrane,
            :math:`R_m`, in :math:`\text{M}\Omega`. Defaults to 1.0.
        batch_size (int, optional): size of input batches for simualtion. Defaults to 1.
        batch_reduction (Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] | None):
            function to reduce adaptation updates over the batch dimension,
            :py:func:`torch.mean` when None. Defaults to None.

    Note:
        ``batch_reduction`` can be one of the functions in PyTorch including but not
        limited to :py:func:`torch.sum`, :py:func:`torch.max` and :py:func:`torch.max`.
        A custom function with similar behavior can also be passed in. Like with the
        included function, it should not keep the original dimensions by default.

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
        rc_adaptation: float | tuple[float, ...],
        spike_increment: float | tuple[float, ...],
        resistance: float = 1.0,
        batch_size: int = 1,
        batch_reduction: (
            Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] | None
        ) = None,
    ):
        # call superclass constructor
        Neuron.__init__(self, shape, batch_size)

        # process adapation attributes
        # tuple-wrap if singleton
        if not hasattr(rc_adaptation, "__iter__"):
            rc_adaptation = (rc_adaptation,)
        if not hasattr(spike_increment, "__iter__"):
            spike_increment = (spike_increment,)

        # prepare converted lists
        rcL, siL = [], []

        # test values
        for idx, (rcA, siA) in enumerate(zip_longest(rc_adaptation, spike_increment)):
            # time constant of adaptation
            if rcA is None:
                rcL.append(rcL[-1])
            else:
                v, e = numeric_limit(f"rc_adaptation[{idx}]", rcA, 0, "gt", float)
                if e:
                    raise e
                rcL.append(v)

            # threshold spike increment
            if siL is None:
                siL.append(siL[-1])
            else:
                siL.append(float(siA))

        # reassign
        rc_adaptation, spike_increment = rcL, siL

        # dynamics attributes
        self.step_time, e = numeric_limit("step_time", step_time, 0, "gt", float)
        if e:
            raise e

        self.tc_membrane, e = numeric_limit("tc_membrane", tc_membrane, 0, "gt", float)
        if e:
            raise e

        self.rest_v, self.thresh_eq_v, e = numeric_relative(
            "rest_v", rest_v, "thresh_eq_v", thresh_eq_v, "lt", float
        )
        if e:
            raise e

        self.reset_v_add = float(reset_v_add)
        self.reset_v_mul = float(reset_v_mul)

        self.refrac_t, e = numeric_limit("refrac_t", refrac_t, 0, "gte", float)
        if e:
            raise e

        self.resistance, e = numeric_limit("resistance", resistance, 0, "neq", float)
        if e:
            raise

        # register adaptation attributes as buffers (for tensor ops and compatibility)
        self.register_buffer(
            "rc_adaptation", torch.tensor(rc_adaptation), persistent=False
        )
        self.register_buffer(
            "adapt_increment", torch.tensor(spike_increment), persistent=False
        )

        # call mixin constructors
        VoltageMixin.__init__(self, torch.full(self.bshape, self.rest_v), False)
        SpikeRefractoryMixin.__init__(self, torch.zeros(self.bshape), False)
        AdaptationMixin.__init__(
            self,
            torch.zeros(*self.shape, self.rc_adaptation.numel()),
            False,
            batch_reduction,
        )

    def _integrate_v(self, masked_inputs):
        """Internal, voltage function for :py:func:`~nf.voltage_thresholding`."""
        return nf.voltage_integration_linear(
            masked_inputs,
            self.voltage,
            step_time=self.dt,
            time_constant=self.tc_membrane,
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
                step_time=self.dt,
                time_constant=1 / self.rc_adaptation,
                spike_increment=self.adapt_increment,
                refracs=(self.refrac if refrac_lock else None),
            )
            # update parameter
            self.adaptation = adaptations

        # return spiking output
        return spikes
