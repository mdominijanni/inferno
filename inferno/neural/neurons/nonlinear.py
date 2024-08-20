from .mixins import AdaptiveCurrentMixin, VoltageMixin, SpikeRefractoryMixin
from .. import functional as nf
from ..base import InfernoNeuron
from ..._internal import argtest
from itertools import zip_longest
import torch
from typing import Callable


class QIF(VoltageMixin, SpikeRefractoryMixin, InfernoNeuron):
    r"""Simulation of quadratic integrate-and-fire (QIF) neuron dynamics.

    Implemented as an approximation using Euler's method.

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
        affinity (float): controls the strength of the membrane
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
            :math:`R_m`, in :math:`\text{M}\Omega`. Defaults to ``1.0``.
        batch_size (int, optional): size of input batches for simulation. Defaults to ``1``.

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
        affinity: float,
        reset_v: float,
        thresh_v: float,
        refrac_t: float,
        time_constant: float,
        resistance: float = 1.0,
        batch_size: int = 1,
    ):
        # call superclass constructor
        InfernoNeuron.__init__(self, shape, batch_size)

        # dynamics attributes
        self.step_time = argtest.gt("step_time", step_time, 0, float)
        self.rest_v = argtest.lt("rest_v", rest_v, crit_v, float, "crit_v")
        self.crit_v = argtest.lte("crit_v", crit_v, thresh_v, float, "thresh_v")
        self.affinity = argtest.gt("affinity", affinity, 0, float)
        self.reset_v = argtest.lt("reset_v", reset_v, thresh_v, float, "thresh_v")
        self.thresh_v = float(thresh_v)
        self.refrac_t = argtest.gte("refrac_t", refrac_t, 0, float)
        self.time_constant = argtest.gt("time_constant", time_constant, 0, float)
        self.resistance = argtest.neq("resistance", resistance, 0, float)

        # call mixin constructors
        VoltageMixin.__init__(self, torch.full(self.batchedshape, self.rest_v))
        SpikeRefractoryMixin.__init__(self, torch.zeros(self.batchedshape), "refrac_t")

    def _integrate_v(self, masked_inputs):
        r"""Internal, voltage integrated function as input for thresholding."""
        return nf.voltage_integration_quadratic(
            masked_inputs,
            self.voltage,
            step_time=self.step_time,
            rest_v=self.rest_v,
            crit_v=self.crit_v,
            affinity=self.affinity,
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
    def dt(self, value: float) -> None:
        self.step_time = argtest.gt("dt", value, 0, float)

    def clear(self, **kwargs) -> None:
        r"""Resets neurons to their resting state."""
        self.voltage = torch.full_like(self.voltage, self.rest_v)
        self.refrac = torch.zeros_like(self.refrac)

    def forward(self, inputs: torch.Tensor, refrac_lock=True, **kwargs) -> torch.Tensor:
        r"""Runs a simulation step of the neuronal dynamics.

        Args:
            inputs (torch.Tensor): presynaptic currents,
                :math:`I(t)`, in :math:`\text{nA}`.
            refrac_lock (bool, optional): if membrane voltages should be fixed while
                in the refractory period. Defaults to ``True``.

        Returns:
            torch.Tensor: if the corresponding neuron generated an action potential.
        """
        # use voltage thresholding function
        spikes, voltages, refracs = nf.voltage_thresholding_constant(
            inputs=inputs,
            refracs=self.refrac,
            dynamics=self._integrate_v,
            voltages=(self.voltage if refrac_lock else None),
            step_time=self.step_time,
            reset_v=self.reset_v,
            thresh_v=self.thresh_v,
            refrac_t=self.refrac_t,
        )

        # update parameters
        self.voltage = voltages
        self.refrac = refracs

        # return spiking output
        return spikes


class Izhikevich(
    AdaptiveCurrentMixin, VoltageMixin, SpikeRefractoryMixin, InfernoNeuron
):
    r"""Simulation of Izhikevich (adaptive quadratic) neuron dynamics.

    Implemented as an approximation using Euler's method.

    .. math::
        \begin{align*}
            V_m(t + \Delta t) &= \frac{\Delta t}{\tau_m} \left[ a \left(V_m(t)
            - V_\text{rest}\right)\left(V_m(t)
            - V_\text{crit}\right) + R_mI(t) \right] + V_m(t) \\
            I(t) &= I_x(t) - \sum_k w_k(t) \\
            w_k(t + \Delta t) &= \frac{\Delta t}{\tau_k}\left[ b_k
            \left[ V_m(t) - V_\text{rest} \right] - w_k(t) \right] + w_k(t)
        \end{align*}

    If a spike was generated at time :math:`t`, then.

    .. math::
        \begin{align*}
            V_m(t) &\leftarrow V_\text{reset} \\
            w_k(t) &\leftarrow w_k(t) + d_k
        \end{align*}

    Args:
        shape (tuple[int, ...] | int): shape of the group of neurons being simulated.
        step_time (float): length of a simulation time step,
            :math:`\Delta t`, in :math:`\text{ms}`.
        rest_v (float): membrane potential difference at equilibrium,
            :math:`V_\text{rest}`, in :math:`\text{mV}`.
        crit_v (float): membrane potential difference at which potential
            naturally increases, :math:`V_\text{crit}`, in :math:`\text{mV}`.
        affinity (float): controls the strength of the membrane
            potential's drift towards :math:`V_\text{rest}` and away from
            :math:`V_\text{crit}`, :math:`a`, unitless.
        reset_v (float): membrane voltage after an action potential is generated,
            :math:`V_\text{reset}`, in :math:`\text{mV}`.
        thresh_v (float): membrane voltage at which action potentials are generated,
            in :math:`\text{mV}`.
        refrac_t (float): length the absolute refractory period, in :math:`\text{ms}`.
        tc_membrane (float): time constant of exponential decay for membrane voltage,
            :math:`\tau_m`, in :math:`\text{ms}`.
        tc_adaptation (float | tuple[float, ...]): time constant of exponential decay
            for threshold adaptations, :math:`\tau_k`, in :math:`\text{ms}`.
        voltage_coupling (float | tuple[float, ...]): strength of coupling to membrane
            voltage, :math:`b_k`, in :math:`\mu\text{S}`.
        spike_increment (float | tuple[float, ...]): amount by which the adaptive
            current is increased after a spike, :math:`d_k`, in :math:`\text{nA}`.
        resistance (float, optional): resistance across the cell membrane,
            :math:`R_m`, in :math:`\text{M}\Omega`. Defaults to ``1.0``.
        batch_size (int, optional): size of input batches for simulation. Defaults to ``1``.

    Note:
        ``batch_reduction`` can be one of the functions in PyTorch including but not
        limited to :py:func:`torch.sum`, :py:func:`torch.mean`, and :py:func:`torch.amax`.
        A custom function with similar behavior can also be passed in. Like with the
        included function, it should not keep the original dimensions by default.

    See Also:
        For more details and references, visit
        :ref:`zoo/neurons-nonlinear:Izhikevich (Adaptive Quadratic)`
        in the zoo.
    """

    def __init__(
        self,
        shape: tuple[int, ...] | int,
        step_time: float,
        *,
        rest_v: float,
        crit_v: float,
        affinity: float,
        reset_v: float,
        thresh_v: float,
        refrac_t: float,
        tc_membrane: float,
        tc_adaptation: float | tuple[float, ...],
        voltage_coupling: float | tuple[float, ...],
        spike_increment: float | tuple[float, ...],
        resistance: float = 1.0,
        batch_size: int = 1,
        batch_reduction: (
            Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] | None
        ) = None,
    ):
        # call superclass constructor
        InfernoNeuron.__init__(self, shape, batch_size)

        # process adaptation attributes
        # tuple-wrap if singleton
        if not hasattr(tc_adaptation, "__iter__"):
            tc_adaptation = (tc_adaptation,)
        if not hasattr(voltage_coupling, "__iter__"):
            voltage_coupling = (voltage_coupling,)
        if not hasattr(spike_increment, "__iter__"):
            spike_increment = (spike_increment,)

        # prepare converted lists
        tc_list, vc_list, si_list = [], [], []

        # test values
        for idx, (tc, vc, si) in enumerate(
            zip_longest(tc_adaptation, voltage_coupling, spike_increment)
        ):
            # time constant of adaptation
            if tc is None:
                tc_list.append(tc_list[-1])
            else:
                tc_list.append(argtest.gt(f"tc_adaptation[{idx}]", tc, 0, float))

            # voltage-current coupling
            if vc_list is None:
                vc_list.append(vc_list[-1])
            else:
                vc_list.append(float(vc))

            # current spike increment
            if si_list is None:
                si_list.append(si_list[-1])
            else:
                si_list.append(float(si))

        # register adaptation attributes as buffers (for tensor ops and compatibility)
        self.register_buffer("tc_adaptation", torch.tensor(tc_list), persistent=False)
        self.register_buffer(
            "adapt_vc_coupling", torch.tensor(vc_list), persistent=False
        )
        self.register_buffer("adapt_increment", torch.tensor(si_list), persistent=False)

        # dynamics attributes
        self.step_time = argtest.gt("step_time", step_time, 0, float)
        self.rest_v = argtest.lt("rest_v", rest_v, crit_v, float, "crit_v")
        self.crit_v = argtest.lte("crit_v", crit_v, thresh_v, float, "thresh_v")
        self.affinity = argtest.gt("affinity", affinity, 0, float)
        self.reset_v = argtest.lt("reset_v", reset_v, thresh_v, float, "thresh_v")
        self.thresh_v = float(thresh_v)
        self.refrac_t = argtest.gte("refrac_t", refrac_t, 0, float)
        self.tc_membrane = argtest.gt("time_constant", tc_membrane, 0, float)
        self.resistance = argtest.neq("resistance", resistance, 0, float)

        # call mixin constructors
        VoltageMixin.__init__(self, torch.full(self.batchedshape, self.rest_v))
        SpikeRefractoryMixin.__init__(self, torch.zeros(self.batchedshape), "refrac_t")
        AdaptiveCurrentMixin.__init__(
            self,
            torch.zeros(*self.shape, self.tc_adaptation.numel()),
            batch_reduction,
        )

    def _integrate_v(self, masked_inputs):
        r"""Internal, voltage integrated function as input for thresholding."""
        return nf.voltage_integration_quadratic(
            masked_inputs,
            self.voltage,
            step_time=self.step_time,
            rest_v=self.rest_v,
            crit_v=self.crit_v,
            affinity=self.affinity,
            time_constant=self.tc_membrane,
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
    def dt(self, value: float) -> None:
        self.step_time = argtest.gt("dt", value, 0, float)

    def clear(self, keep_adaptations: bool = True, **kwargs) -> None:
        r"""Resets neurons to their resting state.

        Args:
            keep_adaptations (bool, optional): if learned adaptations should be
                preserved. Defaults to ``True``.
        """
        self.voltage = torch.full_like(self.voltage, self.rest_v)
        self.refrac = torch.zeros_like(self.refrac)
        if not keep_adaptations:
            self.current_adaptation = torch.zeros_like(self.current_adaptation)

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
                :math:`I-x(t)`, in :math:`\text{nA}`.
            adapt (bool | None, optional): if adaptations should be updated
                based on this step. Defaults to ``None``.
            refrac_lock (bool, optional): if membrane voltages should be fixed
                while in the refractory period. Defaults to ``True``.

        Returns:
            torch.Tensor: if the corresponding neuron generated an action potential.

        Note:
            When ``adapt`` is set to None, adaptations will be updated when the neuron
            is in training mode but not when it is in evaluation mode.
        """
        # use voltage thresholding function
        spikes, voltages, refracs = nf.voltage_thresholding_constant(
            inputs=nf.apply_adaptive_currents(inputs, self.current_adaptation),
            refracs=self.refrac,
            dynamics=self._integrate_v,
            voltages=(self.voltage if refrac_lock else None),
            step_time=self.step_time,
            reset_v=self.reset_v,
            thresh_v=self.thresh_v,
            refrac_t=self.refrac_t,
        )

        # update parameters
        self.voltage = voltages
        self.refrac = refracs

        # conditionally update adaptive thresholds
        if adapt or (adapt is None and self.training):
            # use adaptive thresholds update function
            adaptations = nf.adaptive_currents_linear(
                adaptations=self.current_adaptation,
                voltages=voltages,
                spikes=spikes,
                step_time=self.step_time,
                rest_v=self.rest_v,
                time_constant=self.tc_adaptation,
                voltage_coupling=self.adapt_vc_coupling,
                spike_increment=self.adapt_increment,
                refracs=(self.refrac if refrac_lock else None),
            )
            # update parameter
            self.current_adaptation = adaptations

        # return spiking output
        return spikes


class EIF(VoltageMixin, SpikeRefractoryMixin, InfernoNeuron):
    r"""Simulation of exponential integrate-and-fire (EIF) neuron dynamics.

    Implemented as an approximation using Euler's method.

    .. math::
        V_m(t + \Delta t) = \frac{\Delta t}{\tau_m} \left[
        - \left[V_m(t) - V_\text{rest}\right] +
        \Delta_T \exp \left(\frac{V_m(t) - V_T}{\Delta_T}\right) + R_mI(t)
        \right]+ V_m(t)

    If a spike was generated at time :math:`t`, then.

    .. math::
        V_m(t) \leftarrow V_\text{reset}

    Args:
        shape (tuple[int, ...] | int): shape of the group of neurons being simulated.
        step_time (float): length of a simulation time step,
            :math:`\Delta t`, in :math:`\text{ms}`.
        rest_v (float): membrane potential difference at equilibrium,
            :math:`V_\text{rest}`, in :math:`\text{mV}`.
        rheobase_v (float): membrane potential difference approaching the potential at
            which potential rapidly increases, :math:`V_T`, in :math:`\text{mV}`.
        sharpness (float): steepness of the natural increase in membrane potential
            above the rheobase voltage, :math:`\Delta_T`, in :math:`\text{mV}`.
        reset_v (float): membrane voltage after an action potential is generated,
            :math:`V_\text{reset}`, in :math:`\text{mV}`.
        thresh_v (float): membrane voltage at which action potentials are generated,
            in :math:`\text{mV}`.
        refrac_t (float): length the absolute refractory period, in :math:`\text{ms}`.
        time_constant (float): time constant of exponential decay for membrane voltage,
            :math:`\tau_m`, in :math:`\text{ms}`.
        resistance (float, optional): resistance across the cell membrane,
            :math:`R_m`, in :math:`\text{M}\Omega`. Defaults to ``1.0``.
        batch_size (int, optional): size of input batches for simulation. Defaults to ``1``.

    See Also:
        For more details and references, visit
        :ref:`zoo/neurons-nonlinear:Exponential Integrate-and-Fire (EIF)` in the zoo.
    """

    def __init__(
        self,
        shape: tuple[int, ...] | int,
        step_time: float,
        *,
        rest_v: float,
        rheobase_v: float,
        sharpness: float,
        reset_v: float,
        thresh_v: float,
        refrac_t: float,
        time_constant: float,
        resistance: float = 1.0,
        batch_size: int = 1,
    ):
        # call superclass constructor
        InfernoNeuron.__init__(self, shape, batch_size)

        # dynamics attributes
        self.step_time = argtest.gt("step_time", step_time, 0, float)
        self.rest_v = argtest.lt("rest_v", rest_v, rheobase_v, float, "rheobase_v")
        self.rheobase_v = argtest.lte(
            "rheobase_v", rheobase_v, thresh_v, float, "thresh_v"
        )
        self.sharpness = argtest.gt("sharpness", sharpness, 0, float)
        self.reset_v = argtest.lt("reset_v", reset_v, thresh_v, float, "thresh_v")
        self.thresh_v = float(thresh_v)
        self.refrac_t = argtest.gte("refrac_t", refrac_t, 0, float)
        self.time_constant = argtest.gt("time_constant", time_constant, 0, float)
        self.resistance = argtest.neq("resistance", resistance, 0, float)

        # call mixin constructors
        VoltageMixin.__init__(self, torch.full(self.batchedshape, self.rest_v))
        SpikeRefractoryMixin.__init__(self, torch.zeros(self.batchedshape), "refrac_t")

    def _integrate_v(self, masked_inputs):
        r"""Internal, voltage integrated function as input for thresholding."""
        return nf.voltage_integration_exponential(
            masked_inputs,
            self.voltage,
            step_time=self.step_time,
            rest_v=self.rest_v,
            rheobase_v=self.rheobase_v,
            sharpness=self.sharpness,
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
    def dt(self, value: float) -> None:
        self.step_time = argtest.gt("dt", value, 0, float)

    def clear(self, **kwargs) -> None:
        r"""Resets neurons to their resting state."""
        self.voltage = torch.full_like(self.voltage, self.rest_v)
        self.refrac = torch.zeros_like(self.refrac)

    def forward(self, inputs: torch.Tensor, refrac_lock=True, **kwargs) -> torch.Tensor:
        r"""Runs a simulation step of the neuronal dynamics.

        Args:
            inputs (torch.Tensor): presynaptic currents,
                :math:`I(t)`, in :math:`\text{nA}`.
            refrac_lock (bool, optional): if membrane voltages should be fixed while
                in the refractory period. Defaults to ``True``.

        Returns:
            torch.Tensor: if the corresponding neuron generated an action potential.
        """
        # use voltage thresholding function
        spikes, voltages, refracs = nf.voltage_thresholding_constant(
            inputs=inputs,
            refracs=self.refrac,
            dynamics=self._integrate_v,
            voltages=(self.voltage if refrac_lock else None),
            step_time=self.step_time,
            reset_v=self.reset_v,
            thresh_v=self.thresh_v,
            refrac_t=self.refrac_t,
        )

        # update parameters
        self.voltage = voltages
        self.refrac = refracs

        # return spiking output
        return spikes


class AdEx(AdaptiveCurrentMixin, VoltageMixin, SpikeRefractoryMixin, InfernoNeuron):
    r"""Simulation of adaptive exponential integrate-and-fire (AdEx) neuron dynamics.

    Implemented as an approximation using Euler's method.

    .. math::
        \begin{align*}
            V_m(t + \Delta t) &= \frac{\Delta t}{\tau_m} \left[- \left[V_m(t) - V_\text{rest}\right] +
            \Delta_T \exp \left(\frac{V_m(t) - V_T}{\Delta_T}\right) + R_mI(t) \right]+ V_m(t) \\
            I(t) &= I_x(t) - \sum_k w_k(t) \\
            w_k(t + \Delta t) &= \frac{\Delta t}{\tau_k}\left[ a_k \left[ V_m(t) - V_\text{rest} \right]
            - w_k(t) \right] + w_k(t) \\
        \end{align*}

    If a spike was generated at time :math:`t`, then.

    .. math::
        \begin{align*}
            V_m(t) &\leftarrow V_\text{reset} \\
            w_k(t) &\leftarrow w_k(t) + b_k
        \end{align*}

    Args:
        shape (tuple[int, ...] | int): shape of the group of neurons being simulated.
        step_time (float): length of a simulation time step,
            :math:`\Delta t`, in :math:`\text{ms}`.
        rest_v (float): membrane potential difference at equilibrium,
            :math:`V_\text{rest}`, in :math:`\text{mV}`.
        rheobase_v (float): membrane potential difference approaching the potential at
            which potential rapidly increases, :math:`V_T`, in :math:`\text{mV}`.
        sharpness (float): steepness of the natural increase in membrane potential
            above the rheobase voltage, :math:`\Delta_T`, in :math:`\text{mV}`.
        reset_v (float): membrane voltage after an action potential is generated,
            :math:`V_\text{reset}`, in :math:`\text{mV}`.
        thresh_v (float): membrane voltage at which action potentials are generated,
            in :math:`\text{mV}`.
        refrac_t (float): length the absolute refractory period, in :math:`\text{ms}`.
        tc_membrane (float): time constant of exponential decay for membrane voltage,
            :math:`\tau_m`, in :math:`\text{ms}`.
        tc_adaptation (float | tuple[float, ...]): time constant of exponential decay
            for threshold adaptations, :math:`\tau_k`, in :math:`\text{ms}`.
        voltage_coupling (float | tuple[float, ...]): strength of coupling to membrane
            voltage, :math:`a_k`, in :math:`\mu\text{S}`.
        spike_increment (float | tuple[float, ...]): amount by which the adaptive
            current is increased after a spike, :math:`b_k`, in :math:`\text{nA}`.
        resistance (float, optional): resistance across the cell membrane,
            :math:`R_m`, in :math:`\text{M}\Omega`. Defaults to ``1.0``.
        batch_size (int, optional): size of input batches for simulation. Defaults to ``1``.
        batch_reduction (Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] | None):
            function to reduce adaptation updates over the batch dimension,
            :py:func:`torch.mean` when ``None``. Defaults to ``None``.

    Note:
        ``batch_reduction`` can be one of the functions in PyTorch including but not
        limited to :py:func:`torch.sum`, :py:func:`torch.mean`, and :py:func:`torch.amax`.
        A custom function with similar behavior can also be passed in. Like with the
        included function, it should not keep the original dimensions by default.

    See Also:
        For more details and references, visit
        :ref:`zoo/neurons-nonlinear:Adaptive Exponential Integrate-and-Fire (AdEx)`
        in the zoo.
    """

    def __init__(
        self,
        shape: tuple[int, ...] | int,
        step_time: float,
        *,
        rest_v: float,
        rheobase_v: float,
        sharpness: float,
        reset_v: float,
        thresh_v: float,
        refrac_t: float,
        tc_membrane: float,
        tc_adaptation: float | tuple[float, ...],
        voltage_coupling: float | tuple[float, ...],
        spike_increment: float | tuple[float, ...],
        resistance: float = 1.0,
        batch_size: int = 1,
        batch_reduction: (
            Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] | None
        ) = None,
    ):
        # call superclass constructor
        InfernoNeuron.__init__(self, shape, batch_size)

        # process adaptation attributes
        # tuple-wrap if singleton
        if not hasattr(tc_adaptation, "__iter__"):
            tc_adaptation = (tc_adaptation,)
        if not hasattr(voltage_coupling, "__iter__"):
            voltage_coupling = (voltage_coupling,)
        if not hasattr(spike_increment, "__iter__"):
            spike_increment = (spike_increment,)

        # prepare converted lists
        tc_list, vc_list, si_list = [], [], []

        # test values
        for idx, (tc, vc, si) in enumerate(
            zip_longest(tc_adaptation, voltage_coupling, spike_increment)
        ):
            # time constant of adaptation
            if tc is None:
                tc_list.append(tc_list[-1])
            else:
                tc_list.append(argtest.gt(f"tc_adaptation[{idx}]", tc, 0, float))

            # voltage-current coupling
            if vc_list is None:
                vc_list.append(vc_list[-1])
            else:
                vc_list.append(float(vc))

            # current spike increment
            if si_list is None:
                si_list.append(si_list[-1])
            else:
                si_list.append(float(si))

        # register adaptation attributes as buffers (for tensor ops and compatibility)
        self.register_buffer("tc_adaptation", torch.tensor(tc_list), persistent=False)
        self.register_buffer(
            "adapt_vc_coupling", torch.tensor(vc_list), persistent=False
        )
        self.register_buffer("adapt_increment", torch.tensor(si_list), persistent=False)

        # dynamics attributes
        self.step_time = argtest.gt("step_time", step_time, 0, float)
        self.rest_v = argtest.lt("rest_v", rest_v, rheobase_v, float, "rheobase_v")
        self.rheobase_v = argtest.lte(
            "rheobase_v", rheobase_v, thresh_v, float, "thresh_v"
        )
        self.sharpness = argtest.gt("sharpness", sharpness, 0, float)
        self.reset_v = argtest.lt("reset_v", reset_v, thresh_v, float, "thresh_v")
        self.thresh_v = float(thresh_v)
        self.refrac_t = argtest.gte("refrac_t", refrac_t, 0, float)
        self.tc_membrane = argtest.gt("tc_membrane", tc_membrane, 0, float)
        self.resistance = argtest.neq("resistance", resistance, 0, float)

        # call mixin constructors
        VoltageMixin.__init__(self, torch.full(self.batchedshape, self.rest_v))
        SpikeRefractoryMixin.__init__(self, torch.zeros(self.batchedshape), "refrac_t")
        AdaptiveCurrentMixin.__init__(
            self,
            torch.zeros(*self.shape, self.tc_adaptation.numel()),
            batch_reduction,
        )

    def _integrate_v(self, masked_inputs):
        r"""Internal, voltage integrated function as input for thresholding."""
        return nf.voltage_integration_exponential(
            masked_inputs,
            self.voltage,
            step_time=self.step_time,
            rest_v=self.rest_v,
            rheobase_v=self.rheobase_v,
            sharpness=self.sharpness,
            time_constant=self.tc_membrane,
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
    def dt(self, value: float) -> None:
        self.step_time = argtest.gt("dt", value, 0, float)

    def clear(self, keep_adaptations: bool = True, **kwargs) -> None:
        r"""Resets neurons to their resting state.

        Args:
            keep_adaptations (bool, optional): if learned adaptations should be
                preserved. Defaults to ``True``.
        """
        self.voltage = torch.full_like(self.voltage, self.rest_v)
        self.refrac = torch.zeros_like(self.refrac)
        if not keep_adaptations:
            self.current_adaptation = torch.zeros_like(self.current_adaptation)

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
                :math:`I_x(t)`, in :math:`\text{nA}`.
            adapt (bool | None, optional): if adaptations should be updated
                based on this step. Defaults to ``None``.
            refrac_lock (bool, optional): if membrane voltages should be fixed
                while in the refractory period. Defaults to ``True``.

        Returns:
            torch.Tensor: if the corresponding neuron generated an action potential.

        Note:
            When ``adapt`` is set to None, adaptations will be updated when the neuron
            is in training mode but not when it is in evaluation mode.
        """
        # use voltage thresholding function
        spikes, voltages, refracs = nf.voltage_thresholding_constant(
            inputs=nf.apply_adaptive_currents(inputs, self.current_adaptation),
            refracs=self.refrac,
            dynamics=self._integrate_v,
            voltages=(self.voltage if refrac_lock else None),
            step_time=self.step_time,
            reset_v=self.reset_v,
            thresh_v=self.thresh_v,
            refrac_t=self.refrac_t,
        )

        # update parameters
        self.voltage = voltages
        self.refrac = refracs

        # conditionally update adaptive thresholds
        if adapt or (adapt is None and self.training):
            # use adaptive thresholds update function
            adaptations = nf.adaptive_currents_linear(
                adaptations=self.current_adaptation,
                voltages=voltages,
                spikes=spikes,
                step_time=self.step_time,
                rest_v=self.rest_v,
                time_constant=self.tc_adaptation,
                voltage_coupling=self.adapt_vc_coupling,
                spike_increment=self.adapt_increment,
                refracs=(self.refrac if refrac_lock else None),
            )
            # update parameter
            self.current_adaptation = adaptations

        # return spiking output
        return spikes
