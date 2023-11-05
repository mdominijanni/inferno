import torch
import inferno
from . import (
    _voltage_thresholding_discrete,
    _voltage_thresholding_slope_intercept_discrete,
    apply_adaptive_thresholds,
)
from . import adaptive_thresholds_linear_spike


def leaky_integrate_and_fire_euler(
    inputs: torch.Tensor,
    voltages: torch.Tensor,
    refracs: torch.Tensor,
    *,
    step_time: float | torch.Tensor,
    rest_v: float | torch.Tensor,
    reset_v: float | torch.Tensor,
    thresh_v: float | torch.Tensor,
    refrac_ts: int | torch.Tensor,
    time_constant: float | torch.Tensor,
    resistance: float | torch.Tensor = 1.0,
    lock_voltage_on_refrac: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Runs a simulation step of leaky integrate-and-fire (LIF) dynamics.

    Implemented as an approximation using Euler's method.

    .. math::
        V_m(t + \Delta t) = \frac{\Delta t}{\tau_m} \left[ -[V_m(t) - V_\mathrm{rest}] + R_m I(t) \right] + V_m(t)

    If a spike was generated at time :math:`t`, then.

    .. math::
        V_m(t) \leftarrow V_\mathrm{reset}


    Args:
        inputs (torch.Tensor): presynaptic currents,
            :math:`I(t)`, in :math:`\mathrm{nA}`.
        voltages (torch.Tensor): voltage across the cell membrane,
            :math:`V_m(t)`, in :math:`\mathrm{mV}`.
        refracs (torch.Tensor): number of remaining simulation steps to exit refractory periods.
        step_time (float | torch.Tensor): length of a simulation time step,
            :math:`\Delta t`, in :math:`\mathrm{ms}`.
        rest_v (float | torch.Tensor): membrane potential difference at equilibrium,
            :math:`V_\mathrm{rest}`, in :math:`\mathrm{mV}`.
        reset_v (float | torch.Tensor): membrane voltage after an action potential is generated,
            :math:`V_\mathrm{reset}`, in :math:`\mathrm{mV}`.
        thresh_v (float | torch.Tensor): membrane voltage at which action potentials are generated,
            in :math:`\mathrm{mV}`.
        refrac_ts (int | torch.Tensor): number of time steps the absolute refractory period lasts.
        time_constant (float | torch.Tensor): time constant of exponential decay for membrane voltage,
            :math:`\tau_m`, in :math:`\mathrm{ms}`.
        resistance (float | torch.Tensor, optional): resistance across the cell membrane,
            :math:`R_m`, in :math:`\mathrm{M\Omega}`. Defaults to 1.0.
        lock_voltage_on_refrac (bool, optional): if membrane voltages should be fixed while in the
            refractory period. Defaults to True.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: tuple containing output and updated state:

            spikes: which neurons generated an action potential.

            voltages: updated membrane voltages.

            refracs: updated number of remaining simulation steps to exit refractory periods.

    Shape:
        ``inputs``, ``voltages``, and ``refracs``: :math:`[B] \times N_0 \times \cdots`,
        where the batch dimension :math:`B` is optional.

        **other inputs**:
        `broadcastable <https://pytorch.org/docs/stable/notes/broadcasting.html>`_ with
        ``inputs``, ``voltages``, and ``refracs``.

        **outputs**:
        same shape as ``inputs``, ``voltages``, and ``refracs``.

    Note:
        For more details and references, visit
        :ref:`zoo/neurons-linear:Leaky Integrate-and-Fire (LIF)` in the zoo.
    """

    # update voltages and determine which neurons have spiked
    def volt_fn(masked_inputs):
        decay = step_time / time_constant
        v_in = resistance * masked_inputs
        v_delta = voltages - rest_v
        return decay * (-v_delta + v_in) + voltages

    spikes, voltages, refracs = _voltage_thresholding_discrete(
        inputs=inputs,
        refracs=refracs,
        voltage_fn=volt_fn,
        reset_v=reset_v,
        thresh_v=thresh_v,
        refrac_ts=refrac_ts,
        voltages=(voltages if lock_voltage_on_refrac else None),
    )

    # return generated spikes and updated state
    return spikes, voltages, refracs


def leaky_integrate_and_fire(
    inputs: torch.Tensor,
    voltages: torch.Tensor,
    refracs: torch.Tensor,
    *,
    step_time: float | torch.Tensor,
    rest_v: float | torch.Tensor,
    reset_v: float | torch.Tensor,
    thresh_v: float | torch.Tensor,
    refrac_ts: int | torch.Tensor,
    time_constant: float | torch.Tensor,
    resistance: float | torch.Tensor = 1.0,
    lock_voltage_on_refrac: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Runs a simulation step of leaky integrate-and-fire (LIF) dynamics.

    .. math::
        V_m(t + \Delta t) = \left[V_m(t) - V_\mathrm{rest} - R_mI(t)\right]
        \exp\left(-\frac{t}{\tau_m}\right) + V_\mathrm{rest} + R_mI(t)

    If a spike was generated at time :math:`t`, then.

    .. math::
        V_m(t) \leftarrow V_\mathrm{reset}

    Args:
        inputs (torch.Tensor): presynaptic currents,
            :math:`I(t)`, in :math:`\mathrm{nA}`.
        voltages (torch.Tensor): voltage across the cell membrane,
            :math:`V_m(t)`, in :math:`\mathrm{mV}`.
        refracs (torch.Tensor): number of remaining simulation steps to exit refractory periods.
        step_time (float | torch.Tensor): length of a simulation time step,
            :math:`\Delta t`, in :math:`\mathrm{ms}`.
        rest_v (float | torch.Tensor): membrane potential difference at equilibrium,
            :math:`V_\mathrm{rest}`, in :math:`\mathrm{mV}`.
        reset_v (float | torch.Tensor): membrane voltage after an action potential is generated,
            :math:`V_\mathrm{reset}`, in :math:`\mathrm{mV}`.
        thresh_v (float | torch.Tensor): membrane voltage at which action potentials are generated,
            in :math:`\mathrm{mV}`.
        refrac_ts (int | torch.Tensor): number of time steps the absolute refractory period lasts.
        time_constant (float | torch.Tensor): time constant of exponential decay for membrane voltage,
            :math:`\tau_m`, in :math:`\mathrm{ms}`.
        resistance (float | torch.Tensor, optional): resistance across the cell membrane,
            :math:`R_m`, in :math:`\mathrm{M\Omega}`. Defaults to 1.0.
        lock_voltage_on_refrac (bool, optional): if membrane voltages should be fixed while in the
            refractory period. Defaults to True.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: tuple containing output and updated state:

            spikes: which neurons generated an action potential.

            voltages: updated membrane voltages.

            refracs: updated number of remaining simulation steps to exit refractory periods.

    Shape:
        ``inputs``, ``voltages``, and ``refracs``: :math:`[B] \times N_0 \times \cdots`,
        where the batch dimension :math:`B` is optional.

        **other inputs**:
        `broadcastable <https://pytorch.org/docs/stable/notes/broadcasting.html>`_ with
        ``inputs``, ``voltages``, and ``refracs``.

        **outputs**:
        same shape as ``inputs``, ``voltages``, and ``refracs``.

    Note:
        For more details and references, visit
        :ref:`zoo/neurons-linear:Leaky Integrate-and-Fire (LIF)` in the zoo.
    """
    # compute decay for tensors or primitives
    decay = inferno.exp(-step_time / time_constant)

    # update voltages and determine which neurons have spiked
    def volt_fn(masked_inputs):
        v_in = resistance * masked_inputs
        v_delta = voltages - rest_v
        return v_in + (v_delta - v_in) * decay + rest_v

    spikes, voltages, refracs = _voltage_thresholding_discrete(
        inputs=inputs,
        refracs=refracs,
        voltage_fn=volt_fn,
        reset_v=reset_v,
        thresh_v=thresh_v,
        refrac_ts=refrac_ts,
        voltages=(voltages if lock_voltage_on_refrac else None),
    )

    # return generated spikes and updated state
    return spikes, voltages, refracs


def adaptive_leaky_integrate_and_fire(
    inputs: torch.Tensor,
    voltages: torch.Tensor,
    refracs: torch.Tensor,
    adaptations: torch.Tensor,
    *,
    step_time: float | torch.Tensor,
    rest_v: float | torch.Tensor,
    reset_v: float | torch.Tensor,
    eq_thresh_v: float | torch.Tensor,
    refrac_ts: int | torch.Tensor,
    tc_membrane: float | torch.Tensor,
    tc_adaptation: float | torch.Tensor,
    spike_adapt_increment: float | torch.Tensor,
    resistance: float | torch.Tensor = 1.0,
    lock_voltage_on_refrac: bool = True,
    lock_adaptation_on_refrac: bool = True,
    update_adaptations: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Runs a simulation step of adaptive leaky integrate-and-fire (ALIF) dynamics.

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
        inputs (torch.Tensor): presynaptic currents,
            :math:`I(t)`, in :math:`\mathrm{nA}`.
        voltages (torch.Tensor): voltage across the cell membrane,
            :math:`V_m(t)`, in :math:`\mathrm{mV}`.
        refracs (torch.Tensor): number of remaining simulation steps to exit refractory periods.
        adaptations (torch.Tensor): last adaptations applied to membrane voltage threshold,
            :math:`\theta_k`, in :math:`\mathrm{mV}`.
        step_time (float | torch.Tensor): length of a simulation time step,
            :math:`\Delta t`, in :math:`\mathrm{ms}`.
        rest_v (float | torch.Tensor): membrane potential difference at equilibrium,
            :math:`V_\mathrm{rest}`, in :math:`\mathrm{mV}`.
        reset_v (float | torch.Tensor): membrane voltage after an action potential is generated,
            :math:`V_\mathrm{reset}`, in :math:`\mathrm{mV}`.
        eq_thresh_v (float | torch.Tensor): equilibrium of the firing threshold,
            :math:`\Theta_\infty$`, in :math:`\mathrm{mV}`.
        refrac_ts (int | torch.Tensor): number of time steps the absolute refractory period lasts.
        tc_membrane (float | torch.Tensor): time constant of exponential decay for membrane voltage,
            :math:`\tau_m`, in :math:`\mathrm{ms}`.
        tc_adaptation (float | torch.Tensor): time constant of exponential decay for threshold adaptations,
            :math:`\tau_k`, in :math:`\mathrm{ms}`.
        spike_adapt_increment (float | torch.Tensor): amount by which the adaptive threshold is increased
            after a spike, :math:`a_k`, in :math:`\mathrm{mV}`.
        resistance (float | torch.Tensor, optional): resistance across the cell membrane,
            :math:`R_m`, in :math:`\mathrm{M\Omega}`. Defaults to 1.0.
        lock_voltage_on_refrac (bool, optional): if membrane voltages should be fixed while in the
            refractory period. Defaults to True.
        lock_adaptation_on_refrac (bool, optional): if adaptations not triggered by spikes should be fixed while
            in the refractory period. Defaults to True.
        update_adaptations (bool, optional): if adaptations should be updated. Defaults to True.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: tuple containing output and updated state:

            spikes: which neurons generated an action potential.

            voltages: updated membrane voltage.

            refracs: updated number of remaining simulation steps to exit refractory periods.

            adaptations: updated adaptations for membrane voltage thresholds.

    Shape:
        ``inputs``, ``voltages``, and ``refracs``: :math:`[B] \times N_0 \times \cdots`,
        where the batch dimension :math:`B` is optional.

        ``adaptations``: :math:`N_0 \times \cdots \times k`,
        each slice along the last dimension should be shaped like the neuron group to which the adaptation
        is being applied, where :math:`k` is the number of parameter tuples.

        ``step_time``, ``rest_v``, ``reset_v``, ``thresh_v``, ``refrac_ts``, ``tc_membrane``, ``resistance``:
        `broadcastable <https://pytorch.org/docs/stable/notes/broadcasting.html>`_ with
        ``inputs``, ``voltages``, ``refracs``.

        ``tc_adaptation`` and ``spike_adapt_increment``:
        `broadcastable <https://pytorch.org/docs/stable/notes/broadcasting.html>`_ with ``adaptations``.

        **outputs (except adaptations)**: same shape as ``inputs``, ``voltages``, and ``refracs``.

        **outputs (adaptations)**: :math:`[B] \times N_0 \times \cdots \times k`,
        where the batch dimension :math:`B` is dependent on inputs.

    Note:
        This function doesn't automatically reduce resultant adaptations along the batch dimension,
        this should generally be done by averaging along the :math:`0^\mathrm{th}` (batch) dimension.

    Note:
        If ``update_adaptations`` is false, then returned adaptations will not have a leading batch dimension.

    Note:
        For more details and references, visit
        :ref:`zoo/neurons-linear:Adaptive Leaky Integrate-and-Fire (ALIF)` in the zoo.
    """
    # perform leaky integrate-and-fire step with adapted thresholds
    spikes, voltages, refracs = leaky_integrate_and_fire(
        inputs=inputs,
        voltages=voltages,
        refracs=refracs,
        step_time=step_time,
        rest_v=rest_v,
        reset_v=reset_v,
        thresh_v=apply_adaptive_thresholds(eq_thresh_v, adaptations),
        refrac_ts=refrac_ts,
        time_constant=tc_membrane,
        resistance=resistance,
        lock_voltage_on_refrac=lock_voltage_on_refrac,
    )

    # update adaptative thresholds based on spiking
    if update_adaptations:
        adaptations = adaptive_thresholds_linear_spike(
            adaptations=adaptations,
            postsyn_spikes=spikes,
            step_time=step_time
            if not isinstance(step_time, torch.Tensor)
            else step_time.unsqueeze(-1),
            time_constant=tc_adaptation,
            spike_increment=spike_adapt_increment,
            refracs=(refracs if lock_adaptation_on_refrac else None),
        )

    # return generated spikes and updated state
    return spikes, voltages, refracs, adaptations


generalized_leaky_integrate_and_fire_1 = leaky_integrate_and_fire
r"""Runs a simulation step of generalized leaky integrate-and-fire 1 (GLIF\ :sub:`1`) dynamics.

Alias for :py:func:`~inferno.neural.functional.leaky_integrate_and_fire`.

Note:
    For more details and references, visit
    :ref:`zoo/neurons-linear:generalized leaky integrate-and-fire 1 (glif{sub}\`1\`)` in the zoo.
"""


def generalized_leaky_integrate_and_fire_2(
    inputs: torch.Tensor,
    voltages: torch.Tensor,
    refracs: torch.Tensor,
    adaptations: torch.Tensor,
    *,
    step_time: float | torch.Tensor,
    rest_v: float | torch.Tensor,
    reset_v_add: float | torch.Tensor,
    reset_v_mul: float | torch.Tensor,
    eq_thresh_v: float | torch.Tensor,
    refrac_ts: int | torch.Tensor,
    tc_membrane: float | torch.Tensor,
    tc_adaptation: float | torch.Tensor,
    spike_adapt_increment: float | torch.Tensor,
    resistance: float | torch.Tensor = 1.0,
    lock_voltage_on_refrac: bool = True,
    lock_adaptation_on_refrac: bool = True,
    update_adaptations: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Runs a simulation step of generalized leaky integrate-and-fire 2 (GLIF\ :sub:`2`) dynamics.

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
        inputs (torch.Tensor): presynaptic currents,
            :math:`I(t)`, in :math:`\mathrm{nA}`.
        voltages (torch.Tensor): voltage across the cell membrane,
            :math:`V_m(t)`, in :math:`\mathrm{mV}`.
        refracs (torch.Tensor): number of remaining simulation steps to exit refractory periods.
        adaptations (torch.Tensor): last adaptations applied to membrane voltage threshold,
            :math:`\theta_k`, in :math:`\mathrm{mV}`.
        step_time (float | torch.Tensor): length of a simulation time step,
            :math:`\Delta t`, in :math:`\mathrm{ms}`.
        rest_v (float | torch.Tensor): membrane potential difference at equilibrium,
            :math:`V_\mathrm{rest}`, in :math:`\mathrm{mV}`.
        reset_v_add (float | torch.Tensor): additive parameter controlling reset voltage,
            :math:`b_v`, in :math:`\mathrm{mV}`.
        reset_v_mul (float | torch.Tensor): multiplicative parameter controlling reset voltage,
            :math:`m_v`, unitless.
        eq_thresh_v (float | torch.Tensor): equilibrium of the firing threshold,
            :math:`\Theta_\infty$`, in :math:`\mathrm{mV}`.
        refrac_ts (int | torch.Tensor): number of time steps the absolute refractory period lasts.
        tc_membrane (float | torch.Tensor): time constant of exponential decay for membrane voltage,
            :math:`\tau_m`, in :math:`\mathrm{ms}`.
        tc_adaptation (float | torch.Tensor): time constant of exponential decay for threshold adaptations,
            :math:`\tau_k`, in :math:`\mathrm{ms}`.
        spike_adapt_increment (float | torch.Tensor): amount by which the adaptive threshold is increased
            after a spike, :math:`a_k`, in :math:`\mathrm{mV}`.
        resistance (float | torch.Tensor, optional): resistance across the cell membrane,
            :math:`R_m`, in :math:`\mathrm{M\Omega}`. Defaults to 1.0. Defaults to 1.0.
        lock_voltage_on_refrac (bool, optional): if membrane voltages should be fixed while in the
            refractory period. Defaults to True.
        lock_adaptation_on_refrac (bool, optional): if adaptations not triggered by spikes should be fixed while
            in the refractory period. Defaults to True.
        update_adaptations (bool, optional): if adaptations should be updated. Defaults to True.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: tuple containing output and updated state:

            spikes: which neurons generated an action potential.

            voltages: updated membrane voltage.

            refracs: updated number of remaining simulation steps to exit refractory periods.

            adaptations: updated adaptations for membrane voltage thresholds.

    Shape:
        ``inputs``, ``voltages``, and ``refracs``: :math:`[B] \times N_0 \times \cdots`,
        where the batch dimension :math:`B` is optional.

        ``adaptations``: :math:`N_0 \times \cdots \times k`,
        each slice along the last dimension should be shaped like the neuron group to which the adaptation
        is being applied, where :math:`k` is the number of parameter tuples.

        ``step_time``, ``rest_v``, ``reset_v_add``, ``reset_v_mul``, ``thresh_v``, ``refrac_ts``,
        ``tc_membrane``, ``resistance``:
        `broadcastable <https://pytorch.org/docs/stable/notes/broadcasting.html>`_ with
        ``inputs``, ``voltages``, ``refracs``.

        ``tc_adaptation`` and ``spike_adapt_increment``:
        `broadcastable <https://pytorch.org/docs/stable/notes/broadcasting.html>`_ with ``adaptations``.

        **outputs (except adaptations)**: same shape as ``inputs``, ``voltages``, and ``refracs``.

        **outputs (adaptations)**: :math:`[B] \times N_0 \times \cdots \times k`,
        where the batch dimension :math:`B` is dependent on inputs.

    Note:
        This function doesn't automatically reduce resultant adaptations along the batch dimension,
        this should generally be done by averaging along the :math:`0^\mathrm{th}` (batch) dimension.

    Note:
        If ``update_adaptations`` is false, then returned adaptations will not have a leading batch dimension.

    Note:
        For more details and references, visit
        :ref:`zoo/neurons-linear:generalized leaky integrate-and-fire 2 (glif{sub}\`2\`)` in the zoo.
    """
    # compute decay for tensors or primitives
    decay = inferno.exp(-step_time / tc_membrane)

    # update voltages and determine which neurons have spiked
    def volt_fn(masked_inputs):
        v_in = resistance * masked_inputs
        v_delta = voltages - rest_v
        return v_in + (v_delta - v_in) * decay + rest_v

    spikes, voltages, refracs = _voltage_thresholding_slope_intercept_discrete(
        inputs=inputs,
        refracs=refracs,
        voltage_fn=volt_fn,
        rest_v=rest_v,
        v_slope=reset_v_mul,
        v_intercept=reset_v_add,
        thresh_v=apply_adaptive_thresholds(eq_thresh_v, adaptations),
        refrac_ts=refrac_ts,
        voltages=(voltages if lock_voltage_on_refrac else None),
    )

    # update adaptative thresholds based on spiking
    adaptations = adaptive_thresholds_linear_spike(
        adaptations=adaptations,
        postsyn_spikes=spikes,
        step_time=step_time
        if not isinstance(step_time, torch.Tensor)
        else step_time.unsqueeze(-1),
        time_constant=tc_adaptation,
        spike_increment=spike_adapt_increment,
        refracs=(refracs if lock_adaptation_on_refrac else None),
    )

    # return generated spikes and updated state
    return spikes, voltages, refracs, adaptations
