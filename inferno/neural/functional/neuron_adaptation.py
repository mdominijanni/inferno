import torch


def adaptive_currents_linear(
    adaptations: torch.Tensor,
    voltages: torch.Tensor,
    postsyn_spikes: torch.Tensor,
    *,
    step_time: float | torch.Tensor,
    rest_v: float | torch.Tensor,
    time_constant: float | torch.Tensor,
    voltage_coupling: float | torch.Tensor,
    spike_increment: float | torch.Tensor,
    refracs: torch.Tensor | None = None
) -> torch.Tensor:
    r"""Update adaptive currents based on membrane potential and postsynaptic spikes.

    Implemented as an approximation using Euler's method.

    .. math::
        w_k(t + \Delta t) = \frac{\Delta t}{\tau_k}\left[ a_k \left[ V_m(t) - V_\text{rest} \right]
            - w_k(t) \right] + w_k(t)

    If a spike was generated at time :math:`t`, then.

    .. math::
        w_k(t) \leftarrow w_k(t) + b_k

    Args:
        adaptations (torch.Tensor): last adaptations applied to input current,
            :math:`w_k`, in :math:`\mathrm{nA}`.
        voltages (torch.Tensor): membrane potential difference,
            :math:`V_m(t)`, in :math:`\mathrm{mV}`.
        postsyn_spikes (torch.Tensor): postsynaptic spikes, unitless.
        step_time (float | torch.Tensor): length of a simulation time step,
            :math:`\Delta t`, in :math:`\mathrm{ms}`.
        rest_v (float | torch.Tensor): membrane potential difference at equilibrium,
            :math:`V_\mathrm{rest}`, in :math:`\mathrm{mV}`.
        time_constant (float | torch.Tensor): time constant of exponential decay,
            :math:`\tau_k`, in :math:`\mathrm{ms}`.
        voltage_coupling (float | torch.Tensor): strength of coupling to membrane voltage,
            :math:`a_k`, in :math:`\mathrm{\mu S}`.
        spike_increment (float | torch.Tensor): amount by which the adaptive current is increased after a spike,
            :math:`b_k`, in :math:`\mathrm{nA}`.
        refracs (torch.Tensor | None): amount of remaining time needed to exit refractory periods,
            used for masking changes to adaptation when provided, in :math:`\mathrm{ms}`. Defaults to None.

    Returns:
        torch.Tensor: updated adaptations for input currents.

    Shape:
        ``adaptations``:

            :math:`N_0 \times \cdots \times k`,
            each slice along the last dimension should be shaped like the neuron group to which the adaptation
            is being applied, where :math:`k` is the number of parameter tuples.

        ``voltages``, ``postsyn_spikes``, and ``refracs``:

            :math:`[B] \times N_0 \times \cdots`, where the batch dimension :math:`B` is optional.

        ``rest_v``:

            `Broadcastable <https://pytorch.org/docs/stable/notes/broadcasting.html>`_ with
            ``postsyn_spikes`` and ``voltages``.

        ``step_time``, ``voltage_coupling``, ``spike_increment``, and ``time_constant``:

            `Broadcastable <https://pytorch.org/docs/stable/notes/broadcasting.html>`_ with ``adaptations``.

        **output**:

            :math:`[B] \times N_0 \times \cdots \times k`, where the batch dimension :math:`B` is dependent on inputs.

    Note:
        This function doesn't automatically reduce along the batch dimension, this should generally be done
        by averaging along the :math:`0^\mathrm{th}` dimension.

    Note:
        For more details and references, visit :ref:`zoo/neurons-adaptation:Adaptive Current, Linear` in the zoo.
    """
    # calculate euler step for adaptation update
    euler_step = (step_time / time_constant) * (
        voltage_coupling * (voltages - rest_v).unsqueeze(-1) - adaptations
    )

    # apply euler step
    if refracs is None:
        adaptations = adaptations + euler_step
    else:
        adaptations = adaptations.where(
            refracs.unsqueeze(-1) > 0, adaptations + euler_step
        )

    # post-spike adaptation step
    adaptations = adaptations + (spike_increment * postsyn_spikes.unsqueeze(-1))

    # return updated adaptation state
    return adaptations


def adaptive_thresholds_linear_voltage(
    adaptations: torch.Tensor,
    voltages: torch.Tensor,
    *,
    step_time: float | torch.Tensor,
    rest_v: float | torch.Tensor,
    adaptation_rate: float | torch.Tensor,
    rebound_rate: float | torch.Tensor,
    adaptation_reset_min: float | torch.Tensor | None = None,
    postsyn_spikes: torch.Tensor | None = None,
    refracs: torch.Tensor | None = None
) -> torch.Tensor:
    r"""Update adaptive thresholds based on membrane potential.

    Implemented as an approximation using Euler's method.

    .. math::
        \theta_k(t + \Delta t) = \Delta t \left[a_k \left[ V_m(t) - V_\mathrm{rest} \right]
            - b_k \theta_k(t)\right] + \theta_k(t)

    If a spike was generated at time :math:`t`, then.

    .. math::
        \theta_k(t) \leftarrow \max(\theta_k(t), \theta_\mathrm{reset})

    Args:
        adaptations (torch.Tensor): last adaptations applied to membrane voltage threshold,
            :math:`\theta_k`, in :math:`\mathrm{mV}`.
        voltages (torch.Tensor): membrane potential difference,
            :math:`V_m(t)`, in :math:`\mathrm{mV}`.
        step_time (float | torch.Tensor): length of a simulation time step,
            :math:`\Delta t`, in :math:`\mathrm{ms}`.
        rest_v (float | torch.Tensor): membrane potential difference at equilibrium,
            :math:`V_\mathrm{rest}`, in :math:`\mathrm{mV}`.
        adaptation_rate (float | torch.Tensor): rate constant of exponential decay for membrane voltage term,
            :math:`a_k`, in :math:`\mathrm{ms^{-1}}`.
        rebound_rate (float | torch.Tensor): rate constant of exponential decay for threshold voltage term,
            :math:`b_k`, in :math:`\mathrm{ms^{-1}}`.
        adaptation_reset_min (float | torch.Tensor | None, optional): minimum threshold adaptation permitted after
            a postsynaptic potential. Defaults to None.
        postsyn_spikes (torch.Tensor | None, optional): postsynaptic spikes, unitless. Defaults to None.
        refracs (torch.Tensor | None): amount of remaining time needed to exit refractory periods,
            used for masking changes to adaptation when provided, in :math:`\mathrm{ms}`. Defaults to None.

    Returns:
        torch.Tensor: updated adaptations for membrane voltage thresholds.

    Shape:
        ``adaptations``:

            :math:`N_0 \times \cdots \times k`,
            each slice along the last dimension should be shaped like the neuron group to which the adaptation
            is being applied, where :math:`k` is the number of parameter tuples.

        ``voltages``, ``postsyn_spikes``, and ``refracs``:

            :math:`[B] \times N_0 \times \cdots`, where the batch dimension :math:`B` is optional.

        ``rest_v``:

            Broadcastable <https://pytorch.org/docs/stable/notes/broadcasting.html>`_ with
            ``voltages`` and ``postsyn_spikes``.

        ``step_time``, ``adaptation_rate``, ``rebound_rate``, and ``adaptation_reset_min``:

            `Broadcastable <https://pytorch.org/docs/stable/notes/broadcasting.html>`_ with ``adaptations``.

        **output**:

            :math:`[B] \times N_0 \times \cdots \times k`,
            where the batch dimension :math:`B` is dependent on inputs.

    Note:
        If either ``adaptation_reset_min`` or ``postsyn_spikes`` is None, then the adaptation limiting
        after spiking is not performed.

    Note:
        This function doesn't automatically reduce along the batch dimension, this should generally be done
        by summing along the :math:`0^\mathrm{th}` (batch) dimension.

    Note:
        For more details and references, visit
        :ref:`zoo/neurons-adaptation:Adaptive Threshold, Linear Voltage-Dependent` in the zoo.
    """
    # calculate euler step for adaptation update
    euler_step = step_time * (
        adaptation_rate * (voltages - rest_v).unsqueeze(-1) - rebound_rate * adaptations
    )

    # apply euler step
    if refracs is None:
        adaptations = adaptations + euler_step
    else:
        adaptations = adaptations.where(
            refracs.unsqueeze(-1) > 0, adaptations + euler_step
        )

    # post-spike adaptation step
    if adaptation_reset_min is not None and postsyn_spikes is not None:
        adaptations = adaptations.where(
            postsyn_spikes.unsqueeze(-1) == 0,
            adaptations.clamp_min(adaptation_reset_min),
        )

    # return updated adaptation state
    return adaptations


def adaptive_thresholds_linear_spike(
    adaptations: torch.Tensor,
    postsyn_spikes: torch.Tensor,
    *,
    decay: float | torch.Tensor,
    spike_increment: float | torch.Tensor,
    refracs: torch.Tensor | None = None
) -> torch.Tensor:
    r"""Update adaptive thresholds based on postsynaptic spikes.

    .. math::
        \theta_k(t + \Delta t) = \theta_k(t) \exp\left(-\frac{\Delta t}{\tau_k}\right)

    If a spike was generated at time :math:`t`, then.

    .. math::
        \theta_k(t) \leftarrow \theta_k(t) + a_k

    Args:
        adaptations (torch.Tensor): last adaptations applied to membrane voltage threshold,
            :math:`\theta_k`, in :math:`\mathrm{mV}`.
        postsyn_spikes (torch.Tensor): postsynaptic spikes, unitless.
        decay (float | torch.Tensor): exponential decay for adaptations,
            :math:`\exp\left(-\frac{\Delta t}{\tau_k}\right)`, unitless.
        spike_increment (torch.Tensor): amount by which the adaptive threshold is increased after a spike,
            :math:`a_k`, in :math:`\mathrm{mV}`.
        refracs (torch.Tensor | None): amount of remaining time needed to exit refractory periods,
            used for masking changes to adaptation when provided, in :math:`\mathrm{ms}`. Defaults to None.

    Returns:
        torch.Tensor: updated adaptations for membrane voltage thresholds.

    Shape:
        ``adaptations``:

            :math:`N_0 \times \cdots \times k`,
            each slice along the last dimension should be shaped like the neuron group to which the adaptation
            is being applied, where :math:`k` is the number of parameter tuples.

        ``postsyn_spikes`` and ``refracs``:

            :math:`[B] \times N_0 \times \cdots`,
            where the batch dimension :math:`B` is optional.

        ``decay``, and ``spike_increment``:

            `Broadcastable <https://pytorch.org/docs/stable/notes/broadcasting.html>`_ with ``adaptations``.

        **output**:

            :math:`[B] \times N_0 \times \cdots \times k`,
            where the batch dimension :math:`B` is dependent on inputs.

    Note:
        This function doesn't automatically reduce along the batch dimension, this should generally be done
        by summing along the :math:`0^\mathrm{th}` (batch) dimension.

    Note:
        For more details and references, visit
        :ref:`zoo/neurons-adaptation:Adaptive Threshold, Linear Spike-Dependent` in the zoo.
    """
    # decay adaptations over time
    if refracs is None:
        adaptations = adaptations * decay
    else:
        adaptations = adaptations.where(refracs.unsqueeze(-1) > 0, adaptations * decay)

    # increment adaptations after spiking
    adaptations = adaptations + (spike_increment * postsyn_spikes.unsqueeze(-1))

    # return updated adaptation state
    return adaptations


def apply_adaptive_currents(
    presyn_currents: torch.Tensor,
    adaptations: torch.Tensor,
) -> torch.Tensor:
    r"""Applies simple adapation to presynaptic currents.

    Args:
        presyn_currents (torch.Tensor): presynaptic currents :math:`I_+`, in :math:`\mathrm{nA}`.
        adaptations (torch.Tensor): :math:`k` current adaptations :math:`w_k`, in :math:`\mathrm{nA}`.

    Returns:
        torch.Tensor: adapted presynaptic currents.

    Shape:
        ``presyn_currents``:

            :math:`N_0 \times \cdots \times`,
            equal in shape to the group of neurons being simulated.

        ``adaptations``:

            :math:`\cdots \times k`,
            where :math:`\cdots` is any shape
            `broadcastable <https://pytorch.org/docs/stable/notes/broadcasting.html>`_ with ``presyn_currents``.

        **output**:

            :math:`N_0 \times \cdots \times`,
            same as ``presyn_currents``.

    Note:
        For an example, visit :ref:`zoo/neurons-adaptation:Adaptive Current, Linear` in the zoo.
    """
    # return adjusted currents
    return presyn_currents - torch.sum(adaptations, dim=-1)


def apply_adaptive_thresholds(
    threshold: float | torch.Tensor,
    adaptations: torch.Tensor,
) -> torch.Tensor:
    r"""Applies simple adapation to voltage firing thresholds.

    Args:
        threshold (float | torch.Tensor): baseline firing threshold :math:`\Theta_\infty`, in :math`\mathrm{mV}`.
        adaptations (torch.Tensor): :math:`k` threshold adaptations :math:`\theta_k`, in :math:`\mathrm{mV}`.

    Returns:
            torch.Tensor: adapted firing thresholds.

    Note:
        For an example, visit :ref:`zoo/neurons-adaptation:Adaptive Threshold, Linear Spike-Dependent` in the zoo.
    """
    # return adjusted thresholds
    return threshold + torch.sum(adaptations, dim=-1)
