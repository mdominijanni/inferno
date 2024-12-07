from ... import exp
import torch


def adaptive_currents_linear(
    adaptations: torch.Tensor,
    voltages: torch.Tensor,
    spikes: torch.Tensor,
    *,
    step_time: float | torch.Tensor,
    rest_v: float | torch.Tensor,
    time_constant: float | torch.Tensor,
    voltage_coupling: float | torch.Tensor,
    spike_increment: float | torch.Tensor,
    refracs: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Update adaptive currents based on membrane potential and postsynaptic spikes.

    Implemented as an approximation using Euler's method.

    .. math::
        w_k(t + \Delta t) = \frac{\Delta t}{\tau_k}
        \left[ a_k \left[ V_m(t) - V_\text{rest} \right] - w_k(t) \right] + w_k(t)

    If a spike was generated at time :math:`t`, then.

    .. math::
        w_k(t) \leftarrow w_k(t) + b_k

    Args:
        adaptations (torch.Tensor): last adaptations applied to input current,
            :math:`w_k`, in :math:`\text{nA}`.
        voltages (torch.Tensor): membrane voltages :math:`V_m(t)`,
            in :math:`\text{mV}`.
        spikes (torch.Tensor): if the corresponding neuron generated an
            action potential.
        step_time (float | torch.Tensor): length of a simulation time step,
            in :math:`\text{ms}`.
        rest_v (float | torch.Tensor): membrane potential difference at equilibrium,
            :math:`V_\text{rest}`, in :math:`\text{mV}`.
        time_constant (float | torch.Tensor): time constant of exponential decay,
            :math:`\tau_k`, in :math:`\text{ms}`.
        voltage_coupling (float | torch.Tensor): strength of coupling to membrane
            voltage, :math:`a_k`, in :math:`\mu\text{S}`.
        spike_increment (float | torch.Tensor): amount by which the adaptive current is
            increased after a spike, :math:`b_k`, in :math:`\text{nA}`.
        refracs (torch.Tensor | None): remaining absolute refractory periods,
            in :math:`\text{ms}`, when not ``None``, adaptations of neurons in their
            absolute refractory periods are maintained. Defaults to ``None``.

    Returns:
        torch.Tensor: updated adaptations for input currents, in :math:`\text{nA}`.

    .. admonition:: Shape
        :class: tensorshape

        ``adaptations``:

        :math:`N_0 \times \cdots \times K`

        ``voltages``, ``spikes``, ``refracs``:

        :math:`[B] \times N_0 \times \cdots`

        ``rest_v``:

        `Broadcastable <https://pytorch.org/docs/stable/notes/broadcasting.html>`_ with
        ``voltages``, ``spikes``, and ``refracs``.

        ``step_time``, ``voltage_coupling``, ``spike_increment``, ``time_constant``:

        `Broadcastable <https://pytorch.org/docs/stable/notes/broadcasting.html>`_ with
        ``adaptations``.

        ``return``:

        :math:`[B] \times N_0 \times \cdots \times K`

        Where:
            * :math:`B` is the batch size.
            * :math:`N_0, \ldots` are dimensions of the group of neurons simulated.
            * :math:`K` is the number of sets of adaptation parameters.

    Tip:
        This function doesn't automatically reduce along the batch dimension,
        this should generally be done by averaging along the :math:`0^\text{th}`
        dimension.

    See Also:
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
    adaptations = adaptations + (spike_increment * spikes.unsqueeze(-1))

    # return updated adaptation state
    return adaptations


def adaptive_thresholds_linear_voltage(
    adaptations: torch.Tensor,
    voltages: torch.Tensor,
    *,
    step_time: float | torch.Tensor,
    rest_v: float | torch.Tensor,
    adapt_rate: float | torch.Tensor,
    rebound_rate: float | torch.Tensor,
    adapt_reset_min: float | torch.Tensor | None = None,
    spikes: torch.Tensor | None = None,
    refracs: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Update adaptive thresholds based on membrane potential.

    Implemented as an approximation using Euler's method.

    .. math::
        \theta_k(t + \Delta t) = \Delta t
        \left[a_k \left[ V_m(t) - V_\text{rest} \right]
        - b_k \theta_k(t)\right] + \theta_k(t)

    If a spike was generated at time :math:`t`, then.

    .. math::
        \theta_k(t) \leftarrow \max(\theta_k(t), \theta_\text{reset})

    Args:
        adaptations (torch.Tensor): last adaptations applied to membrane voltage
            threshold, :math:`\theta_k`, in :math:`\text{mV}`.
        voltages (torch.Tensor): membrane potential difference,
            :math:`V_m(t)`, in :math:`\text{mV}`.
        step_time (float | torch.Tensor): length of a simulation time step,
            :math:`\Delta t`, in :math:`\text{ms}`.
        rest_v (float | torch.Tensor): membrane potential difference at equilibrium,
            :math:`V_\text{rest}`, in :math:`\text{mV}`.
        adapt_rate (float | torch.Tensor): rate constant of exponential decay for
            membrane voltage term, :math:`a_k`, in :math:`\text{ms}^{-1}`.
        rebound_rate (float | torch.Tensor): rate constant of exponential decay for
            threshold voltage term, :math:`b_k`, in :math:`\text{ms}^{-1}`.
        adapt_reset_min (float | torch.Tensor | None, optional): lower bound for
            the threshold adaptation permitted after a postsynaptic potential,
            :math:`\theta_\text{reset}`, in :math:`\text{mV}`. Defaults to ``None``.
        spikes (torch.Tensor | None, optional): if the corresponding neuron generated an
            action potential. Defaults to ``None``.
        refracs (torch.Tensor | None): remaining absolute refractory periods,
            in :math:`\text{ms}`, when not ``None``, adaptations of neurons in their
            absolute refractory periods are maintained. Defaults to ``None``.

    Returns:
        torch.Tensor: updated adaptations for membrane voltage threshold,
        in :math:`\text{mV}`.

    .. admonition:: Shape
        :class: tensorshape

        ``adaptations``:

        :math:`N_0 \times \cdots \times K`

        ``voltages``, ``spikes``, ``refracs``:

        :math:`[B] \times N_0 \times \cdots`

        ``rest_v``:

        `Broadcastable <https://pytorch.org/docs/stable/notes/broadcasting.html>`_ with
        ``voltages``, ``spikes``, and ``refracs``.

        ``step_time``, ``adapt_rate``, ``rebound_rate``, ``adapt_reset_min``:

        `Broadcastable <https://pytorch.org/docs/stable/notes/broadcasting.html>`_ with
        ``adaptations``.

        ``return``:

        :math:`[B] \times N_0 \times \cdots \times K`

        Where:
            * :math:`B` is the batch size.
            * :math:`N_0, \ldots` are dimensions of the group of neurons simulated.
            * :math:`K` is the number of sets of adaptation parameters.

    Note:
        If either ``adapt_reset_min`` or ``spikes`` is None, then no lower bound
        will be applied to threshold adaptations.

    Tip:
        This function doesn't automatically reduce along the batch dimension,
        this should generally be done by averaging along the :math:`0^\text{th}`
        dimension.

    See Also:
        For more details and references, visit
        :ref:`zoo/neurons-adaptation:Adaptive Threshold, Linear Voltage-Dependent` in the zoo.
    """
    # calculate euler step for adaptation update
    euler_step = step_time * (
        adapt_rate * (voltages - rest_v).unsqueeze(-1) - rebound_rate * adaptations
    )

    # apply euler step
    if refracs is None:
        adaptations = adaptations + euler_step
    else:
        adaptations = adaptations.where(
            refracs.unsqueeze(-1) > 0, adaptations + euler_step
        )

    # post-spike adaptation step
    if adapt_reset_min is not None and spikes is not None:
        adaptations = adaptations.where(
            spikes.unsqueeze(-1) == 0,
            adaptations.clamp_min(adapt_reset_min),
        )

    # return updated adaptation state
    return adaptations


def adaptive_thresholds_constant_spike(
    adaptations: torch.Tensor,
    spikes: torch.Tensor,
    *,
    step_time: float | torch.Tensor,
    time_constant: float | torch.Tensor,
    spike_increment: float | torch.Tensor,
    refracs: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Update adaptive thresholds based on postsynaptic spikes.

    .. math::
        \theta_k(t + \Delta t) = \theta_k(t) \exp\left(-\frac{\Delta t}{\tau_k}\right)

    If a spike was generated at time :math:`t`, then.

    .. math::
        \theta_k(t) \leftarrow \theta_k(t) + a_k

    Args:
        adaptations (torch.Tensor): last adaptations applied to membrane voltage
            threshold, :math:`\theta_k`, in :math:`\text{mV}`.
        spikes (torch.Tensor): if the corresponding neuron generated an
            action potential.
        step_time (float | torch.Tensor): length of a simulation time step,
            :math:`\Delta t`, in :math:`\text{ms}`.
        time_constant (float | torch.Tensor): time constant of exponential decay for
            the adaptations, :math:`\tau_k`, in :math:`\text{ms}`.
        spike_increment (torch.Tensor): amount by which the adaptive threshold is
            increased after a spike, :math:`a_k`, in :math:`\text{mV}`.
        refracs (torch.Tensor | None): remaining absolute refractory periods,
            in :math:`\text{ms}`, when not ``None``, adaptations of neurons in their
            absolute refractory periods are maintained. Defaults to ``None``.

    Returns:
        torch.Tensor: updated adaptations for membrane voltage threshold,
        in :math:`\text{mV}`.

    .. admonition:: Shape
        :class: tensorshape

        ``adaptations``:

        :math:`N_0 \times \cdots \times K`

        ``spikes``, ``refracs``:

        :math:`[B] \times N_0 \times \cdots`

        ``step_time``, ``time_constant``, ``spike_increment``:

        `Broadcastable <https://pytorch.org/docs/stable/notes/broadcasting.html>`_ with
        ``adaptations``.

        ``return``:

        :math:`[B] \times N_0 \times \cdots \times K`

        Where:
            * :math:`B` is the batch size.
            * :math:`N_0, \ldots` are dimensions of the group of neurons simulated.
            * :math:`K` is the number of sets of adaptation parameters.

    Tip:
        This function doesn't automatically reduce along the batch dimension,
        this should generally be done by averaging along the :math:`0^\text{th}`
        dimension.

    See Also:
        For more details and references, visit
        :ref:`zoo/neurons-adaptation:Adaptive Threshold, Linear Spike-Dependent` in the zoo.
    """
    # decay adaptations over time
    decayed = adaptations * exp(-step_time / time_constant)
    if refracs is None:
        adaptations = decayed
    else:
        adaptations = adaptations.where(refracs.unsqueeze(-1) > 0, decayed)

    # increment adaptations after spiking
    adaptations = adaptations + (spike_increment * spikes.unsqueeze(-1))

    # return updated adaptation state
    return adaptations


def adaptive_thresholds_linear_spike(
    adaptations: torch.Tensor,
    spikes: torch.Tensor,
    *,
    step_time: float | torch.Tensor,
    time_constant: float | torch.Tensor,
    spike_scale: float | torch.Tensor,
    spike_increment: float | torch.Tensor,
    refracs: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Update adaptive thresholds based on postsynaptic spikes.

    .. math::
        \theta_k(t + \Delta t) = \theta_k(t) \exp\left(-\frac{\Delta t}{\tau_k}\right)

    If a spike was generated at time :math:`t`, then.

    .. math::
        \theta_k(t) \leftarrow b_k \theta_k(t) + d_k

    Args:
        adaptations (torch.Tensor): last adaptations applied to membrane voltage
            threshold, :math:`\theta_k`, in :math:`\text{mV}`.
        spikes (torch.Tensor): if the corresponding neuron generated an
            action potential.
        step_time (float | torch.Tensor): length of a simulation time step,
            :math:`\Delta t`, in :math:`\text{ms}`.
        time_constant (float | torch.Tensor): time constant of exponential decay for
            the adaptations, :math:`\tau_k`, in :math:`\text{ms}`.
        spike_scale (torch.Tensor): amount by which the adaptive threshold is
            multiplied after a spike, :math:`d_k`, in :math:`\text{mV}`.
        spike_increment (torch.Tensor): amount by which the adaptive threshold is
            increased after a spike, :math:`d_k`, in :math:`\text{mV}`.
        refracs (torch.Tensor | None): remaining absolute refractory periods,
            in :math:`\text{ms}`, when not ``None``, adaptations of neurons in their
            absolute refractory periods are maintained. Defaults to ``None``.

    Returns:
        torch.Tensor: updated adaptations for membrane voltage threshold,
        in :math:`\text{mV}`.

    .. admonition:: Shape
        :class: tensorshape

        ``adaptations``:

        :math:`N_0 \times \cdots \times K`

        ``spikes``, ``refracs``:

        :math:`[B] \times N_0 \times \cdots`

        ``step_time``, ``time_constant``, ``spike_increment``:

        `Broadcastable <https://pytorch.org/docs/stable/notes/broadcasting.html>`_ with
        ``adaptations``.

        ``return``:

        :math:`[B] \times N_0 \times \cdots \times K`

        Where:
            * :math:`B` is the batch size.
            * :math:`N_0, \ldots` are dimensions of the group of neurons simulated.
            * :math:`K` is the number of sets of adaptation parameters.

    Tip:
        This function doesn't automatically reduce along the batch dimension,
        this should generally be done by averaging along the :math:`0^\text{th}`
        dimension.

    See Also:
        For more details and references, visit
        :ref:`zoo/neurons-adaptation:Adaptive Threshold, Linear Spike-Dependent` in the zoo.
    """
    # decay adaptations over time
    decayed = adaptations * exp(-step_time / time_constant)

    if refracs is None:
        adaptations = decayed
    else:
        adaptations = adaptations.where(refracs.unsqueeze(-1) > 0, decayed)

    # increment adaptations after spiking
    adaptations = adaptations + (
        adaptations * (spike_scale - 1.0) + spike_increment
    ) * spikes.unsqueeze(-1)

    # return updated adaptation state
    return adaptations


def apply_adaptive_currents(
    current: torch.Tensor,
    adaptations: torch.Tensor,
) -> torch.Tensor:
    r"""Applies simple adaptation to presynaptic currents.

    Args:
        current (torch.Tensor): presynaptic currents, :math:`I_+`,
            in :math:`\text{nA}`.
        adaptations (torch.Tensor): :math:`k` current adaptations, :math:`w_k`,
            in :math:`\text{nA}`.

    Returns:
        torch.Tensor: adapted presynaptic currents.

    Note:
        The first :math:`N - 1` dimensions of ``adaptations`` must be broadcastable
        with ``current``.

    See Also:
        For an example, visit :ref:`zoo/neurons-adaptation:Adaptive Current, Linear` in the zoo.
    """
    # return adjusted currents
    return current - torch.sum(adaptations, dim=-1)


def apply_adaptive_thresholds(
    threshold: float | torch.Tensor,
    adaptations: torch.Tensor,
) -> torch.Tensor:
    r"""Applies simple adaptation to voltage firing thresholds.

    Args:
        threshold (float | torch.Tensor): equilibrium of the firing threshold,
            :math:`\Theta_\infty`, in :math:`\text{mV}`.
        adaptations (torch.Tensor): :math:`k` threshold adaptations, :math:`\theta_k`,
            in :math:`\text{mV}`.

    Returns:
        torch.Tensor: adapted firing thresholds.

    Note:
        The first :math:`N - 1` dimensions of ``adaptations`` must be broadcastable
        with ``threshold``.

    See Also:
        For an example, visit :ref:`zoo/neurons-adaptation:Adaptive Threshold, Linear Spike-Dependent` in the zoo.
    """
    # return adjusted thresholds
    return threshold + torch.sum(adaptations, dim=-1)
