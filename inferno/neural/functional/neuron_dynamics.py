from inferno.typing import OneToOne
import torch


def voltage_thresholding(
    inputs: torch.Tensor,
    refracs: torch.Tensor,
    dynamics: OneToOne[torch.Tensor],
    *,
    step_time: float | torch.Tensor,
    reset_v: float | torch.Tensor,
    thresh_v: float | torch.Tensor,
    refrac_t: float | torch.Tensor,
    voltages: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Manage refractory periods, spiking, and voltage based on voltage thresholding.

    Implements the logic, that when.

    .. math::
        V_m(t) \geq \Theta(t)

    Membrane voltages are reset as.

    .. math::
        V_m(t) \leftarrow V_\mathrm{reset}

    Args:
        inputs (torch.Tensor): presynaptic currents, :math:`I(t)`,
            in :math:`\mathrm{nA}`.
        refracs (torch.Tensor): remaining absolute refractory periods,
            in :math:`\mathrm{ms}`.
        dynamics (OneToOne[torch.Tensor]): function which given input currents in
            :math:`\mathrm{nA}` returns the updated membrane voltages, :math:`V_m(t)`,
            in :math:`\mathrm{mV}`.
        step_time (float | torch.Tensor): length of a simulation time step,
            in :math:`\mathrm{ms}`.
        reset_v (float | torch.Tensor): membrane voltage after an action potential
            is generated, :math:`V_\mathrm{reset}`, in :math:`\mathrm{mV}`.
        thresh_v (float | torch.Tensor): membrane voltage at which action potentials
            are generated, \Theta(t), in :math:`\mathrm{mV}`.
        refrac_t (float | torch.Tensor): length the absolute refractory period,
            in :math:`\mathrm{ms}`.
        voltages (torch.Tensor | None): membrane voltages, V_m(t),
            in :math:`\mathrm{mV}`, to maintain while in refractory periods,
            voltages not held if None. Defaults to None.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: tuple of output and
        updated state containing:

            * spikes: if the corresponding neuron generated an action potential.
            * voltages: updated membrane potentials, in :math:`\mathrm{mV}`.
            * refracs: remaining absolute refractory periods, in :math:`\mathrm{ms}`.
    """
    # decrement refractory periods and create mask
    refracs = (refracs - step_time).clamp(min=0)
    mask = refracs == 0

    # compute updated voltages
    if voltages is None:
        voltages = dynamics(inputs * mask)
    else:
        voltages = voltages.where(~mask, dynamics(inputs * mask))

    # determine which neurons have spiked
    spikes = torch.logical_and(mask, voltages >= thresh_v)

    # set refractory period and voltages of fired neurons to their reset state
    refracs = refracs.where(~spikes, refrac_t)
    voltages = voltages.where(~spikes, reset_v)

    # return generated spikes and updated state
    return spikes, voltages, refracs


def voltage_thresholding_slope_intercept(
    inputs: torch.Tensor,
    refracs: torch.Tensor,
    dynamics: OneToOne[torch.Tensor],
    *,
    step_time: float | torch.Tensor,
    rest_v: float | torch.Tensor,
    v_slope: float | torch.Tensor,
    v_intercept: float | torch.Tensor,
    thresh_v: float | torch.Tensor,
    refrac_t: float | torch.Tensor,
    voltages: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Manage refractory periods, spiking, and voltage based on voltage thresholding.

    Implements the logic, that when.

    .. math::
        V_m(t) \geq \Theta(t)

    Membrane voltages are reset as.

    .. math::
        V_m(t) \leftarrow V_\mathrm{rest} + m_v \left[ V_m(t) - V_\mathrm{rest} \right] - b_v

    Args:
        inputs (torch.Tensor): presynaptic currents, :math:`I(t)`,
            in :math:`\mathrm{nA}`.
        refracs (torch.Tensor): remaining absolute refractory periods,
            in :math:`\mathrm{ms}`.
        dynamics (OneToOne[torch.Tensor]): function which given input currents in
            :math:`\mathrm{nA}` returns the updated membrane voltages, :math:`V_m(t)`,
            in :math:`\mathrm{mV}`.
        step_time (float | torch.Tensor): length of a simulation time step,
            in :math:`\mathrm{ms}`.
        rest_v (float | torch.Tensor): membrane potential difference at equilibrium,
            :math:`V_\mathrm{rest}`, in :math:`\mathrm{mV}`.
        v_slope (float | torch.Tensor): additive parameter controlling reset voltage,
            :math:`b_v`, in :math:`\mathrm{mV}`.
        v_intercept (float | torch.Tensor): multiplicative parameter controlling
            reset voltage, :math:`m_v`, unitless.
        thresh_v (float | torch.Tensor): membrane voltage at which action potentials
            are generated, \Theta(t), in :math:`\mathrm{mV}`.
        refrac_t (float | torch.Tensor): length the absolute refractory period,
            in :math:`\mathrm{ms}`.
        voltages (torch.Tensor | None): membrane voltages, V_m(t),
            in :math:`\mathrm{mV}`, to maintain while in refractory periods,
            voltages not held if None. Defaults to None.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: tuple of output and
        updated state containing:

            * spikes: if the corresponding neuron generated an action potential.
            * voltages: updated membrane potentials, in :math:`\mathrm{mV}`.
            * refracs: remaining absolute refractory periods, in :math:`\mathrm{ms}`.
    """
    # decrement refractory periods and create mask
    refracs = (refracs - step_time).clamp(min=0)
    mask = refracs == 0

    # compute updated voltages
    if voltages is None:
        voltages = dynamics(inputs * mask)
    else:
        voltages = voltages.where(~mask, dynamics(inputs * mask))

    # determine which neurons have spiked
    spikes = torch.logical_and(mask, voltages >= thresh_v)

    # set refractory period and voltages of fired neurons to their reset state
    refracs = refracs.where(~spikes, refrac_t)
    voltages = voltages.where(
        ~spikes, rest_v + v_slope * (voltages - rest_v) - v_intercept
    )

    # return generated spikes and updated state
    return spikes, voltages, refracs


def voltage_integration_linear(
    masked_inputs: torch.Tensor,
    voltages: torch.Tensor,
    *,
    decay: float | torch.Tensor,
    rest_v: float | torch.Tensor,
    resistance: float | torch.Tensor,
) -> torch.Tensor:
    r"""Integrates input currents into membrane voltages using linear dynamics.

    .. math::
        V_m(t + \Delta t) = \left[V_m(t) - V_\mathrm{rest} - R_mI(t)\right]
        \alpha + V_\mathrm{rest} + R_mI(t)

    Where :math:`\alpha` is the multiple for exponential decay, typically expressed
    as :math:`\alpha = \exp(-\Delta t / \tau)`, where :math:`\Delta t` is the step time
    and :math:`\tau` is the time constant, in like units of time.

    Args:
        masked_inputs (torch.Tensor): presynaptic currents masked by neurons in their
            absolute refractory period, :math:`I(t)`, in :math:`\mathrm{nA}`.
        voltages (torch.Tensor): membrane voltages :math:`V_m(t)`,
            in :math:`\mathrm{mV}`.
        decay (float | torch.Tensor): exponential decay factor for membrane voltage,
            :math:`\alpha`, unitless.
        rest_v (float | torch.Tensor): membrane potential difference at equilibrium,
            :math:`V_\mathrm{rest}`, in :math:`\mathrm{mV}`.
        resistance (float | torch.Tensor): resistance across the cell membrane,
            :math:`R_m`, in :math:`\mathrm{M\Omega}`.

    Returns:
        torch.Tensor: membrane voltages with inputs integrated, in :math:`\mathrm{mV}`.
    """
    v_in = resistance * masked_inputs
    v_delta = voltages - rest_v
    return v_in + (v_delta - v_in) * decay + rest_v
