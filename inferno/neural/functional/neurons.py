from typing import Callable
import torch


def _voltage_thresholding(
    inputs: torch.Tensor,
    refracs: torch.Tensor,
    voltage_fn: Callable[[torch.Tensor], torch.Tensor],
    step_time: float | torch.Tensor,
    reset_v: float | torch.Tensor,
    thresh_v: float | torch.Tensor,
    refrac_t: float | torch.Tensor,
    voltages: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Manage refractory periods, spiking, and voltage based on voltage thresholding, used internally.

    Implements the logic, that when.

    .. math::
        $$V_m(t) \geq \Theta(t)$$

    Membrane voltages are reset as.

    .. math::
        $$V_m(t) \leftarrow V_\mathrm{reset}$$

    Args:
        inputs (torch.Tensor): presynaptic currents, in :math:`\mathrm{nA}`.
        refracs (torch.Tensor): amount of remaining time needed to exit refractory periods,
            in :math:`\mathrm{ms}`.
        voltage_fn (Callable[[torch.Tensor], torch.Tensor]): function which given input currents,
            in :math:`\mathrm{nA}`, returns the updated membrane voltages, in :math:`\mathrm{mV}`.
        step_time (float | torch.Tensor): length of the simulated step time,
            in :math:`\mathrm{ms}`.
        reset_v (float | torch.Tensor): membrane voltage after an action potential is generated,
            in :math:`\mathrm{mV}`.
        thresh_v (float | torch.Tensor): membrane voltage at which action potentials are generated,
            in :math:`\mathrm{mV}`.
        refrac_t (float | torch.Tensor): length of time the absolute refractory period lasts,
            in :math:`\mathrm{ms}`.
        voltages (torch.Tensor | None): original voltages to keep if in the refractory period,
            in :math:`\mathrm{mV}`. Defaults to None.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: tuple of output and updated state containing:

            - spikes: which neurons generated an action potential.
            - voltages: updated membrane potentials, in :math:`\mathrm{mV}`.
            - refracs: amount of remaining time needed to exit refractory periods,
              in :math:`\mathrm{ms}`.
    """
    # decrement refractory periods and create mask
    refracs = (refracs - step_time).clamp(min=0)
    mask = refracs == 0

    # compute updated voltages
    if voltages is None:
        voltages = voltage_fn(inputs * mask)
    else:
        voltages = voltages.where(~mask, voltage_fn(inputs * mask))

    # determine which neurons have spiked
    spikes = torch.logical_and(mask, voltages >= thresh_v)

    # set refractory period and voltages of fired neurons to their reset state
    refracs = refracs.where(~spikes, refrac_t)
    voltages = voltages.where(~spikes, reset_v)

    # return generated spikes and updated state
    return spikes, voltages, refracs


def _voltage_thresholding_slope_intercept(
    inputs: torch.Tensor,
    refracs: torch.Tensor,
    voltage_fn: Callable[[torch.Tensor], torch.Tensor],
    step_time: float | torch.Tensor,
    rest_v: float | torch.Tensor,
    v_slope: float | torch.Tensor,
    v_intercept: float | torch.Tensor,
    thresh_v: float | torch.Tensor,
    refrac_t: float | torch.Tensor,
    voltages: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Manage refractory periods, spiking, and voltage based on voltage thresholding, used internally.

    Implements the logic, that when.

    .. math::
        $$V_m(t) \geq \Theta(t)$$

    Membrane voltages are reset as.

    .. math::
        $$V_m(t) \leftarrow V_\mathrm{rest} + m_v \left[ V_m(t) - V_\mathrm{rest} \right] - b_v$$

    Args:
        inputs (torch.Tensor): presynaptic currents, in :math:`\mathrm{nA}`.
        refracs (torch.Tensor): amount of remaining time needed to exit refractory periods,
            in :math:`\mathrm{ms}`.
        voltage_fn (Callable[[torch.Tensor], torch.Tensor]): function which given input currents
            in :math:`\mathrm{nA}`, returns the updated membrane voltages, in :math:`\mathrm{mV}`.
        step_time (float | torch.Tensor): length of the simulated step time,
            in :math:`\mathrm{ms}`.
        rest_v (float | torch.Tensor): membrane potential difference at equilibrium,
            in :math:`\mathrm{mV}`.
        v_slope (float | torch.Tensor): multiplicative parameter of the reset voltage,
            unitless.
        v_intercept (float | torch.Tensor): additive parameter of the reset voltage,
            in :math:`\mathrm{mV}`.
        thresh_v (float | torch.Tensor): membrane voltage at which action potentials are generated,
            in :math:`\mathrm{mV}`.
        refrac_t (float | torch.Tensor): length of time the absolute refractory period lasts,
            in :math:`\mathrm{ms}`.
        voltages (torch.Tensor | None): original voltages to keep if in the refractory period,
            in :math:`\mathrm{mV}`. Defaults to None.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: tuple of output and updated state containing:

            - spikes: which neurons generated an action potential.
            - voltages: updated membrane potentials, in :math:`\mathrm{mV}`.
            - refracs: amount of remaining time needed to exit refractory periods,
              in :math:`\mathrm{ms}`.
    """
    # decrement refractory periods and create mask
    refracs = (refracs - step_time).clamp(min=0)
    mask = refracs == 0

    # compute updated voltages
    if voltages is None:
        voltages = voltage_fn(inputs * mask)
    else:
        voltages = voltages.where(~mask, voltage_fn(inputs * mask))

    # determine which neurons have spiked
    spikes = torch.logical_and(mask, voltages >= thresh_v)

    # set refractory period and voltages of fired neurons to their reset state
    refracs = refracs.where(~spikes, refrac_t)
    voltages = voltages.where(
        ~spikes, rest_v + v_slope * (voltages - rest_v) - v_intercept
    )

    # return generated spikes and updated state
    return spikes, voltages, refracs


def apply_adaptive_currents(
    presyn_currents: torch.Tensor,
    adaptations: torch.Tensor,
) -> torch.Tensor:
    # return adjusted currents
    return presyn_currents - torch.sum(adaptations, dim=-1)


def apply_adaptive_thresholds(
    threshold: float | torch.Tensor,
    adaptations: torch.Tensor,
) -> torch.Tensor:
    # return adjusted thresholds
    return threshold + torch.sum(adaptations, dim=-1)
