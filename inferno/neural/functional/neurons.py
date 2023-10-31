from typing import Callable
import torch


def _voltage_thresholding(
    inputs: torch.Tensor,
    refracs: torch.Tensor,
    voltage_fn: Callable[[torch.Tensor], torch.Tensor],
    reset_v: float | torch.Tensor,
    thresh_v: float | torch.Tensor,
    refrac_ts: int | torch.Tensor,
    voltages: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Manage refractory periods, spiking, and voltage based on voltage thresholding.

    Implements the logic, that when.

    .. math::
        $$V_m(t) \geq \Theta(t)$$

    Membrane voltages are reset as.

    .. math::
        $$V_m(t) \leftarrow V_\mathrm{reset}$$

    Args:
        inputs (torch.Tensor): presynaptic currents, in :math:`\mathrm{nA}`.
        refracs (torch.Tensor): number of remaining simulation steps to exit refractory periods.
        voltage_fn (Callable[[torch.Tensor], torch.Tensor]): function which given input currents,
            in :math:`\mathrm{nA}`, returns the updated membrane voltages, in :math:`\mathrm{mV}`.
        reset_v (float | torch.Tensor): membrane voltage after an action potential is generated,
            in :math:`\mathrm{mV}`.
        thresh_v (float | torch.Tensor): membrane voltage at which action potentials are generated,
            in :math:`\mathrm{mV}`.
        refrac_ts (int | torch.Tensor): number of time steps the absolute refractory period lasts.
        voltages (torch.Tensor | None): original voltages to keep if in the refractory period,
            in :math:`\mathrm{mV}`. Defaults to None.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - spikes - generated spikes from changes in membrane potential.
            - voltages - updated membrane potentials, in (:math:`mV`).
            - refracs - updated number of remaining simulation steps to exit refractory periods.
    """
    # decrement refractory periods and create mask
    refracs = (refracs - 1).clamp(min=0)
    mask = refracs == 0

    # compute updated voltages
    if voltages is None:
        voltages = voltage_fn(inputs * mask)
    else:
        voltages = voltages.where(~mask, voltage_fn(inputs * mask))

    # determine which neurons have spiked
    spikes = torch.logical_and(mask, voltages >= thresh_v)

    # set refractory period and voltages of fired neurons to their reset state
    if isinstance(refrac_ts, torch.Tensor):
        refracs = refracs.where(~spikes, refrac_ts)
    else:
        refracs = refracs.where(~spikes, int(refrac_ts))
    voltages = voltages.where(~spikes, reset_v)

    # return generated spikes and updated state
    return spikes, voltages, refracs


def _voltage_thresholding_slope_intercept(
    inputs: torch.Tensor,
    refracs: torch.Tensor,
    voltage_fn: Callable[[torch.Tensor], torch.Tensor],
    rest_v: float | torch.Tensor,
    v_slope: float | torch.Tensor,
    v_intercept: float | torch.Tensor,
    thresh_v: float | torch.Tensor,
    refrac_ts: int | torch.Tensor,
    voltages: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Manage refractory periods, spiking, and voltage based on voltage thresholding.

    Implements the logic, that when.

    .. math::
        $$V_m(t) \geq \Theta(t)$$

    Membrane voltages are reset as.

    .. math::
        $$V_m(t) \leftarrow V_\mathrm{rest} + m_v \left[ V_m(t) - V_\mathrm{rest} \right] - b_v$$

    Args:
        inputs (torch.Tensor): presynaptic currents, in :math:`\mathrm{nA}`.
        refracs (torch.Tensor): number of remaining simulation steps to exit refractory periods.
        voltage_fn (Callable[[torch.Tensor], torch.Tensor]): function which given input currents
            in :math:`\mathrm{nA}`, returns the updated membrane voltages, in :math:`\mathrm{mV}`.
        rest_v (float | torch.Tensor): membrane potential difference at equilibrium,
            in :math:`\mathrm{mV}`.
        v_slope (float | torch.Tensor): multiplicative parameter of the reset voltage,
            unitless.
        v_intercept (float | torch.Tensor): additive parameter of the reset voltage,
            in :math:`\mathrm{mV}`.
        thresh_v (float | torch.Tensor): membrane voltage at which action potentials are generated,
            in :math:`\mathrm{mV}`.
        refrac_ts (int | torch.Tensor): number of time steps the absolute refractory period lasts.
        voltages (torch.Tensor | None): original voltages to keep if in the refractory period,
            in :math:`\mathrm{mV}`. Defaults to None.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - spikes - generated spikes from changes in membrane potential.
            - voltages - updated membrane potentials, in (:math:`mV`).
            - refracs - updated number of remaining simulation steps to exit refractory periods.
    """
    # decrement refractory periods and create mask
    refracs = (refracs - 1).clamp(min=0)
    mask = refracs == 0

    # compute updated voltages
    if voltages is None:
        voltages = voltage_fn(inputs * mask)
    else:
        voltages = voltages.where(~mask, voltage_fn(inputs * mask))

    # determine which neurons have spiked
    spikes = torch.logical_and(mask, voltages >= thresh_v)

    # set refractory period and voltages of fired neurons to their reset state
    if isinstance(refrac_ts, torch.Tensor):
        refracs = refracs.where(~spikes, refrac_ts)
    else:
        refracs = refracs.where(~spikes, int(refrac_ts))
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
