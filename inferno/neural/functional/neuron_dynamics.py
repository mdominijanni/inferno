from ... import exp
from ...types import OneToOne
import torch


def voltage_thresholding_constant(
    inputs: torch.Tensor,
    refracs: torch.Tensor,
    dynamics: OneToOne[torch.Tensor],
    voltages: torch.Tensor | None = None,
    *,
    step_time: float | torch.Tensor,
    reset_v: float | torch.Tensor,
    thresh_v: float | torch.Tensor,
    refrac_t: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Manage refractory periods, spiking, and voltage based on voltage thresholding.

    Implements the logic, that when.

    .. math::
        V_m(t) \geq \Theta(t)

    Membrane voltages are reset as.

    .. math::
        V_m(t) \leftarrow V_\text{reset}

    Args:
        inputs (torch.Tensor): presynaptic currents, :math:`I(t)`,
            in :math:`\text{nA}`.
        refracs (torch.Tensor): remaining absolute refractory periods,
            in :math:`\text{ms}`.
        dynamics (OneToOne[torch.Tensor]): function which given input currents in
            :math:`\text{nA}` returns the updated membrane voltages, :math:`V_m(t)`,
            in :math:`\text{mV}`.
        voltages (torch.Tensor | None, optional): membrane voltages, V_m(t),
            in :math:`\text{mV}`, to maintain while in refractory periods,
            voltages not held if ``None``. Defaults to ``None``.
        step_time (float | torch.Tensor): length of a simulation time step,
            in :math:`\text{ms}`.
        reset_v (float | torch.Tensor): membrane voltage after an action potential
            is generated, :math:`V_\text{reset}`, in :math:`\text{mV}`.
        thresh_v (float | torch.Tensor): membrane voltage at which action potentials
            are generated, \Theta(t), in :math:`\text{mV}`.
        refrac_t (float | torch.Tensor): length the absolute refractory period,
            in :math:`\text{ms}`.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: tuple of output and
        updated state containing:

            * spikes: if the corresponding neuron generated an action potential.
            * voltages: updated membrane potentials, in :math:`\text{mV}`.
            * refracs: remaining absolute refractory periods, in :math:`\text{ms}`.
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


def voltage_thresholding_linear(
    inputs: torch.Tensor,
    refracs: torch.Tensor,
    dynamics: OneToOne[torch.Tensor],
    voltages: torch.Tensor | None = None,
    *,
    step_time: float | torch.Tensor,
    rest_v: float | torch.Tensor,
    v_slope: float | torch.Tensor,
    v_intercept: float | torch.Tensor,
    thresh_v: float | torch.Tensor,
    refrac_t: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Manage refractory periods, spiking, and voltage based on voltage thresholding.

    Implements the logic, that when.

    .. math::
        V_m(t) \geq \Theta(t)

    Membrane voltages are reset as.

    .. math::
        V_m(t) \leftarrow V_\text{rest} + m_v \left[ V_m(t) - V_\text{rest} \right] - b_v

    Args:
        inputs (torch.Tensor): presynaptic currents, :math:`I(t)`,
            in :math:`\text{nA}`.
        refracs (torch.Tensor): remaining absolute refractory periods,
            in :math:`\text{ms}`.
        dynamics (OneToOne[torch.Tensor]): function which given input currents in
            :math:`\text{nA}` returns the updated membrane voltages, :math:`V_m(t)`,
            in :math:`\text{mV}`.
        voltages (torch.Tensor | None, optional): membrane voltages, V_m(t),
            in :math:`\text{mV}`, to maintain while in refractory periods,
            voltages not held if ``None``. Defaults to ``None``.
        step_time (float | torch.Tensor): length of a simulation time step,
            in :math:`\text{ms}`.
        rest_v (float | torch.Tensor): membrane potential difference at equilibrium,
            :math:`V_\text{rest}`, in :math:`\text{mV}`.
        v_slope (float | torch.Tensor): additive parameter controlling reset voltage,
            :math:`b_v`, in :math:`\text{mV}`.
        v_intercept (float | torch.Tensor): multiplicative parameter controlling
            reset voltage, :math:`m_v`, unitless.
        thresh_v (float | torch.Tensor): membrane voltage at which action potentials
            are generated, \Theta(t), in :math:`\text{mV}`.
        refrac_t (float | torch.Tensor): length the absolute refractory period,
            in :math:`\text{ms}`.
        voltages (torch.Tensor | None): membrane voltages, V_m(t),
            in :math:`\text{mV}`, to maintain while in refractory periods,
            voltages not held if ``None``. Defaults to ``None``.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: tuple of output and
        updated state containing:

            * spikes: if the corresponding neuron generated an action potential.
            * voltages: updated membrane potentials, in :math:`\text{mV}`.
            * refracs: remaining absolute refractory periods, in :math:`\text{ms}`.
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
    step_time: float | torch.Tensor,
    time_constant: float | torch.Tensor,
    rest_v: float | torch.Tensor,
    resistance: float | torch.Tensor,
) -> torch.Tensor:
    r"""Integrates input currents into membrane voltages using linear dynamics.

    .. math::
        V_m(t + \Delta t) = \left[V_m(t) - V_\text{rest} - R_mI(t)\right]
        \exp(-\Delta t / \tau_m) + V_\text{rest} + R_mI(t)

    Args:
        masked_inputs (torch.Tensor): presynaptic currents masked by neurons in their
            absolute refractory period, :math:`I(t)`, in :math:`\text{nA}`.
        voltages (torch.Tensor): membrane voltages :math:`V_m(t)`,
            in :math:`\text{mV}`.
        step_time (float | torch.Tensor): length of a simulation time step,
            :math:`\Delta t`, in :math:`\text{ms}`.
        time_constant (float | torch.Tensor): time constant of exponential decay for
            membrane voltage, :math:`\tau_m`, in :math:`\text{ms}`.
        rest_v (float | torch.Tensor): membrane potential difference at equilibrium,
            :math:`V_\text{rest}`, in :math:`\text{mV}`.
        resistance (float | torch.Tensor): resistance across the cell membrane,
            :math:`R_m`, in :math:`\text{M}\Omega`.

    Returns:
        torch.Tensor: membrane voltages with inputs integrated, in :math:`\text{mV}`.
    """
    decay = exp(-step_time / time_constant)
    extvoltage = resistance * masked_inputs
    return rest_v + (voltages - rest_v - extvoltage) * decay + extvoltage


def voltage_integration_quadratic(
    masked_inputs: torch.Tensor,
    voltages: torch.Tensor,
    *,
    step_time: float | torch.Tensor,
    rest_v: float | torch.Tensor,
    crit_v: float | torch.Tensor,
    affinity: float | torch.Tensor,
    time_constant: float | torch.Tensor,
    resistance: float | torch.Tensor,
) -> torch.Tensor:
    r"""Integrates input currents into membrane voltages using quadratic dynamics.

    Implemented as an approximation using Euler's method.

    .. math::
        V_m(t + \Delta t) = \frac{\Delta t}{\tau_m}
        \left[ a \left(V_m(t) - V_\text{rest}\right)\left(V_m(t) - V_\text{crit}\right) + R_mI(t) \right]
        + V_m(t)

    Args:
        masked_inputs (torch.Tensor): presynaptic currents masked by neurons in their
            absolute refractory period, :math:`I(t)`, in :math:`\text{nA}`.
        voltages (torch.Tensor): membrane voltages :math:`V_m(t)`,
            in :math:`\text{mV}`.
        step_time (float | torch.Tensor): length of a simulation time step,
            :math:`\Delta t`, in :math:`\text{ms}`.
        rest_v (float | torch.Tensor): membrane potential difference at equilibrium,
            :math:`V_\text{rest}`, in :math:`\text{mV}`.
        crit_v (float | torch.Tensor): membrane potential difference at which potential
            naturally increases, :math:`V_\text{crit}`, in :math:`\text{mV}`.
        affinity (float | torch.Tensor): controls the strength of the membrane
            potential's drift towards :math:`V_\text{rest}` and away from
            :math:`V_\text{crit}`, :math:`a`, unitless.
        time_constant (float | torch.Tensor): time constant of exponential decay,
            :math:`\tau_m`, in :math:`\text{ms}`.
        resistance (float | torch.Tensor): resistance across the cell membrane,
            :math:`R_m`, in :math:`\text{M}\Omega`.

    Returns:
        torch.Tensor: membrane voltages with inputs integrated, in :math:`\text{mV}`.
    """
    dyn_v = affinity * (voltages - rest_v) * (voltages - crit_v)
    decay = step_time / time_constant
    return voltages + decay * (dyn_v + (resistance * masked_inputs))


def voltage_integration_exponential(
    masked_inputs: torch.Tensor,
    voltages: torch.Tensor,
    *,
    step_time: float | torch.Tensor,
    rest_v: float | torch.Tensor,
    rheobase_v: float | torch.Tensor,
    sharpness: float | torch.Tensor,
    time_constant: float | torch.Tensor,
    resistance: float | torch.Tensor,
) -> torch.Tensor:
    r"""Integrates input currents into membrane voltages using exponential dynamics.

    Implemented as an approximation using Euler's method.

    .. math::
        V_m(t + \Delta t) = \frac{\Delta t}{\tau_m}
        \left[ a \left(V_m(t) - V_\text{rest}\right)\left(V_m(t) - V_\text{crit}\right) + R_mI(t) \right]
        + V_m(t)

    Args:
        masked_inputs (torch.Tensor): presynaptic currents masked by neurons in their
            absolute refractory period, :math:`I(t)`, in :math:`\text{nA}`.
        voltages (torch.Tensor): membrane voltages :math:`V_m(t)`,
            in :math:`\text{mV}`.
        step_time (float | torch.Tensor): length of a simulation time step,
            :math:`\Delta t`, in :math:`\text{ms}`.
        rest_v (float | torch.Tensor): membrane potential difference at equilibrium,
            :math:`V_\text{rest}`, in :math:`\text{mV}`.
        rheobase_v (float | torch.Tensor): membrane potential difference at which
            potential naturally increases, :math:`V_\text{crit}`, in :math:`\text{mV}`.
        sharpness (float | torch.Tensor): steepness of the natural increase in membrane
            potential above the rheobase voltage, :math:`\Delta_T`,
            in :math:`\text{mV}`.
        time_constant (float | torch.Tensor): time constant of exponential decay,
            :math:`\tau_m`, in :math:`\text{ms}`.
        resistance (float | torch.Tensor): resistance across the cell membrane,
            :math:`R_m`, in :math:`\text{M}\Omega`.

    Returns:
        torch.Tensor: membrane voltages with inputs integrated, in :math:`\text{mV}`.
    """
    expdyn_v = sharpness * torch.exp((voltages - rheobase_v) / sharpness)
    decay = step_time / time_constant
    return voltages + decay * (
        -(voltages - rest_v) + expdyn_v + (resistance * masked_inputs)
    )
