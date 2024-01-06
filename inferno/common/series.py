import torch
from inferno.typing import OneToOne


def trace_nearest(
    observation: torch.Tensor,
    trace: torch.Tensor | None,
    *,
    decay: float | torch.Tensor,
    amplitude: int | float | complex | torch.Tensor,
    target: int | float | bool | complex | torch.Tensor,
    tolerance: int | float | torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Performs a trace for a time step, considering the latest match.

    .. math::
        x_{t + \Delta t} =
        \begin{cases}
            a &\lvert f_{t + \Delta t} - f^* \rvert \leq \epsilon \\
            x_t \exp (\Delta t / \tau) &\text{otherwise}
        \end{cases}

    Args:
        observation (torch.Tensor): latest state to consider for the trace, :math:`f`.
        trace (torch.Tensor | None): current value of the trace, :math:`x`, if it exists.
        decay (float | torch.Tensor): exponential decay for adaptations,
            :math:`\exp\left(-\frac{\Delta t}{\tau_k}\right)`, unitless.
        amplitude (int | float | complex | torch.Tensor): value to set trace to for matching elements, :math:`a`.
        target (int | float | bool | complex | torch.Tensor): target value to set trace to, :math:`f^*`.
        tolerance (int | float | torch.Tensor | None, optional): allowable absolute difference to
            still count as a match, :math:`\epsilon`. Defaults to None.

    Returns:
        torch.Tensor: updated trace, incorporating the new observation.
    """
    # construct mask
    if tolerance is None:
        mask = observation == target
    else:
        mask = torch.abs(observation - target) <= tolerance

    # compute new state
    if trace is None:
        return amplitude * mask
    else:
        return torch.where(mask, amplitude, decay * trace)


def trace_nearest_scaled(
    observation: torch.Tensor,
    trace: torch.Tensor | None,
    *,
    decay: float | torch.Tensor,
    amplitude: int | float | complex | torch.Tensor,
    scale: int | float | complex | torch.Tensor,
    matchfn: OneToOne[torch.Tensor],
) -> torch.Tensor:
    r"""Performs a trace for a time step, considering the latest match, scaled by the inputs.

    Similar to :py:func:`trace_nearest`, except rather than checking for a match, with or without
    some permitted tolerance, this requires the inputs to match some predicate function. Integration
    logic also permits the scaling of inputs to affect the trace value, in addition to the additive
    component.

    .. math::
        x_{t + \Delta t} =
        \begin{cases}
            a + Sf &K(f_{t + \Delta t}) \\
            x_t \exp (\Delta t / \tau) &\text{otherwise}
        \end{cases}

    Args:
        observation (torch.Tensor): latest state to consider for the trace, :math:`f`.
        trace (torch.Tensor | None): current value of the trace, :math:`x`.
        decay (float | torch.Tensor): exponential decay for adaptations,
            :math:`\exp\left(-\frac{\Delta t}{\tau_k}\right)`, unitless.
        amplitude (int | float | complex | torch.Tensor): value to add to trace
            for matching elements, :math:`a`.
        scale (int | float | complex | torch.Tensor): value to multiply matching
            inputs by for the trace, :math:`S`.
        matchfn (OneToOne[torch.Tensor]): test if the inputs are considered a
            match for the trace, :math:`K`.

    Returns:
        torch.Tensor: updated trace, incorporating the new observation.

    Note:
        The output of ``matchfn`` must have the datatype of ``torch.bool`` as it is used as a mask.
    """
    # construct mask
    mask = matchfn(observation)

    # compute new state
    if trace is None:
        return (scale * observation + amplitude) * mask
    return torch.where(
        mask, scale * observation + amplitude, decay * trace
    )


def trace_cumulative(
    observation: torch.Tensor,
    trace: torch.Tensor | None,
    *,
    decay: float | torch.Tensor,
    amplitude: int | float | complex | torch.Tensor,
    target: int | float | bool | complex | torch.Tensor,
    tolerance: int | float | torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Performs a trace for a time step, considering all prior matches.

    .. math::
        x_{t + \Delta t} =
        \begin{cases}
            a + x_t \exp (\Delta t / \tau) &\lvert f_{t + \Delta t} - f^* \rvert \leq \epsilon \\
            x_t \exp (\Delta t / \tau) &\text{otherwise}
        \end{cases}

    Args:
        observation (torch.Tensor): latest state to consider for the trace, :math:`f`.
        trace (torch.Tensor | None): current value of the trace, :math:`x`.
        decay (float | torch.Tensor): exponential decay for adaptations,
            :math:`\exp\left(-\frac{\Delta t}{\tau_k}\right)`, unitless.
        amplitude (int | float | complex | torch.Tensor): value to add to trace to for matching elements, :math:`a`.
        target (int | float | bool | complex | torch.Tensor): target value to set trace to, :math:`f^*`.
        tolerance (int | float | torch.Tensor | None, optional): allowable absolute difference to
            still count as a match, :math:`\epsilon`. Defaults to None.

    Returns:
        torch.Tensor: updated trace, incorporating the new observation.
    """
    # construct mask
    if tolerance is None:
        mask = observation == target
    else:
        mask = torch.abs(observation - target) <= tolerance

    # compute new state
    if trace is None:
        amplitude * mask
    else:
        (decay * trace) + (amplitude * mask)


def trace_cumulative_scaled(
    observation: torch.Tensor,
    trace: torch.Tensor | None,
    *,
    decay: float | torch.Tensor,
    amplitude: int | float | complex | torch.Tensor,
    scale: int | float | complex | torch.Tensor,
    matchfn: OneToOne[torch.Tensor],
) -> torch.Tensor:
    r"""Performs a trace for a time step, considering all prior matches, scaled by the inputs.

    Similar to :py:func:`trace_cumulative`, except rather than checking for a match, with or without
    some permitted tolerance, this requires the inputs to match some predicate function. Integration
    logic also permits the scaling of inputs to affect the trace value, in addition to the additive
    component.

    .. math::
        x_{t + \Delta t} =
        \begin{cases}
            a + Sf + x_t \exp (\Delta t / \tau) &K(f_{t + \Delta t}) \\
            x_t \exp (\Delta t / \tau) &\text{otherwise}
        \end{cases}

    Args:
        observation (torch.Tensor): latest state to consider for the trace, :math:`f`.
        trace (torch.Tensor | None): current value of the trace, :math:`x`.
        decay (float | torch.Tensor): exponential decay for adaptations,
            :math:`\exp\left(-\frac{\Delta t}{\tau_k}\right)`, unitless.
        amplitude (int | float | complex | torch.Tensor): value to add to trace to for matching elements, :math:`a`.
        scale (int | float | complex | torch.Tensor): value to multiply matching inputs by for the trace, :math:`S`.
        matchfn (OneToOne[torch.Tensor]): test if the inputs are considered a match for the trace, :math:`K`.

    Returns:
        torch.Tensor: updated trace, incorporating the new observation.

    Note:
        The output of ``matchfn`` must have the datatype of ``torch.bool`` as it is used as a mask.
    """
    # construct mask
    mask = matchfn(observation)

    # compute new state
    if trace is None:
        return (scale * observation + amplitude) * mask
    else:
        return (decay * trace) + (scale * observation + amplitude) * mask


def simple_exponential_smoothing(
    observation: torch.Tensor,
    level: torch.Tensor | None,
    *,
    alpha: float | int | complex | torch.Tensor,
) -> torch.Tensor:
    r"""Performs simple exponential smoothing for a time step.

    .. math::
        \begin{align*}
            s_0 &= x_0 \\
            s_{t + 1} &= \alpha x_{t + 1}  + (1 - \alpha) s_t
        \end{align*}

    Args:
        observation (torch.Tensor): latest state to consider for exponential smoothing,
            :math:`x`.
        level (torch.Tensor | None): current value of the smoothed level,
            :math:`s`.
        alpha (float | int | complex | torch.Tensor): level smoothing factor, :math:`\alpha`.

    Returns:
        torch.Tensor: revised exponentially smoothed value.
    """
    # initial condition
    if level is None:
        return observation

    # standard condition
    else:
        return alpha * observation + (1 - alpha) * level


def holt_linear_smoothing(
    observation: torch.Tensor,
    level: torch.Tensor | None,
    trend: torch.Tensor | None,
    *,
    alpha: float | int | complex | torch.Tensor,
    beta: float | int | complex | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Performs Holt linear smoothing for a time step.

    .. math::
        \begin{align*}
            s_0 &= x_0 \\
            b_0 &= x_1 - x_0 \\
            s_{t + 1} &= \alpha x_{t + 1}  + (1 - \alpha) s_t \\
            b_{t + 1} &= \beta (s_{t + 1} - s_t) + (1 - \beta) b_t
        \end{align*}

    Args:
        observation (torch.Tensor): latest state to consider for exponential smoothing,
            :math:`x_{t + 1}`.
        level (torch.Tensor | None): current value of the smoothed level,
            :math:`s`.
        trend (torch.Tensor | None): current value of the smoothed trend,
            :math:`b`.
        alpha (float | int | complex | torch.Tensor): level smoothing factor, :math:`\alpha`.
        beta (float | int | complex | torch.Tensor): trend smoothing factor, :math:`\beta`.

    Returns:
        tuple[torch.Tensor, int | torch.Tensor]: tuple containing output/updated state:

            level: revised exponentially smoothed level.

            trend: revised exponentially smoothed trend.
    """
    # t=0 condition
    if level is None:
        return observation, None

    # t=1 condition (initialize trend as x1-x0)
    if trend is None:
        trend = observation - level

    # t>0 condition
    s = simple_exponential_smoothing(observation, level + trend, alpha)
    b = simple_exponential_smoothing(s - level, trend, beta)

    return s, b
