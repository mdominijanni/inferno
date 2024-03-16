import torch
from typing import Callable


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
        x(t) =
        \begin{cases}
            A & \lvert h(t) - h^* \rvert \leq \epsilon \\
            x(t - \Delta t) \exp \left(-\frac{\Delta t}{\tau_x}\right)
            & \left[\lvert h(t) - h^* \rvert > \epsilon\right]
        \end{cases}

    Args:
        observation (torch.Tensor): latest state to consider for the trace, :math:`h`.
        trace (torch.Tensor | None): current value of the trace, :math:`x`,
            if not the inital condition.
        decay (float | torch.Tensor): exponential decay for adaptations,
            :math:`\exp\left(-\frac{\Delta t}{\tau_x}\right)`, unitless.
        amplitude (int | float | complex | torch.Tensor): value to set trace to for
            matching elements, :math:`A`.
        target (int | float | bool | complex | torch.Tensor): target value to set
            trace to, :math:`h^*`.
        tolerance (int | float | torch.Tensor | None, optional): allowable absolute
            difference to still count as a match, :math:`\epsilon`. Defaults to None.

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
        x(t) = x(t - \Delta t) \exp \left(-\frac{\Delta t}{\tau_x}\right)
        + A \left[\lvert h(t) - h^* \rvert \leq \epsilon\right]

    Args:
        observation (torch.Tensor): latest state to consider for the trace, :math:`h`.
        trace (torch.Tensor | None): current value of the trace, :math:`x`,
            if not the inital condition.
        decay (float | torch.Tensor): exponential decay for adaptations,
            :math:`\exp\left(-\frac{\Delta t}{\tau_x}\right)`, unitless.
        amplitude (int | float | complex | torch.Tensor): value to add to trace to for
            matching elements, :math:`A`.
        target (int | float | bool | complex | torch.Tensor): target value to set
            trace to, :math:`h^*`.
        tolerance (int | float | torch.Tensor | None, optional): allowable absolute
            difference to still count as a match, :math:`\epsilon`. Defaults to None.

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
        return (decay * trace) + (amplitude * mask)


def trace_nearest_scaled(
    observation: torch.Tensor,
    trace: torch.Tensor | None,
    *,
    decay: float | torch.Tensor,
    amplitude: int | float | complex | torch.Tensor,
    scale: int | float | complex | torch.Tensor,
    matchfn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    r"""Performs a trace for a time step, considering the latest match, scaled by the inputs.

    Similar to :py:func:`trace_nearest`, except rather than checking for a match,
    with or without some permitted tolerance, this requires the inputs to match some
    predicate function. Integration logic also permits the scaling of inputs to affect
    the trace value, in addition to the additive component.

    .. math::
        x(t) =
        \begin{cases}
            sh + A & J(h) \\
            x(t - \Delta t) \exp \left(-\frac{\Delta t}{\tau_x}\right)
            & \neg J(h)
        \end{cases}

    Args:
        observation (torch.Tensor): latest state to consider for the trace, :math:`h`.
        trace (torch.Tensor | None): current value of the trace, :math:`x`,
            if not the inital condition.
        decay (float | torch.Tensor): exponential decay for adaptations,
            :math:`\exp\left(-\frac{\Delta t}{\tau_x}\right)`, unitless.
        amplitude (int | float | complex | torch.Tensor): value to add to trace
            for matching elements, :math:`A`.
        scale (int | float | complex | torch.Tensor): value to multiply matching
            inputs by for the trace, :math:`s`.
        matchfn (Callable[[torch.Tensor], torch.Tensor]): test if the inputs are considered a
            match for the trace, :math:`J`.

    Returns:
        torch.Tensor: updated trace, incorporating the new observation.

    Important:
        The output of ``matchfn`` must have the datatype of ``torch.bool`` as it
        is used as a mask.
    """
    # construct mask
    mask = matchfn(observation)

    # compute new state
    if trace is None:
        return (scale * observation + amplitude) * mask
    return torch.where(mask, scale * observation + amplitude, decay * trace)


def trace_cumulative_scaled(
    observation: torch.Tensor,
    trace: torch.Tensor | None,
    *,
    decay: float | torch.Tensor,
    amplitude: int | float | complex | torch.Tensor,
    scale: int | float | complex | torch.Tensor,
    matchfn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    r"""Performs a trace for a time step, considering all prior matches, scaled by the inputs.

    Similar to :py:func:`trace_cumulative`, except rather than checking for a match,
    with or without some permitted tolerance, this requires the inputs to match some
    predicate function. Integration logic also permits the scaling of inputs to affect
    the trace value, in addition to the additive component.

    .. math::
        x_{t + \Delta t} =
        \begin{cases}
            a + Sf + x_t \exp (\Delta t / \tau) &K(f_{t + \Delta t}) \\
            x_t \exp (\Delta t / \tau) &\text{otherwise}
        \end{cases}

    .. math::
        x(t) = x(t - \Delta t) \exp \left(-\frac{\Delta t}{\tau_x}\right)
        + (sh + A) \left[\lvert J(h) \right]

    Args:
        observation (torch.Tensor): latest state to consider for the trace, :math:`h`.
        trace (torch.Tensor | None): current value of the trace, :math:`x`,
            if not the inital condition.
        decay (float | torch.Tensor): exponential decay for adaptations,
            :math:`\exp\left(-\frac{\Delta t}{\tau_x}\right)`, unitless.
        amplitude (int | float | complex | torch.Tensor): value to add to trace
            to for matching elements, :math:`A`.
        scale (int | float | complex | torch.Tensor): value to multiply matching
            inputs by for the trace, :math:`s`.
        matchfn (Callable[[torch.Tensor], torch.Tensor]): test if the inputs are considered a
            match for the trace, :math:`J`.

    Returns:
        torch.Tensor: updated trace, incorporating the new observation.

    Important:
        The output of ``matchfn`` must have the datatype of ``torch.bool`` as it
        is used as a mask.
    """
    # construct mask
    mask = matchfn(observation)

    # compute new state
    if trace is None:
        return (scale * observation + amplitude) * mask
    else:
        return (decay * trace) + (scale * observation + amplitude) * mask
