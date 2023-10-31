import torch
from .math import exp
from inferno.typing import OneToOne


def trace_nearest(
    observation: torch.Tensor,
    trace: torch.Tensor | None,
    *,
    step_time: float | torch.Tensor,
    time_constant: float | torch.Tensor,
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
        trace (torch.Tensor | None): current value of the trace, :math:`x`, will initialize if set to None.
        step_time (float | torch.Tensor): length of the discrete step time, :math:`\Delta t`.
        time_constant (float | torch.Tensor): time constant of exponential decay, :math:`\tau`.
        amplitude (int | float | complex | torch.Tensor): value to set trace to for matching elements, :math:`a`.
        target (int | float | bool | complex | torch.Tensor): target value to set trace to, :math:`f^*`.
        tolerance (int | float | torch.Tensor | None, optional): allowable absolute difference to
            still count as a match, :math:`\epsilon`. Defaults to None.

    Returns:
        torch.Tensor: updated trace, incorporating the new observation.
    """
    # initialize trace if it doesn't exist
    if trace is None:
        trace = torch.zeros_like(observation)

    # compute decay for either tensors or primitives
    decay = exp(-step_time / time_constant)

    # construct mask
    if tolerance is None:
        mask = observation == target
    else:
        mask = torch.abs(observation - target) <= tolerance

    return torch.where(mask, amplitude, decay * trace)


def trace_nearest_scaled(
    observation: torch.Tensor,
    trace: torch.Tensor | None,
    *,
    step_time: float | torch.Tensor,
    time_constant: float | torch.Tensor,
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
        trace (torch.Tensor | None): current value of the trace, :math:`x`, will initialize if set to None.
        step_time (float | torch.Tensor): length of the discrete step time, :math:`\Delta t`.
        time_constant (float | torch.Tensor): time constant of exponential decay, :math:`\tau`.
        amplitude (int | float | complex | torch.Tensor): value to add to trace to for matching elements, :math:`a`.
        scale (int | float | complex | torch.Tensor): value to multiply matching inputs by for the trace, :math:`S`.
        matchfn (OneToOne[torch.Tensor]): test if the inputs are considered a match for the trace, :math:`K`.

    Returns:
        torch.Tensor: updated trace, incorporating the new observation.

    Note:
        The output of ``matchfn`` must have the datatype of ``torch.bool`` as it is used as a mask.
    """
    # initialize trace if it doesn't exist
    if trace is None:
        trace = torch.zeros_like(observation)

    # compute decay for either tensors or primitives
    decay = exp(-step_time / time_constant)

    return torch.where(
        matchfn(observation), scale * observation + amplitude, decay * trace
    )


def trace_cumulative(
    observation: torch.Tensor,
    trace: torch.Tensor | None,
    *,
    step_time: float | torch.Tensor,
    time_constant: float | torch.Tensor,
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
        trace (torch.Tensor | None): current value of the trace, :math:`x`, will initialize if set to None.
        step_time (float | torch.Tensor): length of the discrete step time, :math:`\Delta t`.
        time_constant (float | torch.Tensor): time constant of exponential decay, :math:`\tau`.
        amplitude (int | float | complex | torch.Tensor): value to add to trace to for matching elements, :math:`a`.
        target (int | float | bool | complex | torch.Tensor): target value to set trace to, :math:`f^*`.
        tolerance (int | float | torch.Tensor | None, optional): allowable absolute difference to
            still count as a match, :math:`\epsilon`. Defaults to None.

    Returns:
        torch.Tensor: updated trace, incorporating the new observation.
    """
    # initialize trace if it doesn't exist
    if trace is None:
        trace = torch.zeros_like(observation)

    # compute decay for either tensors or primitives
    decay = exp(-step_time / time_constant)

    # construct mask
    if tolerance is None:
        mask = observation == target
    else:
        mask = torch.abs(observation - target) <= tolerance

    return (decay * trace) + torch.where(mask, amplitude, 0)


def trace_cumulative_scaled(
    observation: torch.Tensor,
    trace: torch.Tensor | None,
    *,
    step_time: float | torch.Tensor,
    time_constant: float | torch.Tensor,
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
        trace (torch.Tensor | None): current value of the trace, :math:`x`, will initialize if set to None.
        step_time (float | torch.Tensor): length of the discrete step time, :math:`\Delta t`.
        time_constant (float | torch.Tensor): time constant of exponential decay, :math:`\tau`.
        amplitude (int | float | complex | torch.Tensor): value to add to trace to for matching elements, :math:`a`.
        scale (int | float | complex | torch.Tensor): value to multiply matching inputs by for the trace, :math:`S`.
        matchfn (OneToOne[torch.Tensor]): test if the inputs are considered a match for the trace, :math:`K`.

    Returns:
        torch.Tensor: updated trace, incorporating the new observation.

    Note:
        The output of ``matchfn`` must have the datatype of ``torch.bool`` as it is used as a mask.
    """
    # initialize trace if it doesn't exist
    if trace is None:
        trace = torch.zeros_like(observation)

    # compute decay for either tensors or primitives
    decay = exp(-step_time / time_constant)

    return (decay * trace) + torch.where(
        matchfn(observation), scale * observation + amplitude, 0
    )


def cumulative_average(
    observation: torch.Tensor,
    cumavg: torch.Tensor | None,
    *,
    num_samples: int | torch.Tensor,
) -> tuple[torch.Tensor, int | torch.Tensor]:
    r"""Performs the cumulative average calculation for a time step.

    .. math::
        CA_{n + 1} = \frac{x_{n + 1} + n \cdot CA_n}{n + 1}

    Args:
        observation (torch.Tensor): latest state to consider for the average, :math:`x`.
        cumavg (torch.Tensor | None): current value of the average, will initialize if set to None.
        num_samples (int | torch.Tensor): number of samples observed for the average, :math:`n`.

    Raises:
        ValueError: ``num_samples`` was assigned a negative value.

    Returns:
        tuple[torch.Tensor, int | torch.Tensor]: tuple containing output and updated state:

            cumavg: updated cumulative average.

            num_samples: updated number of samples in the average.
    """
    # initialize state
    if cumavg is None:
        cumavg = torch.zeros_like(observation)

    # check that num_samples is non-negative
    if num_samples < 0:
        raise ValueError(f"'num_samples' must be non-negative, received {num_samples}")

    # return new cumavg and number of samples
    return (observation + num_samples * cumavg) / (num_samples + 1), num_samples + 1


def simple_exponential_smoothing(
    observation: torch.Tensor,
    smoothed: torch.Tensor | None,
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
        smoothed (torch.Tensor | None): current value of the smoothed data,
            :math:`s`, will initialize if set to None.
        alpha (float | int | complex | torch.Tensor): data smoothing factor, :math:`\alpha`.

    Returns:
        torch.Tensor: revised exponentially smoothed value.
    """
    # initial condition
    if smoothed is None:
        return observation
    # standard condition
    return alpha * observation + (1 - alpha) * smoothed


def holt_linear_smoothing(
    observation: torch.Tensor,
    smoothed: torch.Tensor | None,
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
        smoothed (torch.Tensor | None): current value of the smoothed data,
            :math:`s`, will initialize if set to None.
        trend (torch.Tensor | None): current value of the smoothed trend,
            :math:`s`, will initialize if set to None.
        alpha (float | int | complex | torch.Tensor): data smoothing factor, :math:`\alpha`.
        beta (float | int | complex | torch.Tensor): trend smoothing factor, :math:`\beta`.

    Returns:
        tuple[torch.Tensor, int | torch.Tensor]: tuple containing output/updated state:

            smoothed: revised exponentially smoothed value.

            trend: revised exponentially trend value.
    """
    # t=0 condition
    if smoothed is None:
        return observation, None
    # t=1 condition (will be equal to x1-x0 at t=1)
    if trend is None:
        trend = observation - smoothed
    # t>1 condition
    s = simple_exponential_smoothing(observation, smoothed + trend, alpha)
    b = simple_exponential_smoothing(s - smoothed, trend, beta)
    return s, b
