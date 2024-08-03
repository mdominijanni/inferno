import math
import torch
from typing import Callable


def exp_trace_nearest(
    observation: torch.Tensor,
    trace: torch.Tensor | None,
    *,
    step_time: float,
    time_constant: float,
    amplitude: int | float | complex,
    target: int | float | bool | complex,
    tolerance: int | float | None = None,
) -> torch.Tensor:
    r"""Performs an exponential nearest-neighbor trace for a time step, parameterized by a time constant.

    .. math::
        x(t) =
        \begin{cases}
            A & \lvert h(t) - h^* \rvert \leq \epsilon \\
            x(t - \Delta t) \exp \left(-\frac{\Delta t}{\tau}\right)
            & \left[\lvert h(t) - h^* \rvert > \epsilon\right]
        \end{cases}

    When ``trace`` is ``None``, the event mask created will be cast to the datatype
    of ``observation``.

    Args:
        observation (torch.Tensor): latest state to consider for the trace, :math:`h`.
        trace (torch.Tensor | None): current value of the trace, :math:`x`,
            if not the initial condition.
        step_time (float): simulation step time, :math:`\Delta t`, in :math:`\text{ms}`.
        time_constant (float): time constant of exponential decay, :math:`\tau`,
            in :math:`\text{ms}`.
        amplitude (int | float | complex): value to set trace to for
            matching elements, :math:`A`.
        target (int | float | bool | complex): target value to set
            trace to, :math:`h^*`.
        tolerance (int | float | None, optional): allowable absolute
            difference to still count as a match, :math:`\epsilon`.
            Defaults to ``None``.

    Returns:
        torch.Tensor: updated trace, incorporating the new observation.

    Caution:
        Both ``step_time`` and ``time_constant`` need to be positive values, but
        this will not be checked for.
    """
    return trace_nearest(
        observation,
        trace,
        decay=math.exp(-step_time / time_constant),
        amplitude=amplitude,
        target=target,
        tolerance=tolerance,
    )


def exprate_trace_nearest(
    observation: torch.Tensor,
    trace: torch.Tensor | None,
    *,
    step_time: float,
    rate_constant: float,
    amplitude: int | float | complex,
    target: int | float | bool | complex,
    tolerance: int | float | None = None,
) -> torch.Tensor:
    r"""Performs an exponential nearest-neighbor trace for a time step, parameterized by a rate constant.

    .. math::
        x(t) =
        \begin{cases}
            A & \lvert h(t) - h^* \rvert \leq \epsilon \\
            x(t - \Delta t) \exp \left(-\lambda\Delta t\right)
            & \left[\lvert h(t) - h^* \rvert > \epsilon\right]
        \end{cases}

    When ``trace`` is ``None``, the event mask created will be cast to the datatype
    of ``observation``.

    Args:
        observation (torch.Tensor): latest state to consider for the trace, :math:`h`.
        trace (torch.Tensor | None): current value of the trace, :math:`x`,
            if not the initial condition.
        step_time (float): simulation step time, :math:`\Delta t`, in :math:`\text{ms}`.
        rate_constant (float): rate constant of exponential decay, :math:`\lambda`,
            in :math:`\text{ms}^{-1}`.
        amplitude (int | float | complex): value to set trace to for
            matching elements, :math:`A`.
        target (int | float | bool | complex): target value to set
            trace to, :math:`h^*`.
        tolerance (int | float | None, optional): allowable absolute
            difference to still count as a match, :math:`\epsilon`.
            Defaults to ``None``.

    Returns:
        torch.Tensor: updated trace, incorporating the new observation.

    Caution:
        Both ``step_time`` and ``rate_constant`` need to be positive values, but
        this will not be checked for.
    """
    return trace_nearest(
        observation,
        trace,
        decay=math.exp(-rate_constant * step_time),
        amplitude=amplitude,
        target=target,
        tolerance=tolerance,
    )


def exp_trace_cumulative(
    observation: torch.Tensor,
    trace: torch.Tensor | None,
    *,
    step_time: float,
    time_constant: float,
    amplitude: int | float | complex,
    target: int | float | bool | complex,
    tolerance: int | float | None = None,
) -> torch.Tensor:
    r"""Performs an exponential all-to-all trace for a time step, parameterized by a time constant.

    .. math::
        x(t) = x(t - \Delta t) \exp \left(-\frac{\Delta t}{\tau}\right)
        + A \left[\lvert h(t) - h^* \rvert \leq \epsilon\right]

    The event mask created will be cast to the datatype of ``observation`` if ``trace``
    is ``None`` and to the datatype of ``trace`` otherwise.

    Args:
        observation (torch.Tensor): latest state to consider for the trace, :math:`h`.
        trace (torch.Tensor | None): current value of the trace, :math:`x`,
            if not the initial condition.
        step_time (float): simulation step time, :math:`\Delta t`, in :math:`\text{ms}`.
        time_constant (float): time constant of exponential decay, :math:`\tau`,
            in :math:`\text{ms}`.
        amplitude (int | float | complex): value to add to trace to for
            matching elements, :math:`A`.
        target (int | float | bool | complex): target value to set
            trace to, :math:`h^*`.
        tolerance (int | float | None, optional): allowable absolute
            difference to still count as a match, :math:`\epsilon`.
            Defaults to ``None``.

    Returns:
        torch.Tensor: updated trace, incorporating the new observation.

    Caution:
        Both ``step_time`` and ``time_constant`` need to be positive values, but
        this will not be checked for.
    """
    return trace_cumulative(
        observation,
        trace,
        decay=math.exp(-step_time / time_constant),
        amplitude=amplitude,
        target=target,
        tolerance=tolerance,
    )


def exprate_trace_cumulative(
    observation: torch.Tensor,
    trace: torch.Tensor | None,
    *,
    step_time: float,
    rate_constant: float,
    amplitude: int | float | complex,
    target: int | float | bool | complex,
    tolerance: int | float | None = None,
) -> torch.Tensor:
    r"""Performs an exponential all-to-all trace for a time step, parameterized by a time constant.

    .. math::
        x(t) = x(t - \Delta t) \exp \left(-\lambda\Delta t\right)
        + A \left[\lvert h(t) - h^* \rvert \leq \epsilon\right]

    The event mask created will be cast to the datatype of ``observation`` if ``trace``
    is ``None`` and to the datatype of ``trace`` otherwise.

    Args:
        observation (torch.Tensor): latest state to consider for the trace, :math:`h`.
        trace (torch.Tensor | None): current value of the trace, :math:`x`,
            if not the initial condition.
        step_time (float): simulation step time, :math:`\Delta t`, in :math:`\text{ms}`.
        rate_constant (float): rate constant of exponential decay, :math:`\lambda`,
            in :math:`\text{ms}^{-1}`.
        amplitude (int | float | complex): value to add to trace to for
            matching elements, :math:`A`.
        target (int | float | bool | complex): target value to set
            trace to, :math:`h^*`.
        tolerance (int | float | None, optional): allowable absolute
            difference to still count as a match, :math:`\epsilon`.
            Defaults to ``None``.

    Returns:
        torch.Tensor: updated trace, incorporating the new observation.

    Caution:
        Both ``step_time`` and ``time_constant`` need to be positive values, but
        this will not be checked for.
    """
    return trace_cumulative(
        observation,
        trace,
        decay=math.exp(-rate_constant * step_time),
        amplitude=amplitude,
        target=target,
        tolerance=tolerance,
    )


def trace_nearest(
    observation: torch.Tensor,
    trace: torch.Tensor | None,
    *,
    decay: float,
    amplitude: int | float | complex,
    target: int | float | bool | complex,
    tolerance: int | float | None = None,
) -> torch.Tensor:
    r"""Performs a trace for a time step, considering the latest match.

    .. math::
        x(t) =
        \begin{cases}
            A & \lvert h(t) - h^* \rvert \leq \epsilon \\
            x(t - \Delta t) \alpha
            & \left[\lvert h(t) - h^* \rvert > \epsilon\right]
        \end{cases}

    When ``trace`` is ``None``, the event mask created will be cast to the datatype
    of ``observation``.

    Args:
        observation (torch.Tensor): latest state to consider for the trace, :math:`h`.
        trace (torch.Tensor | None): current value of the trace, :math:`x`,
            if not the initial condition.
        decay (float): decay term of the trace, :math:`\alpha`, unitless.
        amplitude (int | float | complex): value to set trace to for
            matching elements, :math:`A`.
        target (int | float | bool | complex): target value to set
            trace to, :math:`h^*`.
        tolerance (int | float | None, optional): allowable absolute
            difference to still count as a match, :math:`\epsilon`.
            Defaults to ``None``.

    Returns:
        torch.Tensor: updated trace, incorporating the new observation.

    Important:
        To compute a regular, exponentially decaying trace, this assumes that ``decay``
        is precomputed as :math:`\exp\left(-\frac{\Delta t}{\tau}\right)` or as
        :math:`\exp\left(-\lambda\Delta t\right)`, where :math:`\Delta t` is the
        simulation step time and :math:`\tau` is the decay time constant and
        :math:`\lambda` is the decay rate constant.
    """
    # construct mask
    if tolerance is None:
        mask = observation == target
    else:
        mask = torch.abs(observation - target) <= tolerance

    # compute new state
    if trace is None:
        return amplitude * mask.to(dtype=observation.dtype)
    else:
        return torch.where(mask, amplitude, decay * trace)


def trace_cumulative(
    observation: torch.Tensor,
    trace: torch.Tensor | None,
    *,
    decay: float,
    amplitude: int | float | complex,
    target: int | float | bool | complex,
    tolerance: int | float | None = None,
) -> torch.Tensor:
    r"""Performs a trace for a time step, considering all prior matches.

    .. math::
        x(t) = x(t - \Delta t) \alpha
        + A \left[\lvert h(t) - h^* \rvert \leq \epsilon\right]

    The event mask created will be cast to the datatype of ``observation`` if ``trace``
    is ``None`` and to the datatype of ``trace`` otherwise.

    Args:
        observation (torch.Tensor): latest state to consider for the trace, :math:`h`.
        trace (torch.Tensor | None): current value of the trace, :math:`x`,
            if not the initial condition.
        decay (float): decay term of the trace, :math:`\alpha`, unitless.
        amplitude (int | float | complex): value to add to trace to for
            matching elements, :math:`A`.
        target (int | float | bool | complex): target value to set
            trace to, :math:`h^*`.
        tolerance (int | float | None, optional): allowable absolute
            difference to still count as a match, :math:`\epsilon`.
            Defaults to ``None``.

    Returns:
        torch.Tensor: updated trace, incorporating the new observation.

    Important:
        To compute a regular, exponentially decaying trace, this assumes that ``decay``
        is precomputed as :math:`\exp\left(-\frac{\Delta t}{\tau}\right)` or as
        :math:`\exp\left(-\lambda\Delta t\right)`, where :math:`\Delta t` is the
        simulation step time and :math:`\tau` is the decay time constant and
        :math:`\lambda` is the decay rate constant.
    """
    # construct mask
    if tolerance is None:
        mask = observation == target
    else:
        mask = torch.abs(observation - target) <= tolerance

    # compute new state
    if trace is None:
        return amplitude * mask.to(dtype=observation.dtype)
    else:
        return (decay * trace) + (amplitude * mask.to(dtype=trace.dtype))


def trace_nearest_scaled(
    observation: torch.Tensor,
    trace: torch.Tensor | None,
    *,
    decay: float,
    amplitude: int | float | complex,
    scale: int | float | complex,
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
            x(t - \Delta t) \alpha & \neg J(h)
        \end{cases}

    Args:
        observation (torch.Tensor): latest state to consider for the trace, :math:`h`.
        trace (torch.Tensor | None): current value of the trace, :math:`x`,
            if not the initial condition.
        decay (float): decay term of the trace, :math:`\alpha`, unitless.
        amplitude (int | float | complex): value to add to trace
            for matching elements, :math:`A`.
        scale (int | float | complex): value to multiply matching
            inputs by for the trace, :math:`s`.
        matchfn (Callable[[torch.Tensor], torch.Tensor]): test if the inputs are
            considered a match for the trace, :math:`J`.

    Returns:
        torch.Tensor: updated trace, incorporating the new observation.

    Important:
        To compute a regular, exponentially decaying trace, this assumes that ``decay``
        is precomputed as :math:`\exp\left(-\frac{\Delta t}{\tau}\right)` or as
        :math:`\exp\left(-\lambda\Delta t\right)`, where :math:`\Delta t` is the
        simulation step time and :math:`\tau` is the decay time constant and
        :math:`\lambda` is the decay rate constant.

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
    decay: float,
    amplitude: int | float | complex,
    scale: int | float | complex,
    matchfn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    r"""Performs a trace for a time step, considering all prior matches, scaled by the inputs.

    Similar to :py:func:`trace_cumulative`, except rather than checking for a match,
    with or without some permitted tolerance, this requires the inputs to match some
    predicate function. Integration logic also permits the scaling of inputs to affect
    the trace value, in addition to the additive component.

    .. math::
        x(t) = x(t - \Delta t) \alpha + (sh + A) \left[\lvert J(h) \right]

    Args:
        observation (torch.Tensor): latest state to consider for the trace, :math:`h`.
        trace (torch.Tensor | None): current value of the trace, :math:`x`,
            if not the initial condition.
        decay (float): decay term of the trace, :math:`\alpha`, unitless.
        amplitude (int | float | complex): value to add to trace
            to for matching elements, :math:`A`.
        scale (int | float | complex): value to multiply matching
            inputs by for the trace, :math:`s`.
        matchfn (Callable[[torch.Tensor], torch.Tensor]): test if the inputs are considered a
            match for the trace, :math:`J`.

    Returns:
        torch.Tensor: updated trace, incorporating the new observation.

    Important:
        To compute a regular, exponentially decaying trace, this assumes that ``decay``
        is precomputed as :math:`\exp\left(-\frac{\Delta t}{\tau}\right)` or as
        :math:`\exp\left(-\lambda\Delta t\right)`, where :math:`\Delta t` is the
        simulation step time and :math:`\tau` is the decay time constant and
        :math:`\lambda` is the decay rate constant.

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


def trace_cumulative_value(
    observation: torch.Tensor,
    trace: torch.Tensor | None,
    *,
    decay: float,
    scale: int | float | complex,
) -> torch.Tensor:
    r"""Performs a trace for a time step, considering all prior values.

    .. math::
        x(t) = x(t - \Delta t) \alpha + sh

    Args:
        observation (torch.Tensor): latest state to consider for the trace, :math:`h`.
        trace (torch.Tensor | None): current value of the trace, :math:`x`,
            if not the initial condition.
        decay (float): decay term of the trace, :math:`\alpha`, unitless.
        scale (int | float | complex): value to multiply inputs by for
            the trace, :math:`s`.

    Important:
        To compute a regular, exponentially decaying trace, this assumes that ``decay``
        is precomputed as :math:`\exp\left(-\frac{\Delta t}{\tau}\right)` or as
        :math:`\exp\left(-\lambda\Delta t\right)`, where :math:`\Delta t` is the
        simulation step time and :math:`\tau` is the decay time constant and
        :math:`\lambda` is the decay rate constant.

    Returns:
        torch.Tensor: updated trace, incorporating the new observation.
    """
    # compute new state
    if trace is None:
        return scale * observation
    else:
        return (decay * trace) + (scale * observation)
