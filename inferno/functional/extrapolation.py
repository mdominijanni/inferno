import torch
from typing import Callable


def extrap_previous(
    sample: torch.Tensor,
    sample_at: torch.Tensor,
    prev_data: torch.Tensor,
    next_data: torch.Tensor,
    step_time: float,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Extrapolates out to the previous state.

    .. math::
        \begin{align*}
            X(0) &= X(t_s) \\
            X(\Delta t) &= D(\Delta t)
        \end{align*}

    Args:
        sample (torch.Tensor): sample from which to extrapolate,
            :math:`X(t=t_s)`
        sample_at (torch.Tensor): relative time at which to sample data,
            :math:`t_s`.
        prev_data (torch.Tensor): most recent observation prior to sample time,
            :math:`D(t=0)`.
        next_data (torch.Tensor): most recent observation subsequent to sample time,
            :math:`D(t=\Delta t)`.
        step_time (float): length of time between the prior and subsequent observations,
            :math:`\Delta t`.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: extrapolated data at neighboring steps,
        :math:`(X(t=0), X(t=\Delta t))`.
    """
    return (sample, next_data)


def extrap_next(
    sample: torch.Tensor,
    sample_at: torch.Tensor,
    prev_data: torch.Tensor,
    next_data: torch.Tensor,
    step_time: float,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Extrapolates out to the next state.

    .. math::
        \begin{align*}
            X(0) &= D(0) \\
            X(\Delta t) &= X(t_s)
        \end{align*}

    Args:
        sample (torch.Tensor): sample from which to extrapolate,
            :math:`X(t=t_s)`
        sample_at (torch.Tensor): relative time at which to sample data,
            :math:`t_s`.
        prev_data (torch.Tensor): most recent observation prior to sample time,
            :math:`D(t=0)`.
        next_data (torch.Tensor): most recent observation subsequent to sample time,
            :math:`D(t=\Delta t)`.
        step_time (float): length of time between the prior and subsequent observations,
            :math:`\Delta t`.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: extrapolated data at neighboring steps,
        :math:`(X(t=0), X(t=\Delta t))`.
    """
    return (prev_data, sample)


def extrap_neighbors(
    sample: torch.Tensor,
    sample_at: torch.Tensor,
    prev_data: torch.Tensor,
    next_data: torch.Tensor,
    step_time: float,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Extrapolates out to the neighboring states.

    .. math::
        \begin{align*}
            X(0) &= X(t_s) \\
            X(\Delta t) &= X(t_s)
        \end{align*}

    Args:
        sample (torch.Tensor): sample from which to extrapolate,
            :math:`X(t=t_s)`
        sample_at (torch.Tensor): relative time at which to sample data,
            :math:`t_s`.
        prev_data (torch.Tensor): most recent observation prior to sample time,
            :math:`D(t=0)`.
        next_data (torch.Tensor): most recent observation subsequent to sample time,
            :math:`D(t=\Delta t)`.
        step_time (float): length of time between the prior and subsequent observations,
            :math:`\Delta t`.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: extrapolated data at neighboring steps,
        :math:`(X(t=0), X(t=\Delta t))`.
    """
    return (sample, sample)


def extrap_nearest(
    sample: torch.Tensor,
    sample_at: torch.Tensor,
    prev_data: torch.Tensor,
    next_data: torch.Tensor,
    step_time: float,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Extrapolates out to the closest neighbor.

    .. math::
        \begin{align*}
            X(0) &=
            \begin{cases}
                X(t_s) & t_s \leq \Delta t / 2\\
                D(0) &\text{otherwise}
            \end{cases} \\
            X(\Delta t) &=
            \begin{cases}
                X(t_s) & t_s > \Delta t / 2\\
                D(\Delta t) &\text{otherwise}
            \end{cases}
        \end{align*}

    Args:
        sample (torch.Tensor): sample from which to extrapolate,
            :math:`X(t=t_s)`
        sample_at (torch.Tensor): relative time at which to sample data,
            :math:`t_s`.
        prev_data (torch.Tensor): most recent observation prior to sample time,
            :math:`D(t=0)`.
        next_data (torch.Tensor): most recent observation subsequent to sample time,
            :math:`D(t=\Delta t)`.
        step_time (float): length of time between the prior and subsequent observations,
            :math:`\Delta t`.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: extrapolated data at neighboring steps,
        :math:`(X(t=0), X(t=\Delta t))`.
    """
    cond = sample_at > (step_time / 2)
    return (torch.where(cond, prev_data, sample), torch.where(cond, sample, next_data))


def extrap_linear_forward(
    sample: torch.Tensor,
    sample_at: torch.Tensor,
    prev_data: torch.Tensor,
    next_data: torch.Tensor,
    step_time: float,
    *,
    adjust: Callable[[torch.Tensor], torch.Tensor] | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Extrapolates out linearly to the next state.

    .. math::
        \begin{align*}
            X(0) &= f(D(0)) \\
            X(\Delta t) &= X(0) + \left(\frac{X(t_s) - X(0)}{t_s} \right) \Delta t
        \end{align*}

    Args:
        sample (torch.Tensor): sample from which to extrapolate,
            :math:`X(t=t_s)`
        sample_at (torch.Tensor): relative time at which to sample data,
            :math:`t_s`.
        prev_data (torch.Tensor): most recent observation prior to sample time,
            :math:`D(t=0)`.
        next_data (torch.Tensor): most recent observation subsequent to sample time,
            :math:`D(t=\Delta t)`.
        step_time (float): length of time between the prior and subsequent observations,
            :math:`\Delta t`.
        adjust (Callable[[torch.Tensor], torch.Tensor] | None, optional): function to
            apply to the previous state before extrapolating, identity when ``None``,
            :math:`f`. Defaults to ``None``.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: extrapolated data at neighboring steps,
        :math:`(X(t=0), X(t=\Delta t))`.
    """
    prev_data = adjust(prev_data) if adjust else prev_data
    slope = (sample - prev_data) / sample_at
    return (prev_data, prev_data + slope * step_time)


def extrap_linear_backward(
    sample: torch.Tensor,
    sample_at: torch.Tensor,
    prev_data: torch.Tensor,
    next_data: torch.Tensor,
    step_time: float,
    *,
    adjust: Callable[[torch.Tensor], torch.Tensor] | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Extrapolates out linearly to the previous state.

    .. math::
        \begin{align*}
            X(0) &= X(\Delta t) - \left(\frac{X(\Delta t) - X(t_s)}{\Delta t - t_s} \right) \Delta t \\
            X(\Delta t) &= f(D(\Delta t))
        \end{align*}

    Args:
        sample (torch.Tensor): sample from which to extrapolate,
            :math:`X(t=t_s)`
        sample_at (torch.Tensor): relative time at which to sample data,
            :math:`t_s`.
        prev_data (torch.Tensor): most recent observation prior to sample time,
            :math:`D(t=0)`.
        next_data (torch.Tensor): most recent observation subsequent to sample time,
            :math:`D(t=\Delta t)`.
        step_time (float): length of time between the prior and subsequent observations,
            :math:`\Delta t`.
        adjust (Callable[[torch.Tensor], torch.Tensor] | None, optional): function to
            apply to the next state before extrapolating, identity when ``None``,
            :math:`f`. Defaults to ``None``.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: extrapolated data at neighboring steps,
        :math:`(X(t=0), X(t=\Delta t))`.
    """
    next_data = adjust(next_data) if adjust else next_data
    slope = (next_data - sample) / (step_time - sample_at)
    return (next_data - slope * step_time, next_data)


def extrap_expdecay(
    sample: torch.Tensor,
    sample_at: torch.Tensor,
    prev_data: torch.Tensor,
    next_data: torch.Tensor,
    step_time: float,
    *,
    time_constant: float,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Extrapolates out assuming exponential decay dynamics, parameterized by a time constant.

    .. math::
        \begin{align*}
            X(0) &= X(t_s) \exp \left( \frac{t_s}{\tau} \right) \\
            X(\Delta t) &= X(t_s) \exp \left( -\frac{\Delta t - t_s}{\tau} \right)
        \end{align*}

    Args:
        sample (torch.Tensor): sample from which to extrapolate,
            :math:`X(t=t_s)`
        sample_at (torch.Tensor): relative time at which to sample data,
            :math:`t_s`.
        prev_data (torch.Tensor): most recent observation prior to sample time,
            :math:`D(t=0)`.
        next_data (torch.Tensor): most recent observation subsequent to sample time,
            :math:`D(t=\Delta t)`.
        step_time (float): length of time between the prior and subsequent observations,
            :math:`\Delta t`.
        time_constant (float): time constant of exponential decay, :math:`\tau`.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: extrapolated data at neighboring steps,
        :math:`(X(t=0), X(t=\Delta t))`.
    """
    return (
        sample * torch.exp(sample_at / time_constant),
        sample * torch.exp((sample_at - step_time) / time_constant),
    )


def extrap_expratedecay(
    sample: torch.Tensor,
    sample_at: torch.Tensor,
    prev_data: torch.Tensor,
    next_data: torch.Tensor,
    step_time: float,
    *,
    rate_constant: float,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Extrapolates out assuming exponential decay dynamics, parameterized by a rate constant.

    .. math::
        \begin{align*}
            X(0) &= X(t_s) \exp \left( \lambda t_s \right) \\
            X(\Delta t) &= X(t_s) \exp \left( -\lambda (\Delta t - t_s) \right)
        \end{align*}

    Args:
        sample (torch.Tensor): sample from which to extrapolate,
            :math:`X(t=t_s)`
        sample_at (torch.Tensor): relative time at which to sample data,
            :math:`t_s`.
        prev_data (torch.Tensor): most recent observation prior to sample time,
            :math:`D(t=0)`.
        next_data (torch.Tensor): most recent observation subsequent to sample time,
            :math:`D(t=\Delta t)`.
        step_time (float): length of time between the prior and subsequent observations,
            :math:`\Delta t`.
        rate_constant (float): rate constant of exponential decay, :math:`\lambda`.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: extrapolated data at neighboring steps,
        :math:`(X(t=0), X(t=\Delta t))`.
    """
    return (
        sample * torch.exp(sample_at * rate_constant),
        sample * torch.exp((sample_at - step_time) * rate_constant),
    )
