import torch


def interp_previous(
    prev_data: torch.Tensor,
    next_data: torch.Tensor,
    sample_at: torch.Tensor,
    step_time: float,
    **kwargs,
) -> torch.Tensor:
    r"""Interpolates by selecting the previous state.

    .. math::
        D(t_s) = D(0)

    Args:
        prev_data (torch.Tensor): most recent observation prior to sample time,
            :math:`D(t=0)`.
        next_data (torch.Tensor): most recent observation subsequent to sample time,
            :math:`D(t=\Delta t)`.
        sample_at (torch.Tensor): relative time at which to sample data,
            :math:`t_s`.
        step_time (float): length of time between the prior and subsequent observations,
            :math:`\Delta t`.

    Returns:
        torch.Tensor: interpolated data at sample time, :math:`D(t=t_s)`.
    """
    return prev_data


def interp_next(
    prev_data: torch.Tensor,
    next_data: torch.Tensor,
    sample_at: torch.Tensor,
    step_time: float,
    **kwargs,
) -> torch.Tensor:
    r"""Interpolates by selecting the next state.

    .. math::
        D(t_s) = D(\Delta t)

    Args:
        prev_data (torch.Tensor): most recent observation prior to sample time,
            :math:`D(t=0)`.
        next_data (torch.Tensor): most recent observation subsequent to sample time,
            :math:`D(t=\Delta t)`.
        sample_at (torch.Tensor): relative time at which to sample data,
            :math:`t_s`.
        step_time (float): length of time between the prior and subsequent observations,
            :math:`\Delta t`.

    Returns:
        torch.Tensor: interpolated data at sample time, :math:`D(t=t_s)`.
    """
    return next_data


def interp_nearest(
    prev_data: torch.Tensor,
    next_data: torch.Tensor,
    sample_at: torch.Tensor,
    step_time: float,
    **kwargs,
) -> torch.Tensor:
    r"""Interpolates by selecting the nearest state.

    .. math::
        D(t_s) =
        \begin{cases}
            D(\Delta t) &\Delta t - t_s > t_s\\
            D(0) &\text{otherwise}
        \end{cases}

    Args:
        prev_data (torch.Tensor): most recent observation prior to sample time,
            :math:`D(t=0)`.
        next_data (torch.Tensor): most recent observation subsequent to sample time,
            :math:`D(t=\Delta t)`.
        sample_at (torch.Tensor): relative time at which to sample data,
            :math:`t_s`.
        step_time (float): length of time between the prior and subsequent observations,
            :math:`\Delta t`.

    Returns:
        torch.Tensor: interpolated data at sample time, :math:`D(t=t_s)`.
    """
    return torch.where(sample_at / step_time > 0.5, next_data, prev_data)


def interp_linear(
    prev_data: torch.Tensor,
    next_data: torch.Tensor,
    sample_at: torch.Tensor,
    step_time: float,
    **kwargs,
) -> torch.Tensor:
    r"""Interpolates between previous and next states linearlly.

    .. math::
        D(t_s) = D(0) + \left( \frac{D(\Delta t) - D(0)}{\Delta t} \right)t_s

    Args:
        prev_data (torch.Tensor): most recent observation prior to sample time,
            :math:`D(t=0)`.
        next_data (torch.Tensor): most recent observation subsequent to sample time,
            :math:`D(t=\Delta t)`.
        sample_at (torch.Tensor): relative time at which to sample data,
            :math:`t_s`.
        step_time (float): length of time between the prior and subsequent observations,
            :math:`\Delta t`.

    Returns:
        torch.Tensor: interpolated data at sample time, :math:`D(t=t_s)`.
    """
    slope = (next_data - prev_data) / step_time
    return prev_data + slope * sample_at


def interp_expdecay(
    prev_data: torch.Tensor,
    next_data: torch.Tensor,
    sample_at: torch.Tensor,
    step_time: float,
    *,
    time_constant: float,
    **kwargs,
) -> torch.Tensor:
    r"""Interpolates by exponentially decaying value from previous state, parameterized by a time constant.

    .. math::
        D(t_s) = D(0) \exp\left(-\frac{t_s}{\tau}\right)

    Args:
        prev_data (torch.Tensor): most recent observation prior to sample time,
            :math:`D(t=0)`.
        next_data (torch.Tensor): most recent observation subsequent to sample time,
            :math:`D(t=\Delta t)`.
        sample_at (torch.Tensor): relative time at which to sample data,
            :math:`t_s`.
        step_time (float): length of time between the prior and subsequent observations,
            :math:`\Delta t`.
        time_constant (float): time constant of exponential decay, :math:`\tau`.

    Returns:
        torch.Tensor: interpolated data at sample time, :math:`D(t=t_s)`.
    """
    return prev_data * torch.exp(-sample_at / time_constant)


def interp_expratedecay(
    prev_data: torch.Tensor,
    next_data: torch.Tensor,
    sample_at: torch.Tensor,
    step_time: float,
    *,
    rate_constant: float,
    **kwargs,
) -> torch.Tensor:
    r"""Interpolates by exponentially decaying value from previous state, parameterized by a rate constant.

    .. math::
        D(t_s) = D(0) \exp\left(-\lambda t_s\right)

    Args:
        prev_data (torch.Tensor): most recent observation prior to sample time,
            :math:`D(t=0)`.
        next_data (torch.Tensor): most recent observation subsequent to sample time,
            :math:`D(t=\Delta t)`.
        sample_at (torch.Tensor): relative time at which to sample data,
            :math:`t_s`.
        step_time (float): length of time between the prior and subsequent observations,
            :math:`\Delta t`.
        rate_constant (float): rate constant of exponential decay, :math:`\lambda`.

    Returns:
        torch.Tensor: interpolated data at sample time, :math:`D(t=t_s)`.
    """
    return prev_data * torch.exp(-sample_at * rate_constant)
