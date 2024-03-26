import torch


def previous(
    prev_data: torch.Tensor,
    next_data: torch.Tensor,
    sample_at: torch.Tensor,
    step_time: float,
    **kwargs,
) -> torch.Tensor:
    r"""Interpolates by selecting the previous state.

    .. math::
        D_{t=s} = D_{t=0}

    Args:
        prev_data (torch.Tensor): most recent observation prior to sample time,
            :math:`D_{t=0}`.
        next_data (torch.Tensor): most recent observation subsequent to sample time,
            :math:`D_{t=T}`.
        sample_at (torch.Tensor): relative time at which to sample data, :math:`s`.
        step_data (float): length of time between the prior and subsequent observations,
            :math:`T`.

    Returns:
        torch.Tensor: interpolated data at sample time, :math:`D_{t=s}`.
    """
    return prev_data


def nearest(
    prev_data: torch.Tensor,
    next_data: torch.Tensor,
    sample_at: torch.Tensor,
    step_time: float,
    **kwargs,
) -> torch.Tensor:
    r"""Interpolates by selecting the nearest state.

    .. math::
        D_{t=s} =
        \begin{cases}
            D_{t=T} &T - s > s\\
            D_{t=0} &\text{otherwise}
        \end{cases}

    Args:
        prev_data (torch.Tensor): most recent observation prior to sample time,
            :math:`D_{t=0}`.
        next_data (torch.Tensor): most recent observation subsequent to sample time,
            :math:`D_{t=T}`.
        sample_at (torch.Tensor): relative time at which to sample data, :math:`s`.
        step_data (float): length of time between the prior and subsequent observations,
            :math:`T`.

    Returns:
        torch.Tensor: interpolated data at sample time, :math:`D_{t=s}`.
    """
    return torch.where(sample_at / step_time > 0.5, next_data, prev_data)


def linear(
    prev_data: torch.Tensor,
    next_data: torch.Tensor,
    sample_at: torch.Tensor,
    step_time: float,
    **kwargs,
) -> torch.Tensor:
    r"""Interpolates between previous and next states linearlly.

    .. math::
        D_{t=s} = \frac{D_{t=T} - D_{t=0}}{T} s + D_{t=0}

    Args:
        prev_data (torch.Tensor): most recent observation prior to sample time,
            :math:`D_{t=0}`.
        next_data (torch.Tensor): most recent observation subsequent to sample time,
            :math:`D_{t=T}`.
        sample_at (torch.Tensor): relative time at which to sample data, :math:`s`.
        step_data (float): length of time between the prior and subsequent observations,
            :math:`T`.

    Returns:
        torch.Tensor: interpolated data at sample time, :math:`D_{t=s}`.
    """
    slope = (next_data - prev_data) / step_time
    return prev_data + slope * sample_at


def expdecay(
    prev_data: torch.Tensor,
    next_data: torch.Tensor,
    sample_at: torch.Tensor,
    step_time: float,
    *,
    time_constant: float,
    **kwargs,
) -> torch.Tensor:
    r"""Interpolates by exponentially decaying value from previous state.

    .. math::
        D_{t=s} = D_{t=0} \exp\left(-\frac{s}{\tau}\right)

    Args:
        prev_data (torch.Tensor): most recent observation prior to sample time,
            :math:`D_{t=0}`.
        next_data (torch.Tensor): most recent observation subsequent to sample time,
            :math:`D_{t=T}`.
        sample_at (torch.Tensor): relative time at which to sample data, :math:`s`.
        step_data (float): length of time between the prior and subsequent observations,
            :math:`T`.
        time_constant (float): time constant of exponential decay, :math:`\tau`.

    Returns:
        torch.Tensor: interpolated data at sample time, :math:`D_{t=s}`.
    """
    return prev_data * torch.exp(-sample_at / time_constant)


def expratedecay(
    prev_data: torch.Tensor,
    next_data: torch.Tensor,
    sample_at: torch.Tensor,
    step_time: float,
    *,
    decay_rate: float,
    **kwargs,
) -> torch.Tensor:
    r"""Interpolates by exponentially decaying value from previous state.

    .. math::
        D_{t=s} = D_{t=0} \exp\left(-\lambda s\right)

    Args:
        prev_data (torch.Tensor): most recent observation prior to sample time,
            :math:`D_{t=0}`.
        next_data (torch.Tensor): most recent observation subsequent to sample time,
            :math:`D_{t=T}`.
        sample_at (torch.Tensor): relative time at which to sample data, :math:`s`.
        step_data (float): length of time between the prior and subsequent observations,
            :math:`T`.
        decay_rate (float): rate of exponential decay, :math:`\lambda`.

    Returns:
        torch.Tensor: interpolated data at sample time, :math:`D_{t=s}`.
    """
    return prev_data * torch.exp(-sample_at * decay_rate)
