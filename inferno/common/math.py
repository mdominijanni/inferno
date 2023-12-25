import cmath
from functools import singledispatch
import math
import numpy as np
import torch
from typing import Protocol


@singledispatch
def exp(
    x: int | float | complex | torch.Tensor | np.ndarray | np.number,
) -> float | complex | torch.Tensor | np.ndarray | np.number:
    r"""Type agnostic exponential function.

    .. math::
        y = e^x

    Args:
        x (int | float | complex | torch.Tensor | numpy.ndarray | numpy.number): value by which to raise :math:`e`.

    Returns:
        float | complex | torch.Tensor | numpy.ndarray | numpy.number: :math:`e` raised to the input.
    """
    raise NotImplementedError


@exp.register(int)
@exp.register(float)
def _(x: int | float) -> float:
    return math.exp(x)


@exp.register(complex)
def _(x: complex) -> complex:
    return cmath.exp(x)


@exp.register(torch.Tensor)
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(x)


@exp.register(np.ndarray)
def _(x: np.ndarray) -> np.ndarray:
    return np.exp(x)


@exp.register(np.number)
def _(x: np.number) -> np.number:
    return np.exp(x)


class Interpolation(Protocol):
    r"""Callable used to interpolate in time between two tensors.

    Args:
        prev_data (torch.Tensor): most recent observation prior to sample time.
        next_data (torch.Tensor): most recent observation subsequent to sample time.
        sample_at (torch.Tensor): relative time at which to sample data.
        step_data (float): length of time between the prior and subsequent observations.

    Returns:
        torch.Tensor: interpolated data at sample time.

    Note:
        ``sample_at`` is measured as the amount of time after the time at which
        ``prev_data`` was sampled.
    """

    def __call__(
        self,
        prev_data: torch.Tensor,
        next_data: torch.Tensor,
        sample_at: torch.Tensor,
        step_data: float,
    ) -> torch.Tensor:
        r"""Callback protocol function."""
        ...


def interp_previous(
    prev_data: torch.Tensor,
    next_data: torch.Tensor,
    sample_at: torch.Tensor,
    step_time: float,
) -> torch.Tensor:
    r"""Interpolates by selecting the previous state.

    .. math::
        D_{t=s} = D_{t=0}

    Args:
        prev_data (torch.Tensor): most recent observation prior to sample time, :math:`D_{t=0}`.
        next_data (torch.Tensor): most recent observation subsequent to sample time, :math:`D_{t=T}`.
        sample_at (torch.Tensor): relative time at which to sample data, :math:`s`.
        step_data (float): length of time between the prior and subsequent observations, :math:`T`.

    Returns:
        torch.Tensor: interpolated data at sample time, :math:`D_{t=s}`.
    """
    return prev_data


def interp_nearest(
    prev_data: torch.Tensor,
    next_data: torch.Tensor,
    sample_at: torch.Tensor,
    step_time: float,
) -> torch.Tensor:
    r"""Interpolates by selecting the nearest state.

    .. math::
        D_{t=s} =
        \begin{cases}
            D_{t=T} &T - s > s\\
            D_{t=0} &\text{otherwise}
        \end{cases}

    Args:
        prev_data (torch.Tensor): most recent observation prior to sample time, :math:`D_{t=0}`.
        next_data (torch.Tensor): most recent observation subsequent to sample time, :math:`D_{t=T}`.
        sample_at (torch.Tensor): relative time at which to sample data, :math:`s`.
        step_data (float): length of time between the prior and subsequent observations, :math:`T`.

    Returns:
        torch.Tensor: interpolated data at sample time, :math:`D_{t=s}`.
    """
    return torch.where(sample_at / step_time > 0.5, next_data, prev_data)


def interp_linear(
    prev_data: torch.Tensor,
    next_data: torch.Tensor,
    sample_at: torch.Tensor,
    step_time: float,
) -> torch.Tensor:
    r"""Interpolates between previous and next states linearlly.

    .. math::
        D_{t=s} = \frac{D_{t=T} - D_{t=0}}{T} s + D_{t=0}

    Args:
        prev_data (torch.Tensor): most recent observation prior to sample time, :math:`D_{t=0}`.
        next_data (torch.Tensor): most recent observation subsequent to sample time, :math:`D_{t=T}`.
        sample_at (torch.Tensor): relative time at which to sample data, :math:`s`.
        step_data (float): length of time between the prior and subsequent observations, :math:`T`.

    Returns:
        torch.Tensor: interpolated data at sample time, :math:`D_{t=s}`.
    """
    slope = (next_data - prev_data) / step_time
    return prev_data + slope * sample_at


def interp_exp_decay(
    prev_data: torch.Tensor,
    next_data: torch.Tensor,
    sample_at: torch.Tensor,
    step_time: float,
    *,
    time_constant: float,
) -> torch.Tensor:
    r"""Interpolates by exponentially decaying value from previous state.

    .. math::
        D_{t=s} = D_{t=0} \exp\left(-\frac{s}{\tau}\right)

    Args:
        prev_data (torch.Tensor): most recent observation prior to sample time, :math:`D_{t=0}`.
        next_data (torch.Tensor): most recent observation subsequent to sample time, :math:`D_{t=T}`.
        sample_at (torch.Tensor): relative time at which to sample data, :math:`s`.
        step_data (float): length of time between the prior and subsequent observations, :math:`T`.
        time_constant (float): time constant of exponential decay, :math:`tau`.

    Returns:
        torch.Tensor: interpolated data at sample time, :math:`D_{t=s}`.
    """
    return prev_data * torch.exp(-sample_at / time_constant)
