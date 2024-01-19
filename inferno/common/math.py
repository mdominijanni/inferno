import cmath
import functools
import math
import numpy as np
import torch
from typing import Protocol


@functools.singledispatch
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


def normalize(
    data: torch.Tensor,
    order: int | float,
    scale: float | complex = 1.0,
    dims: int | tuple[int] | None = None,
) -> torch.Tensor:
    r"""Normalizes a tensor.

    Args:
        data (torch.Tensor): data to normalize.
        order (int | float): order of :math:`p`-norm by which to normalize.
        scale (float | complex, optional): desired :math:`p`-norm of elements along
            specified dimensions. Defaults to 1.0.
        dims (int | tuple[int] | None, optional): dimensions along which to normalize,
            all dimensions if None. Defaults to None.

    Returns:
        torch.Tensor: normalized tensor.
    """
    return scale * (
        data / torch.linalg.vector_norm(data, ord=order, dim=dims, keepdim=True)
    )


def simple_exponential_smoothing(
    obs: torch.Tensor,
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
        obs (torch.Tensor): latest state to consider for exponential smoothing,
            :math:`x`.
        level (torch.Tensor | None): current value of the smoothed level,
            :math:`s`.
        alpha (float | int | complex | torch.Tensor): level smoothing factor,
            :math:`\alpha`.

    Returns:
        torch.Tensor: revised exponentially smoothed value.
    """
    # initial condition
    if level is None:
        return obs

    # standard condition
    else:
        return alpha * obs + (1 - alpha) * level


def holt_linear_smoothing(
    obs: torch.Tensor,
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
        obs (torch.Tensor): latest state to consider for exponential smoothing,
            :math:`x_{t + 1}`.
        level (torch.Tensor | None): current value of the smoothed level,
            :math:`s`.
        trend (torch.Tensor | None): current value of the smoothed trend,
            :math:`b`.
        alpha (float | int | complex | torch.Tensor): level smoothing factor,
            :math:`\alpha`.
        beta (float | int | complex | torch.Tensor): trend smoothing factor,
            :math:`\beta`.

    Returns:
        tuple[torch.Tensor, int | torch.Tensor]: tuple containing output/updated state:

            level: revised exponentially smoothed level.

            trend: revised exponentially smoothed trend.
    """
    # t=0 condition
    if level is None:
        return obs, None

    # t=1 condition (initialize trend as x1-x0)
    if trend is None:
        trend = obs - level

    # t>0 condition
    s = simple_exponential_smoothing(obs, level + trend, alpha)
    b = simple_exponential_smoothing(s - level, trend, beta)

    return s, b


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

    See Also:
        Provided interpolation functions include :py:func:`interp_previous`,
        :py:func:`interp_nearest`, :py:func:`interp_linear`, and
        :py:func:`interp_exp_decay`. Some provided functions may require keyword
        arguments. For arguments which require an object of type ``Interpolation``,
        use a :py:func:`~functools.partial` function to fill in keyword arguments.
    """

    def __call__(
        self,
        prev_data: torch.Tensor,
        next_data: torch.Tensor,
        sample_at: torch.Tensor,
        step_data: float,
        /,
    ) -> torch.Tensor:
        r"""Callback protocol function."""
        ...


def interp_previous(
    prev_data: torch.Tensor,
    next_data: torch.Tensor,
    sample_at: torch.Tensor,
    step_time: float,
    /,
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


def interp_nearest(
    prev_data: torch.Tensor,
    next_data: torch.Tensor,
    sample_at: torch.Tensor,
    step_time: float,
    /,
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


def interp_linear(
    prev_data: torch.Tensor,
    next_data: torch.Tensor,
    sample_at: torch.Tensor,
    step_time: float,
    /,
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


def interp_exp_decay(
    prev_data: torch.Tensor,
    next_data: torch.Tensor,
    sample_at: torch.Tensor,
    step_time: float,
    /,
    *,
    time_constant: float,
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
        time_constant (float): time constant of exponential decay, :math:`tau`.

    Returns:
        torch.Tensor: interpolated data at sample time, :math:`D_{t=s}`.
    """
    return prev_data * torch.exp(-sample_at / time_constant)
