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
    r"""Callable used to interpolate in time between two tensors."""

    def __call__(
        self,
        prev_data: torch.Tensor,
        next_data: torch.Tensor,
        sample_at: torch.Tensor,
        step_data: float,
    ) -> torch.Tensor:
        r"""Callback protocol function.

        Args:
            prev_data (torch.Tensor): most recent observation prior to sample time.
            next_data (torch.Tensor): most recent observation subsequent to sample time.
            sample_at (torch.Tensor): relative time at which to sample data.
            step_data (float): length of time between the prior and subsequent observations.

        Returns:
            torch.Tensor: interpolated data at sample time.

        Note:
            `sample_at` is measured as the amount of time after the time at which
            `prev_data` was sampled.
        """
        ...


def interp_previous(
    prev_data: torch.Tensor,
    next_data: torch.Tensor,
    sample_at: torch.Tensor,
    step_time: float,
) -> torch.Tensor:
    return prev_data


def interp_nearest(
    prev_data: torch.Tensor,
    next_data: torch.Tensor,
    sample_at: torch.Tensor,
    step_time: float,
) -> torch.Tensor:
    return torch.where(sample_at / step_time > 0.5, next_data, prev_data)


def interp_linear(
    prev_data: torch.Tensor,
    next_data: torch.Tensor,
    sample_at: torch.Tensor,
    step_time: float,
) -> torch.Tensor:
    slope = (next_data - prev_data) / step_time
    return prev_data + slope * sample_at


def interp_exp_decay(
    prev_data: torch.Tensor,
    next_data: torch.Tensor,
    sample_at: torch.Tensor,
    step_time: float,
    *,
    time_constant: float | torch.Tensor,
) -> torch.Tensor:
    return prev_data * torch.exp(-sample_at / time_constant)


def gen_interp_exp_decay(
    time_constant: float | torch.Tensor,
) -> Interpolation:
    return lambda prev_data, next_data, sample_at, step_time: interp_exp_decay(
        prev_data, next_data, sample_at, step_time, time_constant=time_constant
    )
