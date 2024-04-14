import torch
from typing import Protocol


class HalfBounding(Protocol):
    r"""Callable used to apply bounding to the lower or upper limit of a parameter.

    Args:
        param (torch.Tensor): value of the parameter being bound.
        update (torch.Tensor): value of the partial update to scale.
        limit (float): maximum or minimum value of the updated parameter.

    Returns:
        torch.Tensor: bounded update.
    """

    def __call__(
        self,
        param: torch.Tensor,
        update: torch.Tensor,
        limit: float,
        **kwargs,
    ) -> torch.Tensor:
        r"""Callback protocol function."""
        ...


class FullBounding(Protocol):
    r"""Callable used to apply bounding to the lower and upper limit of a parameter.

    Args:
        param (torch.Tensor): value of the parameter being bound.
        pos (torch.Tensor): value of the positive part of the update to scale.
        neg (torch.Tensor): value of the negative part of the update to scale.
        max (float | None): maximum value of the updated parameter.
        min (float | None): minimum value of the updated parameter.

    Returns:
        torch.Tensor: bounded update.
    """

    def __call__(
        self,
        param: torch.Tensor,
        pos: torch.Tensor,
        neg: torch.Tensor,
        max: float | None,
        min: float | None,
        **kwargs,
    ) -> torch.Tensor:
        r"""Callback protocol function."""
        ...


class Interpolation(Protocol):
    r"""Callable used to interpolate in time between two tensors.

    Here, ``prev_data`` and ``next_data`` should be the nearest two observations,
    and ``sample_at`` should be the length of time, since ``prev_data`` was observered,
    from which to sample. ``step_time`` is the total length of time between the
    observations ``prev_data`` and ``next_data``.

    The result is a single tensor, shaped like ``prev_data`` and ``next_data``,
    containing the interpolated data.

    Args:
        prev_data (torch.Tensor): most recent observation prior to sample time.
        next_data (torch.Tensor): most recent observation subsequent to sample time.
        sample_at (torch.Tensor): relative time at which to sample data.
        step_time (float): length of time between the prior and subsequent observations.

    Returns:
        torch.Tensor: interpolated data at sample time.
    """

    def __call__(
        self,
        prev_data: torch.Tensor,
        next_data: torch.Tensor,
        sample_at: torch.Tensor,
        step_time: float,
        **kwargs,
    ) -> torch.Tensor:
        r"""Callback protocol function."""
        ...


class Extrapolation(Protocol):
    r"""Callable used to extrapolate in time to two tensors.

    Here, ``sample`` should be the tensor which was "sampled" at a time between two
    discrete observations, ``prev_data`` and ``next_data``. ``sample_at`` should be
    the length of time between the nearest discrete observation and this sample.
    ``step_time`` is the total length of time between the nearest two observations.

    The result is a 2-tuple of tensors, both shaped like ``data``, the first being the
    the extrapolated data at the time of the nearest prior discrete observation and the
    second being the extrapolated data at the time of the nearest subsequent discrete
    observations.

    Args:
        sample (torch.Tensor): sample from which to extrapolate.
        sample_at (torch.Tensor): relative time of the data from which to extrapolate.
        prev_data (torch.Tensor): most recent observation prior to sample time.
        next_data (torch.Tensor): most recent observation subsequent to sample time.
        step_time (float): length of time between the prior and subsequent observations.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: extrapolated data at the start and end
        of the step.
    """

    def __call__(
        self,
        sample: torch.Tensor,
        sample_at: torch.Tensor,
        prev_data: torch.Tensor,
        next_data: torch.Tensor,
        step_time: float,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Callback protocol function."""
        ...


class DimensionReduction(Protocol):
    r"""Callable used to reduce the dimensions of a tensor.

    For simpler cases, these will wrap PyTorch methods such as :py:func:`torch.mean` for
    convenience. When the ``kwargs`` are defined with a partial function, these should
    be compatible with parameters in Inferno such as ``batch_reduction`` and should be
    compatible with ``einops.reduce``. To this end, any implementation should maintain
    the default behavior for ``keepdim``.

    Args:
        data (torch.Tensor): tensor to which operations should be applied.
        dim (tuple[int, ...] | int | None, optional): dimension(s) along which the
            reduction should be applied, all dimensions when ``None``.
            Defaults to ``None``.
        keepdim (bool, optional): if the dimensions should be retained in the output.
            Defaults to ``False``.

    Returns:
        torch.Tensor: dimensionally reduced tensor.
    """

    def __call__(
        self,
        data: torch.Tensor,
        dim: tuple[int, ...] | int | None = None,
        keepdim: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        r"""Callback protocol function."""
        ...
