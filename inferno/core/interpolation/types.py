import torch
from typing import Protocol


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
        **kwargs,
    ) -> torch.Tensor:
        r"""Callback protocol function."""
        ...
