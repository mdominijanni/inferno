import torch
from typing import Protocol


class HalfBounding(Protocol):
    r"""Callable used to apply bounding to the lower or upper limit of a parameter.

    Args:
        param (torch.Tensor): value of the parameter being bound.
        update (torch.Tensor): value of the partial update to scale.
        limit (float | torch.Tensor): maximum or minimum value of the updated parameter.

    Returns:
        torch.Tensor: bounded update.
    """

    def __call__(
        self,
        param: torch.Tensor,
        update: torch.Tensor,
        limit: float | torch.Tensor,
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
        max (float | torch.Tensor | None): maximum value of the updated parameter.
        min (float | torch.Tensor | None): minimum value of the updated parameter.

    Returns:
        torch.Tensor: bounded update.
    """

    def __call__(
        self,
        param: torch.Tensor,
        pos: torch.Tensor,
        neg: torch.Tensor,
        max: float | torch.Tensor | None,
        min: float | torch.Tensor | None,
        **kwargs,
    ) -> torch.Tensor:
        r"""Callback protocol function."""
        ...
