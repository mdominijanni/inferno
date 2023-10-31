from __future__ import annotations
from typing import Callable
import torch
from .base import FoldingReducer
import inferno
from inferno._internal import newtensor
from inferno.typing import OneToOneMethod, ManyToOneMethod


class EMAReducer(FoldingReducer):
    r"""Stores the exponential moving average.

    .. math::
        \begin{align*}
            s_0 &= x_0 \\
            s_{t + 1} &= \alpha x_{t + 1}  + (1 - \alpha) s_t
        \end{align*}

    For the smoothed data (state) :math:`s` and observation :math:`x`.

    Args:
        alpha (float | int | torch.Tensor): data smoothing factor, :math:`\alpha`.
        mapmeth (OneToOneMethod[EMAReducer, torch.Tensor] | ManyToOneMethod[EMAReducer, torch.Tensor] | None, optional): transformation
            to apply to inputs, no transformation if None. Defaults to None.
        filtermeth (Callable[[EMAReducer, torch.Tensor], bool] | None, optional): conditional test if transformed
            input should be integrated, always will if None. Defaults to None.
    """

    def __init__(
        self,
        alpha: float | int | torch.Tensor,
        *,
        mapmeth: OneToOneMethod[EMAReducer, torch.Tensor]
        | ManyToOneMethod[EMAReducer, torch.Tensor]
        | None = None,
        filtermeth: Callable[[EMAReducer, torch.Tensor], bool] | None = None,
    ):
        # call superclass constructor, overriding instance methods if specified
        FoldingReducer.__init__(self, mapmeth=mapmeth, filtermeth=filtermeth)

        # register state
        self.register_buffer("_alpha", newtensor(alpha))

    def fold_fn(self, obs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self._alpha * obs + (1 - self._alpha) * state

    def map_fn(self, *inputs: torch.Tensor) -> torch.Tensor:
        return inputs[0]

    def filter_fn(self, inputs: torch.Tensor) -> bool:
        return True

    def init_fn(self, inputs: torch.Tensor) -> torch.Tensor:
        newdata = inferno.zeros(self._data, shape=inputs.shape)
        newdata[:] = inputs
        return (
            newdata
        )
