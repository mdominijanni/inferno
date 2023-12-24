from __future__ import annotations
from .base import FoldingReducer
import inferno
from inferno.typing import OneToOne
import torch


class EventReducer(FoldingReducer):
    r"""Stores the length of time since an element of the input matched a criterion.

    Args:
        step_time (float): length of time between observations, :math:`\Delta t`.
        criterion (OneToOne[torch.Tensor]): function to test if the input is considered
            matches for it to be considered an event.
        history_len (float): length of time over which results should be stored, in the same units as :math:`\Delta t`.

    Note:
        The output of ``criterion`` must have a datatype (:py:class:`torch.dtype`) of
        :py:data:`torch.bool`.
    """

    def __init__(
        self,
        step_time: float,
        criterion: OneToOne[torch.Tensor],
        *,
        history_len: float = 0.0,
    ):
        # call superclass constructor
        FoldingReducer.__init__(self, step_time, history_len)

        # set non-persistent function
        self.criterion = criterion

    def fold(self, obs: torch.Tensor, state: torch.Tensor | None) -> torch.Tensor:
        if state is None:
            return torch.where(self.criterion(obs), 0, float("inf"))
        else:
            return torch.where(self.criterion(obs), 0, state + self.dt)

    def map(self, *inputs: torch.Tensor) -> torch.Tensor:
        return inputs[0]

    def initialize(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.fill_(float("inf"))

    def interpolate(
        self,
        prev_data: torch.Tensor,
        next_data: torch.Tensor,
        sample_at: torch.Tensor,
        step_time: float,
    ) -> torch.Tensor:
        return prev_data + sample_at


class PassthroughReducer(FoldingReducer):
    def __init__(
        self,
        step_time: float,
        *,
        history_len: float = 0.0,
    ):
        # call superclass constructor
        FoldingReducer.__init__(self, step_time, history_len)

    def fold(self, obs: torch.Tensor, state: torch.Tensor | None) -> torch.Tensor:
        return obs

    def map(self, *inputs: torch.Tensor) -> torch.Tensor:
        return inputs[0]

    def initialize(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.fill_(0)

    def interpolate(
        self,
        prev_data: torch.Tensor,
        next_data: torch.Tensor,
        sample_at: torch.Tensor,
        step_time: float,
    ) -> torch.Tensor:
        return inferno.interp_previous(prev_data, next_data, sample_at, step_time)
