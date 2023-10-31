from __future__ import annotations
from .base import FoldingReducer, MappingReducer
import inferno
from inferno.typing import OneToOneMethod, ManyToOneMethod, OneToOne
import torch
from typing import Callable


class EventReducer(FoldingReducer):
    r"""Stores the number of calls since an element of the input matched a criterion.

    Args:
        criterion (OneToOne[torch.Tensor]): function to test if the input is considered matches for it to be considered an event.
        mapmeth (OneToOneMethod[EventReducer, torch.Tensor] | ManyToOneMethod[EventReducer, torch.Tensor] | None, optional): transformation
            to apply to inputs, where the first argument is the reducer itself, the first argument is returned if None. Defaults to None.
        filtermeth (Callable[[EventReducer, torch.Tensor], bool] | None, optional): conditional test if transformed
            input should be integrated, always will if None. Defaults to None.

    Note:
        The output of ``criterion`` must have a datatype (:py:class:`torch.dtype`) of :py:data:`torch.bool`.
    """

    def __init__(
        self,
        criterion: OneToOne[torch.Tensor],
        *,
        mapmeth: OneToOneMethod[EventReducer, torch.Tensor]
        | ManyToOneMethod[EventReducer, torch.Tensor]
        | None = None,
        filtermeth: Callable[[EventReducer, torch.Tensor], bool] | None = None,
    ):
        # call superclass constructor
        FoldingReducer.__init__(self, mapmeth=mapmeth, filtermeth=filtermeth)

        # set non-persistent function
        self._criterion = criterion

    def fold_fn(self, obs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return torch.where(self._criterion(obs), 0, state + 1)

    def map_fn(self, *inputs: torch.Tensor) -> torch.Tensor:
        return inputs[0]

    def filter_fn(self, inputs: torch.Tensor) -> bool:
        return True

    def init_fn(self, inputs: torch.Tensor) -> torch.Tensor:
        return inferno.zeros(self._data, shape=inputs.shape)


class HistoryReducer(MappingReducer):
    r"""Stores the previous values of inputs over a rolling window.

    Args:
        window (int): size of the window over which observations should be recorded.
        mapmeth (OneToOneMethod[HistoryReducer, torch.Tensor] | ManyToOneMethod[HistoryReducer, torch.Tensor] | None, optional): transformation
            to apply to inputs, where the first argument is the reducer itself, the first argument is returned if None. Defaults to None.
        filtermeth (Callable[[HistoryReducer, torch.Tensor], bool] | None, optional): conditional test if transformed
            input should be integrated, always will if None. Defaults to None.
    """

    def __init__(
        self,
        window: int,
        *,
        mapmeth: OneToOneMethod[HistoryReducer, torch.Tensor]
        | ManyToOneMethod[HistoryReducer, torch.Tensor]
        | None = None,
        filtermeth: Callable[[HistoryReducer, torch.Tensor], bool] | None = None,
    ):
        # call superclass constructor
        MappingReducer.__init__(self, window, mapmeth=mapmeth, filtermeth=filtermeth)

    def map_fn(self, *inputs: torch.Tensor) -> torch.Tensor:
        return inputs[0]

    def filter_fn(self, inputs: torch.Tensor) -> bool:
        return True
