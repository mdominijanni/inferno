from __future__ import annotations
from .base import FoldReducer
from ...functional import interp_previous
from ...types import OneToOne
import torch


class EventReducer(FoldReducer):
    r"""Stores the length of time since an element of the input matched a criterion.

    Args:
        step_time (float): length of time between observation.
        criterion (OneToOne[torch.Tensor]): function to test if the input is considered
            matches for it to be considered an event.
        inclusive (bool): if the duration should be inclusive. Defaults to ``False``.
        duration (float, optional): length of time over which results should be
            stored, in the same units as ``step_time``. Defaults to ``0.0``.

    Important:
        The output of ``criterion`` must have a datatype (:py:class:`torch.dtype`) of
        :py:data:`torch.bool`. The datatype returned by :py:meth:`fold` will be the
        same as that of the reducer itself.
    """

    def __init__(
        self,
        step_time: float,
        criterion: OneToOne[torch.Tensor],
        duration: float = 0.0,
        inclusive: bool = False,
    ):
        # call superclass constructor
        FoldReducer.__init__(self, step_time, duration, inclusive, float("inf"))

        # set non-persistent function
        self.criterion = criterion

    def fold(self, obs: torch.Tensor, state: torch.Tensor | None) -> torch.Tensor:
        r"""Application of last prior event.

        Args:
            obs (torch.Tensor): observation to incorporate into state.
            state (torch.Tensor | None): state from the prior time step,
                ``None`` if no prior observations.

        Returns:
            torch.Tensor: state for the current time step.
        """
        if state is None:
            return torch.where(self.criterion(obs), 0, float("inf")).to(
                dtype=self.data.dtype
            )
        else:
            return torch.where(self.criterion(obs), 0, state + self.dt).to(
                dtype=self.data.dtype
            )

    def interpolate(
        self,
        prev_data: torch.Tensor,
        next_data: torch.Tensor,
        sample_at: torch.Tensor,
        step_time: float,
    ) -> torch.Tensor:
        r"""Exact value interpolation between observations.

        Args:
            prev_data (torch.Tensor): most recent observation prior to sample time.
            next_data (torch.Tensor): most recent observation subsequent to sample time.
            sample_at (torch.Tensor): relative time at which to sample data.
            step_time (float): length of time between the prior and
                subsequent observations.

        Returns:
            torch.Tensor: interpolated data at sample time.
        """
        return prev_data + sample_at


class PassthroughReducer(FoldReducer):
    r"""Directly stores prior observations.

    Args:
        step_time (float): length of time between observation.
        duration (float, optional): length of time over which results should be
            stored, in the same units as ``step_time``. Defaults to ``0.0``.
        inclusive (bool): if the duration should be inclusive. Defaults to ``False``.
    """

    def __init__(
        self,
        step_time: float,
        duration: float = 0.0,
        inclusive: bool = False,
    ):
        # call superclass constructor
        FoldReducer.__init__(self, step_time, duration, inclusive, 0)

    def fold(self, obs: torch.Tensor, state: torch.Tensor | None) -> torch.Tensor:
        r"""Application of passthrough.

        Args:
            obs (torch.Tensor): observation to incorporate into state.
            state (torch.Tensor | None): state from the prior time step,
                ``None`` if no prior observations.

        Returns:
            torch.Tensor: state for the current time step.
        """
        return obs

    def interpolate(
        self,
        prev_data: torch.Tensor,
        next_data: torch.Tensor,
        sample_at: torch.Tensor,
        step_time: float,
    ) -> torch.Tensor:
        r"""Previous value interpolation between observations.

        Args:
            prev_data (torch.Tensor): most recent observation prior to sample time.
            next_data (torch.Tensor): most recent observation subsequent to sample time.
            sample_at (torch.Tensor): relative time at which to sample data.
            step_time (float): length of time between the prior and
                subsequent observations.

        Returns:
            torch.Tensor: interpolated data at sample time.
        """
        return interp_previous(prev_data, next_data, sample_at, step_time)
