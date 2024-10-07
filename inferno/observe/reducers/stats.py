from .base import FoldReducer
from ... import exponential_smoothing
from ..._internal import argtest
from ...functional import interp_linear
import torch


class EMAReducer(FoldReducer):
    r"""Stores the exponential moving average.

    .. math::
        \begin{align*}
            s_0 &= x_0 \\
            s_{t + 1} &= \alpha x_{t + 1}  + (1 - \alpha) s_t
        \end{align*}

    For the smoothed data (state) :math:`s` and observation :math:`x` and
    where smoothing factor :math:`\alpha` is as follows.

    .. math::
        \alpha = 1 - \exp \left(\frac{-\Delta t}{\tau}\right)

    For some time constant :math:`\tau`.

    Args:
        step_time (float): length of time between observations, :math:`\Delta t`.
        alpha (float): exponential smoothing factor, :math:`\alpha`.
        duration (float, optional): length of time over which results should be
            stored, in the same units as :math:`\Delta t`. Defaults to ``0.0.``
        inclusive (bool, optional): if the duration should be inclusive. Defaults to ``False``.
        inplace (bool, optional): if write operations should be performed
            in-place. Defaults to ``False``.
    Note:
        ``alpha`` is decoupled from the step time, so if the step time changes, then the
            underlying time constant will change, ``alpha`` will remain the same.
    """

    def __init__(
        self,
        step_time: float,
        alpha: float,
        duration: float = 0.0,
        inclusive: bool = False,
        inplace: bool = False,
    ):
        # call superclass constructor
        FoldReducer.__init__(self, step_time, duration, inclusive, inplace, 0)

        # set state
        self.alpha = argtest.minmax_incl("alpha", alpha, 0, 1, float)

    def fold(self, obs: torch.Tensor, state: torch.Tensor | None) -> torch.Tensor:
        r"""Application of exponential smoothing.

        Args:
            obs (torch.Tensor): observation to incorporate into state.
            state (torch.Tensor | None): state from the prior time step,
                ``None`` if no prior observations.

        Returns:
            torch.Tensor: state for the current time step.
        """
        return exponential_smoothing(obs, state, alpha=self.alpha)

    def interpolate(
        self,
        prev_data: torch.Tensor,
        next_data: torch.Tensor,
        sample_at: torch.Tensor,
        step_time: float,
    ) -> torch.Tensor:
        r"""Linear interpolation between observations.

        Args:
            prev_data (torch.Tensor): most recent observation prior to sample time.
            next_data (torch.Tensor): most recent observation subsequent to sample time.
            sample_at (torch.Tensor): relative time at which to sample data.
            step_time (float): length of time between the prior and subsequent
                observations.

        Returns:
            torch.Tensor: interpolated data at sample time.
        """
        return interp_linear(prev_data, next_data, sample_at, step_time)


class CAReducer(FoldReducer):
    r"""Stores the cumulative average.

    .. math::
        \begin{align*}
            \mu(t) &= \frac{x(t) + n(t - \Delta t) \mu(t - \Delta t)}{n(t)} \\
            n(t) &= \frac{t}{\Delta t}
        \end{align*}

    Args:
        step_time (float): length of the discrete step time, :math:`\Delta t`.
        duration (float, optional): length of time over which results should be
            stored, in the same units as :math:`\Delta t`. Defaults to ``0.0``.
        inclusive (bool, optional): if the duration should be inclusive.
            Defaults to ``False``.
        inplace (bool, optional): if write operations should be performed
            in-place. Defaults to ``False``.
    """

    def __init__(
        self,
        step_time: float,
        duration: float = 0.0,
        inclusive: bool = False,
        inplace: bool = False,
    ):
        # call superclass constructor
        FoldReducer.__init__(self, step_time, duration, inclusive, inplace, 0)

        # set state
        self.register_extra("_count", 0)

    def clear(self, keepshape=False, **kwargs) -> None:
        r"""Reinitializes the reducer's state.

        Args:
            keepshape (bool, optional): if the underlying storage shape should be
                preserved. Defaults to ``False``.
        """
        self._count = 0
        FoldReducer.clear(self, keepshape=keepshape, **kwargs)

    def fold(self, obs: torch.Tensor, state: torch.Tensor | None) -> torch.Tensor:
        r"""Application of summation.

        Args:
            obs (torch.Tensor): observation to incorporate into state.
            state (torch.Tensor | None): state from the prior time step,
                ``None`` if no prior observations.

        Returns:
            torch.Tensor: state for the current time step.
        """
        self._count += 1

        if state is None:
            return obs.to(dtype=self.data.dtype)
        else:
            return state + (obs.to(dtype=state.dtype) - state) / self._count

    def interpolate(
        self,
        prev_data: torch.Tensor,
        next_data: torch.Tensor,
        sample_at: torch.Tensor,
        step_time: float,
    ) -> torch.Tensor:
        r"""Linear interpolation between observations.

        Args:
            prev_data (torch.Tensor): most recent observation prior to sample time.
            next_data (torch.Tensor): most recent observation subsequent to sample time.
            sample_at (torch.Tensor): relative time at which to sample data.
            step_time (float): length of time between the prior and subsequent
                observations.

        Returns:
            torch.Tensor: interpolated data at sample time.
        """
        return interp_linear(prev_data, next_data, sample_at, step_time)
