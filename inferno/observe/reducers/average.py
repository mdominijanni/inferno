from __future__ import annotations
import torch
from .base import FoldingReducer
import inferno


class EMAReducer(FoldingReducer):
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
        alpha (float): exponential smoothing factor, :math:`\alpha`.
        step_time (float): length of time between observations, :math:`\Delta t`.
        history_len (float, optional): length of time over which results should be stored,
            in the same units as :math:`\Delta t`. Defaults to 0.0.

    Note:
        ``alpha`` is decoupled from the step time, so if the step time changes, then the underlying time
        constant will change, ``alpha`` will remain the same.
    """

    def __init__(
        self,
        alpha: float,
        step_time: float,
        *,
        history_len: float = 0.0,
    ):
        # call superclass constructor
        FoldingReducer.__init__(self, step_time, history_len)

        # register state
        self.register_buffer("alpha", torch.tensor(float(alpha)))

    def fold(self, obs: torch.Tensor, state: torch.Tensor | None) -> torch.Tensor:
        r"""Application of exponential smoothing.

        Args:
            obs (torch.Tensor): observation to incorporate into state.
            state (torch.Tensor | None): state from the prior time step,
                None if no prior observations.

        Returns:
            torch.Tensor: state for the current time step.
        """
        if state is None:
            return obs
        else:
            return self.alpha * obs + (1 - self.alpha) * state

    def map(self, *inputs: torch.Tensor) -> torch.Tensor:
        r"""Selection of first input tensor.

        Args:
            *inputs (torch.Tensor): tensor inputs to combine.

        Returns:
            torch.Tensor: resultant observation.
        """
        return inputs[0]

    def initialize(self, inputs: torch.Tensor) -> torch.Tensor:
        r"""Setting of entire state history to zero.

        Args:
            inputs (torch.Tensor): empty tensor of state.

        Raises:
            NotImplementedError: abstract methods must be implemented by subclass.

        Returns:
            torch.Tensor: filled state tensor.
        """
        return inputs.fill_(0)

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
            step_data (float): length of time between the prior and subsequent observations.

        Returns:
            torch.Tensor: interpolated data at sample time.
        """
        return inferno.interp_linear(prev_data, next_data, sample_at, step_time)
