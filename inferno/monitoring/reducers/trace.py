from typing import Any
import warnings

import torch

from inferno._internal import create_tensor
from inferno.monitoring.reducers.abstract import AbstractReducer


class TraceReducer(AbstractReducer):
    """Stores elementwise a trace of a tensor.

    Args:
        amplitude (float, optional): value to set the trace to when the target is matched. Defaults to `1.0`.
        target (Any, optional): value to match when checking for the most recent prior occurrence. Defaults to `1`.

    Kwargs:
        decay (float, optional): real value between zero and one which controls the rate at which the trace goes to zero.
        step_time (float, optional): length of the simulated step time, used for calculating delay, used if `delay` is unspecified.
        time_constant (float, optional): value used alongside `step_time` for calculating delay, used if `delay` is unspecified.

    Raises:
        KeyError: `decay` or both `step_time` and `time_constant` must be provided.

    Warn:
        RuntimeWarning: for trace decay to function correctly, 'decay' should be in the range (0,1).
    """
    def __init__(
        self,
        amplitude: float = 1.0,
        target: Any = 1,
        **kwargs
    ):
        # call superclass constructor
        AbstractReducer.__init__(self)

        # construct decay if step_time and time_constant are specified, otherwise use decay
        if 'decay' not in kwargs:
            if ('step_time' not in kwargs) or ('time_constant' not in kwargs):
                raise KeyError("either 'decay' or both 'step_time' and 'time_constant' must be provided")
            self.register_buffer('decay', torch.exp(create_tensor(-kwargs['step_time'] / kwargs['time_constant'])))
        else:
            self.register_buffer('decay', create_tensor(kwargs['decay']))

        # check if decay is in a valid range
        if (self.decay <= 0.0) or (self.decay >= 1.0):
            warnings.warn("for trace decay to function correctly, 'decay' should be in the range (0,1)", category=RuntimeWarning)

        # register buffers
        self.register_buffer('data', None)
        self.register_buffer('amplitude', create_tensor(amplitude))
        self.register_buffer('target', create_tensor(target))

    def clear(self) -> None:
        """Reinitializes reducer state.
        """
        del self.data
        self.register_buffer('data', None)

    def peak(self) -> torch.Tensor | None:
        """Returns the trace of the observered tensor since the reducer was last cleared.

        Returns:
            torch.Tensor | None: cumulative output stored since the reducer was last cleared.
        """
        return self.data

    def pop(self) -> torch.Tensor | None:
        """Returns the trace of the observered tensor since the reducer was last cleared, then clears the reducer.

        Returns:
            torch.Tensor | None: cumulative output stored since the reducer was last cleared.
        """
        res = self.peak()
        if res is not None:
            res = res.clone()
        self.clear()
        return res

    def forward(self, inputs: torch.Tensor) -> None:
        """Incorporates inputs into cumulative storage for output.

        Args:
            inputs (torch.Tensor): :py:class:`torch.Tensor` of the attribute being monitored.
        """
        if self.data is None:
            self.data = torch.zeros_like(inputs, dtype=self.decay.dtype, device=self.decay.device, requires_grad=False)
        self.data.mul_(self.decay)
        self.data.masked_fill_(inputs == self.target, self.amplitude)


class AdditiveTraceReducer(AbstractReducer):
    """Stores elementwise an additive trace of a tensor.

    Args:
        amplitude (float, optional): value to set the trace to when the target is matched. Defaults to `1.0`.
        target (Any, optional): value to match when checking for the most recent prior occurrence. Defaults to `1`.

    Kwargs:
        decay (float, optional): real value between zero and one which controls the rate at which the trace goes to zero.
        step_time (float, optional): length of the simulated step time, used for calculating delay, used if `delay` is unspecified.
        time_constant (float, optional): value used alongside `step_time` for calculating delay, used if `delay` is unspecified.

    Raises:
        KeyError: `decay` or both `step_time` and `time_constant` must be provided.

    Warn:
        RuntimeWarning: for trace decay to function correctly, 'decay' should be in the range (0,1).
    """
    def __init__(
        self,
        amplitude: float = 1.0,
        target: Any = 1,
        **kwargs
    ):
        # call superclass constructor
        AbstractReducer.__init__(self)

        # construct decay if step_time and time_constant are specified, otherwise use decay
        if 'decay' not in kwargs:
            if ('step_time' not in kwargs) or ('time_constant' not in kwargs):
                raise KeyError("either 'decay' or both 'step_time' and 'time_constant' must be provided")
            if not isinstance(kwargs['step_time'], torch.Tensor) and not isinstance(kwargs['time_constant'], torch.Tensor):
                self.register_buffer('decay', torch.exp(torch.tensor(-kwargs['step_time'] / kwargs['time_constant'])))
            else:
                self.register_buffer('decay', torch.exp(-kwargs['step_time'] / kwargs['time_constant']))
        else:
            self.register_buffer('decay', torch.tensor(kwargs['decay'], requires_grad=False))

        # check if decay is in a valid range
        if (self.decay <= 0.0) or (self.decay >= 1.0):
            warnings.warn("for trace decay to function correctly, 'decay' should be in the range (0,1)", category=RuntimeWarning)

        # register buffers
        self.register_buffer('data', None)
        self.register_buffer('amplitude', torch.tensor(amplitude, requires_grad=False))
        self.register_buffer('target', create_tensor(target))

    def clear(self) -> None:
        """Reinitializes reducer state.
        """
        del self.data
        self.register_buffer('data', None)

    def peak(self) -> torch.Tensor | None:
        """Returns the additive trace of the observered tensor since the reducer was last cleared.

        Returns:
            torch.Tensor | None: cumulative output stored since the reducer was last cleared.
        """
        return self.data

    def pop(self) -> torch.Tensor | None:
        """Returns the additive trace of the observered tensor since the reducer was last cleared, then clears the reducer.

        Returns:
            torch.Tensor | None: cumulative output stored since the reducer was last cleared.
        """
        res = self.peak()
        if res is not None:
            res = res.clone()
        self.clear()
        return res

    def forward(self, inputs: torch.Tensor) -> None:
        """Incorporates inputs into cumulative storage for output.

        Args:
            inputs (torch.Tensor): :py:class:`torch.Tensor` of the attribute being monitored.
        """
        if self.data is None:
            self.data = torch.zeros_like(inputs, dtype=self.decay.dtype, device=self.decay.device, requires_grad=False)
        self.data.mul_(self.decay)
        self.data.add_((inputs == self.target) * self.amplitude)


class ScaledAdditiveTraceReducer(AbstractReducer):
    """Stores elementwise an additive trace of a tensor, where the amplitude is scaled by the inputs.

    Args:
        amplitude (float, optional): value to set the trace to when the target is matched. Defaults to `1.0`.

    Kwargs:
        decay (float, optional): real value between zero and one which controls the rate at which the trace goes to zero.
        step_time (float, optional): length of the simulated step time, used for calculating delay, used if `delay` is unspecified.
        time_constant (float, optional): value used alongside `step_time` for calculating delay, used if `delay` is unspecified.

    Raises:
        KeyError: `decay` or both `step_time` and `time_constant` must be provided.

    Warn:
        RuntimeWarning: for trace decay to function correctly, 'decay' should be in the range (0,1).
    """
    def __init__(
        self,
        amplitude: float = 1.0,
        **kwargs
    ):
        # call superclass constructor
        AbstractReducer.__init__(self)

        # construct decay if step_time and time_constant are specified, otherwise use decay
        if 'decay' not in kwargs:
            if ('step_time' not in kwargs) or ('time_constant' not in kwargs):
                raise KeyError("either 'decay' or both 'step_time' and 'time_constant' must be provided")
            if not isinstance(kwargs['step_time'], torch.Tensor) and not isinstance(kwargs['time_constant'], torch.Tensor):
                self.register_buffer('decay', torch.exp(torch.tensor(-kwargs['step_time'] / kwargs['time_constant'])))
            else:
                self.register_buffer('decay', torch.exp(-kwargs['step_time'] / kwargs['time_constant']))
        else:
            self.register_buffer('decay', torch.tensor(kwargs['decay'], requires_grad=False))

        # check if decay is in a valid range
        if (self.decay <= 0.0) or (self.decay >= 1.0):
            warnings.warn("for trace decay to function correctly, 'decay' should be in the range (0,1)", category=RuntimeWarning)

        # register buffers
        self.register_buffer('data', None)
        self.register_buffer('amplitude', torch.tensor(amplitude, requires_grad=False))

    def clear(self) -> None:
        """Reinitializes reducer state.
        """
        del self.data
        self.register_buffer('data', None)

    def peak(self) -> torch.Tensor | None:
        """Returns the additive trace of the observered tensor since the reducer was last cleared.

        Returns:
            torch.Tensor | None: cumulative output stored since the reducer was last cleared.
        """
        return self.data

    def pop(self) -> torch.Tensor | None:
        """Returns the additive trace of the observered tensor since the reducer was last cleared, then clears the reducer.

        Returns:
            torch.Tensor | None: cumulative output stored since the reducer was last cleared.
        """
        res = self.peak()
        if res is not None:
            res = res.clone()
        self.clear()
        return res

    def forward(self, inputs: torch.Tensor) -> None:
        """Incorporates inputs into cumulative storage for output.

        Args:
            inputs (torch.Tensor): :py:class:`torch.Tensor` of the attribute being monitored.
        """
        if self.data is None:
            self.data = torch.zeros_like(inputs, dtype=self.decay.dtype, device=self.decay.device, requires_grad=False)
        self.data.mul_(self.decay)
        self.data.add_(inputs * self.amplitude)
