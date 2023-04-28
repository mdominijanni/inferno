from typing import Any

import torch

from inferno._internal import create_tensor
from inferno.monitoring.reducers.abstract import AbstractReducer


class LastEventReducer(AbstractReducer):
    """Stores elementwise the number of observations since the observation matched a target value.

    Args:
        target (Any, optional): value to match when checking for the most recent prior occurrence. Defaults to `1`.
    """
    def __init__(
        self,
        target: Any = 1
    ):
        # call superclass constructor
        AbstractReducer.__init__(self)

        # register buffers
        self.register_buffer('data', None)
        self.register_buffer('target', create_tensor(target))

    def clear(self) -> None:
        """Reinitializes reducer state.
        """
        del self.data
        self.register_buffer('data', None)

    def peak(self) -> torch.Tensor | None:
        """Returns the number of observations since the observered tensor's values last matched a target since the reducer was last cleared.

        Returns:
            torch.Tensor | None: cumulative output stored since the reducer was last cleared.
        """
        return self.data

    def pop(self) -> torch.Tensor | None:
        """Returns the number of observations since the observered tensor's values last matched a target since the reducer was last cleared, then clears the reducer.

        Returns:
            torch.Tensor | None: cumulative output stored since the reducer was last cleared.
        """
        res = self.peak()
        if res is not None:
            res = res.clone()
        self.clear()
        return res

    def forward(self, inputs: torch.Tensor) -> None:
        """Check if input matches target value and stores the number of observations since.

        Args:
            inputs (torch.Tensor): :py:class:`torch.Tensor` of the attribute being monitored.
        """
        if self.data is None:
            self.data = torch.zeros_like(inputs, requires_grad=False).float()
            self.data.add_(float('inf'))
        self.data.add_(1)
        self.data.masked_fill_(inputs == self.value, 0)


class FuzzyLastEventReducer(AbstractReducer):
    """Stores elementwise the number of observations since the observation matched a target value within an acceptable error.

    Args:
        target (Any, optional): value to match when checking for the most recent prior occurrence. Defaults to `1`.
        epsilon (Any, optional): error to permit when checking if an input matches the target, must be non-negative. Defaults to `0`.

    Raises:
        ValueError: `epsilon` must be non-negative.
    """
    def __init__(
        self,
        target: Any = 1,
        epsilon: Any = 0
    ):
        # call superclass constructor
        AbstractReducer.__init__(self)

        # check if epsilon is valid
        if epsilon < 0:
            raise ValueError(f"'epsilon' must be non-negative, received {epsilon}")

        # register buffers
        self.register_buffer('data', None)
        self.register_buffer('target', create_tensor(target))
        self.register_buffer('epsilon', create_tensor(epsilon))

    def clear(self) -> None:
        """Reinitializes reducer state.
        """
        del self.data
        self.register_buffer('data', None)

    def peak(self) -> torch.Tensor | None:
        """Returns the number of observations since the observered tensor's values last matched a target since the reducer was last cleared.

        Returns:
            torch.Tensor | None: cumulative output stored since the reducer was last cleared.
        """
        return self.data

    def pop(self) -> torch.Tensor | None:
        """Returns the number of observations since the observered tensor's values last matched a target since the reducer was last cleared, then clears the reducer.

        Returns:
            torch.Tensor | None: cumulative output stored since the reducer was last cleared.
        """
        res = self.peak()
        if res is not None:
            res = res.clone()
        self.clear()
        return res

    def forward(self, inputs: torch.Tensor) -> None:
        """Check if input matches target value and stores the number of observations since.

        Args:
            inputs (torch.Tensor): :py:class:`torch.Tensor` of the attribute being monitored.
        """
        if self.data is None:
            self.data = torch.zeros_like(inputs, requires_grad=False).float()
            self.data.add_(float('inf'))
        self.data.add_(1)
        self.data.masked_fill_(torch.abs(inputs - self.target) <= self.epsilon, 0)
