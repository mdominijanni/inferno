from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class AbstractReducer(nn.Module, ABC):
    """Abstract class for manipulating the stored state of monitors.
    """
    def __init__(self):
        # call superclass constructor
        nn.Module.__init__(self)

    @abstractmethod
    def clear(self, **kwargs) -> None:
        """Reinitializes reducer state.

        Raises:
            NotImplementedError: py:meth:`clear` is abstract and must be implemented by the subclass.
        """
        raise NotImplementedError(f"'AbstractReducer.clear()' is abstract, {type(self).__name__} must implement the 'clear' method")

    @abstractmethod
    def peak(self, **kwargs) -> torch.Tensor | None:
        """Returns the cumulative output since the reducer was last cleared.

        Raises:
            NotImplementedError: :py:meth:`peak` is abstract and must be implemented by the subclass.

        Returns:
            torch.Tensor | None: cumulative output stored since the decoder was last cleared.
        """
        raise NotImplementedError(f"'AbstractReducer.peak()' is abstract, {type(self).__name__} must implement the 'peak' method")

    @abstractmethod
    def pop(self, **kwargs) -> torch.Tensor | None:
        """Returns the cumulative output since the reducer was last cleared, then clears the reducer.

        Raises:
            NotImplementedError: :py:meth:`pop` is abstract and must be implemented by the subclass.

        Returns:
            torch.Tensor | None: cumulative output stored since the reducer was last cleared.
        """
        raise NotImplementedError(f"'AbstractReducer.pop()' is abstract, {type(self).__name__} must implement the 'pop' method")

    @abstractmethod
    def forward(self, inputs: torch.Tensor, **kwargs) -> None:
        """Incorporates inputs into cumulative storage for output.

        Args:
            inputs (torch.Tensor): :py:class:`torch.Tensor` of the attribute being monitored.

        Raises:
            NotImplementedError: :py:meth:`forward` is abstract and must be implemented by the subclass.
        """
        raise NotImplementedError(f"'AbstractReducer.forward()' is abstract, {type(self).__name__} must implement the 'forward' method")
