import torch

from inferno._internal import create_tensor
from inferno.monitoring.reducers.abstract import AbstractReducer


class PassthroughReducer(AbstractReducer):
    """Directly stores all prior states of the monitored tensor attribute.
    """

    def __init__(self):
        # call superclass constructor
        AbstractReducer.__init__(self)

        # register data list
        self.data = []

    def clear(self) -> None:
        """Reinitializes reducer state.
        """
        self.data.clear()

    def peak(self, dim: int = -1) -> torch.Tensor | None:
        """Returns the concatenated tensor of observed states since the reducer was last cleared.

        Args:
            dim (int, optional): the dimension along which concatenation should occur between different time steps. Defaults to `-1`.

        Returns:
            torch.Tensor | None: cumulative output stored since the reducer was last cleared.
        """
        if self.data:
            return torch.stack(self.data, dim=dim)
        else:
            return None

    def pop(self, dim: int = -1) -> torch.Tensor | None:
        """Returns the concatenated tensor of observed states since the reducer was last cleared, then clears the reducer.

        Args:
            dim (int, optional): the dimension along which concatenation should occur between different time steps. Defaults to `-1`.

        Returns:
            torch.Tensor | None: cumulative output stored since the reducer was last cleared.
        """
        res = self.peak(dim)
        if res is not None:
            res = res.clone()
        self.clear()
        return res

    def forward(self, inputs: torch.Tensor, **kwargs) -> None:
        """Incorporates inputs into cumulative storage for output.

        Args:
            inputs (torch.Tensor): :py:class:`torch.Tensor` of the attribute being monitored.
        """
        self.data.append(create_tensor(inputs))


class SinglePassthroughReducer(AbstractReducer):
    """Stores the most recent state of the monitored tensor attribute.
    """

    def __init__(self):
        # call superclass constructor
        AbstractReducer.__init__(self)

        # register data buffer
        self.register_buffer('data', None)

    def clear(self) -> None:
        """Reinitializes reducer state.
        """
        del self.data
        self.register_buffer('data', None)

    def peak(self) -> torch.Tensor | None:
        """Returns the most recent output since the reducer was last cleared.

        Returns:
            torch.Tensor | None: most recent output stored since reducer was last cleared.
        """
        return self.data

    def pop(self) -> torch.Tensor | None:
        """Returns the most recent output since the reducer was last cleared, then clears the reducer.

        Returns:
            torch.Tensor | None: most recent output stored since the reducer was last cleared.
        """
        res = self.peak()
        self.clear()
        return res

    def forward(self, inputs: torch.Tensor, **kwargs) -> None:
        """Puts input into storage for output.

        Args:
            inputs (torch.Tensor): :py:class:`torch.Tensor` of the attribute being monitored.
        """
        self.clear()
        self.data = create_tensor(inputs)
