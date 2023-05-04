import torch

from inferno._internal import create_tensor
from inferno.monitoring.reducers.abstract import AbstractReducer


class PassthroughReducer(AbstractReducer):
    """Directly stores prior states in a window of the monitored tensor.

    Args:
        window (int): size of the window over which to store values.

    Raises:
        ValueError: `window` must be a positive integer.
    """

    def __init__(self, window: int):
        # call superclass constructor
        AbstractReducer.__init__(self)

        # test if window size is valid
        if int(window) < 1:
            raise ValueError(f"'window' must be a positive integer, received {window}")

        # register variables
        self.register_buffer('window', create_tensor(int(window)).to(dtype=torch.int64))
        self.register_buffer('idx', create_tensor(-1).to(dtype=torch.int64))
        self.register_buffer('full', create_tensor(False).to(dtype=torch.bool))
        self.register_buffer('data', torch.empty(0))

    def clear(self) -> None:
        """Reinitializes reducer state.
        """
        dtype = self.data.dtype
        device = self.data.device
        del self.data
        self.register_buffer('data', torch.empty(0, dtype=dtype, device=device))
        self.idx.fill_(-1)

    def peak(self, dim: int = -1) -> torch.Tensor | None:
        """Returns the concatenated tensor of observed states since the reducer was last cleared.

        Args:
            dim (int, optional): the dimension along which concatenation should occur between different time steps. Defaults to `-1`.

        Returns:
            torch.Tensor | None: cumulative output stored since the reducer was last cleared.
        """
        if self.idx != -1:
            pdims = list(range(1, self.data.ndim))
            pdims.insert(dim % self.data.ndim, 0)
            if not self.full:
                imdata = self.data[:(self.idx + 1)]
            else:
                imdata = torch.roll(self.data, -int(self.idx + 1), 0)
            return torch.permute(imdata, pdims)

    def pop(self, dim: int = -1) -> torch.Tensor | None:
        """Returns the concatenated tensor of observed states since the reducer was last cleared, then clears the reducer.

        Args:
            dim (int, optional): the dimension along which concatenation should occur between different time steps. Defaults to `-1`.

        Returns:
            torch.Tensor | None: cumulative output stored since the reducer was last cleared.
        """
        res = self.peak(dim)
        self.clear()
        return res

    def forward(self, inputs: torch.Tensor, **kwargs) -> None:
        """Incorporates inputs into cumulative storage for output.

        Args:
            inputs (torch.Tensor): :py:class:`torch.Tensor` of the attribute being monitored.
        """
        if self.data.numel() == 0:
            self.data = torch.full([int(self.window)] + list(inputs.shape), float('nan'), dtype=self.data.dtype, device=self.data.device)
        self.idx = (self.idx + 1) % self.window
        if (self.idx == self.window - 1) and not self.full:
            self.full.fill_(True)
        self.data[self.idx] = create_tensor(inputs).to(dtype=self.data.dtype, device=self.data.device)


class SinglePassthroughReducer(AbstractReducer):
    """Stores the most recent state of the monitored tensor.

    Note:
        Tensor device and datatype never persist, the device and datatype match that of the latest input.
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
        if self.data is not None:
            return self.data

    def pop(self) -> torch.Tensor | None:
        """Returns the most recent output since the reducer was last cleared, then clears the reducer.

        Returns:
            torch.Tensor | None: most recent output stored since the reducer was last cleared.
        """
        res = self.peak()
        if res is not None:
            res = res.clone()
        self.clear()
        return res

    def forward(self, inputs: torch.Tensor, **kwargs) -> None:
        """Puts input into storage for output.

        Args:
            inputs (torch.Tensor): :py:class:`torch.Tensor` of the attribute being monitored.
        """
        self.clear()
        self.data = create_tensor(inputs)
