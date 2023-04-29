import torch

from inferno._internal import create_tensor
from inferno.monitoring.reducers.abstract import AbstractReducer


class SMAReducer(AbstractReducer):
    """Stores elementwise the simple moving average of a tensor.

    .. math::
        SMA_{w, n} = \\frac{x_{n} + x_{n-1} + \\cdots x_{n-w+1}}{w}

    Where :math:`n` is the current step, :math:`w` is the window size, and :math:`x` is the changing value being averaged.

    Args:
        window (int): size of the window over which to average values.

    Raises:
        ValueError: `window` must be a positive integer.
    """

    def __init__(self, window: int):
        # call superclass constructor
        AbstractReducer.__init__(self)

        # test if window size is valid
        if int(window) < 1:
            raise ValueError(f"'window' must be a positive integer, received {window}")

        # register buffers
        self.register_buffer('window', create_tensor(int(window)).to(dtype=torch.int64))
        self.register_buffer('idx', create_tensor(-1).to(dtype=torch.int64))
        self.register_buffer('data', torch.empty(0))

    def clear(self) -> None:
        """Reinitializes reducer state.
        """
        dtype = self.data.dtype
        device = self.data.device
        del self.data
        self.register_buffer('data', torch.empty(0, dtype=dtype, device=device))
        self.idx.fill_(-1)

    def peak(self) -> torch.Tensor | None:
        """Returns the simple moving average of the observered tensor since the reducer was last cleared.

        Returns:
            torch.Tensor | None: cumulative output stored since the reducer was last cleared.
        """
        if self.idx != -1:
            return torch.nanmean(self.data, dim=0, keepdim=False)

    def pop(self) -> torch.Tensor | None:
        """Returns the cumulative moving average of the observered tensor since the reducer was last cleared, then clears the reducer.

        Returns:
            torch.Tensor | None: cumulative output stored since the reducer was last cleared.
        """
        res = self.peak()
        self.clear()
        return res

    def forward(self, inputs: torch.Tensor) -> None:
        """Incorporates inputs into cumulative storage for output.

        Args:
            inputs (torch.Tensor): :py:class:`torch.Tensor` of the attribute being monitored.
        """
        if self.data.numel() == 0:
            self.data = torch.full([int(self.window)] + list(inputs.shape), float('nan'), dtype=self.data.dtype, device=self.data.device)
        self.idx = (self.idx + 1) % self.window
        self.data[self.idx] = create_tensor(inputs).to(dtype=self.data.dtype, device=self.data.device)


class CMAReducer(AbstractReducer):
    """Stores elementwise the cumulative moving average of a tensor.

    .. math::
        CMA_{n} = \\frac{x_{n} - CMA_{n - 1}}{n} CMA_{n - 1}

    Where :math:`n` is the current step and :math:`x` is the changing value being averaged.
    """

    def __init__(self):
        # call superclass constructor
        AbstractReducer.__init__(self)

        # register buffers
        self.register_buffer('count', create_tensor(0).to(dtype=torch.int64))
        self.register_buffer('data', torch.empty(0))

    def clear(self):
        """Reinitializes reducer state.
        """
        dtype = self.data.dtype
        device = self.data.device
        del self.data
        self.register_buffer('data', torch.empty(0, dtype=dtype, device=device))
        self.count.fill_(0)

    def peak(self) -> torch.Tensor | None:
        """Returns the cumulative moving average of the observered tensor since the reducer was last cleared.

        Returns:
            torch.Tensor | None: cumulative output stored since the reducer was last cleared.
        """
        if self.count != 0:
            return self.data

    def pop(self) -> torch.Tensor | None:
        """Returns the cumulative moving average of the observered tensor since the reducer was last cleared, then clears the reducer.

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
        if self.data.numel() == 0:
            self.data = torch.zeros_like(inputs, dtype=self.count.dtype, device=self.count.device)
        self.data.add_((inputs - self.data) / (self.count + 1))
        self.count.add_(1)


class EMAReducer(AbstractReducer):
    """Stores elementwise the exponential moving average (i.e. exponentially weighted moving average) of a tensor.

    .. math::
        EMA_{n} = \\alpha x_n + (1 - \\alpha) EMA_{n - 1}

    Where :math:`n` is the current step, :math:`\\alpha` is the smoothing factor, and :math:`x` is the changing value being averaged.

    Args:
        alpha (float): the smoothing factor to use for computing the average, must be in the range of 0 to 1.
    """

    def __init__(self, alpha: float):
        # call superclass constructor
        AbstractReducer.__init__(self)

        # test if smoothing factor is valid
        if not 0 <= alpha <= 1:
            raise ValueError(f"'alpha' must be in the range (0, 1), received {alpha}")

        # register buffers
        self.register_buffer('alpha', create_tensor(int(alpha)).to(dtype=torch.int64))
        self.register_buffer('data', torch.empty(0))

    def clear(self):
        """Reinitializes reducer state.
        """
        dtype = self.data.dtype
        device = self.data.device
        del self.data
        self.register_buffer('data', torch.empty(0, dtype=dtype, device=device))

    def peak(self) -> torch.Tensor | None:
        """Returns the exponential moving average of the observered tensor since the reducer was last cleared.

        Returns:
            torch.Tensor | None: cumulative output stored since the reducer was last cleared.
        """
        if self.data.numel > 0:
            return self.data

    def pop(self) -> torch.Tensor | None:
        """Returns the exponential moving average of the observered tensor since the reducer was last cleared, then clears the reducer.

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
        if self.data.numel() == 0:
            self.data = torch.zeros_like(inputs, dtype=self.count.dtype, device=self.count.device)
        self.data.mul_(1 - self.alpha)
        self.data.add_(self.alpha * inputs)
