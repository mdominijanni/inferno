import cmath
import einops as ein
import functools
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@functools.singledispatch
def exp(
    x: int | float | complex | torch.Tensor | np.ndarray | np.number,
) -> float | complex | torch.Tensor | np.ndarray | np.number:
    r"""Type agnostic exponential function.

    .. math::
        y = e^x

    Args:
        x (int | float | complex | torch.Tensor | numpy.ndarray | numpy.number): value
            by which to raise :math:`e`.

    Returns:
        float | complex | torch.Tensor | numpy.ndarray | numpy.number: :math:`e`
        raised to the input.
    """
    raise NotImplementedError


@exp.register(int)
@exp.register(float)
def _(x: int | float) -> float:
    return math.exp(x)


@exp.register(complex)
def _(x: complex) -> complex:
    return cmath.exp(x)


@exp.register(torch.Tensor)
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(x)


@exp.register(np.ndarray)
def _(x: np.ndarray) -> np.ndarray:
    return np.exp(x)


@exp.register(np.number)
def _(x: np.number) -> np.number:
    return np.exp(x)


@functools.singledispatch
def sqrt(
    x: int | float | complex | torch.Tensor | np.ndarray | np.number,
) -> float | complex | torch.Tensor | np.ndarray | np.number:
    r"""Type agnostic square root function.

    .. math::
        y = \sqrt{x}

    Args:
        x (int | float | complex | torch.Tensor | numpy.ndarray | numpy.number): value
            of which to take the square root.

    Returns:
        float | complex | torch.Tensor | numpy.ndarray | numpy.number: square root of
        the input.
    """
    raise NotImplementedError


@sqrt.register(int)
@sqrt.register(float)
def _(x: int | float) -> float:
    return math.sqrt(x)


@sqrt.register(complex)
def _(x: complex) -> complex:
    return cmath.sqrt(x)


@sqrt.register(torch.Tensor)
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(x)


@sqrt.register(np.ndarray)
def _(x: np.ndarray) -> np.ndarray:
    return np.sqrt(x)


@sqrt.register(np.number)
def _(x: np.number) -> np.number:
    return np.sqrt(x)


def normalize(
    data: torch.Tensor,
    order: int | float,
    scale: float | complex = 1.0,
    dim: int | tuple[int, ...] | None = None,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    r"""Normalizes a tensor.

    Args:
        data (torch.Tensor): data to normalize.
        order (int | float): order of :math:`p`-norm by which to normalize.
        scale (float | complex, optional): desired :math:`p`-norm of elements along
            specified dimensions. Defaults to ``1.0``.
        dim (int | tuple[int, ...] | None, optional): dimension(s) along which to normalize,
            all dimensions if ``None``. Defaults to ``None``.
        epsilon (float, optional): value added to the denominator in case of
            zero-valued norms. Defaults to ``1e-12``.

    Returns:
        torch.Tensor: normalized tensor.
    """
    return scale * F.normalize(data, p=order, dim=dim, eps=epsilon)  # type: ignore


def rescale(
    data: torch.Tensor,
    resmin: int | float | torch.Tensor | None,
    resmax: int | float | torch.Tensor | None,
    *,
    srcmin: int | float | torch.Tensor | None = None,
    srcmax: int | float | torch.Tensor | None = None,
    dim: int | tuple[int, ...] | None = None,
) -> torch.Tensor:
    r"""Rescales a tensor (min-max normalization).

    Args:
        data (torch.Tensor): tensor to rescale.
        resmin (int | float | torch.Tensor | None): minimum value for the
            tensor after rescaling, unchanged if ``None``.
        resmax (int | float | torch.Tensor | None): maximum value for the
            tensor after rescaling, unchanged if ``None``.
        srcmin (int | float | torch.Tensor | None, optional): minimum value for the
            tensor before rescaling, computed if ``None``. Defaults to ``None``.
        srcmax (int | float | torch.Tensor | None, optional): maximum value for the
            tensor before rescaling, computed if ``None``. Defaults to ``None``.
        dim (int | tuple[int, ...] | None, optional): dimension(s) along which amin/amax
            are computed if not provided, all dimensions if ``None``. Defaults to ``None``.

    Returns:
        torch.Tensor: rescaled tensor.
    """
    # perform substitutions
    if srcmin is None:
        srcmin = torch.amin(data, dim=dim, keepdim=True)  # type: ignore
    if srcmax is None:
        srcmax = torch.amax(data, dim=dim, keepdim=True)  # type: ignore
    if resmin is None:
        resmin = srcmin
    if resmax is None:
        resmax = srcmax

    # rescale and return
    return resmin + (((data - srcmin) * (resmax - resmin)) / (srcmax - srcmin))


def exponential_smoothing(
    obs: torch.Tensor,
    level: torch.Tensor | None,
    *,
    alpha: float | int | complex | torch.Tensor,
) -> torch.Tensor:
    r"""Performs simple exponential smoothing for a time step.

    .. math::
        \begin{align*}
            s_0 &= x_0 \\
            s_{t + 1} &= \alpha x_{t + 1}  + (1 - \alpha) s_t
        \end{align*}

    Args:
        obs (torch.Tensor): latest state to consider for exponential smoothing,
            :math:`x`.
        level (torch.Tensor | None): current value of the smoothed level,
            :math:`s`.
        alpha (float | int | complex | torch.Tensor): level smoothing factor,
            :math:`\alpha`.

    Returns:
        torch.Tensor: revised exponentially smoothed value.
    """
    # initial condition
    if level is None:
        return obs

    # standard condition
    else:
        return alpha * obs + (1 - alpha) * level


def holt_linear_smoothing(
    obs: torch.Tensor,
    level: torch.Tensor | None,
    trend: torch.Tensor | None,
    *,
    alpha: float | int | complex | torch.Tensor,
    beta: float | int | complex | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""Performs Holt linear smoothing for a time step.

    .. math::
        \begin{align*}
            s_0 &= x_0 \\
            b_0 &= x_1 - x_0 \\
            s_{t + 1} &= \alpha x_{t + 1}  + (1 - \alpha) s_t \\
            b_{t + 1} &= \beta (s_{t + 1} - s_t) + (1 - \beta) b_t
        \end{align*}

    Args:
        obs (torch.Tensor): latest state to consider for exponential smoothing,
            :math:`x_{t + 1}`.
        level (torch.Tensor | None): current value of the smoothed level,
            :math:`s`.
        trend (torch.Tensor | None): current value of the smoothed trend,
            :math:`b`.
        alpha (float | int | complex | torch.Tensor): level smoothing factor,
            :math:`\alpha`.
        beta (float | int | complex | torch.Tensor): trend smoothing factor,
            :math:`\beta`.

    Returns:
        tuple[torch.Tensor, torch.Tensor | None]: tuple containing output/updated state:

            level: revised exponentially smoothed level.

            trend: revised exponentially smoothed trend.
    """
    # t=0 condition
    if level is None:
        return obs, None

    # t=1 condition (initialize trend as x1-x0)
    if trend is None:
        trend = obs - level

    # t>0 condition
    s = exponential_smoothing(obs, level + trend, alpha=alpha)
    b = exponential_smoothing(s - level, trend, alpha=beta)

    return s, b


def isi(
    spikes: torch.Tensor, step_time: float, time_first: bool = True
) -> torch.Tensor:
    r"""Transforms spike trains into interspike intervals.

    The returned tensor will be padded with ``NaN`` values where an interval could not
    be computed but the position existed (e.g. padding at the end of) spike trains
    with fewer spikes. If no intervals could be generated at all, a tensor with a
    time dimension of zero will be returned. The returned tensor will have a floating
    point type, as required for the padding.

    Args:
        spikes (torch.Tensor): spike trains for which to calculate intervals.
        step_time (float): length of the simulation step,
            in :math:`\text{ms}`.
        time_first (bool, optional): if the time dimension is given first rather than
            last. Defaults to ``True``.

    Returns:
        torch.Tensor: interspike intervals for the given spike trains.

    .. admonition:: Shape
        :class: tensorshape

        ``spikes``:

        :math:`T \times N_0 \times \cdots` or :math:`N_0 \times \cdots \times T`

        ``return``:

        :math:`(C - 1) \times N_0 \times \cdots` or :math:`N_0 \times \cdots \times (C - 1)`

        Where:
            * :math:`N_0, \ldots` shape of the generating population (batch, neuron shape, etc).
            * :math:`T` the length of the spike trains.
            * :math:`C` the maximum number of spikes amongst the spike trains.
    """
    # bring time dimension to the end if it is not
    if time_first:
        spikes = ein.rearrange(spikes, "t ... -> ... t")

    # ensure step time is a float
    step_time = float(step_time)

    # pad spikes with true to ensure at least one (req. for split)
    padded = F.pad(spikes, (1, 0), mode="constant", value=True)

    # compute nonzero values
    nz = torch.nonzero(padded)[..., -1]

    # compute split indices (at the added pads)
    splits = torch.nonzero(torch.logical_not(nz)).view(-1).tolist()[1:]

    # split the tensor into various length subtensors (subtract 1 to unshift)
    intervals = torch.tensor_split((nz - 1) * step_time, splits, dim=-1)

    # stack, pad trailing with nan, trim leading pad
    intervals = nn.utils.rnn.pad_sequence(
        intervals, batch_first=True, padding_value=float("nan")
    )[:, 1:]

    # compute intervals
    intervals = torch.diff(intervals, dim=-1)

    # reshape and return
    if time_first:
        return ein.rearrange(intervals.view(*spikes.shape[:-1], -1), "... t -> t ...")
    else:
        return intervals.view(*spikes.shape[:-1], -1)


@torch.no_grad()
def victor_purpura_pair_dist(
    t0: torch.Tensor, t1: torch.Tensor, cost: float | torch.Tensor
) -> torch.Tensor:
    r"""Victor–Purpura distance between a pair of spike trains.

    This function is not fully vectorized and may be slow. It take care when using it
    on performance critical pathways.

    Uses a Needleman–Wunsch approach. Translated from the
    `MATLAB code <http://www-users.med.cornell.edu/~jdvicto/spkd_qpara.html>`_
    by Thomas Kreuz.

    Args:
        t0 (torch.Tensor): spike times of the first spike train.
        t1 (torch.Tensor): spike times of the second spike train.
        cost (float | torch.Tensor): cost to move a spike by one unit of time.

    Returns:
        torch.Tensor: distance between the spike trains for each cost.

    .. admonition:: Shape
        :class: tensorshape

        ``t0``:

        :math:`T_m`

        ``t1``:

        :math:`T_n`

        ``cost`` and ``return``:

        :math:`k`

        Where:
            * :math:`T_m` number of spikes in the first spike train.
            * :math:`T_n` number of spikes in the second spike train.
            * :math:`k`, number of cost values to compute distance for, treated
              as :math:`1` when ``cost`` is a float.

    Warning:
        As in the original algorithm, using ``inf`` as the cost will only return
        the total number of spikes, not accounting for spikes occurring at the same
        time in each spike train.
    """
    # check for cost edge conditions and make tensor if not
    if not isinstance(cost, torch.Tensor):
        if cost == 0.0:
            return torch.tensor([float(abs(t0.numel() - t1.numel()))], device=t0.device)
        elif cost == float("inf"):
            return torch.tensor([float(t0.numel() + t1.numel())], device=t0.device)
        else:
            cost = torch.tensor([float(cost)], device=t0.device)

    # create grid for Needleman–Wunsch
    tckwargs = {"dtype": cost.dtype, "device": cost.device}
    grid = torch.zeros(t0.numel() + 1, t1.numel() + 1, **tckwargs)
    grid[:, 0] = torch.arange(0, t0.numel() + 1, **tckwargs).t()
    grid[0, :] = torch.arange(0, t1.numel() + 1, **tckwargs).t()
    grid = grid.unsqueeze(0).repeat(cost.numel(), 1, 1)

    # dp algorithm
    for r in range(1, t0.numel() + 1):
        for c in range(1, t1.numel() + 1):
            c_add_a = grid[:, r - 1, c] + 1
            c_add_b = grid[:, r, c - 1] + 1
            c_shift = grid[:, r - 1, c - 1] + cost * torch.abs(t0[r - 1] - t1[c - 1])
            grid[:, r, c] = (
                torch.stack((c_add_a, c_add_b, c_shift), 0)
                .nan_to_num(nan=float("inf"))
                .amin(0)
            )

    # return result
    return grid[:, -1, -1]
