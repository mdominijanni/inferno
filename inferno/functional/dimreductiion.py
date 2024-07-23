import torch
from typing import Any, Literal


def sum(
    data: torch.Tensor,
    dim: tuple[int, ...] | int | None = None,
    keepdim: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    r"""Returns a tensor with dimensions reduced via summation.

    This is a wrapper around :py:func:`torch.sum`.

    Args:
        data (torch.Tensor): tensor to which operations should be applied.
        dim (tuple[int, ...] | int | None, optional): dimension(s) along which the
            reduction should be applied, all dimensions when ``None``.
            Defaults to ``None``.
        keepdim (bool, optional): if the dimensions should be retained in the output.
            Defaults to ``False``.

    Returns:
        torch.Tensor: dimensionally reduced tensor.
    """
    return torch.sum(data, dim, keepdim=keepdim)


def nansum(
    data: torch.Tensor,
    dim: tuple[int, ...] | int | None = None,
    keepdim: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    r"""Returns a tensor with dimensions reduced via summation, excluding NaN values.

    This is a wrapper around :py:func:`torch.nansum`.

    Args:
        data (torch.Tensor): tensor to which operations should be applied.
        dim (tuple[int, ...] | int | None, optional): dimension(s) along which the
            reduction should be applied, all dimensions when ``None``.
            Defaults to ``None``.
        keepdim (bool, optional): if the dimensions should be retained in the output.
            Defaults to ``False``.

    Returns:
        torch.Tensor: dimensionally reduced tensor.
    """
    return torch.nansum(data, dim, keepdim=keepdim)


def divsum(
    data: torch.Tensor,
    dim: tuple[int, ...] | int | None = None,
    keepdim: bool = False,
    denom: int | float | complex = 1,
    **kwargs: Any,
) -> torch.Tensor:
    r"""Returns a tensor with dimensions reduced via summation then divided by a constant.

    Args:
        data (torch.Tensor): tensor to which operations should be applied.
        dim (tuple[int, ...] | int | None, optional): dimension(s) along which the
            reduction should be applied, all dimensions when ``None``.
            Defaults to ``None``.
        keepdim (bool, optional): if the dimensions should be retained in the output.
            Defaults to ``False``.
        denom (int | float | complex, optional): value by which to divide the sum.
            Defaults to ``1``.

    Returns:
        torch.Tensor: dimensionally reduced tensor.

    Tip:
        This is useful in cases where the mean over a larger sample is to be computed,
        but only a subset of the sample is being reduced.
    """
    return torch.sum(data, dim, keepdim=keepdim) / denom


def nandivsum(
    data: torch.Tensor,
    dim: tuple[int, ...] | int | None = None,
    keepdim: bool = False,
    denom: int | float | complex = 1,
    **kwargs: Any,
) -> torch.Tensor:
    r"""Returns a tensor with dimensions reduced via summation  then divided by a constant, excluding NaN values.

    This is a wrapper around :py:func:`torch.nansum`.

    Args:
        data (torch.Tensor): tensor to which operations should be applied.
        dim (tuple[int, ...] | int | None, optional): dimension(s) along which the
            reduction should be applied, all dimensions when ``None``.
            Defaults to ``None``.
        keepdim (bool, optional): if the dimensions should be retained in the output.
            Defaults to ``False``.
        denom (int | float | complex, optional): value by which to divide the sum.
            Defaults to ``1``.

    Returns:
        torch.Tensor: dimensionally reduced tensor.

    Tip:
        This is useful in cases where the mean over a larger sample is to be computed,
        but only a subset of the sample is being reduced.
    """
    return torch.nansum(data, dim, keepdim=keepdim) / denom


def min(
    data: torch.Tensor,
    dim: tuple[int, ...] | int | None = None,
    keepdim: bool = False,
    **kwargs,
) -> torch.Tensor:
    r"""Returns a tensor with dimensions reduced by taking the minimum.

    This is a wrapper around :py:func:`torch.amin`.

    Args:
        data (torch.Tensor): tensor to which operations should be applied.
        dim (tuple[int, ...] | int | None, optional): dimension(s) along which the
            reduction should be applied, all dimensions when ``None``.
            Defaults to ``None``.
        keepdim (bool, optional): if the dimensions should be retained in the output.
            Defaults to ``False``.

    Returns:
        torch.Tensor: dimensionally reduced tensor.
    """
    return torch.amin(data, dim, keepdim=keepdim)


def absmin(
    data: torch.Tensor,
    dim: tuple[int, ...] | int | None = None,
    keepdim: bool = False,
    **kwargs,
) -> torch.Tensor:
    r"""Returns a tensor with dimensions reduced by taking the minimum absolute distance from zero.

    The signs of elements in ``data`` are preserved by this operation.

    Args:
        data (torch.Tensor): tensor to which operations should be applied.
        dim (tuple[int, ...] | int | None, optional): dimension(s) along which the
            reduction should be applied, all dimensions when ``None``.
            Defaults to ``None``.
        keepdim (bool, optional): if the dimensions should be retained in the output.
            Defaults to ``False``.

    Returns:
        torch.Tensor: dimensionally reduced tensor.

    Note:
        The signs of values in ``data`` are preserved.
    """
    return torch.copysign(
        torch.amin(data.abs(), dim, keepdim=keepdim),
        torch.amin(data, dim, keepdim=keepdim),
    )


def max(
    data: torch.Tensor,
    dim: tuple[int, ...] | int | None = None,
    keepdim: bool = False,
    **kwargs,
) -> torch.Tensor:
    r"""Returns a tensor with dimensions reduced by taking the maximum.

    This is a wrapper around :py:func:`torch.amax`.

    Args:
        data (torch.Tensor): tensor to which operations should be applied.
        dim (tuple[int, ...] | int | None, optional): dimension(s) along which the
            reduction should be applied, all dimensions when ``None``.
            Defaults to ``None``.
        keepdim (bool, optional): if the dimensions should be retained in the output.
            Defaults to ``False``.

    Returns:
        torch.Tensor: dimensionally reduced tensor.
    """
    return torch.amax(data, dim, keepdim=keepdim)


def absmax(
    data: torch.Tensor,
    dim: tuple[int, ...] | int | None = None,
    keepdim: bool = False,
    **kwargs,
) -> torch.Tensor:
    r"""Returns a tensor with dimensions reduced by taking the maximum absolute distance from zero.

    The signs of elements in ``data`` are preserved by this operation.

    Args:
        data (torch.Tensor): tensor to which operations should be applied.
        dim (tuple[int, ...] | int | None, optional): dimension(s) along which the
            reduction should be applied, all dimensions when ``None``.
            Defaults to ``None``.
        keepdim (bool, optional): if the dimensions should be retained in the output.
            Defaults to ``False``.

    Returns:
        torch.Tensor: dimensionally reduced tensor.

    Note:
        The signs of values in ``data`` are preserved.
    """
    return torch.copysign(
        torch.amax(data.abs(), dim, keepdim=keepdim),
        torch.amax(data, dim, keepdim=keepdim),
    )


def mean(
    data: torch.Tensor,
    dim: tuple[int, ...] | int | None = None,
    keepdim: bool = False,
    **kwargs,
) -> torch.Tensor:
    r"""Returns a tensor with dimensions reduced by taking the arithmetic mean.

    This is a wrapper around :py:func:`torch.mean`.

    Args:
        data (torch.Tensor): tensor to which operations should be applied.
        dim (tuple[int, ...] | int | None, optional): dimension(s) along which the
            reduction should be applied, all dimensions when ``None``.
            Defaults to ``None``.
        keepdim (bool, optional): if the dimensions should be retained in the output.
            Defaults to ``False``.

    Returns:
        torch.Tensor: dimensionally reduced tensor.
    """
    return torch.mean(data, dim, keepdim=keepdim)


def nanmean(
    data: torch.Tensor,
    dim: tuple[int, ...] | int | None = None,
    keepdim: bool = False,
    **kwargs,
) -> torch.Tensor:
    r"""Returns a tensor with dimensions reduced by taking the arithmetic mean, excluding NaN values.

    This is a wrapper around :py:func:`torch.nanmean`.

    Args:
        data (torch.Tensor): tensor to which operations should be applied.
        dim (tuple[int, ...] | int | None, optional): dimension(s) along which the
            reduction should be applied, all dimensions when ``None``.
            Defaults to ``None``.
        keepdim (bool, optional): if the dimensions should be retained in the output.
            Defaults to ``False``.

    Returns:
        torch.Tensor: dimensionally reduced tensor.
    """
    return torch.nanmean(data, dim, keepdim=keepdim)


def quantile(
    data: torch.Tensor,
    dim: tuple[int, ...] | int | None = None,
    keepdim: bool = False,
    q: float = 0.5,
    interpolation: Literal[
        "linear", "lowest", "higher", "nearest", "midpoint"
    ] = "linear",
    **kwargs,
) -> torch.Tensor:
    r"""Returns a tensor with dimensions reduced by taking the arithmetic mean.

    This is a wrapper around :py:func:`torch.quantile` with arguments rearranged and
    defaults provided for compatibility. By default, the median is computed (``q=0.5``).

    The interpolation methods are as follows, when ``a`` is the value with the lower
    discrete index, ``b`` is the value with the higher discrete index, and ``k`` is
    the continuous index for the quantile.

    * ``"linear"``: ``a + (b - a) * k % 1``
    * ``"lower"``: ``a``
    * ``"higher"``: ``b``
    * ``"nearest"``: ``b if (k % 1) > 0.5 else a``
    * ``"midpoint"``: ``(a + b) / 2``

    Args:
        data (torch.Tensor): tensor to which operations should be applied.
        dim (tuple[int, ...] | int | None, optional): dimension(s) along which the
            reduction should be applied, all dimensions when ``None``.
            Defaults to ``None``.
        keepdim (bool, optional): if the dimensions should be retained in the output.
            Defaults to ``False``.
        q (float, optional): :math:`q^\text{th}` quantile to take. Defaults to ``0.5``.
        interpolation (str, optional): method of interpolation when the quantile lies
            between two data points. Defaults to ``"linear"``.

    Returns:
        torch.Tensor: dimensionally reduced tensor.
    """
    return torch.quantile(data, q, dim, keepdim=keepdim, interpolation=interpolation)


def nanquantile(
    data: torch.Tensor,
    dim: tuple[int, ...] | int | None = None,
    keepdim: bool = False,
    q: float = 0.5,
    interpolation: Literal[
        "linear", "lowest", "higher", "nearest", "midpoint"
    ] = "linear",
    **kwargs,
) -> torch.Tensor:
    r"""Returns a tensor with dimensions reduced by taking the arithmetic mean, excluding NaN values.

    This is a wrapper around :py:func:`torch.nanquantile` with arguments rearranged and
    defaults provided for compatibility. By default, the median is computed (``q=0.5``).

    The interpolation methods are as follows, when ``a`` is the value with the lower
    discrete index, ``b`` is the value with the higher discrete index, and ``k`` is
    the continuous index for the quantile.

    * ``"linear"``: ``a + (b - a) * k % 1``
    * ``"lower"``: ``a``
    * ``"higher"``: ``b``
    * ``"nearest"``: ``b if (k % 1) > 0.5 else a``
    * ``"midpoint"``: ``(a + b) / 2``

    Args:
        data (torch.Tensor): tensor to which operations should be applied.
        dim (tuple[int, ...] | int | None, optional): dimension(s) along which the
            reduction should be applied, all dimensions when ``None``.
            Defaults to ``None``.
        keepdim (bool, optional): if the dimensions should be retained in the output.
            Defaults to ``False``.
        q (float, optional): :math:`q^\text{th}` quantile to take. Defaults to ``0.5``.
        interpolation (str, optional): method of interpolation when the quantile lies
            between two data points. Defaults to ``"linear"``.

    Returns:
        torch.Tensor: dimensionally reduced tensor.
    """
    return torch.nanquantile(data, q, dim, keepdim=keepdim, interpolation=interpolation)


def median(
    data: torch.Tensor,
    dim: tuple[int, ...] | int | None = None,
    keepdim: bool = False,
    **kwargs,
) -> torch.Tensor:
    r"""Returns a tensor with dimensions reduced by taking median.

    This is an alias for ``quantile(..., q=0.5, interpolation="midpoint")``.

    Args:
        data (torch.Tensor): tensor to which operations should be applied.
        dim (tuple[int, ...] | int | None, optional): dimension(s) along which the
            reduction should be applied, all dimensions when ``None``.
            Defaults to ``None``.
        keepdim (bool, optional): if the dimensions should be retained in the output.
            Defaults to ``False``.

    Returns:
        torch.Tensor: dimensionally reduced tensor.
    """
    return torch.quantile(data, 0.5, dim, keepdim=keepdim, interpolation="midpoint")


def nanmedian(
    data: torch.Tensor,
    dim: tuple[int, ...] | int | None = None,
    keepdim: bool = False,
    **kwargs,
) -> torch.Tensor:
    r"""Returns a tensor with dimensions reduced by taking median, excluding NaN values.

    This is an alias for ``nanquantile(..., q=0.5, interpolation="midpoint")``.

    Args:
        data (torch.Tensor): tensor to which operations should be applied.
        dim (tuple[int, ...] | int | None, optional): dimension(s) along which the
            reduction should be applied, all dimensions when ``None``.
            Defaults to ``None``.
        keepdim (bool, optional): if the dimensions should be retained in the output.
            Defaults to ``False``.

    Returns:
        torch.Tensor: dimensionally reduced tensor.
    """
    return torch.nanquantile(data, 0.5, dim, keepdim=keepdim, interpolation="midpoint")


def geomean(
    data: torch.Tensor,
    dim: tuple[int, ...] | int | None = None,
    keepdim: bool = False,
    **kwargs,
) -> torch.Tensor:
    r"""Returns a tensor with dimensions reduced by taking the geometric mean.

    The geometric mean is calculated by taking the arithmetic mean of the log,
    where zero values are ignored. If all elements being reduced are zero, then the
    output is zero.

    Args:
        data (torch.Tensor): tensor to which operations should be applied.
        dim (tuple[int, ...] | int | None, optional): dimension(s) along which the
            reduction should be applied, all dimensions when ``None``.
            Defaults to ``None``.
        keepdim (bool, optional): if the dimensions should be retained in the output.
            Defaults to ``False``.

    Returns:
        torch.Tensor: dimensionally reduced tensor.
    """
    logdata = data.log()
    counts = torch.sum(data != 0, dim, keepdim=keepdim)
    return torch.where(
        counts.bool(),
        torch.exp(
            torch.sum(
                torch.where(logdata != float("-inf"), logdata, 0), dim, keepdim=keepdim
            )
            / counts
        ),
        0,
    )


def nangeomean(
    data: torch.Tensor,
    dim: tuple[int, ...] | int | None = None,
    keepdim: bool = False,
    **kwargs,
) -> torch.Tensor:
    r"""Returns a tensor with dimensions reduced by taking the geometric mean, excluding NaN values.

    The geometric mean is calculated by taking the arithmetic mean of the log,
    where zero and ``NaN`` values are ignored. If all elements being reduced are zero
    or ``NaN``, then the output is zero.

    Args:
        data (torch.Tensor): tensor to which operations should be applied.
        dim (tuple[int, ...] | int | None, optional): dimension(s) along which the
            reduction should be applied, all dimensions when ``None``.
            Defaults to ``None``.
        keepdim (bool, optional): if the dimensions should be retained in the output.
            Defaults to ``False``.

    Returns:
        torch.Tensor: dimensionally reduced tensor.
    """
    sanitized = torch.nan_to_num(data.log(), nan=0.0, neginf=0.0)
    return torch.nan_to_num(
        torch.exp(
            torch.sum(sanitized, dim, keepdim=keepdim)
            / torch.sum(sanitized != 0, dim, keepdim=keepdim)
        ),
        nan=0.0,
    )
