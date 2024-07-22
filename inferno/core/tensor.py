import torch
from collections.abc import Sequence
from typing import Any, Callable


def zeros(
    tensor: torch.Tensor,
    *,
    shape: Sequence[int] | torch.Size | None = None,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    requires_grad: bool | None = None,
) -> torch.Tensor:
    r"""Returns a tensor based on input filled with zeros.

    Args:
        tensor (torch.Tensor): determines default output properties.
        shape (Sequence[int] | torch.Size | None, optional): overrides shape from
            ``tensor`` if specified. Defaults to ``None``.
        dtype (torch.dtype | None, optional): overrides data type from ``tensor``
            if specified. Defaults to ``None``.
        layout (torch.layout | None, optional): overrides layout from ``tensor``
            if specified. Defaults to ``None``.
        device (torch.device | None, optional): overrides device from ``tensor``
            if specified. Defaults to ``None``.
        requires_grad (bool | None, optional): overrides gradient requirement from
            ``tensor`` if specified. Defaults to ``None``.

    Returns:
        torch.Tensor: tensor like ``tensor``, modified by parameters, filled with ``0``.

    Note:
        To construct a scalar, set ``shape`` to ``()``.
    """
    shape = tensor.shape if shape is None else shape
    dtype = tensor.dtype if dtype is None else dtype
    layout = tensor.layout if layout is None else layout
    device = tensor.device if device is None else device
    requires_grad = tensor.requires_grad if requires_grad is None else requires_grad

    return torch.zeros(
        shape, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad
    )


def ones(
    tensor: torch.Tensor,
    *,
    shape: Sequence[int] | torch.Size | None = None,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    requires_grad: bool | None = None,
) -> torch.Tensor:
    r"""Returns a tensor based on input filled with ones.

    Args:
        tensor (torch.Tensor): determines default output properties.
        shape (Sequence[int] | torch.Size | None, optional): overrides shape from
            ``tensor`` if specified. Defaults to ``None``.
        dtype (torch.dtype | None, optional): overrides data type from ``tensor``
            if specified. Defaults to ``None``.
        layout (torch.layout | None, optional): overrides layout from ``tensor``
            if specified. Defaults to ``None``.
        device (torch.device | None, optional): overrides device from ``tensor``
            if specified. Defaults to ``None``.
        requires_grad (bool | None, optional): overrides gradient requirement from
            ``tensor`` if specified. Defaults to ``None``.

    Returns:
        torch.Tensor: tensor like ``tensor``, modified by parameters, filled with ``1``.

    Note:
        To construct a scalar, set ``shape`` to ``()``.
    """
    shape = tensor.shape if shape is None else shape
    dtype = tensor.dtype if dtype is None else dtype
    layout = tensor.layout if layout is None else layout
    device = tensor.device if device is None else device
    requires_grad = tensor.requires_grad if requires_grad is None else requires_grad

    return torch.ones(
        shape, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad
    )


def empty(
    tensor: torch.Tensor,
    *,
    shape: Sequence[int] | torch.Size | None = None,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    requires_grad: bool | None = None,
) -> torch.Tensor:
    r"""Returns an uninitialized tensor based on input.

    Args:
        tensor (torch.Tensor): determines default output properties.
        shape (Sequence[int] | torch.Size | None, optional): overrides shape from
            ``tensor`` if specified. Defaults to ``None``.
        dtype (torch.dtype | None, optional): overrides data type from ``tensor``
            if specified. Defaults to ``None``.
        layout (torch.layout | None, optional): overrides layout from ``tensor``
            if specified. Defaults to ``None``.
        device (torch.device | None, optional): overrides device from ``tensor``
            if specified. Defaults to ``None``.
        requires_grad (bool | None, optional): overrides gradient requirement from
            ``tensor`` if specified. Defaults to ``None``.

    Returns:
        torch.Tensor: uninitialized tensor like ``tensor``, modified by parameters.

    Note:
        To construct a scalar, set ``shape`` to ``()``.
    """
    shape = tensor.shape if shape is None else shape
    dtype = tensor.dtype if dtype is None else dtype
    layout = tensor.layout if layout is None else layout
    device = tensor.device if device is None else device
    requires_grad = tensor.requires_grad if requires_grad is None else requires_grad

    return torch.empty(
        shape, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad
    )


def full(
    tensor: torch.Tensor,
    value: bool | int | float | complex,
    *,
    shape: Sequence[int] | torch.Size | None = None,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    requires_grad: bool | None = None,
) -> torch.Tensor:
    r"""Returns a tensor based on input filled with specified value.

    Args:
        tensor (torch.Tensor): determines default output properties.
        value (bool | int | float | complex): value with to fill the output.
        shape (Sequence[int] | torch.Size | None, optional): overrides shape from
            ``tensor`` if specified. Defaults to ``None``.
        dtype (torch.dtype | None, optional): overrides data type from ``tensor``
            if specified. Defaults to ``None``.
        layout (torch.layout | None, optional): overrides layout from ``tensor``
            if specified. Defaults to ``None``.
        device (torch.device | None, optional): overrides device from ``tensor``
            if specified. Defaults to ``None``.
        requires_grad (bool | None, optional): overrides gradient requirement from
            ``tensor`` if specified. Defaults to ``None``.

    Returns:
        torch.Tensor: tensor like ``tensor``, modified by parameters, filled
        with ``value``.

    Note:
        To construct a scalar, set ``shape`` to ``()``.
    """
    shape = tensor.shape if shape is None else shape
    dtype = tensor.dtype if dtype is None else dtype
    layout = tensor.layout if layout is None else layout
    device = tensor.device if device is None else device
    requires_grad = tensor.requires_grad if requires_grad is None else requires_grad

    return torch.full(
        shape,
        fill_value=value,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )


def fullc(
    tensor: torch.Tensor,
    value: bool | int | float | complex,
    *,
    shape: Sequence[int] | torch.Size | None = None,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    requires_grad: bool | None = None,
) -> torch.Tensor:
    r"""Returns a float or complex tensor based on input filled with specified value.

    This is like :py:func:`full` except if ``dtype`` is ``None`` and the datatype of
    ``tensor`` is neither floating point nor complex, the default float type will
    be used.

    Args:
        tensor (torch.Tensor): determines default output properties.
        value (bool | int | float | complex): value with to fill the output.
        shape (Sequence[int] | torch.Size | None, optional): overrides shape from
            ``tensor`` if specified. Defaults to ``None``.
        dtype (torch.dtype | None, optional): overrides data type from ``tensor``
            if specified. Defaults to ``None``.
        layout (torch.layout | None, optional): overrides layout from ``tensor``
            if specified. Defaults to ``None``.
        device (torch.device | None, optional): overrides device from ``tensor``
            if specified. Defaults to ``None``.
        requires_grad (bool | None, optional): overrides gradient requirement from
            ``tensor`` if specified. Defaults to ``None``.

    Returns:
        torch.Tensor: tensor like ``tensor``, modified by parameters, filled
        with ``value``.

    Note:
        To construct a scalar, set ``shape`` to ``()``.
    """
    shape = tensor.shape if shape is None else shape
    dtype = (
        (
            tensor.dtype
            if tensor.is_floating_point() or tensor.is_complex()
            else torch.get_default_dtype()
        )
        if dtype is None
        else dtype
    )
    layout = tensor.layout if layout is None else layout
    device = tensor.device if device is None else device
    requires_grad = tensor.requires_grad if requires_grad is None else requires_grad

    return torch.full(
        shape,
        fill_value=value,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )


def uniform(
    tensor: torch.Tensor,
    *,
    shape: Sequence[int] | torch.Size | None = None,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    requires_grad: bool | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    r"""Returns a tensor based on input filled with random values sampled uniformly.

    Args:
        tensor (torch.Tensor): determines default output properties.
        shape (Sequence[int] | torch.Size | None, optional): overrides shape from
            ``tensor`` if specified. Defaults to ``None``.
        dtype (torch.dtype | None, optional): overrides data type from ``tensor``
            if specified. Defaults to ``None``.
        layout (torch.layout | None, optional): overrides layout from ``tensor``
            if specified. Defaults to ``None``.
        device (torch.device | None, optional): overrides device from ``tensor``
            if specified. Defaults to ``None``.
        requires_grad (bool | None, optional): overrides gradient requirement from
            ``tensor`` if specified. Defaults to ``None``.
        generator (torch.Generator | None, optional): pseudorandom number generator
            for sampling. Defaults to ``None``.

    Returns:
        torch.Tensor: tensor like ``tensor``, modified by parameters, with elements
        sampled from :math:`\mathcal{U}(0, 1)`.

    Note:
        To construct a scalar, set ``shape`` to ``()``.

    See Also:
        See :py:func:`torch.rand` for the function which this extends.
    """
    shape = tensor.shape if shape is None else shape
    dtype = tensor.dtype if dtype is None else dtype
    layout = tensor.layout if layout is None else layout
    device = tensor.device if device is None else device
    requires_grad = tensor.requires_grad if requires_grad is None else requires_grad

    return torch.rand(
        shape,
        generator=generator,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )


def normal(
    tensor: torch.Tensor,
    *,
    shape: Sequence[int] | torch.Size | None = None,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    requires_grad: bool | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    r"""Returns a tensor based on input filled with random values sampled normally.

    Args:
        tensor (torch.Tensor): determines default output properties.
        shape (Sequence[int] | torch.Size | None, optional): overrides shape from
            ``tensor`` if specified. Defaults to ``None``.
        dtype (torch.dtype | None, optional): overrides data type from ``tensor``
            if specified. Defaults to ``None``.
        layout (torch.layout | None, optional): overrides layout from ``tensor``
            if specified. Defaults to ``None``.
        device (torch.device | None, optional): overrides device from ``tensor``
            if specified. Defaults to ``None``.
        requires_grad (bool | None, optional): overrides gradient requirement from
            ``tensor`` if specified. Defaults to ``None``.
        generator (torch.Generator | None, optional): pseudorandom number generator
            for sampling. Defaults to ``None``.

    Returns:
        torch.Tensor: tensor like ``tensor``, modified by parameters, with elements
        sampled from :math:`\mathcal{N}(0, 1)`.

    Note:
        To construct a scalar, set ``shape`` to ``()``.

    See Also:
        See :py:func:`torch.randn` for the function which this extends.
    """
    shape = tensor.shape if shape is None else shape
    dtype = tensor.dtype if dtype is None else dtype
    layout = tensor.layout if layout is None else layout
    device = tensor.device if device is None else device
    requires_grad = tensor.requires_grad if requires_grad is None else requires_grad

    return torch.randn(
        shape,
        generator=generator,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )


def scalar(
    value: bool | int | float | complex,
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    requires_grad: bool | None = None,
) -> torch.Tensor:
    r"""Returns a scalar tensor based on input with specified value.

    Shortcut for :py:func:`full` with ``shape=()``.

    Args:
        value (bool | int | float | complex): value with to fill the output.
        tensor (torch.Tensor): determines default output properties.
        dtype (torch.dtype | None, optional): overrides data type from ``tensor``
            if specified. Defaults to ``None``.
        layout (torch.layout | None, optional): overrides layout from ``tensor``
            if specified. Defaults to ``None``.
        device (torch.device | None, optional): overrides device from ``tensor``
            if specified. Defaults to ``None``.
        requires_grad (bool | None, optional): overrides gradient requirement from
            ``tensor`` if specified. Defaults to ``None``.

    Returns:
        torch.Tensor: scalar tensor like ``tensor``, modified by parameters, filled
        with ``value``.
    """
    return full(
        tensor,
        value,
        shape=(),
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )


def astensors(
    *values: Any, conversion: Callable[[Any], torch.Tensor] | None = None
) -> tuple[torch.Tensor, ...] | torch.Tensor:
    r"""Converts inputs into tensors.

    If any value is a tensor, it will be used as a reference and non-tensor
    inputs will be converted using :py:func:`scalar`. If there are no tensors,
    then all elements will be converted into tensors using ``conversion``. When
    determining a reference, the leftmost tensor will be used.

    Args:
        *values (Any): values to convert into tensors
        conversion (Callable[[Any], torch.Tensor] | None): method to convert values if
            none are tensors, the default if unspecified. Defaults to ``None``.

    Returns:
        tuple[torch.Tensor, ...] | torch.Tensor: converted values.
    """
    # get the first tensor to use as a reference point
    ref = None
    for val in values:
        if isinstance(val, torch.Tensor):
            ref = val
            break

    # configure the conversion to use
    if ref is None:
        if conversion is None:
            conversion = lambda x: torch.tensor(x)  # noqa:E731;
        cf = conversion
    else:
        conversion = lambda x: scalar(x, ref)  # noqa:E731;
        cf = lambda x: x if isinstance(x, torch.Tensor) else conversion(x)  # noqa:E731;

    # return tensor values
    if len(values) == 1:
        return cf(values[0])
    else:
        return tuple(map(cf, values))
