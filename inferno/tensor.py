import torch


def zeros(
    tensor: torch.Tensor,
    *,
    shape: tuple[int] | torch.Size | None = None,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    requires_grad: bool | None = None
) -> torch.Tensor:
    r"""Returns a tensor based on input filled with zeros.

    Args:
        tensor (torch.Tensor): determines default output properties.
        shape (tuple[int] | torch.Size | None, optional): overrides shape from
            ``tensor`` if specified. Defaults to None.
        dtype (torch.dtype | None, optional): overrides data type from ``tensor``
            if specified. Defaults to None.
        layout (torch.layout | None, optional): overrides layout from ``tensor``
            if specified. Defaults to None.
        device (torch.device | None, optional): overrides device from ``tensor``
            if specified. Defaults to None.
        requires_grad (bool | None, optional): overrides gradient requirement from
            ``tensor`` if specified. Defaults to None.

    Returns:
        torch.Tensor: tensor like ``tensor``, modified by parameters, filled with 0.

    Note:
        To construct a scalar, set ``shape`` to ``((),)``.
    """
    shape = tensor.shape if shape is None else shape
    dtype = tensor.dtype if dtype is None else dtype
    layout = tensor.layout if layout is None else layout
    device = tensor.device if device is None else device
    requires_grad = tensor.requires_grad if requires_grad is None else requires_grad

    return torch.zeros(
        *shape, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad
    )


def ones(
    tensor: torch.Tensor,
    *,
    shape: tuple[int] | torch.Size | None = None,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    requires_grad: bool | None = None
) -> torch.Tensor:
    r"""Returns a tensor based on input filled with ones.

    Args:
        tensor (torch.Tensor): determines default output properties.
        shape (tuple[int] | torch.Size | None, optional): overrides shape from
            ``tensor`` if specified. Defaults to None.
        dtype (torch.dtype | None, optional): overrides data type from ``tensor``
            if specified. Defaults to None.
        layout (torch.layout | None, optional): overrides layout from ``tensor``
            if specified. Defaults to None.
        device (torch.device | None, optional): overrides device from ``tensor``
            if specified. Defaults to None.
        requires_grad (bool | None, optional): overrides gradient requirement from
            ``tensor`` if specified. Defaults to None.

    Returns:
        torch.Tensor: tensor like ``tensor``, modified by parameters, filled with 1.

    Note:
        To construct a scalar, set ``shape`` to ``((),)``.
    """
    shape = tensor.shape if shape is None else shape
    dtype = tensor.dtype if dtype is None else dtype
    layout = tensor.layout if layout is None else layout
    device = tensor.device if device is None else device
    requires_grad = tensor.requires_grad if requires_grad is None else requires_grad

    return torch.ones(
        *shape, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad
    )


def empty(
    tensor: torch.Tensor,
    *,
    shape: tuple[int] | torch.Size | None = None,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    requires_grad: bool | None = None
) -> torch.Tensor:
    r"""Returns an uninitialized tensor based on input.

    Args:
        tensor (torch.Tensor): determines default output properties.
        shape (tuple[int] | torch.Size | None, optional): overrides shape from
            ``tensor`` if specified. Defaults to None.
        dtype (torch.dtype | None, optional): overrides data type from ``tensor``
            if specified. Defaults to None.
        layout (torch.layout | None, optional): overrides layout from ``tensor``
            if specified. Defaults to None.
        device (torch.device | None, optional): overrides device from ``tensor``
            if specified. Defaults to None.
        requires_grad (bool | None, optional): overrides gradient requirement from
            ``tensor`` if specified. Defaults to None.

    Returns:
        torch.Tensor: uninitialized tensor like ``tensor``, modified by parameters.

    Note:
        To construct a scalar, set ``shape`` to ``((),)``.
    """
    shape = tensor.shape if shape is None else shape
    dtype = tensor.dtype if dtype is None else dtype
    layout = tensor.layout if layout is None else layout
    device = tensor.device if device is None else device
    requires_grad = tensor.requires_grad if requires_grad is None else requires_grad

    return torch.empty(
        *shape, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad
    )


def full(
    tensor: torch.Tensor,
    value: bool | int | float | complex,
    *,
    shape: tuple[int] | torch.Size | None = None,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    requires_grad: bool | None = None
) -> torch.Tensor:
    r"""Returns a tensor based on input filled with specified value.

    Args:
        tensor (torch.Tensor): determines default output properties.
        value (bool | int | float | complex): value with to fill the output.
        shape (tuple[int] | torch.Size | None, optional): overrides shape from
            ``tensor`` if specified. Defaults to None.
        dtype (torch.dtype | None, optional): overrides data type from ``tensor``
            if specified. Defaults to None.
        layout (torch.layout | None, optional): overrides layout from ``tensor``
            if specified. Defaults to None.
        device (torch.device | None, optional): overrides device from ``tensor``
            if specified. Defaults to None.
        requires_grad (bool | None, optional): overrides gradient requirement from
            ``tensor`` if specified. Defaults to None.

    Returns:
        torch.Tensor: tensor like ``tensor``, modified by parameters, filled
        with ``value``.

    Note:
        To construct a scalar, set ``shape`` to ``((),)``.
    """
    shape = tensor.shape if shape is None else shape
    dtype = tensor.dtype if dtype is None else dtype
    layout = tensor.layout if layout is None else layout
    device = tensor.device if device is None else device
    requires_grad = tensor.requires_grad if requires_grad is None else requires_grad

    return torch.full(
        *shape,
        fill_value=value,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad
    )


def uniform(
    tensor: torch.Tensor,
    *,
    shape: tuple[int] | torch.Size | None = None,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    requires_grad: bool | None = None,
    generator: torch.Generator | None = None
) -> torch.Tensor:
    r"""Returns a tensor based on input filled with random values sampled uniformly.

    Args:
        tensor (torch.Tensor): determines default output properties.
        shape (tuple[int] | torch.Size | None, optional): overrides shape from
            ``tensor`` if specified. Defaults to None.
        dtype (torch.dtype | None, optional): overrides data type from ``tensor``
            if specified. Defaults to None.
        layout (torch.layout | None, optional): overrides layout from ``tensor``
            if specified. Defaults to None.
        device (torch.device | None, optional): overrides device from ``tensor``
            if specified. Defaults to None.
        requires_grad (bool | None, optional): overrides gradient requirement from
            ``tensor`` if specified. Defaults to None.
        generator (torch.Generator | None, optional): pseudorandom number generator
            for sampling. Defaults to None.

    Returns:
        torch.Tensor: tensor like ``tensor``, modified by parameters, filled
        with ``value``.

    Note:
        To construct a scalar, set ``shape`` to ``((),)``.

    See Also:
        See :py:func:`torch.rand` for the function which this extends.
    """
    shape = tensor.shape if shape is None else shape
    dtype = tensor.dtype if dtype is None else dtype
    layout = tensor.layout if layout is None else layout
    device = tensor.device if device is None else device
    requires_grad = tensor.requires_grad if requires_grad is None else requires_grad

    return torch.rand(
        *shape,
        generator=generator,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad
    )


def normal(
    tensor: torch.Tensor,
    *,
    shape: tuple[int] | torch.Size | None = None,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    requires_grad: bool | None = None,
    generator: torch.Generator | None = None
) -> torch.Tensor:
    r"""Returns a tensor based on input filled with random values sampled normally.

    Args:
        tensor (torch.Tensor): determines default output properties.
        shape (tuple[int] | torch.Size | None, optional): overrides shape from
            ``tensor`` if specified. Defaults to None.
        dtype (torch.dtype | None, optional): overrides data type from ``tensor``
            if specified. Defaults to None.
        layout (torch.layout | None, optional): overrides layout from ``tensor``
            if specified. Defaults to None.
        device (torch.device | None, optional): overrides device from ``tensor``
            if specified. Defaults to None.
        requires_grad (bool | None, optional): overrides gradient requirement from
            ``tensor`` if specified. Defaults to None.
        generator (torch.Generator | None, optional): pseudorandom number generator
            for sampling. Defaults to None.

    Returns:
        torch.Tensor: tensor like ``tensor``, modified by parameters, filled
        with ``value``.

    Note:
        To construct a scalar, set ``shape`` to ``((),)``.

    See Also:
        See :py:func:`torch.randn` for the function which this extends.
    """
    shape = tensor.shape if shape is None else shape
    dtype = tensor.dtype if dtype is None else dtype
    layout = tensor.layout if layout is None else layout
    device = tensor.device if device is None else device
    requires_grad = tensor.requires_grad if requires_grad is None else requires_grad

    return torch.randn(
        *shape,
        generator=generator,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad
    )
