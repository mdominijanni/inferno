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
        shape (tuple[int] | torch.Size | None, optional): overrides shape from ``tensor`` if specified.
            Defaults to None.
        dtype (torch.dtype | None, optional): overrides data type from ``tensor`` if specified.
            Defaults to None.
        layout (torch.layout | None, optional): overrides layout from ``tensor`` if specified.
            Defaults to None.
        device (torch.device | None, optional): overrides device from ``tensor`` if specified.
            Defaults to None.
        requires_grad (bool | None, optional): overrides gradient requirement from ``tensor`` if specified.
            Defaults to None.

    Returns:
        torch.Tensor: tensor like the input ``tensor`` modified by parameters, filled with scalar 0.
    """
    shape = tensor.shape if shape is None else shape
    dtype = tensor.dtype if dtype is None else dtype
    layout = tensor.layout if layout is None else layout
    device = tensor.device if device is None else device
    requires_grad = tensor.requires_grad if requires_grad is None else requires_grad

    return torch.zeros(
        *shape,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad
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
        shape (tuple[int] | torch.Size | None, optional): overrides shape from ``tensor`` if specified.
            Defaults to None.
        dtype (torch.dtype | None, optional): overrides data type from ``tensor`` if specified.
            Defaults to None.
        layout (torch.layout | None, optional): overrides layout from ``tensor`` if specified.
            Defaults to None.
        device (torch.device | None, optional): overrides device from ``tensor`` if specified.
            Defaults to None.
        requires_grad (bool | None, optional): overrides gradient requirement from ``tensor`` if specified.
            Defaults to None.

    Returns:
        torch.Tensor: tensor like the input ``tensor`` modified by parameters, filled with scalar 1.
    """
    shape = tensor.shape if shape is None else shape
    dtype = tensor.dtype if dtype is None else dtype
    layout = tensor.layout if layout is None else layout
    device = tensor.device if device is None else device
    requires_grad = tensor.requires_grad if requires_grad is None else requires_grad

    return torch.ones(
        *shape,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad
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
        shape (tuple[int] | torch.Size | None, optional): overrides shape from ``tensor`` if specified.
            Defaults to None.
        dtype (torch.dtype | None, optional): overrides data type from ``tensor`` if specified.
            Defaults to None.
        layout (torch.layout | None, optional): overrides layout from ``tensor`` if specified.
            Defaults to None.
        device (torch.device | None, optional): overrides device from ``tensor`` if specified.
            Defaults to None.
        requires_grad (bool | None, optional): overrides gradient requirement from ``tensor`` if specified.
            Defaults to None.

    Returns:
        torch.Tensor: uninitialized tensor like the input ``tensor`` modified by parameters.
    """
    shape = tensor.shape if shape is None else shape
    dtype = tensor.dtype if dtype is None else dtype
    layout = tensor.layout if layout is None else layout
    device = tensor.device if device is None else device
    requires_grad = tensor.requires_grad if requires_grad is None else requires_grad

    return torch.empty(
        *shape,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad
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
        value (bool | int | float | complex): scalar value with which to fill output tensor.
        shape (tuple[int] | torch.Size | None, optional): overrides shape from ``tensor`` if specified.
            Defaults to None.
        dtype (torch.dtype | None, optional): overrides data type from ``tensor`` if specified.
            Defaults to None.
        layout (torch.layout | None, optional): overrides layout from ``tensor`` if specified.
            Defaults to None.
        device (torch.device | None, optional): overrides device from ``tensor`` if specified.
            Defaults to None.
        requires_grad (bool | None, optional): overrides gradient requirement from ``tensor`` if specified.
            Defaults to None.

    Returns:
        torch.Tensor: tensor like the input ``tensor`` modified by parameters, filled with scalar ``value``.
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
