import torch


def zeros(
    tensor: torch.Tensor,
    shape: tuple[int] | torch.Size | None = None,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    requires_grad: bool | None = None
):
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


def empty(
    tensor: torch.Tensor,
    shape: tuple[int] | torch.Size | None = None,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    requires_grad: bool | None = None
):
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
