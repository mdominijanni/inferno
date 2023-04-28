from typing import Any

import torch


def get_nested_attr(obj: object, attr: str) -> Any:
    """Accesses and returns an object attribute recursively using dot notation.

    For example, if we have an object `obj` and a string `"so1.so2.so3"`, this function will
    retrieve `obj.so1.so2.so3`. This is performed recursively using `__getattr__()` and
    `__getattribute__()` when the former fails.

    Args:
        obj (object): object from which to retrieve the nested attribute.
        attr (str): string in dot notation for the nested attribute to retrieve, excluding the initial dot.

    Raises:
        AttributeError: object does not contain the specified attribute.

    Returns:
        Any: nested attribute of `obj` specified by `attr`.
    """
    def _multigetattr(obj, attr):
        try:
            return obj.__getattr__(attr)
        except AttributeError:
            try:
                return obj.__getattribute__(attr)
            except AttributeError:
                raise AttributeError(f'\'{type(obj).__name__}\' object has no attribute \'{attr}\'')

    pre, _, post = attr.partition('.')
    if post == '':
        return _multigetattr(obj, pre)
    else:
        return get_nested_attr(_multigetattr(obj, pre), post)


def create_tensor(obj: object) -> torch.Tensor:
    if isinstance(obj, torch.Tensor):
        return obj.clone().detach()
    else:
        return torch.tensor(obj)


def agnostic_mean(tensor: torch.Tensor, dim: int | None = None, keepdim: bool = False) -> torch.Tensor:
    if dim is None:
        return torch.mean(tensor.float()).type(tensor.dtype)
    else:
        return torch.mean(tensor.float(), dim=dim, keepdim=keepdim).type(tensor.dtype)
