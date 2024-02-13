from functools import reduce
import torch
from typing import Any


def regroup(
    flatseq: tuple[Any, ...], groups: tuple[int | tuple, ...]
) -> tuple[Any, ...]:
    return tuple(
        flatseq[g] if isinstance(g, int) else regroup(flatseq, g) for g in groups
    )


def newtensor(obj: Any) -> torch.Tensor:
    r"""Creates a new tensor from an existing object, tensor or otherwise.

    Args:
        obj (Any): object off of which new tensor should be constructed.

    Returns:
       torch.Tensor: newly constructed tensor.
    """
    try:
        return obj.clone().detach()
    except AttributeError:
        return torch.tensor(obj)


def rgetattr(obj: object, attr: str, *default) -> Any:
    r"""Accesses and returns an object attribute recursively using dot notation.

    For example, if we have an object ``obj`` and a string ``"so1.so2.so3"``,
    this function will retrieve ``obj.so1.so2.so3``. This is performed recursively
    using ``getattr()``.

    Args:
        obj (object): object from which to retrieve the nested attribute.
        attr (str): string in dot notation for the nested attribute to retrieve,
            excluding the initial dot.
        *default (Any, optional): if specified, including with None, it will be
            returned if attr is not found.

    Returns:
        Any: nested attribute of ``obj`` specified by ``attr``,
        or ``default`` if it is specified and ``attr`` is not found.

    Note:
        If a default is specified, it will be returned if at any point in the chain,
        the attribute is not found. If multiple values are passed with ``*default``,
        only the first will be used.
    """

    try:
        return reduce(getattr, [obj] + attr.split("."))

    except AttributeError:
        if default:
            return default[0]
        else:
            raise


def rsetattr(obj: object, attr: str, val: Any):
    r"""Sets an object attribute recursively using dot notation.

    For example, if we have an object ``obj`` and a string ``"so1.so2.so3"``,
    to which some value ``v`` is being assigned, this function will retrieve
    ``obj.so1.so2`` recursively using ``getattr()``,  then assign ``v`` to ``so3``
    in the object ``so2`` using ``setattr()``.

    Args:
        obj (object): object to which the nested attribute will be set.
        attr (str): string in dot notation for the nested attribute to set,
            excluding the initial dot.
        val (Any): value to which the attribute will be set.
    """
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)
