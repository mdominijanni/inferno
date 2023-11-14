from functools import reduce
import torch
from typing import Any


def areinstances(classinfo, *objects, mode="all"):
    match mode.lower():
        case "all":
            return all((isinstance(obj, classinfo) for obj in objects))
        case "any":
            return any((isinstance(obj, classinfo) for obj in objects))
        case _:
            raise ValueError(f"invalid mode '{mode}' given, must be 'all' or 'any'.")


def newtensor(obj):
    try:
        return obj.clone().detach()
    except AttributeError:
        return torch.tensor(obj)


def rgetattr(obj: object, attr: str, *args) -> Any:
    r"""Accesses and returns an object attribute recursively using dot notation.

    For example, if we have an object ``obj`` and a string ``"so1.so2.so3"``, this function will
    retrieve ``obj.so1.so2.so3``. This is performed recursively using ``getattr()``.

    Args:
        obj (object): object from which to retrieve the nested attribute.
        attr (str): string in dot notation for the nested attribute to retrieve, excluding the initial dot.
        default (Any, optional): if specified, including with None, it will be returned if attr is not found.

    Returns:
        Any: nested attribute of ``obj`` specified by ``attr``,
        or ``default`` if it is specified and ``attr`` is not found.

    Note:
        If a default is specified, it will be returned if at any point in the chain the attribute is not found.
    """

    def getattr_(obj, attr):
        return getattr(obj, attr, *args)

    return reduce(getattr_, [obj] + attr.split("."))


def rsetattr(obj: object, attr: str, val: Any):
    r"""Sets an object attribute recursively using dot notation.

    For example, if we have an object ``obj`` and a string ``"so1.so2.so3"``, to which some value
    ``v`` is being assigned, this function will retrieve ``obj.so1.so2`` recursively using ``getattr()``,
    then assign ``v`` to ``so3`` in the object ``so2`` using ``setattr()``.

    Args:
        obj (object): object to which the nested attribute will be set.
        attr (str): string in dot notation for the nested attribute to set, excluding the initial dot.
        val (Any): value to which the attribute will be set.
    """
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)
