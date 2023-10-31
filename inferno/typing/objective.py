from typing import Protocol, TypeVar

K = TypeVar("K")


class ShapeConstructor(Protocol[K]):
    r"""Callable type taking variadic integer aruments and returning an object.
    """

    def __call__(self, *shape: int) -> K:
        r"""Callback protocol function."""
        ...
