from typing import TypeVar

T = TypeVar("T")


def identity(inputs: T, **kwargs) -> T:
    r"""Placeholder function that returns the input.

    Args:
        inputs (T): input and return value.

    Returns:
        T: value given as input.
    """
    return inputs


def varidentity(*inputs: T, **kwargs) -> tuple[T, ...]:
    r"""Placeholder variadic function that returns the input.

    Args:
        *inputs (T): input and return value.

    Returns:
        tuple[T, ...]: value given as input.
    """
    return inputs


def tuplewrap(inputs: T, **kwargs) -> tuple[T]:
    r"""Placeholder function that wraps the input in a tuple.

    Args:
        inputs (T): input and return value.

    Returns:
        tuple[T]: value given as input.
    """
    return (inputs,)
