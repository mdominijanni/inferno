from typing import Protocol, TypeVar

T = TypeVar("T")
K = TypeVar("K")


class OneToOne(Protocol[T]):
    r"""Callable type taking one input of a type
    and returning one output of the same type.
    """

    def __call__(self, inputs: T) -> T:
        r"""Callback protocol function."""
        ...


class ManyToOne(Protocol[T]):
    r"""Callable type taking an arbitrary number of inputs of a type
    and returning one output of the same type.
    """

    def __call__(self, *inputs: T) -> T:
        r"""Callback protocol function."""
        ...


class OneToMany(Protocol[T]):
    r"""Callable type taking one input of a type
    and returning an arbitrary number of outputs of the same type.
    """

    def __call__(self, inputs: T) -> tuple[T]:
        r"""Callback protocol function."""
        ...


class ManyToMany(Protocol[T]):
    r"""Callable type taking an arbitrary number of inputs of a type
    and returning the same number of outputs of the same type.
    """

    def __call__(self, *inputs: T) -> tuple[T]:
        r"""Callback protocol function."""
        ...


class OneToOneMethod(Protocol[K, T]):
    r"""Callable type taking a module and one input of a type
    and returning one output of the same type.
    """

    def __call__(self, module: K, inputs: T) -> T:
        r"""Callback protocol function."""
        ...


class ManyToOneMethod(Protocol[K, T]):
    r"""Callable type taking a module and an arbitrary number of inputs of a type
    and returning one output of the same type.
    """

    def __call__(self, module: K, *inputs: T) -> T:
        r"""Callback protocol function."""
        ...


class OneToManyMethod(Protocol[K, T]):
    r"""Callable type taking a module and one input of a type
    and returning an arbitrary number of outputs of the same type.
    """

    def __call__(self, module: K, inputs: T) -> tuple[T]:
        r"""Callback protocol function."""
        ...


class ManyToManyMethod(Protocol[K, T]):
    r"""Callable type taking a module and an arbitrary number of inputs of a type
    and returning the same number of outputs of the same type.
    """

    def __call__(self, module: K, *inputs: T) -> tuple[T]:
        r"""Callback protocol function."""
        ...
