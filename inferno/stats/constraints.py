from functools import reduce
from itertools import chain
import torch
from typing import Callable


def validate(
    var: bool | int | float | complex | torch.Tensor | None,
    test: Callable[[torch.Tensor], torch.Tensor],
) -> bool | torch.Tensor | None:
    r"""Checks validity of a variable.

    Args:
        var (bool | int | float | complex | torch.Tensor | None):
            variable to test validity of.
        test (Callable[[torch.Tensor], torch.Tensor]): validity test with
            boolean result.

    Returns:
        torch.Tensor | bool | None: if the variable is valid.
    """
    # none if no test variable is provided
    if var is None:
        return None

    # variable was provided
    else:
        # convert to scalar tensor if non-tensor
        if not isinstance(var, torch.Tensor):
            var = torch.tensor(var)

        # return builtin if tensor is dimensionless
        if var.ndim == 0:
            return test(var).item()
        else:
            return test(var)


def _conj(*variables: bool | torch.Tensor | None) -> torch.Tensor:
    # non-scalar tensors must be on the same device
    return reduce(
        lambda x, y: torch.logical_and(x, y),
        map(
            lambda x: x if isinstance(x, torch.Tensor) else torch.tensor(x),
            filter(lambda x: x is not None, chain(variables, [True])),
        ),
    )


def _disj(*variables: bool | torch.Tensor | None) -> torch.Tensor:
    # non-scalar tensors must be on the same device
    return reduce(
        lambda x, y: torch.logical_or(x, y),
        map(
            lambda x: x if isinstance(x, torch.Tensor) else torch.tensor(x),
            filter(lambda x: x is not None, chain(variables, [False])),
        ),
    )


def conjunction(
    var: bool | int | float | complex | torch.Tensor | None,
    *constraints: Callable[
        [bool | int | float | complex | torch.Tensor | None], bool | torch.Tensor | None
    ],
) -> bool | torch.Tensor | None:
    return validate(var, lambda x: _conj(*map(lambda c: c(x), constraints)))


def disjunction(
    var: bool | int | float | complex | torch.Tensor | None,
    *constraints: Callable[
        [bool | int | float | complex | torch.Tensor | None], bool | torch.Tensor | None
    ],
) -> bool | torch.Tensor | None:
    return validate(var, lambda x: _disj(*map(lambda c: c(x), constraints)))


def negate(var: bool | torch.Tensor | None) -> bool | torch.Tensor | None:
    return validate(var, lambda x: torch.logical_not(x))


def splitcomplexand(
    var: bool | int | float | complex | torch.Tensor | None,
    proc: Callable[[torch.Tensor], torch.Tensor],
) -> bool | torch.Tensor | None:
    def test(x):
        try:
            return torch.logical_and(proc(x.imag), proc(x.real))
        except RuntimeError:
            return proc(x)

    return validate(var, test)


def splitcomplexor(
    var: bool | int | float | complex | torch.Tensor | None,
    proc: Callable[[torch.Tensor], torch.Tensor],
) -> bool | torch.Tensor | None:
    def test(x):
        try:
            return torch.logical_or(proc(x.imag), proc(x.real))
        except RuntimeError:
            return proc(x)

    return validate(var, test)


def greater(
    var: bool | int | float | complex | torch.Tensor | None,
    comparator: float,
) -> bool | torch.Tensor | None:
    return splitcomplexand(var, lambda x: x > comparator)


def lesser(
    var: bool | int | float | complex | torch.Tensor | None,
    comparator: float,
) -> bool | torch.Tensor | None:
    return splitcomplexand(var, lambda x: x < comparator)


def nonnan(
    var: bool | int | float | complex | torch.Tensor | None,
) -> bool | torch.Tensor | None:
    return validate(var, lambda x: x == x)


def ltposinf(
    var: bool | int | float | complex | torch.Tensor | None,
) -> bool | torch.Tensor | None:
    return lesser(var, float("inf"))


def gtneginf(
    var: bool | int | float | complex | torch.Tensor | None,
) -> bool | torch.Tensor | None:
    return greater(var, float("-inf"))


def finite(
    var: bool | int | float | complex | torch.Tensor | None,
) -> bool | torch.Tensor | None:
    return conjunction(var, ltposinf, gtneginf)


def positive(
    var: bool | int | float | complex | torch.Tensor | None,
) -> bool | torch.Tensor | None:
    return greater(var, 0)


def negative(
    var: bool | int | float | complex | torch.Tensor | None,
) -> bool | torch.Tensor | None:
    return lesser(var, 0)


def nonnegative(
    var: bool | int | float | complex | torch.Tensor | None,
) -> bool | torch.Tensor | None:
    return negate(negative(var))


def nonpositive(
    var: bool | int | float | complex | torch.Tensor | None,
) -> bool | torch.Tensor | None:
    return negate(positive(var))


def zvimag(
    var: bool | int | float | complex | torch.Tensor | None,
) -> bool | torch.Tensor | None:
    def test(x):
        try:
            return x.imag == 0
        except RuntimeError:
            return torch.ones_like(x, dtype=torch.bool)

    return validate(var, test)


def nonfractional(
    var: bool | int | float | complex | torch.Tensor | None,
) -> bool | torch.Tensor | None:
    return splitcomplexand(var, lambda x: x % 1 == 0)


def real(
    var: bool | int | float | complex | torch.Tensor | None,
) -> bool | torch.Tensor | None:
    return conjunction(var, finite, zvimag)


def posreal(
    var: bool | int | float | complex | torch.Tensor,
) -> bool | torch.Tensor:
    return conjunction(var, real, positive)


def negreal(
    var: bool | int | float | complex | torch.Tensor,
) -> bool | torch.Tensor:
    return conjunction(var, real, negative)


def nonposreal(
    var: bool | int | float | complex | torch.Tensor,
) -> bool | torch.Tensor:
    return conjunction(var, real, nonpositive)


def nonnegreal(
    var: bool | int | float | complex | torch.Tensor | None,
) -> bool | torch.Tensor | None:
    return conjunction(var, real, nonnegative)


def gaussinteger(
    var: bool | int | float | complex | torch.Tensor | None,
) -> bool | torch.Tensor | None:
    return conjunction(var, finite, nonfractional)


def integer(
    var: bool | int | float | complex | torch.Tensor | None,
) -> bool | torch.Tensor | None:
    return conjunction(var, real, nonfractional)


def posinteger(
    var: bool | int | float | complex | torch.Tensor | None,
) -> bool | torch.Tensor | None:
    return conjunction(var, integer, positive)


def neginteger(
    var: bool | int | float | complex | torch.Tensor | None,
) -> bool | torch.Tensor | None:
    return conjunction(var, integer, negative)


def nonposinteger(
    var: bool | int | float | complex | torch.Tensor | None,
) -> bool | torch.Tensor | None:
    return conjunction(var, integer, nonpositive)


def nonneginteger(
    var: bool | int | float | complex | torch.Tensor | None,
) -> bool | torch.Tensor | None:
    return conjunction(var, integer, nonnegative)


def permissive(
    var: bool | int | float | complex | torch.Tensor | None,
    constraint: Callable[
        [bool | int | float | complex | torch.Tensor | None], bool | torch.Tensor | None
    ],
) -> bool | torch.Tensor | None:
    return disjunction(var, constraint, lambda x: negate(nonnan(x)))
