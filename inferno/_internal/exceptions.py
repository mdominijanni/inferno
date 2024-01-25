from typing import Any, Literal


def numeric_limit(
    name: str,
    val: Any,
    lim: Any,
    op: Literal["lt", "lte", "gt", "gte", "neq"],
    cast: type | None = None,
) -> Any:
    r"""Casts a value and compares to a limit, raising an error if the comparison is false.

    Args:
        name (str): variable name string for the error message.
        val (Any): value being tested.
        lim (Any): comparison value.
        op (Literal["lt", "lte", "gt", "gte", "neq"]): operation for comparison.
        cast (type | None, optional): type to cast value to. Defaults to None.

    Returns:
        Any: casted value.
    """
    # cast value if a type is given
    if val is not None:
        val = cast(val)

    # perform comparison
    match str(op).lower():
        # less than
        case "lt":
            # return cast value if valid
            if val < lim:
                return val

            # special message for testing around zero
            if lim == 0:
                raise ValueError(f"{name} must be negative, received {val}.")
            else:
                raise ValueError(f"{name} must be less than {lim}, received {val}.")

        # less than or equal
        case "lte":
            # return cast value if valid
            if val <= lim:
                return val

            # special message for testing around zero
            if lim == 0:
                raise ValueError(f"{name} must be nonpositive, received {val}.")
            else:
                raise ValueError(
                    f"{name} must be less than or equal to {lim}, received {val}."
                )

        # greater than
        case "gt":
            # return cast value if valid
            if val > lim:
                return val

            # special message for testing around zero
            if lim == 0:
                raise ValueError(f"{name} must be positive, received {val}.")
            else:
                raise ValueError(f"{name} must be greater than {lim}, received {val}.")

        # greater than or equal
        case "gte":
            # return cast value if valid
            if val >= lim:
                return val

            # special message for testing around zero
            if lim == 0:
                raise ValueError(f"{name} must be nonnegative, received {val}.")
            else:
                raise ValueError(
                    f"{name} must be greater than or equal to {lim}, received {val}."
                )

        # not equal
        case "neq":
            # return cast value if valid
            if val != lim:
                return val

            # special message for testing around zero
            if lim == 0:
                raise ValueError(f"{name} must be nonzero, received {val}.")
            else:
                raise ValueError(f"{name} must not be equal to {lim}, received {val}.")

        # invalid operator
        case _:
            raise ValueError(
                "operator must be one of "
                "'lt, 'lte', 'gt', 'gte', or 'neq', "
                f"received '{str(op).lower()}'."
            )


def numeric_interval(
    name: str,
    val: Any,
    lower: Any,
    upper: Any,
    op: Literal["closed", "open", "left-open", "right-open", "c", "o", "lo", "ro"],
    cast: type | None = None,
) -> Any:
    """Casts a value and compares to a range, raising an error if it falls outside.

    Args:
        name (str): variable name string for the error message.
        val (Any): value being tested.
        lower (Any): lower bound of the range.
        upper (Any): upper bound of the range.
        op (Literal["closed", "open", "left-open", "right-open", "c", "o", "lo", "ro"]):
            kind of range to test.
        cast (type | None, optional): type used for casting. Defaults to None.

    Returns:
        Any: casted value.
    """
    # cast value if a type is given
    if val is not None:
        val = cast(val)

    # perform comparison
    match str(op).lower():
        # less than
        case "closed" | "c":
            # return cast value if valid
            if lower <= val <= upper:
                return val

            raise ValueError(
                f"{name} must exist on the interval [{lower}, {upper}], "
                f"received {val}."
            )

        # less than or equal
        case "open" | "o":
            # return cast value if valid
            if lower < val < upper:
                return val

            raise ValueError(
                f"{name} must exist on the interval ({lower}, {upper}), "
                f"received {val}."
            )

        # greater than
        case "left-open" | "lo":
            # return cast value if valid
            if lower < val <= upper:
                return val

            raise ValueError(
                f"{name} must exist on the interval ({lower}, {upper}], "
                f"received {val}."
            )

        # greater than or equal
        case "right-open" | "ro":
            # return cast value if valid
            if lower <= val < upper:
                return val

            raise ValueError(
                f"{name} must exist on the interval [{lower}, {upper}), "
                f"received {val}."
            )

        # invalid operator
        case _:
            raise ValueError(
                "operator must be one of "
                "'closed', 'open', 'left-open', 'right-open', 'c', 'o', 'lo', or 'ro', "
                f"received '{str(op).lower()}'."
            )


def instance_of(
    name: str,
    obj: Any,
    typespec: type | tuple[type, ...],
) -> None:
    """Checks if an object is an instance of a type or tuple thereof and raise an error if not.

    Args:
        name (str): variable name string for the error message.
        obj (Any): object being tested.
        typespec (type | tuple[type, ...]): type or tuple of types being tested.
    """

    # inner function for type names
    def typename(atype):
        if hasattr(atype, "__name__"):
            return atype.__name__
        else:
            return atype

    # test if object meets type specification
    if not isinstance(obj, typespec):
        # type specification is tuple of types
        if isinstance(typespec, tuple):
            raise TypeError(
                f"{typename(obj)} {name} must be an instance of one of "
                f"{', '.join(typename(ts) for ts in typespec)}."
            )

        # type specification is not a tuple
        else:
            raise TypeError(
                f"{typename(obj).__name__} {name} must be an instance of "
                f"{typename(typespec)}."
            )


def attr_members(
    name: str,
    obj: Any,
    *attr: str,
) -> None:
    """Checks if an object contains one or more attributes and raises an error if it doesn't.

    Args:
        name (str): variable name string for the error message.
        obj (Any): object being tested.
        *attr (str): attributes being checked for.
    """

    sep = "', '"
    if not all(map(lambda a: hasattr(obj, a), attr)):
        raise RuntimeError(
            f"{type(obj).__name__} {name} is missing the "
            f"attributes: '{sep.join(filter(lambda a: hasattr(obj, a), attr))}'."
        )
