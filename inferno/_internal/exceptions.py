from typing import Any, Literal


def numeric_limit(
    name: str,
    val: Any,
    lim: Any,
    op: Literal["lt", "lte", "gt", "gte", "neq"],
    cast: type | None = None,
) -> Any:
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


def instance_of(
    name: str,
    obj: Any,
    typespec: type | tuple[type, ...],
) -> None:
    # inner function for type names
    def typename(atype):
        if hasattr(atype, '__name__'):
            return atype.__name__
        else:
            return atype

    # test if object meets type specification
    if not isinstance(obj, typespec):
        # type specification is tuple of types
        if not isinstance(typespec, type):
            raise TypeError(
                f"{typename(obj)} {name} must be an instance of one of "
                f"{', '.join(typename(ts) for ts in typespec)}."
            )

        # type specification is type
        else:
            raise TypeError(
                f"{type(obj).__name__} {name} must be an instance of "
                f"{typespec.__name__}."
            )


def attr_members(
    name: str,
    obj: Any,
    attr: str | tuple[str, ...],
) -> None:
    # inner function for type names
    def typename(atype):
        if hasattr(atype, '__name__'):
            return atype.__name__
        else:
            return atype

    # single attribute case
    if isinstance(attr, str):
        if not hasattr(obj, attr):
            raise RuntimeError(
                f"{typename(obj)} {name} must have the " f"attribute {attr}."
            )

    # multiple attribute case
    else:
        if not all(map(lambda a: hasattr(obj, a), attr)):
            raise RuntimeError(
                f"{typename(obj)} {name} must have the "
                f"attributes {', '.join(attr)}."
            )
