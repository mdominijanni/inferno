from types import UnionType
from typing import Any, Literal


def typenameof(type_: type | UnionType) -> str:
    if isinstance(type_, UnionType):
        " | ".join(typenameof(st) for st in type_.__args__)
    else:
        if hasattr(type_, "__name__"):
            return type_.__name__
        else:
            return str(type_)


def numeric_limit(
    name: str,
    val: Any,
    lim: Any,
    op: Literal["lt", "lte", "gt", "gte", "neq"],
    cast: type | None = None,
    *,
    wraptxt: str | tuple[str, str] = "'",
    wrapnum: str | tuple[str, str] = "",
) -> tuple[Any, ValueError | None]:
    r"""Casts a value and compares it against a limit, returns an error if the comparison is false.

    Args:
        name (str): variable name string for the error message.
        val (Any): value being tested.
        lim (Any): comparison value.
        op (Literal["lt", "lte", "gt", "gte", "neq"]): operation for comparison.
        cast (type | None, optional): type to cast value to. Defaults to None.
        wraptxt (str | tuple[str, str], optional): string to insert around variable name
            in the message, separate prepend/postpend if a tuple. Defaults to "'".
        wrapnum (str | tuple[str, str], optional): string to insert around the value and
            limit in the message, separate prepend/postpend if a tuple. Defaults to "".

    Returns:
        tuple[Any, ValueError | None]: casted value and an error if generated.
    """
    # cast value if a type is given
    if cast is not None:
        val = cast(val)

    # create message strings
    wxl, wxr = (wraptxt, wraptxt) if isinstance(wraptxt, str) else wraptxt
    wnl, wnr = (wrapnum, wrapnum) if isinstance(wrapnum, str) else wrapnum
    prefix = f"{wxl}{name}{wxr} must be"
    negprefix = f"{wxl}{name}{wxr} must not be"
    limstr = f"{wnl}{lim}{wnr}"
    suffix = f"received {wnl}{val}{wnr}"

    # perform comparison
    match str(op).lower():
        # less than
        case "lt":
            # no error case
            if val < lim:
                return val, None

            # special message for testing around zero
            if lim == 0:
                return val, ValueError(f"{prefix} negative, {suffix}.")

            # normal message
            else:
                return val, ValueError(f"{prefix} less than {limstr}, {suffix}.")

        # less than or equal
        case "lte":
            # no error case
            if val <= lim:
                return val, None

            # special message for testing around zero
            if lim == 0:
                return val, ValueError(f"{prefix} nonpositive, {suffix}.")

            # normal message
            else:
                return val, ValueError(
                    f"{prefix} less than or equal to {limstr}, {suffix}."
                )

        # greater than
        case "gt":
            # no error case
            if val > lim:
                return val, None

            # special message for testing around zero
            if lim == 0:
                return val, ValueError(f"{prefix} positive, {suffix}.")

            # normal message
            else:
                return val, ValueError(f"{prefix} greater than {limstr}, {suffix}.")

        # greater than or equal
        case "gte":
            # no error case
            if val >= lim:
                return val, None

            # special message for testing around zero
            if lim == 0:
                return val, ValueError(f"{prefix} nonnegative, {suffix}.")

            # normal message
            else:
                return val, ValueError(
                    f"{prefix} greater than or equal to {limstr}, {suffix}."
                )

        # not equal
        case "neq":
            # no error case
            if val != lim:
                return val, None

            # special message for testing around zero
            if lim == 0:
                return val, ValueError(f"{prefix} nonzero, {suffix}.")

            # normal message
            else:
                return val, ValueError(f"{negprefix} equal to {limstr}, {suffix}.")

        # invalid operator
        case _:
            raise ValueError(
                "operator must be one of "
                "'lt, 'lte', 'gt', 'gte', or 'neq', "
                f"received '{str(op).lower()}'."
            )


def multiple_numeric_limit(
    name: str,
    vals: tuple[Any, ...],
    lim: Any,
    op: Literal["lt", "lte", "gt", "gte", "neq"],
    cast: type | None = None,
    allow_empty: bool = True,
    *,
    wraptxt: str | tuple[str, str] = "'",
    wrapnum: str | tuple[str, str] = "",
) -> tuple[Any, ValueError | None]:
    r"""Casts value in a sequence and compares each against a limit, returns an error if the comparison is false.

    Args:
        name (str): variable name string for the error message.
        vals (tuple[Any, ...]): sequence of values being tested.
        lim (Any): comparison value.
        op (Literal["lt", "lte", "gt", "gte", "neq"]): operation for comparison.
        cast (type | None, optional): type to cast value to. Defaults to None.
        allow_empty (bool, optional): if an empty sequence of values is valid.
            Defaults to True.
        wraptxt (str | tuple[str, str], optional): string to insert around variable name
            in the message, separate prepend/postpend if a tuple. Defaults to "'".
        wrapnum (str | tuple[str, str], optional): string to insert around the values
            and limit in the message, separate prepend/postpend if a tuple.
            Defaults to "".

    Returns:
        tuple[Any, ValueError | None]: casted value and an error if generated.
    """
    # cast value if a type is given
    if cast is not None:
        vals = tuple(cast(v) for v in vals)

    # create message strings
    wxl, wxr = (wraptxt, wraptxt) if isinstance(wraptxt, str) else wraptxt
    wnl, wnr = (wrapnum, wrapnum) if isinstance(wrapnum, str) else wrapnum
    prefix = f"all elements of {wxl}{name}{wxr} must be"
    negprefix = f"all elements of {wxl}{name}{wxr} must not be"
    limstr = f"{wnl}{lim}{wnr}"
    suffix = f"received {wnl}({','.join(str(v) for v in vals)}){wnr}"

    # check for empty condition
    if len(vals) == 0:
        if allow_empty:
            return vals, None
        else:
            return vals, ValueError(f"{wxl}{name}{wxr} cannot be empty.")

    # perform comparison
    match str(op).lower():
        # less than
        case "lt":
            # no error case
            if all(v < lim for v in vals):
                return vals, None

            # special message for testing around zero
            if lim == 0:
                return vals, ValueError(f"{prefix} negative, {suffix}.")

            # normal message
            else:
                return vals, ValueError(f"{prefix} less than {limstr}, {suffix}.")

        # less than or equal
        case "lte":
            # no error case
            if all(v <= lim for v in vals):
                return vals, None

            # special message for testing around zero
            if lim == 0:
                return vals, ValueError(f"{prefix} nonpositive, {suffix}.")

            # normal message
            else:
                return vals, ValueError(
                    f"{prefix} less than or equal to {limstr}, {suffix}."
                )

        # greater than
        case "gt":
            # no error case
            if all(v > lim for v in vals):
                return vals, None

            # special message for testing around zero
            if lim == 0:
                return vals, ValueError(f"{prefix} positive, {suffix}.")

            # normal message
            else:
                return vals, ValueError(f"{prefix} greater than {limstr}, {suffix}.")

        # greater than or equal
        case "gte":
            # no error case
            if all(v >= lim for v in vals):
                return vals, None

            # special message for testing around zero
            if lim == 0:
                return vals, ValueError(f"{prefix} nonnegative, {suffix}.")

            # normal message
            else:
                return vals, ValueError(
                    f"{prefix} greater than or equal to {limstr}, {suffix}."
                )

        # not equal
        case "neq":
            # no error case
            if all(v != lim for v in vals):
                return vals, None

            # special message for testing around zero
            if lim == 0:
                return vals, ValueError(f"{prefix} nonzero, {suffix}.")

            # normal message
            else:
                return vals, ValueError(f"{negprefix} equal to {limstr}, {suffix}.")

        # invalid operator
        case _:
            raise ValueError(
                "operator must be one of "
                "'lt, 'lte', 'gt', 'gte', or 'neq', "
                f"received '{str(op).lower()}'."
            )


def numeric_relative(
    lhname: str,
    lhval: Any,
    rhname: str,
    rhval: Any,
    op: Literal["lt", "lte", "gt", "gte", "neq"],
    cast: type | None = None,
    *,
    wraptxt: str | tuple[str, str] = "'",
    wrapnum: str | tuple[str, str] = "",
) -> tuple[Any, Any, ValueError | None]:
    r"""Casts a two values and compares them against each other, returns an error if the comparison is false.

    Args:
        lhname (str): variable name string for the error message (left hand side).
        lhval (Any): value being tested (left hand side).
        rhname (str): variable name string for the error message (right hand side).
        rhval (Any): value being tested (right hand side).
        op (Literal["lt", "lte", "gt", "gte", "neq"]): operation for comparison.
        cast (type | None, optional): type to cast value to. Defaults to None.
        wraptxt (str | tuple[str, str], optional): string to insert around variable name
            in the message, separate prepend/postpend if a tuple. Defaults to "'".
        wrapnum (str | tuple[str, str], optional): string to insert around the values
            in the message, separate prepend/postpend if a tuple. Defaults to "".

    Returns:
        tuple[Any, Any, ValueError | None]: casted values, left first, and an
        error if generated.
    """
    # cast value if a type is given
    if cast is not None:
        lhval = cast(lhval)
        rhval = cast(rhval)

    # create message strings
    wxl, wxr = (wraptxt, wraptxt) if isinstance(wraptxt, str) else wraptxt
    wnl, wnr = (wrapnum, wrapnum) if isinstance(wrapnum, str) else wrapnum
    prefix = f"{wxl}{lhname}{wxr}, {wnl}{lhval}{wnr}, must be"
    negprefix = f"{wxl}{lhname}{wxr}, {wnl}{lhval}{wnr}, must not be"
    suffix = f"{wxl}{rhname}{wxr}, {wnl}{rhval}{wnr}"

    # perform comparison
    match str(op).lower():
        # less than
        case "lt":
            # no error case
            if lhval < rhval:
                return lhval, rhval, None

            return lhval, rhval, ValueError(f"{prefix} less than {suffix}.")

        # less than or equal
        case "lte":
            # no error case
            if lhval <= rhval:
                return lhval, rhval, None

            return lhval, rhval, ValueError(f"{prefix} less than or equal to {suffix}.")

        # greater than
        case "gt":
            # no error case
            if lhval > rhval:
                return lhval, rhval, None

            return lhval, rhval, ValueError(f"{prefix} greater than {suffix}.")

        # greater than or equal
        case "gte":
            # no error case
            if lhval >= rhval:
                return lhval, rhval, None

            return (
                lhval,
                rhval,
                ValueError(f"{prefix} greater than or equal to {suffix}."),
            )

        # not equal
        case "neq":
            # no error case
            if lhval != rhval:
                return lhval, rhval, None

            return lhval, rhval, ValueError(f"{negprefix} equal to {suffix}.")

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
    *,
    wraptxt: str | tuple[str, str] = "'",
    wrapnum: str | tuple[str, str] = "",
) -> tuple[Any, ValueError | None]:
    """Casts a value and compares to a range, returns an error if it falls outside.

    Args:
        name (str): variable name string for the error message.
        val (Any): value being tested.
        lower (Any): lower bound of the range.
        upper (Any): upper bound of the range.
        op (Literal["closed", "open", "left-open", "right-open", "c", "o", "lo", "ro"]):
            kind of range to test.
        cast (type | None, optional): type used for casting. Defaults to None.
        wraptxt (str | tuple[str, str], optional): string to insert around variable name
            in the message, separate prepend/postpend if a tuple. Defaults to "'".
        wrapnum (str | tuple[str, str], optional): string to insert around the value in
            the message, separate prepend/postpend if a tuple. Defaults to "".

    Returns:
        tuple[Any, ValueError | None]: casted value and an error if generated.
    """
    wxl, wxr = (wraptxt, wraptxt) if isinstance(wraptxt, str) else wraptxt
    wnl, wnr = (wrapnum, wrapnum) if isinstance(wrapnum, str) else wrapnum
    prefix = f"{wxl}{name}{wxr} must exist on the interval"
    suffix = f"received {wnl}{val}{wnr}"

    # cast value if a type is given
    if cast is not None:
        val = cast(val)

    # perform comparison
    match str(op).lower():
        # less than
        case "closed" | "c":
            # no error case
            if lower <= val <= upper:
                return val, None

            raise ValueError(f"{prefix} [{lower}, {upper}], {suffix}")

        # less than or equal
        case "open" | "o":
            # no error case
            if lower < val < upper:
                return val, None

            raise ValueError(f"{prefix} ({lower}, {upper}), {suffix}")

        # greater than
        case "left-open" | "lo":
            # no error case
            if lower < val <= upper:
                return val, None

            raise ValueError(f"{prefix} ({lower}, {upper}], {suffix}")

        # greater than or equal
        case "right-open" | "ro":
            # no error case
            if lower <= val < upper:
                return val, None

            raise ValueError(f"{prefix} [{lower}, {upper}), {suffix}")

        # invalid operator
        case _:
            raise ValueError(
                "operator must be one of: "
                "'closed', 'open', 'left-open', 'right-open', 'c', 'o', 'lo', 'ro'; "
                f"received '{str(op).lower()}'."
            )


def instance_of(
    name: str,
    obj: Any,
    typespec: type | tuple[type, ...],
    *,
    wrapname: str | tuple[str, str] = "'",
) -> TypeError | None:
    """Checks if an object is an instance of a type or tuple thereof and returns an error if not.

    Args:
        name (str): variable name string for the error message.
        obj (Any): object being tested.
        typespec (type | tuple[type, ...]): type or tuple of types being tested.
        wrapname (str | tuple[str, str], optional): string to insert around variable
            name in the message, separate prepend/postpend if a tuple. Defaults to "'".

    Returns:
        TypeError | None: generated error if any.
    """
    # build prefix
    wnl, wnr = (wrapname, wrapname) if isinstance(wrapname, str) else wrapname
    prefix = f"{typenameof(name)} {wnl}{name}{wnr} must be an instance of"

    # test if object meets type specification
    if not isinstance(obj, typespec):
        # type specification is tuple of types
        if isinstance(typespec, tuple):
            return TypeError(
                f"{prefix} one of: {', '.join(typenameof(ts) for ts in typespec)}."
            )

        # type specification is not a tuple
        else:
            return TypeError(f"{prefix} {typenameof(typespec)}.")


def attr_members(
    name: str,
    obj: Any,
    *attr: str,
    wrapname: str | tuple[str, str] = "'",
    wrapattr: str | tuple[str, str] = "'",
) -> RuntimeError | None:
    """Checks if an object contains one or more attributes and returns an error if it doesn't.

    Args:
        name (str): variable name string for the error message.
        obj (Any): object being tested.
        *attr (str): attributes being checked for.
        wrapname (str | tuple[str, str], optional): string to insert around variable
            name in the message, separate prepend/postpend if a tuple. Defaults to "'".
        wrapattr (str | tuple[str, str], optional): string to insert around attribute
            names in the message, separate prepend/postpend if a tuple. Defaults to "'".

    Returns:
        RuntimeError | None: generated error if any.
    """
    # build seperator and prefix
    wnl, wnr = (wrapname, wrapname) if isinstance(wrapname, str) else wrapname
    wal, war = (wrapattr, wrapattr) if isinstance(wrapattr, str) else wrapattr
    prefix = f"{typenameof(name)} {wnl}{name}{wnr} is missing the attribute(s)"
    sep = f"{wal}, {war}"

    # test of object has attributes
    if not all(map(lambda a: hasattr(obj, a), attr)):
        return RuntimeError(
            f"{prefix}: {wal}{sep.join(filter(lambda a: hasattr(obj, a), attr))}{war}."
        )
