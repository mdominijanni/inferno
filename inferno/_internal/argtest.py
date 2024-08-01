from .utils import rgetattr
from collections.abc import Iterable, Sequence
import math
from types import UnionType
from typing import Any, Callable


def _typename(type_: type | UnionType, sep: str = ", ") -> str:
    r"""Module Internal: gets the name of a given type or UnionType.

    Args:
        type_ (type | UnionType): type to get the name of.
        sep (str, optional): added separator between types in a union. Defaults to ``", "``.

    Returns:
        str: name of a type (including a union type).
    """
    if isinstance(type_, UnionType):
        return sep.join(_typename(subtype) for subtype in type_.__args__)
    elif hasattr(type_, "__name__"):
        return type_.__name__
    else:
        return str(type_)


def _prefixstr(prefix: str | None) -> str:
    r"""Module Internal: converts prefix into display string.

    Args:
        prefix (str | None): prefix to add to error message.

    Returns:
        str: string to display in the error message as the prefix.
    """
    if prefix:
        return f"{prefix}"
    else:
        return ""


def _valuestr(value: Any) -> str:
    r"""Module Internal: converts test value into display string.

    Args:
        value (Any): value to add to error message.

    Returns:
        str: string to display in the error message as the value.
    """
    if isinstance(value, str):
        return f"'{value}'"
    else:
        return f"{value}"


def _cast(name: str, value: Any, cast: type | None, prefix: str | None) -> Any:
    r"""Module Internal: casts a value using appropriate conditional tests.

    Args:
        name (str): display name of the variable tested.
        value (Any): variable being testing.
        cast (type | None): type of the cast, if any, which is being performed.
        prefix (str | None): error message prefix.

    Returns:
        Any: casted value.
    """
    if cast is None:
        return value
    elif cast is int:
        return integer(name, value, prefix)
    else:
        return cast(value)


def ofsequence(
    name: str, values: Sequence[Any], test: Callable, *args, **kwargs
) -> tuple[Any, ...]:
    r"""Runs an argument test on every value in a sequence.

    Args:
        name (str): display name of the variable tested.
        values (Sequence[Any]): sequence of values to test.
        test (Callable): callable with which to test.
        args (Any): positional arguments passed to the test call.
        kwargs (Any): keyword arguments passed to the test call.

    Returns:
        Any: tuple of values after being cast.

    Note:
        The variable's display name is augmented with the index as "name[index]".

    Note:
        The first two arguments of ``test`` should be ``name`` and ``value``, the
        rest determined by ``args`` and ``kwargs``. This fits the public functions
        of :py:module:`argtest`. These first two arguments are passed positionally.
    """
    return tuple(
        test(f"{name}[{idx}]", val, *args, **kwargs) for idx, val in enumerate(values)
    )


def oneof(
    name: str,
    value: Any,
    *targets: Any,
    op: Callable[[Any], Any] | None = None,
    prefix: str | None = None,
) -> Any:
    r"""Checks if a value is one of multiple options

    Args:
        name (str): display name of the variable tested.
        value (Any): variable being testing.
        *targets (Any): values to test against.
        op (Callable[[Any], Any] | None, optional): operation to apply to ``value``
            before comparison. Defaults to ``None``.
        prefix (str | None, optional): error message prefix. Defaults to ``None``.

    Returns:
        Any: input value with operation applied.
    """
    v = op(value) if op else value
    if any(map(lambda x: v == x, targets)):
        return v
    else:
        val, pfx = _valuestr(value), _prefixstr(prefix)
        raise ValueError(
            f"{pfx}'{name}' of {val} must be one of: {(_valuestr(t) for t in targets)}"
        )


def onedefined(*nvpairs: tuple[str, Any], prefix: str | None = None) -> tuple[Any, ...]:
    r"""Checks if at least one value in a sequence is not None.
    Args:
        *nvpairs (tuple[str, Any]): tuples each containing the display name of a
            variable being tested and the variable itself.
        prefix (str | None, optional): error message prefix. Defaults to ``None``.

    Returns:
        tuple[Any, ...]: values tested.
    """
    if any(map(lambda nvp: nvp[1] is not None, nvpairs)):
        return tuple(nvp[1] for nvp in nvpairs)
    else:
        namestr, pfx = "', '".join(nvp[0] for nvp in nvpairs), _prefixstr(prefix)
        raise RuntimeError(f"{pfx}at least one of '{namestr}' must not be None")


def integer(name: str, value: Any, prefix: str | None = None) -> int:
    r"""Checks if a value can be interpreted as an integer.

    Args:
        name (str): display name of the variable tested.
        value (Any): variable being testing.
        prefix (str | None, optional): error message prefix. Defaults to ``None``.

    Returns:
        int: value casted as an integer.
    """
    # error display strings
    msg = (
        f"{_prefixstr(prefix)}'{name}' of {_valuestr(value)} "
        f"cannot be interpreted as an integer"
    )

    # string conversion
    if isinstance(value, str):
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                raise ValueError(msg)

    # numeric test
    if value % 1 == 0:
        return int(value)
    else:
        raise ValueError(msg)


def dimensions(
    name: str,
    value: Any,
    lower: int | None,
    upper: int | None,
    permit_none: bool = False,
    wrap_output: bool = False,
    prefix: str | None = None,
) -> int | tuple[int, ...] | None:
    r"""Checks if a value can be interpreted as a shape or dimensions.

    Generally, set ``permit_none`` to `True` for dimensions, and ``wrap_output``
    to ``True`` for a shape.

    Args:
        name (str): display name of the variable tested.
        value (Any): variable being testing.
        lower (Any | None): inclusive lower index bound, if any.
        upper (Any | None): inclusive upper index bound, if any.
        permit_none (bool, optional): if a value of ``None`` should be permitted.
            Defaults to ``False``.
        wrap_output (bool, optional): if non-``None`` singleton output should be wrapped.
            Defaults to ``False``.
        prefix (str | None, optional): error message prefix to display.
            Defaults to ``None``.

    Returns:
        int | tuple[int, ...] | None: input after standardization.
    """
    if value is None:
        if permit_none:
            return None
        else:
            val, pfx = _valuestr(value), _prefixstr(prefix)
            raise TypeError(f"{pfx}'{name}' ({val}) cannot be None")

    else:
        match (lower is not None, upper is not None):
            case (False, False):
                func = integer
                args, kwargs = (), {"prefix": prefix}
            case (False, True):
                func = lte
                args, kwargs = (upper,), {"cast": int, "prefix": prefix}
            case (True, False):
                func = gte
                args, kwargs = (lower,), {"cast": int, "prefix": prefix}
            case (True, True):
                func = minmax_incl
                args, kwargs = (lower, upper), {"cast": int, "prefix": prefix}

        if isinstance(value, Iterable) and not isinstance(value, str):
            return ofsequence(name, value, func, *args, **kwargs)
        elif wrap_output:
            return (func(name, value, *args, **kwargs),)
        else:
            return func(name, value, *args, **kwargs)


def likesign(
    name: str,
    value: Any,
    signed: Any,
    cast: type | None = None,
    signed_name: str | None = None,
    prefix: str | None = None,
) -> Any:
    r"""Casts a value and then tests it is has the same sign as another.

    Args:
        name (str): display name of the variable tested.
        value (Any): variable being testing.
        signed (Any): signed value to match.
        cast (type | None, optional): type to which value should be cast.
            Defaults to ``None``.
        signed_name (str | None, optional): name defining the signed value,
            if not a constant. Defaults to ``None``.
        prefix (str | None, optional): error message prefix to display.
            Defaults to ``None``.

    Returns:
        Any: value (and limit if given a name) after being cast.

    Note:
        When "signed_name" is not ``None``, it will be casted as well and will be
        displayed in the error message. Use this if the signed value is not constant
        but is dependent upon another value.
    """
    # casts given value to appropriate type
    casted = _cast(name, value, cast, prefix)

    # conditionally casts the limit
    if signed_name:
        casted_signed = _cast(signed_name, signed, cast, prefix)
    else:
        casted_signed = signed

    # test condition, non-constant limit message
    if math.copysign(1.0, casted) == math.copysign(1.0, casted_signed):
        return casted
    elif signed_name:
        val, lim, pfx = _valuestr(value), _valuestr(signed), _prefixstr(prefix)
        raise ValueError(
            f"{pfx}'{name}' ({val}) have the same sign as '{signed_name}' ({lim})"
        )
    else:
        val, pfx = _valuestr(value), _prefixstr(prefix)
        raise ValueError(f"{pfx}'{name}' ({val}) have the same sign as {signed}")


def lt(
    name: str,
    value: Any,
    limit: Any,
    cast: type | None = None,
    limit_name: str | None = None,
    prefix: str | None = None,
) -> Any:
    r"""Casts a value and then tests it is less than a limit.

    Args:
        name (str): display name of the variable tested.
        value (Any): variable being testing.
        limit (Any): permitted exclusive upper bound.
        cast (type | None, optional): type to which value should be cast.
            Defaults to ``None``.
        limit_name (str | None, optional): name defining the limit, if not a constant.
            Defaults to ``None``.
        prefix (str | None, optional): error message prefix to display.
            Defaults to ``None``.

    Returns:
        Any: value (and limit if given a name) after being cast.

    Note:
        When "limit_name" is not ``None``, it will be casted as well and will be
        displayed in the error message. Use this if the limit is not constant but is
        dependent upon another value.
    """
    # casts given value to appropriate type
    casted = _cast(name, value, cast, prefix)

    # conditionally casts the limit
    if limit_name:
        casted_limit = _cast(limit_name, limit, cast, prefix)
    else:
        casted_limit = limit

    # test condition with special message around zero, and non-constant limit message
    if casted < casted_limit:
        return casted
    elif limit_name:
        val, lim, pfx = _valuestr(value), _valuestr(limit), _prefixstr(prefix)
        raise ValueError(
            f"{pfx}'{name}' ({val}) must be less than '{limit_name}' ({lim})"
        )
    elif limit == 0:
        val, pfx = _valuestr(value), _prefixstr(prefix)
        raise ValueError(f"{pfx}'{name}' ({val}) must be negative")
    else:
        val, pfx = _valuestr(value), _prefixstr(prefix)
        raise ValueError(f"{pfx}'{name}' ({val}) must be less than {limit}")


def lte(
    name: str,
    value: Any,
    limit: Any,
    cast: type | None = None,
    limit_name: str | None = None,
    prefix: str | None = None,
) -> Any:
    r"""Casts a value and then tests it is less or equal to than a limit.

    Args:
        name (str): display name of the variable tested.
        value (Any): variable being testing.
        limit (Any): permitted inclusive upper bound.
        cast (type | None, optional): type to which value should be cast.
            Defaults to ``None``.
        limit_name (str | None, optional): name defining the limit, if not a constant.
            Defaults to ``None``.
        prefix (str | None, optional): error message prefix to display.
            Defaults to ``None``.

    Returns:
        Any: value (and limit if given a name) after being cast.

    Note:
        When "limit_name" is not ``None``, it will be casted as well and will be
        displayed in the error message. Use this if the limit is not constant but is
        dependent upon another value.
    """
    # casts given value to appropriate type
    casted = _cast(name, value, cast, prefix)

    # conditionally casts the limit
    if limit_name:
        casted_limit = _cast(limit_name, limit, cast, prefix)
    else:
        casted_limit = limit

    # test condition with special message around zero
    if casted <= casted_limit:
        return casted
    elif limit_name:
        val, lim, pfx = _valuestr(value), _valuestr(limit), _prefixstr(prefix)
        raise ValueError(
            f"{pfx}'{name}' ({val}) must be less than or equal to '{limit_name}' ({lim})"
        )
    elif limit == 0:
        val, pfx = _valuestr(value), _prefixstr(prefix)
        raise ValueError(f"{pfx}'{name}' ({val}) must be nonpositive")
    else:
        val, pfx = _valuestr(value), _prefixstr(prefix)
        raise ValueError(f"{pfx}'{name}' ({val}) must be less than or equal to {limit}")


def gt(
    name: str,
    value: Any,
    limit: Any,
    cast: type | None = None,
    limit_name: str | None = None,
    prefix: str | None = None,
) -> Any:
    r"""Casts a value and then tests it is greater than a limit.

    Args:
        name (str): display name of the variable tested.
        value (Any): variable being testing.
        limit (Any): permitted exclusive lower bound.
        cast (type | None, optional): type to which value should be cast.
            Defaults to ``None``.
        limit_name (str | None, optional): name defining the limit, if not a constant.
            Defaults to ``None``.
        prefix (str | None, optional): error message prefix to display.
            Defaults to ``None``.

    Returns:
        Any: value (and limit if given a name) after being cast.

    Note:
        When "limit_name" is not ``None``, it will be casted as well and will be
        displayed in the error message. Use this if the limit is not constant but is
        dependent upon another value.
    """
    # casts given value to appropriate type
    casted = _cast(name, value, cast, prefix)

    # conditionally casts the limit
    if limit_name:
        casted_limit = _cast(limit_name, limit, cast, prefix)
    else:
        casted_limit = limit

    # test condition with special message around zero
    if casted > casted_limit:
        return casted
    elif limit_name:
        val, lim, pfx = _valuestr(value), _valuestr(limit), _prefixstr(prefix)
        raise ValueError(
            f"{pfx}'{name}' ({val}) must be greater than '{limit_name}' ({lim})"
        )
    elif limit == 0:
        val, pfx = _valuestr(value), _prefixstr(prefix)
        raise ValueError(f"{pfx}'{name}' of {val} must be positive")
    else:
        val, pfx = _valuestr(value), _prefixstr(prefix)
        raise ValueError(f"{pfx}'{name}' of {val} must be greater than {limit}")


def gte(
    name: str,
    value: Any,
    limit: Any,
    cast: type | None = None,
    limit_name: str | None = None,
    prefix: str | None = None,
) -> Any:
    r"""Casts a value and then tests it is greater or equal to than a limit.

    Args:
        name (str): display name of the variable tested.
        value (Any): variable being testing.
        limit (Any): permitted inclusive lower bound.
        cast (type | None, optional): type to which value should be cast.
            Defaults to ``None``.
        limit_name (str | None, optional): name defining the limit, if not a constant.
            Defaults to ``None``.
        prefix (str | None, optional): error message prefix to display.
            Defaults to ``None``.

    Returns:
        Any: value (and limit if given a name) after being cast.

    Note:
        When "limit_name" is not ``None``, it will be casted as well and will be
        displayed in the error message. Use this if the limit is not constant but is
        dependent upon another value.
    """
    # casts given value to appropriate type
    casted = _cast(name, value, cast, prefix)

    # conditionally casts the limit
    if limit_name:
        casted_limit = _cast(limit_name, limit, cast, prefix)
    else:
        casted_limit = limit

    # test condition with special message around zero
    if casted >= casted_limit:
        return casted
    elif limit_name:
        val, lim, pfx = _valuestr(value), _valuestr(limit), _prefixstr(prefix)
        raise ValueError(
            f"{pfx}'{name}' ({val}) must be greater than or equal to '{limit_name}' ({lim})"
        )
    elif limit == 0:
        val, pfx = _valuestr(value), _prefixstr(prefix)
        raise ValueError(f"{pfx}'{name}' of {val} must be nonnegative")
    else:
        val, pfx = _valuestr(value), _prefixstr(prefix)
        raise ValueError(
            f"{pfx}'{name}' of {val} must be greater than or equal to {limit}"
        )


def neq(
    name: str,
    value: Any,
    limit: Any,
    cast: type | None = None,
    limit_name: str | None = None,
    prefix: str | None = None,
) -> Any:
    r"""Casts a value and then tests it is not equal to a limit.

    Args:
        name (str): display name of the variable tested.
        value (Any): variable being testing.
        limit (Any): forbidden value.
        cast (type | None, optional): type to which value should be cast.
            Defaults to ``None``.
        limit_name (str | None, optional): name defining the limit, if not a constant.
            Defaults to ``None``.
        prefix (str | None, optional): error message prefix to display.
            Defaults to ``None``.

    Returns:
        Any: value (and limit if given a name) after being cast.

    Note:
        When "limit_name" is not ``None``, it will be casted as well and will be
        displayed in the error message. Use this if the limit is not constant but is
        dependent upon another value.
    """
    # casts given value to appropriate type
    casted = _cast(name, value, cast, prefix)

    # conditionally casts the limit
    if limit_name:
        casted_limit = _cast(limit_name, limit, cast, prefix)
    else:
        casted_limit = limit

    # test condition with special message around zero
    if casted != casted_limit:
        return casted
    elif limit_name:
        val, lim, pfx = _valuestr(value), _valuestr(limit), _prefixstr(prefix)
        raise ValueError(f"{pfx}'{name}' ({val}) must not equal '{limit_name}' ({lim})")
    elif limit == 0:
        val, pfx = _valuestr(value), _prefixstr(prefix)
        raise ValueError(f"{pfx}'{name}' of {val} must be nonzero")
    else:
        val, pfx = _valuestr(value), _prefixstr(prefix)
        raise ValueError(f"{pfx}'{name}' of {val} must not equal {limit}")


def minmax_incl(
    name: str,
    value: Any,
    lower: Any,
    upper: Any,
    cast: type | None = None,
    prefix: str | None = None,
) -> Any:
    r"""Casts a value and then tests if is on a closed interval.

    Args:
        name (str): display name of the variable tested.
        value (Any): variable being testing.
        lower (Any): permitted inclusive lower bound.
        upper (Any): permitted inclusive upper bound.
        cast (type | None, optional): type to which value should be cast.
            Defaults to ``None``.
        prefix (str | None, optional): error message prefix to display.
            Defaults to ``None``.

    Returns:
        Any: value after being cast.
    """
    # casts given value to appropriate type
    casted = _cast(name, value, cast, prefix)

    # test condition
    if lower <= casted <= upper:
        return value
    else:
        val, pfx = _valuestr(value), _prefixstr(prefix)
        raise ValueError(
            f"{pfx}'{name}' of {val} must be on the interval [{lower}, {upper}]"
        )


def minmax_excl(
    name: str,
    value: Any,
    lower: Any,
    upper: Any,
    cast: type | None = None,
    prefix: str | None = None,
) -> Any:
    r"""Casts a value and then tests if is on an open interval.

    Args:
        name (str): display name of the variable tested.
        value (Any): variable being testing.
        lower (Any): permitted exclusive lower bound.
        upper (Any): permitted exclusive upper bound.
        cast (type | None, optional): type to which value should be cast.
            Defaults to ``None``.
        prefix (str | None, optional): error message prefix to display.
            Defaults to ``None``.

    Returns:
        Any: value after being cast.
    """
    # casts given value to appropriate type
    casted = _cast(name, value, cast, prefix)

    # test condition
    if lower < casted < upper:
        return value
    else:
        val, pfx = _valuestr(value), _prefixstr(prefix)
        raise ValueError(
            f"{pfx}'{name}' of {val} must be on the interval ({lower}, {upper})"
        )


def min_excl_max_incl(
    name: str,
    value: Any,
    lower: Any,
    upper: Any,
    cast: type | None = None,
    prefix: str | None = None,
) -> Any:
    r"""Casts a value and then tests if is on a left-open interval.

    Args:
        name (str): display name of the variable tested.
        value (Any): variable being testing.
        lower (Any): permitted exclusive lower bound.
        upper (Any): permitted inclusive upper bound.
        cast (type | None, optional): type to which value should be cast.
            Defaults to ``None``.
        prefix (str | None, optional): error message prefix to display.
            Defaults to ``None``.

    Returns:
        Any: value after being cast.
    """
    # casts given value to appropriate type
    casted = _cast(name, value, cast, prefix)

    # test condition
    if lower < casted <= upper:
        return value
    else:
        val, pfx = _valuestr(value), _prefixstr(prefix)
        raise ValueError(
            f"{pfx}'{name}' of {val} must be on the interval ({lower}, {upper}]"
        )


def min_incl_max_excl(
    name: str,
    value: Any,
    lower: Any,
    upper: Any,
    cast: type | None = None,
    prefix: str | None = None,
) -> Any:
    r"""Casts a value and then tests if is on a right-open interval.

    Args:
        name (str): display name of the variable tested.
        value (Any): variable being testing.
        lower (Any): permitted inclusive lower bound.
        upper (Any): permitted exclusive upper bound.
        cast (type | None, optional): type to which value should be cast.
            Defaults to ``None``.
        prefix (str | None, optional): error message prefix to display.
            Defaults to ``None``.

    Returns:
        Any: value after being cast.
    """
    # casts given value to appropriate type
    casted = _cast(name, value, cast, prefix)

    # test condition
    if lower <= casted < upper:
        return value
    else:
        val, pfx = _valuestr(value), _prefixstr(prefix)
        raise ValueError(
            f"{pfx}'{name}' of {val} must be on the interval [{lower}, {upper})"
        )


def instance(
    name: str,
    obj: Any,
    typespec: type | UnionType | tuple[type | UnionType, ...],
    prefix: str | None = None,
) -> Any:
    r"""Checks if an object is an instance of a given type spec.

    Args:
        name (str): display name of the object being tested.
        obj (Any): object being tested.
        typespec (type | UnionType | tuple[type | UnionType, ...]): allowed types.
        prefix (str | None, optional): error message prefix to display.
            Defaults to ``None``.

    Returns:
        Any: tested object.
    """
    if isinstance(obj, typespec):
        return obj
    elif isinstance(typespec, tuple):
        objt, pfx = _typename(obj), _prefixstr(prefix)
        tspec = ", ".join(_typename(ts, sep=", ") for ts in typespec)
        raise TypeError(
            f"{pfx}'{name}' is an instance of {objt}, must be an instance of: {tspec}"
        )
    else:
        objt, tspec, pfx = _typename(obj), _typename(typespec), _prefixstr(prefix)
        raise TypeError(
            f"{pfx}'{name}' is an instance of {objt}, must be an instance of: {tspec}"
        )


def members(name: str, obj: Any, *attr: str, prefix: str | None = None) -> Any:
    r"""Checks if an object has given attributes.

    Args:
        name (str): display name of the object being tested.
        obj (Any): object being tested.
        attr (str): required attributes.
        prefix (str | None, optional): error message prefix to display.
            Defaults to ``None``.

    Returns:
        Any: tested object.
    """
    if all(map(lambda a: hasattr(obj, a), attr)):
        return obj
    else:
        objt, sa, pfx = _typename(obj), ", ".join(attr), _prefixstr(prefix)
        raise RuntimeError(
            f"{pfx}'{name}' of type {objt} is missing the attribute(s): {sa}"
        )


def nestedmembers(name: str, obj: Any, *attr: str, prefix: str | None = None) -> Any:
    r"""Checks if an object has given nested attributes.

    Args:
        name (str): display name of the object being tested.
        obj (Any): object being tested.
        attr (str): required nested attributes.
        prefix (str | None, optional): error message prefix to display.
            Defaults to ``None``.

    Returns:
        Any: tested object.
    """
    try:
        for a in attr:
            lastattr = a
            _ = rgetattr(obj, a)
    except AttributeError:
        objt, sa, pfx = _typename(obj), ", ".join(lastattr), _prefixstr(prefix)
        raise RuntimeError(
            f"{pfx}'{name}' of type {objt} is missing the attribute(s): {sa}"
        )
    else:
        return obj


def identifier(name: str, value: str, prefix: str | None = None) -> str:
    r"""Checks if a string is a valid identifier. Does not check for existence.

    Args:
        name (str): display name of the object being tested.
        value (str): value (Any): variable being testing.
        prefix (str | None, optional): error message prefix to display.
            Defaults to ``None``.

    Returns:
        str: tested object.
    """
    if not isinstance(value, str):
        pfx = _prefixstr(prefix)
        raise TypeError(f"{pfx}'{name}' must be a string, but is a {_typename(value)}")
    elif not value.isidentifier():
        val, pfx = _valuestr(value), _prefixstr(prefix)
        raise ValueError(f"{pfx}'{name}' ({val}) is not a valid identifier")
    else:
        return value


def nestedidentifier(name: str, value: str, prefix: str | None = None) -> str:
    r"""Checks if a string is a valid identifier, or dot separated identifiers. Does not check for existence.

    Args:
        name (str): display name of the object being tested.
        value (str): value (Any): variable being testing.
        prefix (str | None, optional): error message prefix to display.
            Defaults to ``None``.

    Returns:
        str: tested object.
    """
    if not isinstance(value, str):
        pfx = _prefixstr(prefix)
        raise TypeError(f"{pfx}'{name}' must be a string, but is a {_typename(value)}")
    elif any(map(lambda s: not s.isidentifier(), value.split("."))):
        val, pfx = _valuestr(value), _prefixstr(prefix)
        raise ValueError(
            f"{pfx}'{name}' ({val}) is not a valid dot-separated identifier"
        )
    else:
        return value


def index(
    name: str,
    value: int,
    length: int,
    length_name: str | None = None,
    prefix: str | None = None,
) -> int:
    r"""Checks that an index is in range for a given length sequence.

    Args:
        name (str): display name of the variable tested.
        value (int): variable being testing.
        length (int): length of the sequence being tested against
        length_name (str | None, optional): name of the sequence, if not a constant.
            Defaults to ``None``.
        prefix (str | None, optional): error message prefix. Defaults to ``None``.

    Returns:
        int: value tested
    """
    casted = _cast(name, value, int, prefix)

    if not value == casted:
        pfx = _prefixstr(prefix)
        raise TypeError(f"{pfx}'{name}' must be an int, but is a {_typename(value)}")
    if -length <= casted < length:
        return casted
    elif casted == 0:
        pfx = _prefixstr(prefix)
        if length_name:
            raise ValueError(
                f"{pfx}cannot index the zero-length sequence '{length_name}'"
            )
        else:
            raise ValueError(f"{pfx}cannot index a zero-length sequence")
    elif length_name:
        val, nlgh, lgh, pfx = (
            _valuestr(value),
            _valuestr(-length),
            _valuestr(length),
            _prefixstr(prefix),
        )
        raise ValueError(
            f"{pfx}'{name}' ({val}) must be on the interval [{nlgh}, {lgh}) for "
            f"indexing '{length_name}' of length {lgh}"
        )
    else:
        val, nlgh, lgh, pfx = (
            _valuestr(value),
            _valuestr(-length),
            _valuestr(length),
            _prefixstr(prefix),
        )
        raise ValueError(
            f"{pfx}'{name}' ({val}) must be on the interval [{nlgh}, {lgh}) for "
            f"indexing a sequence of length {lgh}"
        )
