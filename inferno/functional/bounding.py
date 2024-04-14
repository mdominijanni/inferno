from ..core.tensor import zeros
import torch


def bound_upper_power(
    param: torch.Tensor,
    update: torch.Tensor,
    limit: float,
    *,
    power: float,
    **kwargs,
) -> torch.Tensor:
    r"""Computes the scaled update of upper-bound power parameter dependence.

    This is sometimes also referred to as "soft parameter dependence".

    .. math::
        U_+ = (P_\text{max} - P)^{\mu_+} U_+

    Args:
        param (torch.Tensor): parameter with update bounding, :math:`P`.
        update (torch.Tensor): potentiative update being applied, :math:`U_+`.
        limit (float): value of the upper bound, :math:`P_\text{max}`.
        power (float): exponent of parameter dependence, :math:`\mu_+`.

    Returns:
        torch.Tensor: bounded update.
    """
    return ((limit - param) ** power) * update


def bound_lower_power(
    param: torch.Tensor,
    update: torch.Tensor,
    limit: float,
    *,
    power: float,
    **kwargs,
) -> torch.Tensor:
    r"""Computes the scaled update of lower-bound power parameter dependence.

    This is sometimes also referred to as "soft parameter dependence".

    .. math::
        U_- = (P - P_\text{min})^{\mu_-} U_-

    Args:
        param (torch.Tensor): parameter with update bounding, :math:`P`.
        update (torch.Tensor): depressive update being applied, :math:`U_-`.
        limit (float): value of the upper bound, :math:`P_\text{min}`.
        power (float): exponent of parameter dependence, :math:`\mu_-`.

    Returns:
        torch.Tensor: bounded update.
    """
    return ((param - limit) ** power) * update


def bound_power(
    param: torch.Tensor,
    pos: torch.Tensor,
    neg: torch.Tensor,
    max: float | None,
    min: float | None,
    *,
    upper_power: float,
    lower_power: float,
    **kwargs,
) -> torch.Tensor:
    r"""Computes the scaled update of power parameter dependence.

    This is sometimes also referred to as "soft parameter dependence".

    .. math::
        U = (P_\text{max} - P)^{\mu_+} U_+ - (P - P_\text{min})^{\mu_-} U_-

    Args:
        param (torch.Tensor): parameter with update bounding, :math:`P`.
        pos (torch.Tensor): potentiative update being applied, :math:`U_+`.
        neg (torch.Tensor): depressive update being applied, :math:`U_-`.
        max (float | None): value of the upper bound,
            :math:`P_\text{max}`.
        min (float | None): value of the lower bound,
            :math:`P_\text{min}`.
        upper_power (float): exponent of upper-bound parameter
            dependence, :math:`\mu_+`.
        lower_power (float): exponent of lower-bound parameter
            dependence, :math:`\mu_-`.

    Returns:
        torch.Tensor: bounded update.
    """
    # update subcomponents
    if max is not None:
        pos = bound_upper_power(param, pos, max, upper_power)
    if min is not None:
        neg = bound_lower_power(param, neg, min, lower_power)

    # combined update
    return pos - neg


def bound_upper_scaled_power(
    param: torch.Tensor,
    update: torch.Tensor,
    limit: float,
    *,
    power: float,
    range: float,
    **kwargs,
) -> torch.Tensor:
    r"""Computes the scaled update of upper-bound scaled power parameter dependence.

    .. math::
        U_+ = \left(\frac{P_\text{max} - P}{P_\text{max} - P_\text{min}}\right)^{\mu_+} U_+

    Args:
        param (torch.Tensor): parameter with update bounding, :math:`P`.
        update (torch.Tensor): potentiative update being applied, :math:`U_+`.
        limit (float): value of the upper bound, :math:`P_\text{max}`.
        power (float): exponent of parameter dependence, :math:`\mu_+`.
        range (float): absolute difference between the upper and lower
            bounds, :math:`P_\text{max} - P_\text{min}`.

    Returns:
        torch.Tensor: bounded update.
    """
    return (((limit - param) / range) ** power) * update


def bound_lower_scaled_power(
    param: torch.Tensor,
    update: torch.Tensor,
    limit: float,
    *,
    power: float,
    range: float,
    **kwargs,
) -> torch.Tensor:
    r"""Computes the scaled update of lower-bound scaled power parameter dependence.

    .. math::
        U_- = \left(\frac{P - P_\text{min}}{P_\text{max} - P_\text{min}}\right)^{\mu_-} U_-

    Args:
        param (torch.Tensor): parameter with update bounding, :math:`P`.
        update (torch.Tensor): depressive update being applied, :math:`U_-`.
        limit (float): value of the upper bound, :math:`P_\text{min}`.
        power (float): exponent of parameter dependence, :math:`\mu_-`.
        range (float): absolute difference between the upper and lower
            bounds, :math:`P_\text{max} - P_\text{min}`.

    Returns:
        torch.Tensor: bounded update.
    """
    return (((param - limit) / range) ** power) * update


def bound_scaled_power(
    param: torch.Tensor,
    pos: torch.Tensor,
    neg: torch.Tensor,
    max: float | None,
    min: float | None,
    *,
    upper_power: float,
    lower_power: float,
    **kwargs,
) -> torch.Tensor:
    r"""Computes the scaled update of scaled power parameter dependence.

    This is sometimes also referred to as "soft parameter dependence".

    .. math::
        U = \left(\frac{P_\text{max} - P}{P_\text{max} - P_\text{min}}\right)^{\mu_+} U_+
        - \left(\frac{P - P_\text{min}}{P_\text{max} - P_\text{min}}\right)^{\mu_-} U_-

    Args:
        param (torch.Tensor): parameter with update bounding, :math:`P`.
        pos (torch.Tensor): potentiative update being applied, :math:`U_+`.
        neg (torch.Tensor): depressive update being applied, :math:`U_-`.
        max (float | None): value of the upper bound,
            :math:`P_\text{max}`.
        min (float | None): value of the lower bound,
            :math:`P_\text{min}`.
        upper_power (float): exponent of upper-bound parameter
            dependence, :math:`\mu_+`.
        lower_power (float): exponent of lower-bound parameter
            dependence, :math:`\mu_-`.

    Returns:
        torch.Tensor: bounded update.
    """
    # update subcomponents
    if max is not None:
        pos = bound_upper_scaled_power(param, pos, max, upper_power, max - min)
    if min is not None:
        neg = bound_lower_scaled_power(param, neg, min, lower_power, max - min)

    # combined update
    return pos - neg


def bound_upper_multiplicative(
    param: torch.Tensor,
    update: torch.Tensor,
    limit: float,
    **kwargs,
) -> torch.Tensor:
    r"""Computes the scaled update of upper-bound multiplicative parameter dependence.

    This is sometimes also referred to as "soft parameter dependence" and is equivalent
    to power dependence with an exponent of 1.

    .. math::
        U_+ = (P_\text{max} - P) U_+

    Args:
        param (torch.Tensor): parameter with update bounding, :math:`P`.
        update (torch.Tensor): potentiative update being applied, :math:`U_+`.
        limit (float): value of the upper bound, :math:`P_\text{max}`.

    Returns:
        torch.Tensor: bounded update.
    """
    return (limit - param) * update


def bound_lower_multiplicative(
    param: torch.Tensor,
    update: torch.Tensor,
    limit: float,
    **kwargs,
) -> torch.Tensor:
    r"""Computes the scaled update of lower-bound multiplicative parameter dependence.

    This is sometimes also referred to as "soft parameter dependence" and is equivalent
    to power dependence with an exponent of 1.

    .. math::
        U_- = (P - P_\text{min}) U_-

    Args:
        param (torch.Tensor): parameter with update bounding, :math:`P`.
        update (torch.Tensor): depressive update being applied, :math:`U_-`.
        limit (float): value of the upper bound, :math:`P_\text{min}`.

    Returns:
        torch.Tensor: bounded update.
    """
    return (param - limit) * update


def bound_multiplicative(
    param: torch.Tensor,
    pos: torch.Tensor,
    neg: torch.Tensor,
    max: float | None,
    min: float | None,
    **kwargs,
) -> torch.Tensor:
    r"""Computes the scaled update of multiplicative parameter dependence.

    This is sometimes also referred to as "soft parameter dependence" and is equivalent
    to power dependence with an exponent of 1.

    .. math::
        U = (P_\text{max} - P) U_+ - (P - P_\text{min}) U_-

    Args:
        param (torch.Tensor): parameter with update bounding, :math:`P`.
        pos (torch.Tensor): potentiative update being applied, :math:`U_+`.
        neg (torch.Tensor): depressive update being applied, :math:`U_-`.
        max (float | None): value of the upper bound,
            :math:`P_\text{max}`.
        min (float | None): value of the lower bound,
            :math:`P_\text{min}`.

    Returns:
        torch.Tensor: bounded update.
    """
    # update subcomponents
    if max is not None:
        pos = bound_upper_multiplicative(param, pos, max)
    if min is not None:
        neg = bound_lower_multiplicative(param, neg, min)

    # combined update
    return pos - neg


def bound_upper_scaled_multiplicative(
    param: torch.Tensor,
    update: torch.Tensor,
    limit: float,
    range: float,
    **kwargs,
) -> torch.Tensor:
    r"""Computes the scaled update of upper-bound scaled multiplicative parameter dependence.

    This is sometimes also referred to as "soft parameter dependence" and is equivalent
    to power dependence with an exponent of 1.

    .. math::
        U_+ = \left(\frac{P_\text{max} - P}{P_\text{max} - P_\text{min}}\right) U_+

    Args:
        param (torch.Tensor): parameter with update bounding, :math:`P`.
        update (torch.Tensor): potentiative update being applied, :math:`U_+`.
        limit (float): value of the upper bound, :math:`P_\text{max}`.
        range (float): absolute difference between the upper and lower
            bounds, :math:`P_\text{max} - P_\text{min}`.

    Returns:
        torch.Tensor: bounded update.
    """
    return (limit - param) / range * update


def bound_lower_scaled_multiplicative(
    param: torch.Tensor,
    update: torch.Tensor,
    limit: float,
    range: float,
    **kwargs,
) -> torch.Tensor:
    r"""Computes the scaled update of lower-bound scaled multiplicative parameter dependence.

    This is sometimes also referred to as "soft parameter dependence" and is equivalent
    to power dependence with an exponent of 1.

    .. math::
        U_- = \left(\frac{P - P_\text{min}}{P_\text{max} - P_\text{min}}\right) U_-

    Args:
        param (torch.Tensor): parameter with update bounding, :math:`P`.
        update (torch.Tensor): depressive update being applied, :math:`U_-`.
        limit (float): value of the upper bound, :math:`P_\text{min}`.
        range (float): absolute difference between the upper and lower
            bounds, :math:`P_\text{max} - P_\text{min}`.

    Returns:
        torch.Tensor: bounded update.
    """
    return (param - limit) / range * update


def bound_scaled_multiplicative(
    param: torch.Tensor,
    pos: torch.Tensor,
    neg: torch.Tensor,
    max: float | None,
    min: float | None,
    **kwargs,
) -> torch.Tensor:
    r"""Computes the scaled update of multiplicative parameter dependence.

    This is sometimes also referred to as "soft parameter dependence" and is equivalent
    to power dependence with an exponent of 1.

    .. math::
        U = \left(\frac{P_\text{max} - P}{P_\text{max} - P_\text{min}}\right) U_+
        - \left(\frac{P - P_\text{min}}{P_\text{max} - P_\text{min}}\right) U_-

    Args:
        param (torch.Tensor): parameter with update bounding, :math:`P`.
        pos (torch.Tensor): potentiative update being applied, :math:`U_+`.
        neg (torch.Tensor): depressive update being applied, :math:`U_-`.
        max (float | None): value of the upper bound,
            :math:`P_\text{max}`.
        min (float | None): value of the lower bound,
            :math:`P_\text{min}`.

    Returns:
        torch.Tensor: bounded update.
    """
    # update subcomponents
    if max is not None:
        pos = bound_upper_scaled_multiplicative(param, pos, max, max - min)
    if min is not None:
        neg = bound_lower_scaled_multiplicative(param, neg, min, max - min)

    # combined update
    return pos - neg


def bound_upper_sharp(
    param: torch.Tensor,
    update: torch.Tensor,
    limit: float,
    **kwargs,
) -> torch.Tensor:
    r"""Computes the scaled update of upper-bound sharp parameter dependence.

    This is sometimes also referred to as "hard parameter dependence".

    .. math::
        U_+ = \Theta(P_\text{max} - P) U_+

    Where

    .. math::
        \Theta(x) =
        \begin{cases}
            1 &x \geq 0 \\
            0 & x < 0
        \end{cases}

    Args:
        param (torch.Tensor): parameter with update bounding, :math:`P`.
        update (torch.Tensor): potentiative update being applied, :math:`U_+`.
        limit (float): value of the upper bound, :math:`P_\text{max}`.

    Returns:
        torch.Tensor: bounded update.
    """
    diff = limit - param
    return torch.heaviside(diff, zeros(diff, shape=())) * update


def bound_lower_sharp(
    param: torch.Tensor,
    update: torch.Tensor,
    limit: float,
    **kwargs,
) -> torch.Tensor:
    r"""Computes the scaled update of lower-bound sharp parameter dependence.

    This is sometimes also referred to as "hard parameter dependence".

    .. math::
        U_- = \Theta(P - P_\text{min}) U_-

    Where

    .. math::
        \Theta(x) =
        \begin{cases}
            1 &x \geq 0 \\
            0 & x < 0
        \end{cases}

    Args:
        param (torch.Tensor): parameter with update bounding, :math:`P`.
        update (torch.Tensor): depressive update being applied, :math:`U_-`.
        limit (float): value of the upper bound, :math:`P_\text{min}`.

    Returns:
        torch.Tensor: bounded update.
    """
    diff = param - limit
    return torch.heaviside(diff, zeros(diff, shape=())) * update


def bound_sharp(
    param: torch.Tensor,
    pos: torch.Tensor,
    neg: torch.Tensor,
    max: float | None,
    min: float | None,
    **kwargs,
) -> torch.Tensor:
    r"""Computes the scaled update of sharp parameter dependence.

    This is sometimes also referred to as "hard parameter dependence".

    .. math::
        U = \Theta(P_\text{max} - P) U_+ - \Theta(P - P_\text{min}) U_-

    Where

    .. math::
        \Theta(x) =
        \begin{cases}
            1 &x \geq 0 \\
            0 & x < 0
        \end{cases}


    Args:
        param (torch.Tensor): parameter with update bounding, :math:`P`.
        pos (torch.Tensor): potentiative update being applied, :math:`U_+`.
        neg (torch.Tensor): depressive update being applied, :math:`U_-`.
        max (float | None): value of the upper bound,
            :math:`P_\text{max}`.
        min (float | None): value of the lower bound,
            :math:`P_\text{min}`.

    Returns:
        torch.Tensor: bounded update.
    """
    # update subcomponents
    if max is not None:
        pos = bound_upper_sharp(param, pos, max)
    if min is not None:
        neg = bound_lower_sharp(param, neg, min)

    # combined update
    return pos - neg
