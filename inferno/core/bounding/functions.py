from ..tensor import zeros
import torch


def upper_power(
    param: torch.Tensor,
    update: torch.Tensor,
    limit: float | torch.Tensor,
    *,
    power: float | torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    r"""Computes the scaled update of upper-bound power parameter dependence.

    This is sometimes also referred to as "soft parameter dependence".

    .. math::
        U_+ = (P_\text{max} - P)^{\mu_+} U_+

    Args:
        param (torch.Tensor): parameter with update bounding, :math:`P`.
        update (torch.Tensor): potentiative update being applied, :math:`U_+`.
        limit (float | torch.Tensor): value of the upper bound, :math:`P_\text{max}`.
        power (float | torch.Tensor): exponent of parameter dependence, :math:`\mu_+`.

    Returns:
        torch.Tensor: bounded update.
    """
    return ((limit - param) ** power) * update


def lower_power(
    param: torch.Tensor,
    update: torch.Tensor,
    limit: float | torch.Tensor,
    *,
    power: float | torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    r"""Computes the scaled update of lower-bound power parameter dependence.

    This is sometimes also referred to as "soft parameter dependence".

    .. math::
        U_- = (P - P_\text{min})^{\mu_-} U_-

    Args:
        param (torch.Tensor): parameter with update bounding, :math:`P`.
        update (torch.Tensor): depressive update being applied, :math:`U_-`.
        limit (float | torch.Tensor): value of the upper bound, :math:`P_\text{min}`.
        power (float | torch.Tensor): exponent of parameter dependence, :math:`\mu_-`.

    Returns:
        torch.Tensor: bounded update.
    """
    return ((param - limit) ** power) * update


def power(
    param: torch.Tensor,
    pos: torch.Tensor,
    neg: torch.Tensor,
    max: float | torch.Tensor | None,
    min: float | torch.Tensor | None,
    *,
    upper_power: float | torch.Tensor,
    lower_power: float | torch.Tensor,
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
        max (float | torch.Tensor | None): value of the upper bound,
            :math:`P_\text{max}`.
        min (float | torch.Tensor | None): value of the lower bound,
            :math:`P_\text{min}`.
        upper_power (float | torch.Tensor): exponent of upper-bound parameter
            dependence, :math:`\mu_+`.
        lower_power (float | torch.Tensor): exponent of lower-bound parameter
            dependence, :math:`\mu_-`.

    Returns:
        torch.Tensor: bounded update.
    """
    # update subcomponents
    if max is not None:
        pos = upper_power(param, pos, max, upper_power)
    if min is not None:
        neg = lower_power(param, neg, min, lower_power)

    # combined update
    return pos - neg


def upper_multiplicative(
    param: torch.Tensor,
    update: torch.Tensor,
    limit: float | torch.Tensor,
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
        limit (float | torch.Tensor): value of the upper bound, :math:`P_\text{max}`.

    Returns:
        torch.Tensor: bounded update.
    """
    return (limit - param) * update


def lower_multiplicative(
    param: torch.Tensor,
    update: torch.Tensor,
    limit: float | torch.Tensor,
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
        limit (float | torch.Tensor): value of the upper bound, :math:`P_\text{min}`.

    Returns:
        torch.Tensor: bounded update.
    """
    return (param - limit) * update


def multiplicative(
    param: torch.Tensor,
    pos: torch.Tensor,
    neg: torch.Tensor,
    max: float | torch.Tensor | None,
    min: float | torch.Tensor | None,
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
        max (float | torch.Tensor | None): value of the upper bound,
            :math:`P_\text{max}`.
        min (float | torch.Tensor | None): value of the lower bound,
            :math:`P_\text{min}`.

    Returns:
        torch.Tensor: bounded update.
    """
    # update subcomponents
    if max is not None:
        pos = upper_multiplicative(param, pos, max)
    if min is not None:
        neg = lower_multiplicative(param, neg, min)

    # combined update
    return pos - neg


def upper_sharp(
    param: torch.Tensor,
    update: torch.Tensor,
    limit: float | torch.Tensor,
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
        limit (float | torch.Tensor): value of the upper bound, :math:`P_\text{max}`.

    Returns:
        torch.Tensor: bounded update.
    """
    diff = limit - param
    return torch.heaviside(diff, zeros(diff, shape=())) * update


def lower_sharp(
    param: torch.Tensor,
    update: torch.Tensor,
    limit: float | torch.Tensor,
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
        limit (float | torch.Tensor): value of the upper bound, :math:`P_\text{min}`.

    Returns:
        torch.Tensor: bounded update.
    """
    diff = param - limit
    return torch.heaviside(diff, zeros(diff, shape=())) * update


def sharp(
    param: torch.Tensor,
    pos: torch.Tensor,
    neg: torch.Tensor,
    max: float | torch.Tensor | None,
    min: float | torch.Tensor | None,
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
        max (float | torch.Tensor | None): value of the upper bound,
            :math:`P_\text{max}`.
        min (float | torch.Tensor | None): value of the lower bound,
            :math:`P_\text{min}`.

    Returns:
        torch.Tensor: bounded update.
    """
    # update subcomponents
    if max is not None:
        pos = upper_sharp(param, pos, max)
    if min is not None:
        neg = lower_sharp(param, neg, min)

    # combined update
    return pos - neg
