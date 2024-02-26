import inferno
import torch
from typing import Protocol


class HalfBounding(Protocol):
    r"""Callable used to apply bounding to the lower or upper limit of a parameter.

    Args:
        param (torch.Tensor): value of the parameter being bound.
        update (torch.Tensor): value of the partial update to scale.
        limit (float | torch.Tensor): maximum or minimum value of the updated parameter.

    Returns:
        torch.Tensor: bounded update.

    See Also:
        Provided parameter half-bounding functions include
        :py:func:`upper_power_dependence`, :py:func:`upper_lower_dependence`,
        :py:func:`upper_multiplicative_dependence`,
        :py:func:`lower_multiplicative_dependence`, :py:func`upper_sharp_dependence`,
        and :py:func:`lower_sharp_dependence`.
    """

    def __call__(
        self,
        param: torch.Tensor,
        update: torch.Tensor,
        limit: float | torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        r"""Callback protocol function."""
        ...


class FullBounding(Protocol):
    r"""Callable used to apply bounding to the lower and upper limit of a parameter.

    Args:
        param (torch.Tensor): value of the parameter being bound.
        pos_update (torch.Tensor): value of the positive part of the update to scale.
        neg_update (torch.Tensor): value of the negative part of the update to scale.
        max_limit (float | torch.Tensor | None): maximum value of the updated parameter.
        min_limit (float | torch.Tensor | None): minimum value of the updated parameter.

    Returns:
        torch.Tensor: bounded update.

    See Also:
        Provided parameter bounding functions include
        :py:func:`power_dependence`, :py:func:`multiplicative_dependence`, and
        :py:func:`sharp_dependence`.
    """

    def __call__(
        self,
        param: torch.Tensor,
        pos_update: torch.Tensor,
        neg_update: torch.Tensor,
        max_limit: float | torch.Tensor | None,
        min_limit: float | torch.Tensor | None,
        **kwargs,
    ) -> torch.Tensor:
        r"""Callback protocol function."""
        ...


def upper_power_dependence(
    param: torch.Tensor,
    update: torch.Tensor,
    limit: float | torch.Tensor,
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


def lower_power_dependence(
    param: torch.Tensor,
    update: torch.Tensor,
    limit: float | torch.Tensor,
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


def power_dependence(
    param: torch.Tensor,
    pos_update: torch.Tensor,
    neg_update: torch.Tensor,
    max_limit: float | torch.Tensor | None,
    min_limit: float | torch.Tensor | None,
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
        pos_update (torch.Tensor): potentiative update being applied, :math:`U_+`.
        neg_update (torch.Tensor): depressive update being applied, :math:`U_-`.
        max_limit (float | torch.Tensor | None): value of the upper bound,
            :math:`P_\text{max}`.
        min_limit (float | torch.Tensor | None): value of the lower bound,
            :math:`P_\text{min}`.
        upper_power (float | torch.Tensor): exponent of upper-bound parameter
            dependence, :math:`\mu_+`.
        lower_power (float | torch.Tensor): exponent of lower-bound parameter
            dependence, :math:`\mu_-`.

    Returns:
        torch.Tensor: bounded update.
    """
    # update subcomponents
    if max_limit is not None:
        pos_update = upper_power_dependence(param, pos_update, max_limit, upper_power)
    if min_limit is not None:
        neg_update = lower_power_dependence(param, neg_update, min_limit, lower_power)

    # combined update
    return pos_update - neg_update


def upper_multiplicative_dependence(
    param: torch.Tensor,
    update: torch.Tensor,
    limit: float | torch.Tensor,
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


def lower_multiplicative_dependence(
    param: torch.Tensor,
    update: torch.Tensor,
    limit: float | torch.Tensor,
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


def multiplicative_dependence(
    param: torch.Tensor,
    pos_update: torch.Tensor,
    neg_update: torch.Tensor,
    max_limit: float | torch.Tensor | None,
    min_limit: float | torch.Tensor | None,
    **kwargs,
) -> torch.Tensor:
    r"""Computes the scaled update of multiplicative parameter dependence.

    This is sometimes also referred to as "soft parameter dependence" and is equivalent
    to power dependence with an exponent of 1.

    .. math::
        U = (P_\text{max} - P) U_+ - (P - P_\text{min}) U_-

    Args:
        param (torch.Tensor): parameter with update bounding, :math:`P`.
        pos_update (torch.Tensor): potentiative update being applied, :math:`U_+`.
        neg_update (torch.Tensor): depressive update being applied, :math:`U_-`.
        max_limit (float | torch.Tensor | None): value of the upper bound,
            :math:`P_\text{max}`.
        min_limit (float | torch.Tensor | None): value of the lower bound,
            :math:`P_\text{min}`.

    Returns:
        torch.Tensor: bounded update.
    """
    # update subcomponents
    if max_limit is not None:
        pos_update = upper_multiplicative_dependence(param, pos_update, max_limit)
    if min_limit is not None:
        neg_update = lower_multiplicative_dependence(param, neg_update, min_limit)

    # combined update
    return pos_update - neg_update


def upper_sharp_dependence(
    param: torch.Tensor,
    update: torch.Tensor,
    limit: float | torch.Tensor,
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
    return torch.heaviside(diff, inferno.zeros(diff, shape=())) * update


def lower_sharp_dependence(
    param: torch.Tensor,
    update: torch.Tensor,
    limit: float | torch.Tensor,
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
    return torch.heaviside(diff, inferno.zeros(diff, shape=())) * update


def sharp_dependence(
    param: torch.Tensor,
    pos_update: torch.Tensor,
    neg_update: torch.Tensor,
    max_limit: float | torch.Tensor | None,
    min_limit: float | torch.Tensor | None,
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
        pos_update (torch.Tensor): potentiative update being applied, :math:`U_+`.
        neg_update (torch.Tensor): depressive update being applied, :math:`U_-`.
        max_limit (float | torch.Tensor | None): value of the upper bound,
            :math:`P_\text{max}`.
        min_limit (float | torch.Tensor | None): value of the lower bound,
            :math:`P_\text{min}`.

    Returns:
        torch.Tensor: bounded update.
    """
    # update subcomponents
    if max_limit is not None:
        pos_update = upper_sharp_dependence(param, pos_update, max_limit)
    if min_limit is not None:
        neg_update = lower_sharp_dependence(param, neg_update, min_limit)

    # combined update
    return pos_update - neg_update
