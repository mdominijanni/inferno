import functools  # noqa:F401, for docstrings
import inferno
import torch
from typing import Protocol


class UpdateBounding(Protocol):
    r"""Callable used to apply bounding to amplitudes.

    Args:
        parameter (torch.Tensor): parameter being bound.
        amplitude (float | torch.Tensor): amplitude of the update without bounding
            (e.g. the learning rate).

    Returns:
        torch.Tensor: bound update amplitudes.

    See Also:
        Provided parameter bounding functions include
        :py:func:`power_upper_bounding`, :py:func:`power_lower_bounding`,
        :py:func:`hard_upper_bounding`, and :py:func:`hard_lower_bounding`.
        Use a :py:func:`~functools.partial` function to fill in keyword arguments.
    """

    def __call__(
        self,
        parameter: torch.Tensor,
        amplitude: float | torch.Tensor,
        /,
    ) -> torch.Tensor:
        r"""Callback protocol function."""
        ...


def power_upper_bounding(
    parameter: torch.Tensor,
    amplitude: float | torch.Tensor,
    /,
    max: float | torch.Tensor,
    power: float = 1.0,
) -> torch.Tensor:
    r"""Applies soft parameter bounding on potentiative update amplitude.

    .. math::

        A_+ = (v_\text{max} - w)^{\mu_+} \eta_+

    Args:
        parameter (torch.Tensor): model parameter, :math:`v`.
        amplitude (float | torch.Tensor): amplitude of the update excluding parameter
            dependence, :math:`\eta_+`.
        max (float | torch.Tensor): upper bound of parameter, :math:`v_\text{max}`.
        power (float, optional): exponent of parameter dependence, :math:`\mu_+`.
            Defaults to 1.0.

    Returns:
        torch.Tensor: amplitudes :math:`A_+` after applying parameter bound.

    See Also:
        For more details and references, visit
        :ref:`zoo/learning-stdp:Weight Dependence, Soft Bounding` in the zoo.
    """
    return ((max - parameter) ** power) * amplitude


def power_lower_bounding(
    parameter: torch.Tensor,
    amplitude: float | torch.Tensor,
    /,
    min: float | torch.Tensor,
    power: float = 1.0,
) -> torch.Tensor:
    r"""Applies soft parameter bounding on depressive update amplitude.

    .. math::

        A_- = (w - v_\text{min})^{\mu_-} \eta_-


    Args:
        parameter (torch.Tensor): model parameter, :math:`v`.
        amplitude (float | torch.Tensor): amplitude of the update excluding parameter
            dependence, :math:`\eta_-`.
        min (float | torch.Tensor): lower bound of parameter, :math:`v_\text{min}`.
        power (float, optional): exponent of parameter dependence, :math:`\mu_-`.
            Defaults to 1.0.

    Returns:
        torch.Tensor: amplitudes :math:`A_-` after applying parameter bound.

    See Also:
        For more details and references, visit
        :ref:`zoo/learning-stdp:Weight Dependence, Soft Bounding` in the zoo.
    """
    return ((parameter - min) ** power) * amplitude


def hard_upper_bounding(
    parameter: torch.Tensor,
    amplitude: float | torch.Tensor,
    /,
    max: float | torch.Tensor,
) -> torch.Tensor:
    r"""Applies hard parameter bounding on potentiative update amplitude.

    .. math::

        A_+ = \Theta(v_\text{max} - w) \eta_+

    Where

    .. math::

        \Theta(x) =
        \begin{cases}
            1 &x \geq 0 \\
            0 & x < 0
        \end{cases}

    Args:
        parameter (torch.Tensor): model parameter, :math:`v`.
        amplitude (float | torch.Tensor): amplitude of the update excluding parameter
            dependence, :math:`\eta_+`.
        max (float | torch.Tensor): upper bound of parameter, :math:`v_\text{max}`.

    Returns:
        torch.Tensor: amplitudes :math:`A_+` after applying parameter bound.

    See Also:
        For more details and references, visit
        :ref:`zoo/learning-stdp:Weight Dependence, Hard Bounding` in the zoo.
    """
    diff = max - parameter
    return torch.heaviside(diff, inferno.zeros(diff, shape=())) * amplitude


def hard_lower_bounding(
    parameter: torch.Tensor,
    amplitude: float | torch.Tensor,
    /,
    min: float | torch.Tensor,
) -> torch.Tensor:
    r"""Applies hard parameter bounding on depressive update amplitude.

    .. math::

        A_- = \Theta(w - v_\text{min}) \eta_-

    .. math::

        \Theta(x) =
        \begin{cases}
            1 &x \geq 0 \\
            0 & x < 0
        \end{cases}

    Args:
        parameter (torch.Tensor): model parameter, :math:`v`.
        amplitude (float | torch.Tensor): amplitude of the update excluding parameter
            dependence, :math:`\eta_-`.
        min (float | torch.Tensor): lower bound of parameter, :math:`v_\text{min}`.

    Returns:
        torch.Tensor: amplitudes :math:`A_-` after applying parameter bound.

    See Also:
        For more details and references, visit
        :ref:`zoo/learning-stdp:Weight Dependence, Hard Bounding` in the zoo.
    """
    diff = parameter - min
    return torch.heaviside(diff, inferno.zeros(diff, shape=())) * amplitude
