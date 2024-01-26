import functools  # noqa:F401, for docstrings
import inferno
import torch
from typing import Protocol


class BindWeights(Protocol):
    r"""Callable used to apply weight bounding to amplitudes.

    Args:
        weights (torch.Tensor): model weights.
        amplitude (float | torch.Tensor): amplitude of the update without weight
            bounding (e.g. the learning rate).

    Returns:
        torch.Tensor: weight-bound update amplitudes.

    See Also:
        Provided weight bounding functions include
        :py:func:`wdep_soft_upper_bounding`, :py:func:`wdep_soft_lower_bounding`,
        :py:func:`wdep_hard_upper_bounding`, and :py:func:`wdep_hard_lower_bounding`.
        Use a :py:func:`~functools.partial` function to fill in keyword arguments.
    """

    def __call__(
        self,
        weights: torch.Tensor,
        amplitude: float | torch.Tensor,
        /,
    ) -> torch.Tensor:
        r"""Callback protocol function."""
        ...


def wdep_soft_upper_bounding(
    weights: torch.Tensor,
    amplitude: float | torch.Tensor,
    /,
    wmax: float | torch.Tensor,
    power: float = 1.0,
) -> torch.Tensor:
    r"""Applies soft weight bounding on potentiative update amplitude.

    .. math::

        A_+ = (w_\text{max} - w)^{\mu_+} \eta_+

    Args:
        weights (torch.Tensor): model weights, :math:`w`.
        amplitude (float | torch.Tensor): amplitude of the update excluding weight
            dependence, :math:`\eta_+`.
        wmax (float | torch.Tensor): upper bound of weights, :math:`w_\text{max}`.
        power (float, optional): exponent of weight dependence, :math:`\mu_+`.
            Defaults to 1.0.

    Returns:
        torch.Tensor: amplitudes :math:`A_+` after applying weight bound.

    See Also:
        For more details and references, visit
        :ref:`zoo/learning-stdp:Weight Dependence, Soft Bounding` in the zoo.
    """
    return ((wmax - weights) ** power) * amplitude


def wdep_soft_lower_bounding(
    weights: torch.Tensor,
    amplitude: float | torch.Tensor,
    /,
    wmin: float | torch.Tensor,
    power: float = 1.0,
) -> torch.Tensor:
    r"""Applies soft weight bounding on depressive update amplitude.

    .. math::

        A_- = (w - w_\text{min})^{\mu_-} \eta_-


    Args:
        weights (torch.Tensor): model weights, :math:`w`.
        amplitude (float | torch.Tensor): amplitude of the update excluding weight
            dependence, :math:`\eta_-`.
        wmin (float | torch.Tensor): lower bound of weights, :math:`w_\text{min}`.
        power (float, optional): exponent of weight dependence, :math:`\mu_-`.
            Defaults to 1.0.

    Returns:
        torch.Tensor: amplitudes :math:`A_-` after applying weight bound.

    See Also:
        For more details and references, visit
        :ref:`zoo/learning-stdp:Weight Dependence, Soft Bounding` in the zoo.
    """
    return ((weights - wmin) ** power) * amplitude


def wdep_hard_upper_bounding(
    weights: torch.Tensor,
    amplitude: float | torch.Tensor,
    /,
    wmax: float | torch.Tensor,
) -> torch.Tensor:
    r"""Applies hard weight bounding on potentiative update amplitude.

    .. math::

        A_+ = \Theta(w_\text{max} - w) \eta_+

    Where

    .. math::

        \Theta(x) =
        \begin{cases}
            1 &x \geq 0 \\
            0 & x < 0
        \end{cases}

    Args:
        weights (torch.Tensor): model weights, :math:`w`.
        amplitude (float | torch.Tensor): amplitude of the update excluding weight
            dependence, :math:`\eta_+`.
        wmax (float | torch.Tensor): upper bound of weights, :math:`w_\text{max}`.

    Returns:
        torch.Tensor: amplitudes :math:`A_+` after applying weight bound.

    See Also:
        For more details and references, visit
        :ref:`zoo/learning-stdp:Weight Dependence, Hard Bounding` in the zoo.
    """
    diff = wmax - weights
    return torch.heaviside(diff, inferno.zeros(diff, shape=())) * amplitude


def wdep_hard_lower_bounding(
    weights: torch.Tensor,
    amplitude: float | torch.Tensor,
    /,
    wmin: float | torch.Tensor,
) -> torch.Tensor:
    r"""Applies hard weight bounding on depressive update amplitude.

    .. math::

        A_- = \Theta(w - w_\text{min}) \eta_-

    .. math::

        \Theta(x) =
        \begin{cases}
            1 &x \geq 0 \\
            0 & x < 0
        \end{cases}

    Args:
        weights (torch.Tensor): model weights, :math:`w`.
        amplitude (float | torch.Tensor): amplitude of the update excluding weight
            dependence, :math:`\eta_-`.
        wmin (float | torch.Tensor): lower bound of weights, :math:`w_\text{min}`.

    Returns:
        torch.Tensor: amplitudes :math:`A_-` after applying weight bound.

    See Also:
        For more details and references, visit
        :ref:`zoo/learning-stdp:Weight Dependence, Hard Bounding` in the zoo.
    """
    diff = weights - wmin
    return torch.heaviside(diff, inferno.zeros(diff, shape=())) * amplitude
