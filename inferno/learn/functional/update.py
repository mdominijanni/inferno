import functools  # noqa:F401, for docstrings
import inferno
import torch
from typing import Protocol


class WeightDependence(Protocol):
    r"""Callable used to apply weight dependence to amplitudes.

    Args:
        weights (torch.Tensor): model weights.
        bound (float | torch.Tensor): minimum or maximum value for weights.
        amplitude (float | torch.Tensor): amplitude of the update without weight
            dependence (e.g. the learning rate).

    Returns:
        torch.Tensor: weight-dependant update amplitudes.

    See Also:
        Provided weight dependence functions include
        :py:func:`wdep_soft_upper_bounding`, :py:func:`wdep_soft_lower_bounding`,
        :py:func:`wdep_hard_upper_bounding`, and :py:func:`wdep_hard_lower_bounding`.
        Some provided functions may require keyword arguments. For arguments which
        require an object of type ``WeightDependence``,
        use a :py:func:`~functools.partial` function to fill in keyword arguments.
    """

    def __call__(
        self,
        weights: torch.Tensor,
        bound: float | torch.Tensor,
        amplitude: float | torch.Tensor,
        /,
    ) -> torch.Tensor:
        r"""Callback protocol function."""
        ...


def wdep_soft_upper_bounding(
    weights: torch.Tensor,
    wmax: float | torch.Tensor,
    amplitude: float | torch.Tensor,
    /,
    power: float = 1.0,
) -> torch.Tensor:
    r"""Applies soft weight bounding on potentiative update amplitude.

    .. math::

        A_+ = (w_\text{max} - w)^{\mu_-} \eta_+

    Where

    .. math::

        \Theta(x) =
        \begin{cases}
            1 &x \geq 0 \\
            0 & x < 0
        \end{cases}


    Args:
        weights (torch.Tensor): model weights, :math:`w`.
        wmax (float | torch.Tensor): upper bound of weights, :math:`w_\text{max}`.
        amplitude (float | torch.Tensor): amplitude of the update excluding weight
            dependence, :math:`\eta_+`.
        power (float, optional): exponent of weight dependence, :math:`\mu_+`.
            Defaults to 1.0.

    Returns:
        torch.Tensor: amplitudes :math:`A_+` after applying weight bound.
    """
    return ((wmax - weights) ** power) * amplitude


def wdep_soft_lower_bounding(
    weights: torch.Tensor,
    wmin: float | torch.Tensor,
    amplitude: float | torch.Tensor,
    /,
    power: float = 1.0,
) -> torch.Tensor:
    r"""Applies soft weight bounding on depressive update amplitude.

    .. math::

        A_- = (w - w_\text{min})^{\mu_-} \eta_-


    Args:
        weights (torch.Tensor): model weights, :math:`w`.
        wmin (float | torch.Tensor): lower bound of weights, :math:`w_\text{min}`.
        amplitude (float | torch.Tensor): amplitude of the update excluding weight
            dependence, :math:`\eta_-`.
        power (float, optional): exponent of weight dependence, :math:`\mu_-`.
            Defaults to 1.0.

    Returns:
        torch.Tensor: amplitudes :math:`A_-` after applying weight bound.

    """
    return ((weights - wmin) ** power) * amplitude


def wdep_hard_upper_bounding(
    weights: torch.Tensor,
    wmax: float | torch.Tensor,
    amplitude: float | torch.Tensor,
    /,
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
        wmax (float | torch.Tensor): upper bound of weights, :math:`w_\text{max}`.
        amplitude (float | torch.Tensor): amplitude of the update excluding weight
            dependence, :math:`\eta_+`.

    Returns:
        torch.Tensor: amplitudes :math:`A_+` after applying weight bound.
    """
    diff = wmax - weights
    return torch.heaviside(diff, inferno.zeros(diff, shape=((),))) * amplitude


def wdep_hard_lower_bounding(
    weights: torch.Tensor,
    wmin: float | torch.Tensor,
    amplitude: float | torch.Tensor,
    /,
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
        wmin (float | torch.Tensor): lower bound of weights, :math:`w_\text{min}`.
        amplitude (float | torch.Tensor): amplitude of the update excluding weight
            dependence, :math:`\eta_-`.

    Returns:
        torch.Tensor: amplitudes :math:`A_-` after applying weight bound.
    """
    diff = weights - wmin
    return torch.heaviside(diff, inferno.zeros(diff, shape=((),))) * amplitude
