import inferno
import torch


def hard_bounding_pot(weight: torch.Tensor, wmax: float | torch.Tensor) -> torch.Tensor:
    r"""Multiplier for hard bounding of potentiative update.

    .. math::
        \frac{A_+}{\eta_+} = \Theta(w_\mathrm{max} - w)

    Args:
        weight (torch.Tensor): weights to bound.
        wmax (float | torch.Tensor): upper bound of weights.

    Returns:
        torch.Tensor: multiplier for bounded LTP update, shaped like ``weight``.

    Note:
        For more details, visit :ref:`zoo/learning-stdp:Hard Weight Bounding` in the zoo.
    """
    diff = wmax - weight
    return torch.heaviside(diff, inferno.zeros(diff, shape=((),)))


def hard_bounding_dep(weight: torch.Tensor, wmin: float | torch.Tensor) -> torch.Tensor:
    r"""Multiplier for hard bounding of depressive update.

    .. math::
        \frac{A_-}{\eta_-} = \Theta(w - w_\mathrm{min})

    Args:
        weight (torch.Tensor): weights to bound.
        wmin (float | torch.Tensor): lower bound of weights.

    Returns:
        torch.Tensor: multiplier for bounded LTD update, shaped like ``weight``.

    Note:
        For more details, visit :ref:`zoo/learning-stdp:Hard Weight Bounding` in the zoo.
    """
    diff = weight - wmin
    return torch.heaviside(diff, inferno.zeros(diff, shape=((),)))


def soft_bounding_pot(
    weight: torch.Tensor, wmax: float | torch.Tensor, wdpexp: float | torch.Tensor = 1.0
) -> torch.Tensor:
    r"""Multiplier for soft bounding (weight dependence) of potentiative update.

    .. math::
        \frac{A_+}{\eta_+} = (w_\mathrm{max} - w)^{\mu_+}

    Args:
        weight (torch.Tensor): weights to bound.
        wmax (float | torch.Tensor): upper bound of weights.
        wdpexp (float | torch.Tensor, optional): exponent of weight dependence (LTP). Defaults to 1.0.

    Returns:
        torch.Tensor: multiplier for bounded LTP update, shaped like ``weight``.

    Note:
        For more details, visit
        :ref:`zoo/learning-stdp:Soft Weight Bounding (Weight Dependence)` in the zoo.
    """
    return (wmax - weight) ** wdpexp


def soft_bounding_dep(
    weight: torch.Tensor, wmin: float | torch.Tensor, wddexp: float | torch.Tensor = 1.0
) -> torch.Tensor:
    r"""Multiplier for soft bounding (weight dependence) of depressive update.

    .. math::
        \frac{A_-}{\eta_-} = (w - w_\mathrm{min})^{\mu_-}

    Args:
        weight (torch.Tensor): weights to bound.
        wmin (float | torch.Tensor): lower bound of weights.
        wdpexp (float | torch.Tensor, optional): exponent of weight dependence (LTD). Defaults to 1.0.

    Returns:
        torch.Tensor: multiplier for bounded LTD update, shaped like ``weight``.

    Note:
        For more details, visit
        :ref:`zoo/learning-stdp:Soft Weight Bounding (Weight Dependence)` in the zoo.
    """
    return (weight - wmin) ** wddexp


def stdp_pot(
    trace_presyn: torch.Tensor,
    spike_postsyn: torch.Tensor,
    learning_rate: float | torch.Tensor,
):
    pass
