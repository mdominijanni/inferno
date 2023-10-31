import einops as ein
import math
import torch
import torch.nn.functional as F


def dense_linear(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    biases: torch.Tensor | None = None,
):
    r"""Applies a linear all-to-all connection on inputs.

    .. math::
        y = \mathbf{x} \mathbf{W}^\intercal + \mathbf{b}

    Args:
        inputs (torch.Tensor): inputs to linear transformation, :math:`\mathbf{x}`.
        weights (torch.Tensor): weights for linear transformation, :math:`\mathbf{W}`.
        biases (torch.Tensor | None, optional): bias for linear transformation, :math:`\mathbf{b}`. Defaults to None.

    Raises:
        RuntimeError: shape of ``inputs`` and ``weights`` are incompatible.

    Returns:
        torch.Tensor: linearly transformed inputs.

    Shape:
        ``inputs``: :math:`[B] \times N_0 \times \cdots`,
        where the batch dimension :math:`B` is optional.

        ``weights``: :math:`M \times \prod N_0 \cdots`

        ``biases``: :math:`M`

        **outputs**:
        :math:`[B] \times M`,
        where the batch dimension :math:`B` is contingent on the input.
    """
    # [N] dimensional input
    if inputs.ndim == 1 and inputs.shape[0] == weights.shape[1]:  # [N]
        return F.linear(inputs, weights, biases)

    # [B, N0, ...] dimensional input
    elif math.prod(inputs.shape[1:]) == weights.shape[1]:  # [B, N0, ...]
        return F.linear(ein.rearrange(inputs, "b ... -> b (...)"), weights, biases)

    # [N0, ...] dimensional input
    elif math.prod(inputs.shape) == weights.shape[1]:  # [N0, ...]
        return F.linear(ein.rearrange(inputs, "... -> (...)"), weights, biases)

    # invalid input shape
    else:
        raise RuntimeError(
            f"`inputs` of shape {tuple(inputs.shape)} and "
            f"`weights` of shape {tuple(weights.shape)} are incompatible."
        )


def direct_linear(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    biases: torch.Tensor | None = None,
):
    r"""Applies a linear one-to-one connection on inputs.

    .. math::
        y = \sum \mathbf{x} \left[\mathbf{W} \odot I\right]^\intercal + \mathbf{b}

    Args:
        inputs (torch.Tensor): inputs to linear transformation, :math:`\mathbf{x}`.
        weights (torch.Tensor): weights for linear transformation, :math:`\mathbf{W}`.
        biases (torch.Tensor | None, optional): bias for linear transformation, :math:`\mathbf{b}`. Defaults to None.

    Raises:
        RuntimeError: ``weights`` are of an invalid shape (non-square).

    Returns:
        torch.Tensor: linearly transformed inputs.

    Shape:
        ``inputs``: :math:`[B] \times N_0 \times \cdots`,
        where the batch dimension :math:`B` is optional.

        ``weights``: :math:`\prod N_0 \cdots \times \prod N_0 \cdots` or
        :math:`\prod N_0 \cdots`

        ``biases``: :math:`\prod N_0 \cdots`

        **outputs**:
        :math:`[B] \times \prod N_0 \cdots`,
        where the batch dimension :math:`B` is contingent on the input.
    """
    # 1D weights
    if weights.ndim == 1:
        return dense_linear(inputs, torch.diagflat(weights), biases)

    # 2D weights
    elif weights.ndim == 2 and weights.shape[0] == weights.shape[1]:
        return dense_linear(
            inputs,
            weights * torch.eye(weights.shape[0], out=torch.empty_like(weights)),
            biases,
        )

    # invalid weight shape
    else:
        raise RuntimeError(
            f"`weights` are of an invalid shape {tuple(weights.shape)} "
            "must be a 2D tensor with equal-sized dimensions."
        )


def lateral_linear(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    biases: torch.Tensor | None = None,
):
    r"""Applies a linear all-to-"all-but-one" connection on inputs.

    .. math::
        y = \sum \mathbf{x} \left[\mathbf{W} \odot (1 - I)\right]^\intercal + \mathbf{b}

    Args:
        inputs (torch.Tensor): inputs to linear transformation, :math:`\mathbf{x}`.
        weights (torch.Tensor): weights for linear transformation, :math:`\mathbf{W}`.
        biases (torch.Tensor | None, optional): bias for linear transformation, :math:`\mathbf{b}`. Defaults to None.

    Raises:
        RuntimeError: ``weights`` are of an invalid shape (non-square).

    Returns:
        torch.Tensor: linearly transformed inputs.

    Shape:
        ``inputs``: :math:`[B] \times N_0 \times \cdots`,
        where the batch dimension :math:`B` is optional

        ``weights``: :math:`\prod N_0 \cdots \times \prod N_0`

        ``biases``: :math:`\prod N_0 \cdots`

        **outputs**:
        :math:`[B] \times \prod N_0 \cdots`,
        where the batch dimension :math:`B` is contingent on the input.
    """
    # 2D weights
    if weights.ndim == 2 and weights.shape[0] == weights.shape[1]:
        return dense_linear(
            inputs,
            weights * (1 - torch.eye(weights.shape[0], out=torch.empty_like(weights))),
            biases,
        )

    # invalid weight shape
    else:
        raise RuntimeError(
            f"`weights` are of an invalid shape {tuple(weights.shape)} "
            "must be a 2D tensor with equal-sized dimensions."
        )
