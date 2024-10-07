import torch


def exp_stdp_post_kernel(
    diff: torch.Tensor,
    learning_rate: float,
    time_constant: float,
    **kwargs,
) -> torch.Tensor:
    r"""Returns the postsynaptic update for exponential spike-timing dependent plasticity.

    .. math::
        K_\text{post}(t_\Delta(t)) =
        \eta \exp\left(-\frac{\lvert t_\Delta(t) \rvert}{\tau} \right) [t_\Delta(t) \geq 0]

    Args:
        diff (torch.Tensor): duration of time, possibly adjusted, between presynaptic
            and postsynaptic spikes, :math:`t_\Delta(t)`, in :math:`\text{ms}`.
        learning_rate (float): learning rate for update component, :math:`\eta`.
        time_constant (float): time constant of exponential decay for update component,
            :math:`\tau`, in :math:`\text{ms}`.

    Returns:
        torch.Tensor: unreduced update component.

    Note:
        See :py:class:`~inferno.functional.protocols.SpikeTimeHalfKernel`
        for more information on the expected inputs and outputs.
    """
    return torch.exp(diff.abs() / (-time_constant)) * (
        learning_rate * (diff >= 0).to(dtype=diff.dtype)
    )


def exp_stdp_pre_kernel(
    diff: torch.Tensor,
    learning_rate: float,
    time_constant: float,
    **kwargs,
) -> torch.Tensor:
    r"""Returns the presynaptic update for exponential spike-timing dependent plasticity.

    .. math::
        K_\text{pre}(t_\Delta(t)) =
        \eta \exp\left(-\frac{\lvert t_\Delta(t) \rvert}{\tau} \right) [t_\Delta(t) < 0]

    Args:
        diff (torch.Tensor): duration of time, possibly adjusted, between presynaptic
            and postsynaptic spikes, :math:`t_\Delta(t)`, in :math:`\text{ms}`.
        learning_rate (float): learning rate for update component, :math:`\eta`.
        time_constant (float): time constant of exponential decay for update component,
            :math:`\tau`, in :math:`\text{ms}`.

    Returns:
        torch.Tensor: unreduced update component.

    Note:
        See :py:class:`~inferno.functional.protocols.SpikeTimeHalfKernel`
        for more information on the expected inputs and outputs.
    """
    return torch.exp(diff.abs() / (-time_constant)) * (
        learning_rate * (diff < 0).to(dtype=diff.dtype)
    )
