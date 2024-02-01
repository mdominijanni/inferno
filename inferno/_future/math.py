import torch
import torch.nn as nn
import torch.nn.functional as F


def isi(spikes: torch.Tensor, step_time: float) -> torch.Tensor:
    r"""Transforms spike trains into inter-spike intervals.

    The returned tensor will be padded with ``NaN``s where an interval could not
    be computed but the position existed (e.g. padded to the end of) spike trains
    with fewer spikes. If no intervals could be generated at all, a tensor with a
    final dimension of zero will be returned.

    Args:
        spikes (torch.Tensor): spike trains to calculate intervals for.
        step_time (float): length of the simulation step, in :math:`\text{ms}`.

    Returns:
        torch.Tensor: interspike intervals for the given spike trains.

    .. admonition:: Shape
        :class: tensorshape

        ``spikes``:

        :math:`N_0 \times \cdots \times T`

        ``return``:

        :math:`N_0 \times \cdots (C - 1)`

        Where:
            * :math:`N_0, \ldots` shape of the generating population (batch, neuron shape, etc).
            * :math:`T` the length of the spike trains.
            * :math:`C` the maximum number of spikes amongst the spike trains.
    """
    # pad spikes with true to ensure at least one (req. for split)
    padded = F.pad(spikes, (1, 0), mode="constant", value=True)

    # compute nonzero values
    nz = torch.nonzero(padded)[..., -1]

    # compute split indices (at the added pads)
    splits = torch.nonzero(torch.logical_not(nz)).view(-1).tolist()[1:]

    # split the tensor into various length subtensors (subtract 1 to unshift)
    intervals = torch.tensor_split((nz - 1) * float(step_time), splits, dim=-1)

    # stack, pad trailing with nan, trim leading pad
    intervals = nn.utils.rnn.pad_sequence(
        intervals, batch_first=True, padding_value=float("nan")
    )[:, 1:]

    # compute intervals
    intervals = torch.diff(intervals, dim=-1)

    # reshape and return
    return intervals.view(*spikes.shape[:-1], -1)
