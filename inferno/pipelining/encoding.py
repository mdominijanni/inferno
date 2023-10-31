import torch


def interval_poisson(
    inputs: torch.Tensor,
    step_time: float,
    num_steps: int,
    batch_dim: int | None = None,
    rates_as_probabilities: bool = False,
    return_generator: bool = True
):
    # convert inputs as either probabilities or frequency in Hz, to rates for a Poisson distribution
    if rates_as_probabilities:
        as_rates = None
        raise NotImplementedError('probabilistic rate not yet implemented')
    else:
        as_rates = lambda x: torch.nan_to_num(1 / x, posinf=0) * (1000 / step_time)

    # add a batch dimension if none is given, otherwise move it to the zeroth position
    if batch_dim is None:
        as_batch = lambda x: x.unsqueeze(0)
    else:
        as_batch = lambda x: x.movedim(int(batch_dim) % len(tuple(x.shape)), 0)

    with torch.no_grad():
        # apply the afformentioned to get to propertly shaped rates
        rates = as_batch(as_rates(inputs))
        # get the intervals as sampled from a Poisson distribution, increment non-zero inputs by one to avoid collisions
        intervals = torch.poisson(rates.expand([num_steps + 1] + [-1] * rates.ndim))
        intervals[:, rates != 0] += (intervals[:, rates != 0] == 0).to(dtype=intervals.dtype)
        # convert intervals to times through cumulative summation and replace values of the length of the spike sequence
        times = torch.cumsum(intervals, dim=0)
        times[times >= num_steps] = 0
        # use times as indices and scatter accordingly
        spikes = torch.zeros_like(times).scatter_(0, times.long(), 1)[1:].bool()

    if return_generator:
        return (elem for elem in spikes)
    else:
        return spikes
