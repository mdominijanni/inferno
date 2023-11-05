import torch
import torch.nn as nn
from typing import Literal
from .. import Synapse, SynapseConstructor


class PassthroughSynapse(Synapse):
    r"""Synapse which directly returns values inputted.

    This generally acts as a "placeholder" synapse, when the inputs which are given are
    the same as what the connection should have. Spikes are determined by non-zero inputs
    to the :py:meth:`forward` call (negative or positive).

    Args:
        shape (tuple[int, ...] | int): shape of the group of synapses being simulated.
        step_time (float): length of a simulation time step, in :math:`\mathrm{ms}`.
        batch_size (int, optional): size of input batches for simualtion. Defaults to 1.
        delay (int, optional): maximum delay to support. Defaults to None.
        interpolation ("nearest" | "previous", optional): interpolation mode for float-type selectors.
            Defaults to "nearest".

    Note:
        if ``delay`` is None, the internal data structure will be different, and :py:meth:`dcurrent`
        and :py:meth:`dspike` will be invalid.

    Raises:
        ValueError: ``step_time`` must be positive.
        RuntimeError: ``interpolation`` must be 'nearest' or 'previous'.
    """

    def __init__(
        self,
        shape: tuple[int, ...] | int,
        step_time: float,
        *,
        batch_size: int = 1,
        delay: int | None = None,
        interpolation: Literal["nearest", "previous"] = "nearest",
    ):
        # tuplize shape
        try:
            shape = (int(shape),)
        except TypeError:
            shape = tuple(int(s) for s in shape)

        # call superclass constructor
        if delay is not None:
            currents = nn.Parameter(
                torch.zeros(batch_size, int(delay) + 1, *shape),
                requires_grad=False,
            )
            Synapse.__init__(
                self, shape, batch_size, batched_parameters=(("currents", currents))
            )
        else:
            Synapse.__init__(self, shape, batch_size, batched_parameters=("currents",))

        # check that the step time is valid
        if float(step_time) <= 0:
            raise ValueError(
                f"step time must be positive, received {float(step_time)}."
            )

        # register extras
        self.register_extra("step_time", float(step_time))
        self.register_extra("delay_max", None if delay is None else int(delay))

        # non-persistant function
        match interpolation:
            case "nearest":
                self.interp = torch.round
            case "previous":
                self.interp = torch.ceil
            case _:
                raise RuntimeError(
                    f"invalid `interpolation` of '{interpolation}' received"
                )

        # set values for parameters
        self.currents.fill_(0)

    @classmethod
    def partial(
        cls, interpolation: Literal["nearest", "previous"] = "nearest"
    ) -> SynapseConstructor:
        r"""Returns a function with a common signature for synapse construction.

        Args:
            interpolation ("nearest" | "previous", optional): interpolation mode for float-type selectors.
                Defaults to "nearest".

        Returns:
           SynapseConstructor: partial constructor for synapse.
        """

        def constructor(shape, step_time, batch_size, delay):
            return cls(shape, step_time, batch_size, delay, interpolation=interpolation)

        return constructor

    @property
    def dt(self) -> float:
        r"""Length of the simulation time step, in milliseconds.

        Returns:
            float: length of the simulation time step
        """
        return self.step_time

    @dt.setter
    def dt(self, value: float):
        self.step_time = float(value)

    @property
    def delay(self) -> int | None:
        r"""Maximum supported delay, as a multiple of simulation time steps.

        Returns:
            int | None: maximum number of buffered time steps.
        """
        return self.delay_max

    @property
    def current(self) -> torch.Tensor:
        r"""Currents of the synapses.

        Args:
            value (torch.Tensor): new synapse currents.

        Returns:
            torch.Tensor: currents of the synapses.

        Note:
            This will return the currents over the entire delay history.
        """
        return self.currents.data

    @current.setter
    def current(self, value: torch.Tensor):
        self.currents.data = value

    @property
    def spike(self) -> torch.Tensor:
        r"""Spikes to the synapses.

        Returns:
            torch.Tensor: spikes to the synapses.

        Note:
            This will return the spikes over the entire delay history.
        """
        return self.currents.data != 0

    def dcurrent(self, selector: torch.Tensor, out=None) -> torch.Tensor:
        r"""Returns currents at times specified by delays.

        Args:
            selector (torch.Tensor): delays for selection of currents.
            out (torch.Tensor | None, optional): output tensor, if required. Defaults to None.

        Returns:
            torch.Tensor: currents selected at the given times.

        Note:
            It is expected that if ``selector`` is of a floating point datatype, as assessed
            by :py:func:`torch.Tensor.is_floating_point`, then it will be assumed to be in
            :math:`ms`. If it is not, it will be assumed to be an integer multiple of step time.
        """
        if self.delay is None:
            raise RuntimeError("`currents` is invalid with delayed synapse")
        else:
            if selector.is_floating_point():
                return torch.gather(
                    self.current,
                    1,
                    self.interp(selector / self.dt)
                    .clamp(min=0, max=self.max_delay)
                    .long(),
                    out=out,
                )
            else:
                return torch.gather(self.current, 1, selector.long(), out=out)

    def dspike(self, selector: torch.Tensor, out=None) -> torch.Tensor:
        r"""Returns spikes at times specified by delays.

        Args:
            selector (torch.Tensor): delays for selection of spikes.
            out (torch.Tensor | None, optional): output tensor, if required. Defaults to None.

        Returns:
            torch.Tensor: spikes selected at the given times.

        Note:
            It is expected that if ``selector`` is of a floating point datatype, as assessed
            by :py:func:`torch.Tensor.is_floating_point`, then it will be assumed to be in
            :math:`ms`. If it is not, it will be assumed to be an integer multiple of step time.
        """
        if self.delay is None:
            raise RuntimeError("`spikes` is invalid with delayed synapse")
        else:
            if selector.is_floating_point():
                return torch.gather(
                    self.spike,
                    1,
                    self.interp(selector / self.dt)
                    .clamp(min=0, max=self.max_delay)
                    .long(),
                    out=out,
                )
            else:
                return torch.gather(self.spike, 1, selector.long(), out=out)

    def clear(self):
        r"""Resets synapses to their resting state."""
        self.currents.fill_(0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        r"""Runs a simulation step of the synaptic kinetics.

        Args:
            inputs (torch.Tensor): spike-like input.

        Returns:
            torch.Tensor: currents resulting from the kinetics.
        """
        if self.delay is None:
            self.currents.data[:] = inputs
            return self.currents.data
        else:
            self.currents.data = torch.roll(self.currents.data, shifts=1, dims=1)
            self.currents.data[:, 0] = inputs
            return self.currents.data[:, 0]
