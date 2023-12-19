import inferno
import torch
import torch.nn as nn
from typing import Literal
import warnings
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
        delay (float, optional): maximum delay to support. Defaults to None.
        interpolation ("nearest" | "previous", optional): interpolation mode for non-integer multiple selectors.
            Defaults to "nearest".
        spikes_from_currents (bool, optional): if spiking input should be assumed from current-like input.
            Defaults to True.
        current_overbound (float | None, optional): value to replace values out of bounds, use values at
                final record if none. Defaults to 0.0.
        spike_overbound (bool | None, optional): value to replace values out of bounds, use values at
                final record if none. Defaults to False.

    Note:
        If ``delay`` is None, the internal data structure will be different, and :py:meth:`current_at`
        and :py:meth:`spike_at` will be invalid.

    Note:
        When ``interpolation`` is set to `"nearest"`, the closest record will be used, even if it is in the future.
        When it is set to `"previous"` the most recent will be used.

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
        delay: float | None = None,
        interpolation: Literal["nearest", "previous"] = "previous",
        spikes_from_currents: bool = True,
        current_overbound: float | None = 0.0,
        spike_overbound: bool | None = False,
    ):
        # tuplize shape
        try:
            shape = (int(shape),)
        except TypeError:
            shape = tuple(int(s) for s in shape)

        # check that the step time is valid
        if float(step_time) <= 0:
            raise ValueError(
                f"step time must be greater than zero, received {float(step_time)}"
            )

        # check that the delay is valid
        if delay is not None and float(delay) <= 0:
            raise ValueError(
                f"delay, if not none, must be positive, received {float(delay)}"
            )

        # call superclass constructor
        Synapse.__init__(self, shape, step_time, batch_size, delay)

        # register parameters
        self.register_parameter(
            "current_", nn.Parameter(torch.zeros(*self.bshape, self.hsize), False)
        )
        self.register_constrained("current_")

        if not spikes_from_currents:
            self.register_parameter(
                "spike_",
                nn.Parameter(
                    torch.zeros(*self.bshape, self.hsize, dtype=torch.bool), False
                ),
            )
            self.register_constrained("spike_")

        # register extras
        self.register_extra("current_overbound", current_overbound)
        self.register_extra("spike_overbound", spike_overbound)

        # add non-persistant interpolation
        match interpolation.lower():
            case "nearest":
                self.interpolation = inferno.interp_nearest
            case "previous":
                self.interpolation = inferno.interp_previous
            case _:
                raise RuntimeError(
                    f"invalid `interpolation` of '{interpolation}' received"
                )

        # function for conversion of currents to spikes
        self._current_to_spike = lambda c: c > 0

    @classmethod
    def partialconstructor(
        cls,
        interpolation: Literal["nearest", "previous"] = "previous",
        spikes_from_currents: bool = True,
        current_overbound: float | None = 0.0,
        spike_overbound: bool | None = False,
    ) -> SynapseConstructor:
        r"""Returns a function with a common signature for synapse construction.

        Args:
            interpolation ("nearest" | "previous", optional): interpolation mode for non-integer multiple selectors.
                Defaults to "nearest".
            spikes_from_currents (bool, optional): if spiking input should be assumed from current-like input.
                Defaults to True.
            current_overbound (float | None, optional): value to replace values out of bounds, use values at
                    final record if none. Defaults to 0.0.
            spike_overbound (bool | None, optional): value to replace values out of bounds, use values at
                    final record if none. Defaults to False.

        Returns:
           SynapseConstructor: partial constructor for synapse.
        """

        def constructor(shape, step_time, batch_size, delay):
            return cls(
                shape,
                step_time,
                batch_size,
                delay,
                interpolation=interpolation,
                spikes_from_currents=spikes_from_currents,
                current_overbound=current_overbound,
                spike_overbound=spike_overbound,
            )

        return constructor

    def clear(self):
        r"""Resets synapses to their resting state."""
        self.reset("current_", 0)
        if hasattr(self, "spike_"):
            self.reset("spike_", False)

    def forward(
        self, inputs: torch.Tensor, spikes: torch.Tensor | None = None, **kwargs
    ) -> torch.Tensor:
        r"""Runs a simulation step of the synaptic kinetics.

        Args:
            inputs (torch.Tensor): main input to the synapse, treated like a current.
            spikes (torch.Tensor | None, optional): spiking input if distinct from input current.
                Defaults to None.

        Returns:
            torch.Tensor: currents resulting from the kinetics.
        """
        if hasattr(self, "spike_") and spikes is None:
            raise RuntimeError(
                f"{type(self).__name__} object initialized with `spikes_from_currents` "
                "set to True, requires `spikes` argument on __call__()."
            )
        if not hasattr(self, "spike_") and spikes is not None:
            warnings.warn(
                f"{type(self).__name__} object initialized with `spikes_from_currents` "
                "set to False, ignoring `spikes`.",
                category=RuntimeWarning,
            )

        self.current = inputs

        if spikes is not None:
            self.pushto("spikes_", spikes)

        return self.current

    @property
    def current(self) -> torch.Tensor:
        r"""Currents of the synapses at the present time.

        Args:
            value (torch.Tensor): new synapse currents.

        Returns:
            torch.Tensor: currents of the synapses.
        """
        return self.latest("current_")

    @current.setter
    def current(self, value: torch.Tensor):
        self.pushto("current_", value)

    @property
    def spike(self) -> torch.Tensor:
        r"""Spikes to the synapses at the present time.

        Returns:
            torch.Tensor: spikes to the synapses.

        Note:
            Spikes are computed as inputs greater than zero. If currents rather than spikes
            are passed into this object's :py:meth:`forward` method, this property will
            return incorrect values.
        """
        if hasattr(self, "spike_"):
            return self.latest("spike_")
        else:
            return self._current_to_spike(self.current)

    def current_at(self, selector: torch.Tensor) -> torch.Tensor:
        r"""Returns currents at times specified by delays.

        Args:
            selector (torch.Tensor): delays for selection of currents.

        Shape:
            ``selector``:

                :math:`B \times N_0 \times \cdots \times D`, where :math:`B` is the number of batches,
                :math:`N_0 \times \cdots` is the underlying shape, and :math:`D` is the number of delay selectors.

            **outputs**

                Same shape as ``selector``.

        Returns:
            torch.Tensor: currents selected at the given times.
        """
        if not self.delayed:
            if self.current_overbound is not None:
                return torch.where(selector == 0, self.current, self.current_overbound)
            else:
                return self.current
        else:
            res = self.select(
                name="current_",
                time=selector.clamp(min=0, max=((self.hsize - 1) * self.dt)),
                interpolation=self.interpolation,
            )
            if self.current_overbound is not None:
                return torch.where(
                    selector == selector.clamp(min=0, max=((self.hsize - 1) * self.dt)),
                    res,
                    self.current_overbound,
                )
            else:
                return res

    def spike_at(self, selector: torch.Tensor) -> torch.Tensor:
        r"""Returns spikes at times specified by delays.

        Args:
            selector (torch.Tensor): delays for selection of spikes.

        Shape:
            ``selector``:

                :math:`B \times N_0 \times \cdots \times D`, where :math:`B` is the number of batches,
                :math:`N_0 \times \cdots` is the underlying shape, and :math:`D` is the number of delay selectors.

            **outputs**

                Same shape as ``selector``.

        Returns:
            torch.Tensor: spikes selected at the given times.
        """
        if hasattr(self, "spike_"):
            if not self.delayed:
                if self.spike_overbound is not None:
                    return torch.where(selector == 0, self.spike, self.spike_overbound)
                else:
                    return self.spike
            else:
                res = self.select(
                    name="spike_",
                    time=selector.clamp(min=0, max=((self.hsize - 1) * self.dt)),
                    interpolation=self.interpolation,
                )
                if self.spike_overbound is not None:
                    return torch.where(
                        selector == selector.clamp(min=0, max=((self.hsize - 1) * self.dt)),
                        res,
                        self.spike_overbound,
                    )
                else:
                    return res
        else:
            if not self.delayed:
                if self.spike_overbound is not None:
                    return torch.where(selector == 0, self._current_to_spike(self.current), self.spike_overbound)
                else:
                    return self._current_to_spike(self.current)
            else:
                res = self.select(
                    name="current_",
                    time=selector.clamp(min=0, max=((self.hsize - 1) * self.dt)),
                    interpolation=self.interpolation,
                )
                if self.spike_overbound is not None:
                    return torch.where(
                        selector == selector.clamp(min=0, max=((self.hsize - 1) * self.dt)),
                        self._current_to_spike(res),
                        self.spike_overbound,
                    )
                else:
                    return self._current_to_spike(res)
