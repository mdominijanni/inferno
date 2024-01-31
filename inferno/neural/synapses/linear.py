import inferno
import torch
from typing import Literal
from .mixins import DelayedSpikeCurrentMixin
from .. import Synapse, SynapseConstructor


class PassthroughSynapse(DelayedSpikeCurrentMixin, Synapse):
    r"""Synapse which directly returns values inputted.

    This acts as a "placeholder" synapse for when the inputs should be the same
    as for the connection, except potentially shiften in time.

    Args:
        shape (tuple[int, ...] | int): shape of the group of synapses being simulated.
        step_time (float): length of a simulation time step, in :math:`\text{ms}`.
        delay (float, optional): maximum supported delay, in :math:`\text{ms}`.
        interp_mode (Literal["nearest", "previous"], optional): interpolation mode
            for selectors between observations. Defaults to "nearest".
        interp_tol (float, optional): maximum difference in time from an observation
            to treat as co-occurring, in :math:`\text{ms}`. Defaults to 0.0.
        derive_spikes (bool, optional): if spikes should be computed from internal
            currents. Defaults to True.
        current_overbound (float | None, optional): value to replace currents out of
            bounds, uses values at observation limits if None. Defaults to 0.0.
        spike_overbound (bool | None, optional): value to replace spikes out of bounds,
            uses values at observation limits if None. Defaults to False.
        batch_size (int, optional): size of input batches for simualtion. Defaults to 1.

    Important:
        When ``derive_spikes`` is set to True, only the parameter ``current`` is
        maintained, and ``spike`` is computed as currents greater than or equal to 1.
        This is normally unproblematic, although issues my arise if added subthreshold
        currents exceed 1.

    Note:
        When ``interp_mode`` is set to ``"nearest"``, the closest observation will be
        used, even if it occurred after the selected time. When set to ``"previous"``
        the the closest prior observation is selected.
    """

    def __init__(
        self,
        shape: tuple[int, ...] | int,
        step_time: float,
        *,
        delay: float = 0.0,
        interp_mode: Literal["nearest", "previous"] = "previous",
        interp_tol: float = 0.0,
        derive_spikes: bool = True,
        current_overbound: float | None = 0.0,
        spike_overbound: bool | None = False,
        batch_size: int = 1,
    ):
        # call superclass constructor
        Synapse.__init__(self, shape, step_time, delay, batch_size)

        # determine mixin arguments
        currents = torch.zeros(*self.bshape, self.hsize)

        if derive_spikes:
            spikes = lambda c: c >= 1  # noqa: E731
        else:
            spikes = torch.zeros(*self.bshape, self.hsize, dtype=torch.bool)

        match interp_mode.lower():
            case "nearest":
                interp = inferno.interp_nearest
            case "previous":
                interp = inferno.interp_previous
            case _:
                raise RuntimeError(
                    f"invalid `interp_mode` '{interp_mode}' received, must be one of "
                    "'nearest' or 'previous'."
                )

        # call mixin constructor
        DelayedSpikeCurrentMixin.__init__(
            self,
            currents=currents,
            spikes=spikes,
            current_interp=interp,
            spike_interp=interp,
            tolerance=interp_tol,
            current_overval=current_overbound,
            spike_overval=spike_overbound,
            requires_grad=False,
        )

    @classmethod
    def partialconstructor(
        cls,
        interp_mode: Literal["nearest", "previous"] = "previous",
        interp_tol: float = 0.0,
        derive_spikes: bool = True,
        current_overbound: float | None = 0.0,
        spike_overbound: bool | None = False,
    ) -> SynapseConstructor:
        r"""Returns a function with a common signature for synapse construction.

        Args:
            interp_mode (Literal["nearest", "previous"], optional): interpolation mode
                for selectors between observations. Defaults to "nearest".
            interp_tol (float, optional): maximum difference in time from an observation
                to treat as co-occurring, in :math:`\text{ms}`. Defaults to 0.0.
            derive_spikes (bool, optional): if inputs will represent currents and
                spikes, with spikes derived therefrom. Defaults to True.
            current_overbound (float | None, optional): value to replace currents out of
                bounds, uses values at observation limits if None. Defaults to 0.0.
            spike_overbound (bool | None, optional): value to replace spikes out of
                bounds, uses values at observation limits if None. Defaults to False.

        Returns:
           SynapseConstructor: partial constructor for synapse.
        """

        def constructor(
            shape: tuple[int, ...] | int,
            step_time: float,
            delay: float,
            batch_size: int,
        ):
            return cls(
                shape=shape,
                step_time=step_time,
                delay=delay,
                interp_mode=interp_mode,
                interp_tol=interp_tol,
                derive_spikes=derive_spikes,
                current_overbound=current_overbound,
                spike_overbound=spike_overbound,
                batch_size=batch_size,
            )

        return constructor

    def clear(self, **kwargs):
        r"""Resets synapses to their resting state."""
        self.reset("current_", 0.0)
        if not self.spikesderived:
            self.reset("spike_", False)

    def forward(self, *inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""Runs a simulation step of the synaptic dynamics.

        Args:
            *inputs (torch.Tensor): inputs to the synapse.

        Returns:
            torch.Tensor: synaptic currents after simulation step.

        Important:
            ``*inputs`` can either be a single tensor representing the input spikes.
            The first will always be the input spikes, and any subsequent tensors will
            be treated as additional currents.
        """
        if len(inputs) == 1:
            self.current = inputs[0]
            self.spikes = inputs[0]
        else:
            currents = torch.sum(torch.stack(inputs[1:], dim=0), dim=0)
            self.current = inputs[0] + currents
            self.spikes = inputs[0]

        return self.current
