from ... import RecordTensor, VirtualTensor
from ..._internal import argtest
from ...functional import Interpolation
from ...types import OneToOne
from ..base import InfernoSynapse
import torch
from typing import Any, Callable


def _synparam_at(
    value: RecordTensor,
    selector: torch.Tensor,
    interpolation: Interpolation,
    interp_kwargs: dict[str, Any],
    tolerance: float,
    overbound: Any | None,
    transform: OneToOne[torch.Tensor] | None = None,
) -> torch.Tensor:
    r"""Internal, generalized selector function for synaptic parameters.

    Args:
        value (RecordTensor): record tensor to access.
        selector (torch.Tensor): time before present for which synaptic parameters
            should be retrieved, in :math:`\text{ms}`.
        interpolation (Interpolation): interpolation function used when selecting
            prior values.
        interp_kwargs (dict[str, Any]): keyword arguments passed into the interpolation
            function.
        tolerance (float): maximum difference in time from an observation
            to treat as co-occurring, in :math:`\text{ms}`.
        overbound (Any | None): value to replace parameter values out of bounds,
            uses values at observation limits if ``None``.
        transform (OneToOne[torch.Tensor] | None, optional): function applied to
            retrieved values before returning, identity if ``None``. Defaults to ``None``.

    Returns:
        torch.Tensor: selected synaptic parameter values.
    """
    # identity transform if none is specified
    if not transform:
        transform = lambda x: x  # noqa: E731

    # undelayed access
    if value.recordsz == 1:
        # bounded selector for overbounding
        bounded_selector = 0

        # retrieve most recent value
        res = transform(value.peek())

    # delayed access
    else:
        # bound the selector
        bounded_selector = selector.clamp(min=0, max=value.duration)

        # select values using RecordTensor
        res = transform(
            value.select(
                bounded_selector,
                interpolation,
                tolerance=tolerance,
                interp_kwargs=interp_kwargs,
            )
        )

    # apply overbound if specified
    if overbound is not None:
        res = torch.where(
            (selector - bounded_selector).abs() <= tolerance, res, overbound
        )

    # return parameter values at delayed indices
    return res


class CurrentMixin:
    r"""Mixin for synapses with current primitive.

    Args:
        data (torch.Tensor): initial synaptic currents, in :math:`\text{nA}`.
        interpolation (Interpolation): interpolation function used when selecting
            prior currents.
        interp_kwargs (dict[str, Any]): keyword arguments passed into the interpolation
            function.
        overbound (float | None): value to replace currents out of bounds, uses values
            at observation limits if ``None``.
        tolerance (float): maximum difference in time from an observation
            to treat as co-occurring, in :math:`\text{ms}`.
    """

    def __init__(
        self,
        currents: torch.Tensor,
        interpolation: Interpolation,
        interp_kwargs: dict[str, Any],
        overbound: float | None,
        tolerance: float,
    ):
        _ = argtest.instance("self", self, InfernoSynapse)
        RecordTensor.create(
            self,
            "current_",
            self.dt,
            self.delay,
            currents,
            persist_data=True,
            persist_constraints=False,
            persist_temporal=False,
            strict=True,
            live=False,
            inclusive=True,
        )
        self.add_delayed("current_")
        self.add_batched("current_")
        self.__interp = interpolation
        self.__interp_kwargs = interp_kwargs
        self.__overbound = overbound if overbound is None else float(overbound)
        self.__tolerance = float(tolerance)

    @property
    def current(self) -> torch.Tensor:
        r"""Currents of the synapses at present, in nanoamperes.

        Args:
            value (torch.Tensor): new synapse currents.

        Returns:
            torch.Tensor: present synaptic currents.
        """
        return self.current_.peek()

    @current.setter
    def current(self, value: torch.Tensor) -> None:
        self.current_.push(value, self.inplace)

    def current_at(self, selector: torch.Tensor) -> torch.Tensor:
        r"""Retrieves previous synaptic currents, in nanoamperes.

        Args:
            selector (torch.Tensor): time before present for which synaptic currents
                should be retrieved, in :math:`\text{ms}`.

        Returns:
            torch.Tensor: selected synaptic currents.

        .. admonition:: Shape
            :class: tensorshape

            ``selector``:

            :math:`B \times N_0 \times \cdots \times [D]`

            ``return``:

            :math:`B \times N_0 \times \cdots \times [D]`

            Where:
                * :math:`B` is the batch size.
                * :math:`N_0 \times \cdots` is the shape of the synapse.
                * :math:`D` is the number of selectors per synapse.
        """
        return _synparam_at(
            self.current_,
            selector,
            self.__interp,
            self.__interp_kwargs,
            self.__tolerance,
            self.__overbound,
            None,
        )


class SpikeMixin:
    r"""Mixin for synapses with spike primitive.

    Args:
        spikes (torch.Tensor): initial input spikes.
        interpolation (Interpolation): interpolation function used when selecting
            prior spikes.
        interp_kwargs (dict[str, Any]): keyword arguments passed into the interpolation
            function.
        overbound (bool | None): value to replace spikes out of bounds, uses values at
            observation limits if ``None``.
        tolerance (float): maximum difference in time from an observation
            to treat as co-occurring, in :math:`\text{ms}`.
    """

    def __init__(
        self,
        spikes: torch.Tensor,
        interpolation: Interpolation,
        interp_kwargs: dict[str, Any],
        overbound: bool | None,
        tolerance: float,
    ):
        _ = argtest.instance("self", self, InfernoSynapse)
        RecordTensor.create(
            self,
            "spike_",
            self.dt,
            self.delay,
            spikes,
            persist_data=True,
            persist_constraints=False,
            persist_temporal=False,
            strict=True,
            live=False,
            inclusive=True,
        )
        self.add_delayed("spike_")
        self.add_batched("spike_")
        self.__interp = interpolation
        self.__interp_kwargs = interp_kwargs
        self.__overbound = overbound if overbound is None else bool(overbound)
        self.__tolerance = float(tolerance)

    @property
    def spike(self) -> torch.Tensor:
        r"""Spike input to the synapses at present.

        Args:
            value (torch.Tensor): new spike input.

        Returns:
            torch.Tensor: present spike input.
        """
        return self.spike_.peek()

    @spike.setter
    def spike(self, value: torch.Tensor) -> None:
        self.spike_.push(value.bool(), self.inplace)

    def spike_at(self, selector: torch.Tensor) -> torch.Tensor:
        r"""Retrieves previous spike inputs.

        Args:
            selector (torch.Tensor): time before present for which spike inputs
                should be retrieved, in :math:`\text{ms}`.

        Returns:
            torch.Tensor: selected spike inputs.

        .. admonition:: Shape
            :class: tensorshape

            ``selector``:

            :math:`B \times N_0 \times \cdots \times [D]`

            ``return``:

            :math:`B \times N_0 \times \cdots \times [D]`

            Where:
                * :math:`B` is the batch size.
                * :math:`N_0 \times \cdots` is the shape of the synapse.
                * :math:`D` is the number of selectors per synapse.
        """
        return _synparam_at(
            self.spike_,
            selector,
            self.__interp,
            self.__interp_kwargs,
            self.__overbound,
            self.__tolerance,
            None,
        ).to(dtype=self.spike_.value.dtype, device=self.spike_.value.device)


class CurrentDerivedSpikeMixin(CurrentMixin):
    r"""Mixin for synapses with current and spikes derived therefrom.

    Args:
        currents (torch.Tensor): initial synaptic currents, in :math:`\text{nA}`.
        to_spikes (Callable[[InfernoSynapse, torch.dtype, torch.device, torch.Tensor], torch.Tensor]): function
            which takes the synapse, data type, device, and a tensor of currents, and
            returns the corresponding spikes.
        interp (Interpolation): interpolation function used when selecting
            prior currents and spikes derived therefrom.
        interp_kwargs (dict[str, Any]): keyword arguments passed into the
            interpolation function.
        current_overbound (float | None): value to replace currents out of
            bounds, uses values at observation limits if ``None``.
        spike_overbound (bool | None): value to replace spikes out of bounds,
            uses values at observation limits if ``None``.
        tolerance (float): maximum difference in time from an observation
            to treat as co-occurring, in :math:`\text{ms}`.
    """

    def __init__(
        self,
        currents: torch.Tensor,
        to_spikes: Callable[
            [InfernoSynapse, torch.dtype, torch.device, torch.Tensor], torch.Tensor
        ],
        interp: Interpolation,
        interp_kwargs: dict[str, Any],
        current_overbound: float | None,
        spike_overbound: bool | None,
        tolerance: float,
    ):
        CurrentMixin.__init__(
            self,
            currents,
            interp,
            interp_kwargs,
            current_overbound,
            tolerance,
        )
        self.__to_spike = to_spikes
        self.__interp = interp
        self.__interp_kwargs = interp_kwargs
        self.__spike_overbound = (
            None if spike_overbound is None else bool(spike_overbound)
        )
        self.__tolerance = argtest.gte("tolerance", tolerance, 0, float)

        VirtualTensor.create(
            self,
            "spike_",
            "_derived_spike",
            dtype=torch.bool,
            persist=False,
        )

    def _derived_spike(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        r"""Used by VirtualTensor for spikes.

        Args:
            dtype (torch.dtype): data type for the spikes.
            device (torch.device): device for the spikess.

        Returns:
            torch.Tensor: calculated spikess.
        """
        return self.__to_spike(self, dtype, device, self.current)

    @property
    def spike(self) -> torch.Tensor:
        r"""Spike input to the synapses at present.

        Args:
            value (torch.Tensor): new spike input.

        Returns:
            torch.Tensor: present spike input.

        Note:
            The setter does nothing as spikes are derived from currents.
        """
        return self.spike_.value

    @spike.setter
    def spike(self, value: torch.Tensor) -> None:
        pass

    def spike_at(self, selector: torch.Tensor) -> torch.Tensor:
        r"""Retrieves previous spike inputs.

        Args:
            selector (torch.Tensor): time before present for which spike inputs
                should be retrieved, in :math:`\text{ms}`.

        Returns:
            torch.Tensor: selected spike inputs.

        .. admonition:: Shape
            :class: tensorshape

            ``selector``:

            :math:`B \times N_0 \times \cdots \times [D]`

            ``return``:

            :math:`B \times N_0 \times \cdots \times [D]`

            Where:
                * :math:`B` is the batch size.
                * :math:`N_0 \times \cdots` is the shape of the synapse.
                * :math:`D` is the number of selectors per synapse.
        """
        return _synparam_at(
            self.current_,
            selector,
            self.__interp,
            self.__interp_kwargs,
            self.__tolerance,
            self.__spike_overbound,
            lambda d, m=self: m.__to_spike(m, m.spike_.dtype, m.spike_.device, d),
        ).to(dtype=self.spike_.dtype, device=self.spike_.device)


class SpikeDerivedCurrentMixin(SpikeMixin):
    r"""Mixin for synapses with spikes and currents derived therefrom.

    Args:
        spikes (torch.Tensor): initial input spikes.
        to_currents (Callable[[InfernoSynapse, torch.dtype, torch.device, torch.Tensor], torch.Tensor]): function
            which takes the synapse, data type, device, and a tensor of spikes, and
            returns the corresponding current.
        interp (Interpolation): interpolation function used when selecting
            prior spikes and currents derived therefrom.
        interp_kwargs (dict[str, Any]): keyword arguments passed into the
            interpolation function.
        current_overbound (float | None): value to replace currents out of
            bounds, uses values at observation limits if ``None``.
        spike_overbound (bool | None): value to replace spikes out of bounds,
            uses values at observation limits if ``None``.
        tolerance (float): maximum difference in time from an observation
            to treat as co-occurring, in :math:`\text{ms}`.
    """

    def __init__(
        self,
        spikes: torch.Tensor,
        to_currents: Callable[
            [InfernoSynapse, torch.dtype, torch.device, torch.Tensor], torch.Tensor
        ],
        interp: Interpolation,
        interp_kwargs: dict[str, Any],
        current_overbound: float | None,
        spike_overbound: bool | None,
        tolerance: float,
    ):
        SpikeMixin.__init__(
            self, spikes, interp, interp_kwargs, spike_overbound, tolerance
        )
        self.__to_current = to_currents
        self.__interp = interp
        self.__interp_kwargs = interp_kwargs
        self.__current_overbound = (
            None if current_overbound is None else float(current_overbound)
        )
        self.__tolerance = argtest.gte("tolerance", tolerance, 0, float)

        VirtualTensor.create(
            self,
            "current_",
            "_derived_current",
            persist=False,
        )

    def _derived_current(
        self, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        r"""Used by VirtualTensor for currents.

        Args:
            dtype (torch.dtype): data type for the currents.
            device (torch.device): device for the currents.

        Returns:
            torch.Tensor: calculated currents.
        """
        return self.__to_current(self, dtype, device, self.spike)

    @property
    def current(self) -> torch.Tensor:
        r"""Currents of the synapses at present, in nanoamperes.

        Args:
            value (torch.Tensor): new synapse currents.

        Returns:
            torch.Tensor: present synaptic currents.

        Note:
            The setter does nothing as currents are derived from spikes.
        """
        return self.current_.value

    @current.setter
    def current(self, value: torch.Tensor) -> None:
        pass

    def current_at(self, selector: torch.Tensor) -> torch.Tensor:
        r"""Retrieves previous synaptic currents, in nanoamperes.

        Args:
            selector (torch.Tensor): time before present for which synaptic currents
                should be retrieved, in :math:`\text{ms}`.

        Returns:
            torch.Tensor: selected synaptic currents.

        .. admonition:: Shape
            :class: tensorshape

            ``selector``:

            :math:`B \times N_0 \times \cdots \times [D]`

            ``return``:

            :math:`B \times N_0 \times \cdots \times [D]`

            Where:
                * :math:`B` is the batch size.
                * :math:`N_0 \times \cdots` is the shape of the synapse.
                * :math:`D` is the number of selectors per synapse.
        """
        return _synparam_at(
            self.spike_,
            selector,
            self.__interp,
            self.__interp_kwargs,
            self.__tolerance,
            self.__current_overbound,
            lambda d, m=self: m.__to_current(m, m.current_.dtype, m.current_.device, d),
        ).to(dtype=self.current_.dtype, device=self.current_.device)


class SpikeCurrentMixin(CurrentMixin, SpikeMixin):
    r"""Mixin for synapses with primitive current and spikes.

    Args:
        currents (torch.Tensor): initial synaptic currents, in :math:`\text{nA}`.
        spikes (torch.Tensor): initial input spikes.
        current_interp (Interpolation): interpolation function used when selecting
            prior currents.
        current_interp_kwargs (dict[str, Any]): keyword arguments passed into the
            interpolation function for currents.
        spike_interp (Interpolation): interpolation function used when selecting
            prior spikes.
        spike_interp_kwargs (dict[str, Any]): keyword arguments passed into the
            interpolation function for spikes.
        current_overbound (float | None): value to replace currents out of
            bounds, uses values at observation limits if ``None``.
        spike_overbound (bool | None): value to replace spikes out of bounds,
            uses values at observation limits if ``None``.
        tolerance (float): maximum difference in time from an observation
            to treat as co-occurring, in :math:`\text{ms}`.
    """

    def __init__(
        self,
        currents: torch.Tensor,
        spikes: torch.Tensor,
        current_interp: Interpolation,
        current_interp_kwargs: dict[str, Any],
        spike_interp: Interpolation,
        spike_interp_kwargs: dict[str, Any],
        current_overbound: float | None,
        spike_overbound: bool | None,
        tolerance: float,
    ):
        # call superclass mixin constructors
        CurrentMixin.__init__(
            self,
            currents,
            current_interp,
            current_interp_kwargs,
            current_overbound,
            tolerance,
        )
        SpikeMixin.__init__(
            self,
            spikes,
            spike_interp,
            spike_interp_kwargs,
            spike_overbound,
            tolerance,
        )
