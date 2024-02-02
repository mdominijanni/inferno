from inferno import HistoryModule, Interpolation
from inferno._internal import instance_of, numeric_limit
from inferno.typing import OneToOne
import torch
import torch.nn as nn


class CurrentMixin:
    r"""Mixin for synapses with current.

    Args:
        currents (torch.Tensor): initial synaptic currents, in :math:`\text{nA}`.
        requires_grad (bool, optional): if the parameters created require gradients.
            Defaults to False.

    Caution:
        This must be added to a class which inherits from
        :py:class:`HistoryModule`, and the constructor for this
        mixin must be called after the module constructor.

    Note:
        This registers a parameter ``current_`` and sets it as constrained.
    """

    def __init__(
        self,
        currents: torch.Tensor,
        requires_grad=False,
    ):
        e = instance_of("self", self, HistoryModule)
        if e:
            raise e
        self.register_parameter("current_", nn.Parameter(currents, requires_grad))
        self.register_constrained("current_")

    @property
    def current(self) -> torch.Tensor:
        r"""Currents of the synapses at present, in nanoamperes.

        Args:
            value (torch.Tensor): new synapse currents.

        Returns:
            torch.Tensor: present synaptic currents.
        """
        return self.latest("current_")

    @current.setter
    def current(self, value: torch.Tensor):
        self.pushto("current_", value)


class SpikeCurrentMixin(CurrentMixin):
    r"""Mixin for synapses with current and managed spikes.

    Args:
        currents (torch.Tensor): initial synaptic currents, in :math:`\text{nA}`.
        spikes (torch.Tensor | OneToOne[torch.Tensor]): initial input spikes,
            or a function to derive them from synaptic current.
        requires_grad (bool, optional): if the parameters created require gradients.
            Defaults to False.

    Caution:
        This must be added to a class which inherits from
        :py:class:`HistoryModule`, and the constructor for this
        mixin must be called after the module constructor.

    Note:
        This registers a parameter ``current_`` and sets it as constrained.
        If ``spikes`` is a :py:class:`~torch.Tensor`, this also registers
        a parameter ``spike_`` and sets it as constrained.

    Note:
        If ``spikes`` is not a :py:class:`~torch.Tensor`, this sets an attribute
        ``_tospike`` which is managed internally.
    """

    def __init__(
        self,
        currents: torch.Tensor,
        spikes: torch.Tensor | OneToOne[torch.Tensor],
        requires_grad=False,
    ):
        # call superclass constructor
        CurrentMixin.__init__(self, currents, requires_grad)

        # register synaptic spikes or add conversion method attribute
        if isinstance(spikes, torch.Tensor):
            self.register_parameter("spike_", nn.Parameter(spikes, requires_grad))
            self.register_constrained("spike_")
        else:
            self._tospike = spikes

    @property
    def spikesderived(self) -> bool:
        """If input spikes are derived from synaptic currents."""
        return not hasattr(self, "spike_")

    @property
    def spike(self) -> torch.Tensor:
        r"""Spike input to the synapses at present.

        Args:
            value (torch.Tensor): new spike input.

        Returns:
            torch.Tensor: present spike input.

        Note:
            If :py:attr:`spikesderived` is True, then the setter does not alter state.
        """
        if self.spikesderived:
            return self._tospike(self.current)
        else:
            return self.latest("spike_")

    @spike.setter
    def spike(self, value: torch.Tensor):
        if not self.spikesderived:
            self.pushto("spike_", value)


class DelayedSpikeCurrentMixin(SpikeCurrentMixin):
    r"""Mixin for synapses with current and managed spikes with selector methods.

    Args:
        currents (torch.Tensor): initial synaptic currents, in :math:`\text{nA}`.
        spikes (torch.Tensor | OneToOne[torch.Tensor]): initial input spikes,
            or a function to derive them from synaptic current.
        current_interp (Interpolation): interpolation function used when selecting
            prior currents.
        spike_interp (Interpolation): interpolation function used when selecting
            prior spikes.
        current_overval (float | None): value to replace currents out of
            bounds, uses values at observation limits if None.
        spike_overval (bool | None): value to replace spikes out of bounds,
            uses values at observation limits if None.
        tolerance (float): maximum difference in time from an observation
            to treat as co-occurring, in :math:`\text{ms}`.
        requires_grad (bool, optional): if the parameters created require gradients.
            Defaults to False.

    Caution:
        This must be added to a class which inherits from
        :py:class:`HistoryModule`, and the constructor for this
        mixin must be called after the module constructor.

    Note:
        This registers a parameter ``current_`` and sets it as constrained.
        If ``spikes`` is a :py:class:`~torch.Tensor`, this also registers
        a parameter ``spike_`` and sets it as constrained.

    Note:
        This always sets the following interally managed attributes: ``_currentinterp``,
        ``_spikeinterp``, ``_currentobv``, ``_spikeobv``, ``_interptol``, and
        ``_synparam_at``. If ``spikes`` is not a :py:class:`~torch.Tensor`, this
        additionally sets an attribute ``_tospike`` which is managed internally.
    """

    def __init__(
        self,
        currents: torch.Tensor,
        spikes: torch.Tensor | OneToOne[torch.Tensor],
        current_interp: Interpolation,
        spike_interp: Interpolation,
        tolerance: float,
        current_overval: float | None,
        spike_overval: bool | None,
        requires_grad=False,
    ):
        # call superclass constructor
        SpikeCurrentMixin.__init__(self, currents, spikes, requires_grad)

        # set managed internal attributes
        self._currentinterp = current_interp
        self._spikeinterp = spike_interp
        self._currentobv = None if current_overval is None else float(current_overval)
        self._spikeobv = None if spike_overval is None else float(spike_overval)
        self._interptol, e = numeric_limit("tolerance", tolerance, 0, "gte", float)
        if e:
            raise e

    def _synparam_at(
        self,
        target_current: bool,
        selector: torch.Tensor,
        interpolation: Interpolation,
        overbound: float | None,
    ) -> torch.Tensor:
        """Internal, generalized selector function for spikes and currents."""
        # undelayed case
        if self.hsize == 1:
            # most recent current
            res = self.current if target_current else self.spike

            # apply overbound value if specified
            if overbound is not None:
                # nonzero tolerance
                if self._interptol:
                    res = torch.where(selector.abs() <= self._interptol, res, overbound)

                # zero tolerance
                else:
                    res = torch.where(selector == 0, res, overbound)

        # delayed case
        else:
            # bound the selector
            bounded_selector = selector.clamp(min=0, max=self.hlen)

            # currents selected using HistoryModule
            res = self.select(
                name=("current_" if target_current else "spike_"),
                time=bounded_selector,
                interpolation=interpolation,
                tolerance=self._interptol,
            )

            # apply overbound value if specified
            if overbound is not None:
                # nonzero tolerance
                if self._interptol:
                    res = torch.where(
                        (selector - bounded_selector).abs() <= self._interptol,
                        res,
                        overbound,
                    )

                # zero tolerance
                else:
                    res = torch.where(selector == bounded_selector, res, overbound)

        # return currents
        return res

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
        return self._synparam_at(True, selector, self._currentinterp, self._currentobv)

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
        # derived case
        if self.spikesderived:
            # undelayed case
            if self.hsize == 1:
                # most recent current
                res = self.spike

                # apply overbound value if specified
                if self._spikeobv is not None:
                    # nonzero tolerance
                    if self._interptol:
                        res = torch.where(
                            selector.abs() <= self._interptol, res, self._spikeobv
                        )

                    # zero tolerance
                    else:
                        res = torch.where(selector == 0, res, self._spikeobv)

            # delayed case
            else:
                # delayed currents without overbound value
                res = self.tospike_(
                    self._synparam_at(True, selector, self._spikeinterp, None)
                )

                # apply overbound value if specified
                if self._spikeobv is not None:
                    res = torch.where(
                        torch.logical_and(
                            selector >= -self._interptol,
                            selector <= self.hlen + self._interptol,
                        ),
                        res,
                        self._spikeobv,
                    )

        # stored case
        else:
            res = self._synparam_at(False, selector, self._spikeinterp, self._spikeobv)

        # return spikes
        return res
