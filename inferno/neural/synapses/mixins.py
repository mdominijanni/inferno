from ... import HistoryModule, Interpolation
from inferno._internal import instance_of, numeric_limit, attr_members
from ...core.types import OneToOne
import torch
import torch.nn as nn
from typing import Any


class CurrentMixin:
    r"""Mixin for synapses with current primitive.

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
    def current(self, value: torch.Tensor) -> None:
        self.pushto("current_", value)


class SpikeMixin:
    r"""Mixin for synapses with spike primitive.

    Args:
        spikes (torch.Tensor): initial input spikes.
        requires_grad (bool, optional): if the parameters created require gradients.
            Defaults to False.

    Caution:
        This must be added to a class which inherits from
        :py:class:`HistoryModule`, and the constructor for this
        mixin must be called after the module constructor.

    Note:
        This registers a parameter ``spike_`` and sets it as constrained.
    """

    def __init__(
        self,
        spikes: torch.Tensor,
        requires_grad=False,
    ):
        e = instance_of("self", self, HistoryModule)
        if e:
            raise e
        self.register_parameter(
            "spike_", nn.Parameter(spikes.to(dtype=torch.bool), requires_grad)
        )
        self.register_constrained("spike_")

    @property
    def spike(self) -> torch.Tensor:
        r"""Spike input to the synapses at present.

        Args:
            value (torch.Tensor): new spike input.

        Returns:
            torch.Tensor: present spike input.
        """
        return self.latest("spike_")

    @spike.setter
    def spike(self, value: torch.Tensor) -> None:
        self.pushto("spike_", value)


class CurrentDerivedSpikeMixin(CurrentMixin):
    r"""Mixin for synapses with current and spikes derived therefrom.

    Args:
        currents (torch.Tensor): initial synaptic currents, in :math:`\text{nA}`.
        requires_grad (bool, optional): if the parameters created require gradients.
            Defaults to False.

    Caution:
        This must be added to a class which inherits from
        :py:class:`HistoryModule`, and the constructor for this
        mixin must be called after the module constructor.

    Important:
        This must be added to a class which has a method named ``_to_spike``, which
        takes a tensor of currents and returns a tensor of spikes.

    Note:
        This registers a parameter ``current_`` and sets it as constrained.
    """

    def __init__(
        self,
        currents: torch.Tensor,
        requires_grad=False,
    ):
        # test for members
        e = attr_members("self", self, "_to_spike")
        if e:
            raise e

        # call superclass mixin constructor
        CurrentMixin.__init__(self, currents=currents, requires_grad=requires_grad)

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
        return self._to_spike(self.current)

    @spike.setter
    def spike(self, value: torch.Tensor) -> None:
        pass


class SpikeDerivedCurrentMixin(SpikeMixin):
    r"""Mixin for synapses with spikes and currents derived therefrom.

    Args:
        spikes (torch.Tensor): initial input spikes.
        requires_grad (bool, optional): if the parameters created require gradients.
            Defaults to False.

    Caution:
        This must be added to a class which inherits from
        :py:class:`HistoryModule`, and the constructor for this
        mixin must be called after the module constructor.

    Important:
        This must be added to a class which has a method named ``_to_current``, which
        takes a tensor of spikes and returns a tensor of currents.

    Note:
        This registers a parameter ``spike_`` and sets it as constrained.
    """

    def __init__(
        self,
        spikes: torch.Tensor,
        requires_grad=False,
    ):
        # test for members
        e = attr_members("self", self, "_to_current")
        if e:
            raise e

        # call superclass mixin constructor
        SpikeMixin.__init__(self, spikes=spikes, requires_grad=requires_grad)

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
        return self._to_current(self.spike)

    @current.setter
    def current(self, value: torch.Tensor) -> None:
        pass


class SpikeCurrentMixin(CurrentMixin, SpikeMixin):
    r"""Mixin for synapses with primitive current and spikes.

    Args:
        currents (torch.Tensor): initial synaptic currents, in :math:`\text{nA}`.
        spikes (torch.Tensor): initial input spikes.
        requires_grad (bool, optional): if the parameters created require gradients.
            Defaults to False.

    Caution:
        This must be added to a class which inherits from
        :py:class:`HistoryModule`, and the constructor for this
        mixin must be called after the module constructor.

    Note:
        * This registers a parameter ``current_`` and sets it as constrained.
        * This registers a parameter ``spike_`` and sets it as constrained.
    """

    def __init__(
        self,
        currents: torch.Tensor,
        spikes: torch.Tensor,
        requires_grad=False,
    ):
        # call superclass mixin constructors
        CurrentMixin.__init__(self, currents, requires_grad)
        SpikeMixin.__init__(self, spikes, requires_grad)


def _synparam_at(
    module: HistoryModule,
    dataloc: str,
    selector: torch.Tensor,
    interpolation: Interpolation,
    tolerance: float,
    overbound: Any | None,
    transform: OneToOne[torch.Tensor] | None = None,
) -> torch.Tensor:
    r"""Internal, generalized selector function for synaptic parameters.

    Args:
        module (HistoryModule): module from which to access parameters.
        dataloc (str): attribute name of the underlying data from which to select.
        selector (torch.Tensor): time before present for which synaptic parameters
            should be retrieved, in :math:`\text{ms}`.
        interpolation (Interpolation): interpolation function used when selecting
            prior values.
        tolerance (float): maximum difference in time from an observation
            to treat as co-occurring, in :math:`\text{ms}`.
        overbound (Any | None): value to replace parameter values out of bounds,
            uses values at observation limits if None.
        transform (OneToOne[torch.Tensor] | None, optional): function applied to
            retrieved values before returning, identity if None. Defaults to None.

    Returns:
        torch.Tensor: selected synaptic parameter values.
    """
    # identity transform if none is specified
    if not transform:
        transform = lambda x: x  # noqa: E731

    # undelayed access
    if module.recordsz == 1:
        # bounded selector for overbounding
        bounded_selector = 0

        # retrieve most recent value
        res = transform(module.latest_(dataloc))

    # delayed access
    else:
        # bound the selector
        bounded_selector = selector.clamp(min=0, max=module.duration)

        # select values using HistoryModule
        res = transform(
            module.select(
                name=dataloc,
                time=bounded_selector,
                interpolation=interpolation,
                tolerance=tolerance,
            )
        )

    # apply overbound if specified
    if overbound is not None:
        res = torch.where(
            (selector - bounded_selector).abs() <= tolerance, res, overbound
        )

    # return parameter values at delayed indices
    return res


class DelayedSpikeCurrentAccessorMixin:
    r"""Mixin for synapses with delayed current and spike selector methods.

    Args:
        primitive_currents (bool): if the synaptic currents are not derived from
            another synaptic parameter.
        primitive_spikes (bool): if the input spikes are not derived from
            another synaptic parameter.
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

    Caution:
        This must be added to a class which inherits from
        :py:class:`HistoryModule`, and the constructor for this
        mixin must be called after the module constructor.

    Important:
        * This must be added to a class which has either a method named ``_to_current``,
          or a constrained parameter/buffer ``current_``.
        * This must be added to a class which has either a method named ``_to_spike``,
          or a constrained parameter/buffer ``spike_``.

    Note:
        This always sets the following interally managed attributes:
        ``_current_interp``, ``_spike_interp``, ``_current_ob_val``, ``_spike_ob_val``,
        ``_primitive_currents``, ``_primitive_spikes``, and ``_interp_tol``.
    """

    def __init__(
        self,
        primitive_currents: bool,
        primitive_spikes: bool,
        current_interp: Interpolation,
        spike_interp: Interpolation,
        tolerance: float,
        current_overval: float | None,
        spike_overval: bool | None,
    ):
        # check for valid attributes
        if not primitive_currents and not primitive_spikes:
            raise RuntimeError(
                "at least one of 'primitive_currents' or "
                "'primitive_spikes' must be true."
            )
        if not primitive_currents and not hasattr(self, "_to_current"):
            raise RuntimeError(
                "if 'primitive_currents' is false, '_to_current' must be an attribute."
            )
        if primitive_currents and not hasattr(self, "current_"):
            raise RuntimeError(
                "if 'primitive_currents' is true, 'current_' must be an attribute."
            )
        if not primitive_spikes and not hasattr(self, "_to_spike"):
            raise RuntimeError(
                "if 'primitive_spikes' is false, '_to_spike' must be an attribute."
            )
        if primitive_spikes and not hasattr(self, "spike_"):
            raise RuntimeError(
                "if 'primitive_spikes' is true, 'spike_' must be an attribute."
            )

        # set managed internal attributes
        self._primitive_currents = primitive_currents
        self._primitive_spikes = primitive_spikes
        self._current_interp = current_interp
        self._spike_interp = spike_interp
        self._current_ob_val = (
            None if current_overval is None else float(current_overval)
        )
        self._spike_ob_val = None if spike_overval is None else bool(spike_overval)
        self._interp_tol, e = numeric_limit("tolerance", tolerance, 0, "gte", float)
        if e:
            raise e

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
        if self._primitive_currents:
            return _synparam_at(
                self,
                "current_",
                selector,
                self._current_interp,
                self._interp_tol,
                self._current_ob_val,
                None,
            )
        else:
            return _synparam_at(
                self,
                "spike_",
                selector,
                self._current_interp,
                self._interp_tol,
                self._current_ob_val,
                self._to_current,
            )

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
        if self._primitive_spikes:
            return _synparam_at(
                self,
                "spike_",
                selector,
                self._spike_interp,
                self._interp_tol,
                self._spike_ob_val,
                None,
            )
        else:
            return _synparam_at(
                self,
                "current_",
                selector,
                self._spike_interp,
                self._interp_tol,
                self._spike_ob_val,
                self._to_spike,
            )
