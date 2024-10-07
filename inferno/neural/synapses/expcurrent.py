from ... import RecordTensor
from .mixins import SpikeCurrentMixin, SpikeMixin, _synparam_at
from ..base import InfernoSynapse
from ..._internal import argtest
from ...functional import interp_nearest, interp_previous, interp_expdecay
from collections.abc import Sequence
import math
import torch
from typing import Literal


class SingleExponentialCurrent(SpikeCurrentMixin, InfernoSynapse):
    r"""Instantly applied exponentially decaying current-based synapse.

    .. math::
        I(t + \Delta t) = I(t) \exp\left(-\frac{\Delta t}{\tau}\right)
        + \frac{Q}{\tau} [t = t_f]

    Attributes:
        spike_: :py:class:`~inferno.RecordTensor` interface for spikes.
        current_: :py:class:`~inferno.RecordTensor` interface for currents.

    Args:
        shape (Sequence[int] | int): shape of the group of synapses being simulated.
        step_time (float): length of a simulation time step, :math:`\Delta t`,
            in :math:`\text{ms}`.
        spike_charge (float): charge carried by each presynaptic spike, :math:`Q`,
            in :math:`\text{pC}`.
        time_constant (float): exponential time constant for current decay, :math:`\tau`,
            in :math:`\text{ms}`.
        delay (float, optional): maximum supported delay, in :math:`\text{ms}`.
            Defaults to ``0.0``.
        spike_interp_mode (Literal["nearest", "previous"], optional): interpolation mode
            for spike selectors between observations. Defaults to ``"nearest"``.
        interp_tol (float, optional): maximum difference in time from an observation
            to treat as co-occurring, in :math:`\text{ms}`. Defaults to ``0.0``.
        current_overbound (float | None, optional): value to replace currents out of
            bounds, uses values at observation limits if ``None``. Defaults to ``0.0``.
        spike_overbound (bool | None, optional): value to replace spikes out of bounds,
            uses values at observation limits if ``None``. Defaults to ``False``.
        batch_size (int, optional): size of input batches for simulation. Defaults to ``1``.
        inplace (bool): if write operations on :py:class:`~inferno.RecordTensor` attributes
            should be performed with in-place operations. Defaults to ``False``.

    See Also:
        For more details and references, visit
        :ref:`zoo/synapses-current:Single Exponential` in the zoo.
    """

    def __init__(
        self,
        shape: Sequence[int] | int,
        step_time: float,
        *,
        spike_charge: float,
        time_constant: float,
        delay: float = 0.0,
        spike_interp_mode: Literal["nearest", "previous"] = "previous",
        interp_tol: float = 0.0,
        current_overbound: float | None = 0.0,
        spike_overbound: bool | None = False,
        batch_size: int = 1,
        inplace: bool = False,
    ):
        # call superclass constructor
        InfernoSynapse.__init__(self, shape, step_time, delay, batch_size, inplace)

        # synapse attributes
        self.spike_charge = argtest.neq("spike_charge", spike_charge, 0, float)
        self.time_constant = argtest.gt("time_constant", time_constant, 0, float)

        match spike_interp_mode.lower():
            case "nearest":
                spike_interp_mode = interp_nearest
            case "previous":
                spike_interp_mode = interp_previous
            case _:
                raise RuntimeError(
                    f"invalid ispike_interp_modenterp_mode '{spike_interp_mode}' received, "
                    "must be one of 'nearest' or 'previous'."
                )

        # call mixin constructor
        SpikeCurrentMixin.__init__(
            self,
            torch.zeros(*self.batchedshape),
            torch.zeros(*self.batchedshape, dtype=torch.bool),
            current_interp=interp_expdecay,
            current_interp_kwargs={"time_constant": self.time_constant},
            spike_interp=spike_interp_mode,
            spike_interp_kwargs={},
            current_overbound=current_overbound,
            spike_overbound=spike_overbound,
            tolerance=interp_tol,
        )

    @classmethod
    def partialconstructor(
        cls,
        spike_charge: float,
        time_constant: float,
        spike_interp_mode: Literal["nearest", "previous"] = "previous",
        interp_tol: float = 0.0,
        current_overbound: float | None = 0.0,
        spike_overbound: bool | None = False,
        inplace: bool = False,
    ):
        r"""Returns a function with a common signature for synapse construction.

        Args:
            spike_charge (float): charge carried by each presynaptic spike, in :math:`\text{pC}`.
            time_constant (float): exponential time constant for current decay, in :math:`\text{ms}`.
            spike_interp_mode (Literal["nearest", "previous"], optional): interpolation mode
                for spike selectors between observations. Defaults to ``"nearest"``.
            interp_tol (float, optional): maximum difference in time from an observation
                to treat as co-occurring, in :math:`\text{ms}`. Defaults to ``0.0``.
            current_overbound (float | None, optional): value to replace currents out of
                bounds, uses values at observation limits if ``None``. Defaults to ``0.0``.
            spike_overbound (bool | None, optional): value to replace spikes out of bounds,
                uses values at observation limits if ``None``. Defaults to ``False``.
            inplace (bool): if write operations on :py:class:`~inferno.RecordTensor` attributes
                should be performed with in-place operations. Defaults to ``False``.

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
                spike_charge=spike_charge,
                time_constant=time_constant,
                delay=delay,
                spike_interp_mode=spike_interp_mode,
                interp_tol=interp_tol,
                current_overbound=current_overbound,
                spike_overbound=spike_overbound,
                batch_size=batch_size,
                inplace=inplace,
            )

        return constructor

    def clear(self, **kwargs) -> None:
        r"""Resets synapses to their resting state."""
        self.spike_.reset(False)
        self.current_.reset(0.0)

    def forward(self, *inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""Runs a simulation step of the synaptic dynamics.

        Args:
            *inputs (torch.Tensor): input spikes to the synapse.

        Returns:
            torch.Tensor: synaptic currents after simulation step.
        """
        self.spike = inputs[0].bool()
        self.current = (
            self.current * math.exp(-self.dt / self.time_constant)
            + (self.spike_charge / self.time_constant) * inputs[0]
        )
        return self.current


class DoubleExponentialCurrent(SpikeMixin, InfernoSynapse):
    r"""Exponentially applied exponentially decaying current-based synapse.

    .. math::
        \begin{align*}
            I(t) &= I_d(t) - I_r(t) \\
            I_d(t + \Delta t) &= I_d(t) \exp \left(-\frac{\Delta t}{\tau_d}\right)
            + \frac{Q}{\tau_d - \tau_r} \left[t = t_f\right] \\
            I_r(t + \Delta t) &= I_r(t) \exp \left(-\frac{\Delta t}{\tau_r}\right)
            + \frac{Q}{\tau_d - \tau_r} \left[t = t_f\right]
        \end{align*}

    Attributes:
        spike_: :py:class:`~inferno.RecordTensor` interface for spikes.
        pos_current_: :py:class:`~inferno.RecordTensor` interface for added currents.
        neg_current_: :py:class:`~inferno.RecordTensor` interface for subtracted currents.

    Args:
        shape (Sequence[int] | int): shape of the group of synapses being simulated.
        step_time (float): length of a simulation time step, :math:`\Delta t`,
            in :math:`\text{ms}`.
        spike_charge (float): charge carried by each presynaptic spike, :math:`Q`,
            in :math:`\text{pC}`.
        tc_decay (float): exponential time constant for current decay, :math:`\tau_d`,
            in :math:`\text{ms}`.
        tc_rise (float): exponential time constant for current rise, :math:`\tau_r`,
            in :math:`\text{ms}`.
        delay (float, optional): maximum supported delay, in :math:`\text{ms}`.
            Defaults to ``0.0``.
        spike_interp_mode (Literal["nearest", "previous"], optional): interpolation mode
            for spike selectors between observations. Defaults to ``"nearest"``.
        interp_tol (float, optional): maximum difference in time from an observation
            to treat as co-occurring, in :math:`\text{ms}`. Defaults to ``0.0``.
        current_overbound (float | None, optional): value to replace currents out of
            bounds, uses values at observation limits if ``None``. Defaults to ``0.0``.
        spike_overbound (bool | None, optional): value to replace spikes out of bounds,
            uses values at observation limits if ``None``. Defaults to ``False``.
        batch_size (int, optional): size of input batches for simulation. Defaults to ``1``.
        inplace (bool): if write operations on :py:class:`~inferno.RecordTensor` attributes
            should be performed with in-place operations. Defaults to ``False``.

    See Also:
        For more details and references, visit
        :ref:`zoo/synapses-current:Double Exponential` in the zoo.
    """

    def __init__(
        self,
        shape: Sequence[int] | int,
        step_time: float,
        *,
        spike_charge: float,
        tc_decay: float,
        tc_rise: float,
        delay: float = 0.0,
        spike_interp_mode: Literal["nearest", "previous"] = "previous",
        interp_tol: float = 0.0,
        current_overbound: float | None = 0.0,
        spike_overbound: bool | None = False,
        batch_size: int = 1,
        inplace: bool = False,
    ):
        # call superclass constructor
        InfernoSynapse.__init__(self, shape, step_time, delay, batch_size, inplace)

        # synapse attributes
        self.spike_charge = argtest.neq("spike_charge", spike_charge, 0, float)
        self.tc_rise = argtest.gt("tc_rise", tc_rise, 0, float)
        self.tc_decay = argtest.gt("tc_decay", tc_decay, tc_rise, float, "tc_rise")

        match spike_interp_mode.lower():
            case "nearest":
                spike_interp_mode = interp_nearest
            case "previous":
                spike_interp_mode = interp_previous
            case _:
                raise RuntimeError(
                    f"invalid ispike_interp_modenterp_mode '{spike_interp_mode}' received, "
                    "must be one of 'nearest' or 'previous'."
                )

        # call mixin constructor
        SpikeMixin.__init__(
            self,
            torch.zeros(*self.batchedshape, dtype=torch.bool),
            interpolation=spike_interp_mode,
            interp_kwargs={},
            overbound=spike_overbound,
            tolerance=interp_tol,
        )

        # create separate current RecordTensors
        RecordTensor.create(
            self,
            "pos_current_",
            self.dt,
            self.delay,
            torch.zeros(*self.batchedshape),
            persist_data=True,
            persist_constraints=False,
            persist_temporal=False,
            strict=True,
            live=False,
            inclusive=True,
        )
        self.add_delayed("pos_current_")
        self.add_batched("pos_current_")

        RecordTensor.create(
            self,
            "neg_current_",
            self.dt,
            self.delay,
            torch.zeros(*self.batchedshape),
            persist_data=True,
            persist_constraints=False,
            persist_temporal=False,
            strict=True,
            live=False,
            inclusive=True,
        )
        self.add_delayed("neg_current_")
        self.add_batched("neg_current_")

        # current interpolation properties
        self.__current_overbound = current_overbound
        self.__tolerance = float(interp_tol)

    @classmethod
    def partialconstructor(
        cls,
        spike_charge: float,
        tc_decay: float,
        tc_rise: float,
        spike_interp_mode: Literal["nearest", "previous"] = "previous",
        interp_tol: float = 0.0,
        current_overbound: float | None = 0.0,
        spike_overbound: bool | None = False,
        inplace: bool = False,
    ):
        r"""Returns a function with a common signature for synapse construction.

        Args:
            spike_charge (float): charge carried by each presynaptic spike, in :math:`\text{pC}`.
            tc_decay (float): exponential time constant for current decay, in :math:`\text{ms}`.
            tc_rise (float): exponential time constant for current rise, in :math:`\text{ms}`.
            spike_interp_mode (Literal["nearest", "previous"], optional): interpolation mode
                for spike selectors between observations. Defaults to ``"nearest"``.
            interp_tol (float, optional): maximum difference in time from an observation
                to treat as co-occurring, in :math:`\text{ms}`. Defaults to ``0.0``.
            current_overbound (float | None, optional): value to replace currents out of
                bounds, uses values at observation limits if ``None``. Defaults to ``0.0``.
            spike_overbound (bool | None, optional): value to replace spikes out of bounds,
                uses values at observation limits if ``None``. Defaults to ``False``.
            inplace (bool): if write operations on :py:class:`~inferno.RecordTensor` attributes
                should be performed with in-place operations. Defaults to ``False``.

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
                spike_charge=spike_charge,
                tc_decay=tc_decay,
                tc_rise=tc_rise,
                delay=delay,
                spike_interp_mode=spike_interp_mode,
                interp_tol=interp_tol,
                current_overbound=current_overbound,
                spike_overbound=spike_overbound,
                batch_size=batch_size,
                inplace=inplace,
            )

        return constructor

    def clear(self, **kwargs) -> None:
        r"""Resets synapses to their resting state."""
        self.spike_.reset(False)
        self.pos_current_.reset(0.0)
        self.neg_current_.reset(0.0)

    @property
    def current(self) -> torch.Tensor:
        r"""Currents of the synapses at present, in nanoamperes.

        Args:
            value (torch.Tensor): new synapse currents.

        Returns:
            torch.Tensor: present synaptic currents.

        Important:
            The setter for this property does nothing as `current` is a derived value.
            Use the :py:attr:`pos_current` and :py:attr:`neg_current` setters for this
            instead.
        """
        return self.pos_current_.peek() - self.neg_current_.peek()

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
        # undelayed access (spike and current RecordTensors have the same recordsz)
        if self.spike_.recordsz == 1:
            # bounded selector for overbounding
            bounded_selector = 0

            # retrieve most recent value
            res = self.pos_current_.peek() - self.neg_current_.peek()

        # delayed access
        else:
            # bound the selector
            bounded_selector = selector.clamp(min=0, max=self.spike_.duration)

            # select values using RecordTensor
            res = self.pos_current_.select(
                bounded_selector,
                interp_expdecay,
                tolerance=self.__tolerance,
                interp_kwargs={"time_constant": self.tc_decay},
            ) - self.neg_current_.select(
                bounded_selector,
                interp_expdecay,
                tolerance=self.__tolerance,
                interp_kwargs={"time_constant": self.tc_rise},
            )

        # apply overbound if specified
        if self.__current_overbound is not None:
            res = torch.where(
                (selector - bounded_selector).abs() <= self.__tolerance,
                res,
                self.__current_overbound,
            )

        # return parameter values at delayed indices
        return res

    @property
    def pos_current(self) -> torch.Tensor:
        r"""Positive component of currents of the synapses at present, in nanoamperes.

        Args:
            value (torch.Tensor): new positive component of synapse currents.

        Returns:
            torch.Tensor: present positive component of synaptic currents.
        """
        return self.pos_current_.peek()

    @pos_current.setter
    def pos_current(self, value: torch.Tensor) -> None:
        self.pos_current_.push(value, self.inplace)

    def pos_current_at(self, selector: torch.Tensor) -> torch.Tensor:
        r"""Retrieves positive component of previous synaptic currents, in nanoamperes.

        Args:
            selector (torch.Tensor): time before present for which positive component of
                synaptic currents should be retrieved, in :math:`\text{ms}`.

        Returns:
            torch.Tensor: selected positive component of synaptic currents.

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
            self.pos_current_,
            selector,
            interp_expdecay,
            {"time_constant": self.tc_decay},
            self.__tolerance,
            self.__current_overbound,
            None,
        )

    @property
    def neg_current(self) -> torch.Tensor:
        r"""Negative component of currents of the synapses at present, in nanoamperes.

        Args:
            value (torch.Tensor): new negative component of synapse currents.

        Returns:
            torch.Tensor: present negative component of synaptic currents.
        """
        return self.neg_current_.peek()

    @neg_current.setter
    def neg_current(self, value: torch.Tensor) -> None:
        self.neg_current_.push(value, self.inplace)

    def neg_current_at(self, selector: torch.Tensor) -> torch.Tensor:
        r"""Retrieves negative component of previous synaptic currents, in nanoamperes.

        Args:
            selector (torch.Tensor): time before present for which negative component of
                synaptic currents should be retrieved, in :math:`\text{ms}`.

        Returns:
            torch.Tensor: selected negative component of synaptic currents.

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
            self.neg_current_,
            selector,
            interp_expdecay,
            {"time_constant": self.tc_rise},
            self.__tolerance,
            self.__current_overbound,
            None,
        )

    def forward(self, *inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""Runs a simulation step of the synaptic dynamics.

        Args:
            *inputs (torch.Tensor): input spikes to the synapse.

        Returns:
            torch.Tensor: synaptic currents after simulation step.
        """
        self.spike = inputs[0].bool()
        self.pos_current = (
            self.pos_current * math.exp(-self.dt / self.tc_decay)
            + (self.spike_charge / (self.tc_decay - self.tc_rise)) * inputs[0]
        )
        self.neg_current = (
            self.neg_current * math.exp(-self.dt / self.tc_rise)
            + (self.spike_charge / (self.tc_decay - self.tc_rise)) * inputs[0]
        )
        return self.current
