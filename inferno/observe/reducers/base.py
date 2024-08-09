from __future__ import annotations
from ... import Module, RecordTensor
from ..._internal import argtest
from abc import ABC, abstractmethod
import torch
from typing import Any


class Reducer(Module, ABC):
    r"""Abstract base class for the recording of inputs over time."""

    def __init__(self):
        Module.__init__(self)

    @property
    def latest(self) -> torch.Tensor:
        r"""Return's the reducer's current state.

        If :py:meth:`peek` has multiple options, this should be considered as the
        default. Unless overridden, :py:meth:`peek` is called without arguments.

        Returns:
            torch.Tensor: reducer's current state.
        """
        return self.peek()

    @abstractmethod
    def clear(self, **kwargs) -> None:
        r"""Reinitializes the reducer's state."""
        raise NotImplementedError(
            f"'Reducer.clear()' is abstract, {type(self).__name__} "
            "must implement the 'clear' method"
        )

    @abstractmethod
    def view(self, *args, **kwargs) -> torch.Tensor | None:
        r"""Returns the reducer's state at a given time."""
        raise NotImplementedError(
            f"'Reducer.view()' is abstract, {type(self).__name__} "
            "must implement the 'peek' method"
        )

    @abstractmethod
    def dump(self, *args, **kwargs) -> torch.Tensor | None:
        r"""Returns the reducer's state over all observations."""
        raise NotImplementedError(
            f"'Reducer.dump()' is abstract, {type(self).__name__} "
            "must implement the 'peek' method"
        )

    @abstractmethod
    def peek(self, *args, **kwargs) -> torch.Tensor | None:
        r"""Returns the reducer's current state."""
        raise NotImplementedError(
            f"'Reducer.peek()' is abstract, {type(self).__name__} "
            "must implement the 'peek' method"
        )

    @abstractmethod
    def push(self, inputs: torch.Tensor, **kwargs) -> None:
        r"""Incorporates inputs into the reducer's state."""
        raise NotImplementedError(
            f"'Reducer.push()' is abstract, {type(self).__name__} "
            "must implement the 'push' method"
        )

    def forward(self, *inputs: torch.Tensor, **kwargs) -> None:
        """Initializes state and incorporates inputs into the reducer's state."""
        raise NotImplementedError(
            f"'Reducer.forward()' is abstract, {type(self).__name__} "
            "must implement the 'forward' method"
        )


class RecordReducer(Reducer, ABC):
    r"""Abstract base class for the reducers utilizing multiple RecordTensors.

    Args:
        step_time (float): length of time between observations.
        duration (float): length of time for which observations should be stored.
        inclusive (bool, optional): if the duration should be inclusive. Defaults to ``False``.
        inplace (bool, optional): if write operations should be performed
            in-place. Defaults to ``False``.
    """

    def __init__(
        self,
        step_time: float,
        duration: float,
        inclusive: bool = False,
        inplace: bool = False,
    ):
        # call superclass constructor
        Reducer.__init__(self)

        # validate parameters
        self.__step_time = argtest.gt("step_time", step_time, 0, float)
        self.__duration = argtest.gte("duration", duration, 0, float)
        self.__inclusive = bool(inclusive)
        self.__inplace = bool(inplace)

        # collection of record names
        self.__records = set()

    def add_record(self, *attr: str) -> None:
        """Add a record attribute

        Args:
            *attr (str): names of the attributes to set as records.
        """
        for a in attr:
            if not hasattr(self, a):
                raise RuntimeError(f"no attribute '{a}' exists")
            elif not isinstance(getattr(self, a), RecordTensor):
                raise TypeError(
                    f"attribute '{a}' specifies a {type(getattr(self, a).__name__)}, not a RecordTensor"
                )
            else:
                getattr(self, a).dt = self.__step_time
                getattr(self, a).duration = self.__duration
                getattr(self, a).inclusive = self.__inclusive
                self.__records.add(a)

    @property
    def dt(self) -> float:
        r"""Length of time between stored values in history.

        Args:
            value (float): new time step length.

        Returns:
            float: length of the time step.

        Note:
            Altering this property will reset the reducer.

        Note:
            In the same units as :py:attr:`duration`.
        """
        return self.__step_time

    @dt.setter
    def dt(self, value: float) -> None:
        value = argtest.gt("dt", value, 0, float)
        if value != self.__step_time:
            for rec in self.__records:
                getattr(self, rec).dt = value
            self.__step_time = value

    @property
    def duration(self) -> float:
        r"""Length of time over which prior values are stored.

        Args:
            value (float): new length of the history to store.

        Returns:
            float: length of the history.

        Note:
            Altering this property will reset the reducer.

        Note:
            In the same units as :py:attr:`dt`.
        """
        return self.__duration

    @duration.setter
    def duration(self, value: float) -> None:
        value = argtest.gt("duration", value, 0, float)
        if value != self.__duration:
            for rec in self.__records:
                getattr(self, rec).duration = value
            self.__step_time = value

    @property
    def inplace(self) -> bool:
        r"""If write operations should be performed in-place.

        Args:
            value (bool): if write operations should be performed in-place.

        Returns:
            bool: if write operations should be performed in-place.

        Note:
            Generally if gradient computation is required, this should be set to
            ``False``.
        """
        return self.__inplace

    @inplace.setter
    def inplace(self, value: bool) -> None:
        self.__inplace = bool(value)


class FoldReducer(RecordReducer, ABC):
    r"""Subclassable reducer performing a fold operation between previous state and an observation.

    Args:
        step_time (float): length of time between observations.
        duration (float): length of time for which observations should be stored.
        inclusive (bool, optional): if the duration should be inclusive.
            Defaults to ``False``.
        inplace (bool, optional): if write operations should be performed
            in-place. Defaults to ``False``.
        fill (Any, optional): value with which to fill the stored record on clearing and
            initialization. Defaults to ``0``.
    """

    def __init__(
        self,
        step_time: float,
        duration: float,
        inclusive: bool = False,
        inplace: bool = False,
        fill: Any = 0,
    ):
        # call superclass constructor
        RecordReducer.__init__(self, step_time, duration, inclusive, inplace)

        # register data buffer and helpers
        RecordTensor.create(
            self,
            "data_",
            self.dt,
            self.duration,
            torch.empty(0),
            persist_data=True,
            persist_constraints=False,
            persist_temporal=False,
            strict=True,
            live=False,
            inclusive=inclusive,
        )
        self.add_record("data_")
        self.register_extra("_initial", True)
        self.__fill = fill

    @property
    def data(self) -> torch.Tensor:
        r"""Length of the simulation time step, in milliseconds.

        The shape must be equivalent to the original, this allows for calls to
        be made to methods such as :py:meth:`~torch.Tensor.to`.

        Args:
            value (torch.Tensor): new data storage tensor.

        Returns:
            torch.Tensor: data storage tensor.

        Important:
            The order of the data tensor is not equivalent to the historical order.
            Use :py:meth:`dump` for this.
        """
        return self.data_.value

    @data.setter
    def data(self, value: torch.Tensor) -> None:
        if value.shape != self.data_.value.shape:
            raise RuntimeError(
                "shape of data cannot be changed, received value of shape "
                f"{tuple(value.shape)}, required value of "
                f"shape {tuple(self.data_.value.shape)}"
            )
        self.data_.value = value

    def clear(self, keepshape=False, **kwargs) -> None:
        r"""Reinitializes the reducer's state.

        Args:
            keepshape (bool, optional): if the underlying storage shape should be
                preserved. Defaults to ``False``.
        """
        if keepshape:
            self.data_.reset(self.__fill)
        else:
            self.data_.deinitialize(False)
        self._initial = True

    @abstractmethod
    def fold(self, *args: torch.Tensor | None) -> torch.Tensor:
        r"""Calculation of the next state given an observation and prior state.

        Args:
            *args (torch.Tensor | None): positional arguments for folding, all but
                the last will be observations, the final will be the reduced state.

        Raises:
            NotImplementedError: abstract methods must be implemented by subclass.

        Returns:
            torch.Tensor: state for the current time step.
        """
        raise NotImplementedError(
            f"'FoldReducer.fold()' is abstract, {type(self).__name__} "
            "must implement the 'fold' method"
        )

    @abstractmethod
    def interpolate(
        self,
        prev_data: torch.Tensor,
        next_data: torch.Tensor,
        sample_at: torch.Tensor,
        step_time: float | torch.Tensor,
    ) -> torch.Tensor:
        r"""Manner of sampling state between observations.

        Args:
            prev_data (torch.Tensor): most recent observation prior to sample time.
            next_data (torch.Tensor): most recent observation subsequent to sample time.
            sample_at (torch.Tensor): relative time at which to sample data.
            step_time (float | torch.Tensor): length of time between the prior and
                subsequent observations.

        Raises:
            NotImplementedError: abstract methods must be implemented by subclass.

        Returns:
            torch.Tensor: interpolated data at sample time.
        """
        raise NotImplementedError(
            f"'FoldReducer.interpolate()' is abstract, {type(self).__name__} "
            "must implement the 'interpolate' method"
        )

    def view(
        self,
        time: float | torch.Tensor,
        tolerance: float = 1e-7,
    ) -> torch.Tensor | None:
        r"""Returns the reducer's state at a given time.

        Before any samples have been added and before the data tensor has been
        initialized, this will return None. This will fail if any values in ``time``
        fall outside of the possible range.

        Args:
            time (float | torch.Tensor): times, measured before present, at which
                to select from.
            tolerance (float, optional): maximum difference in time from a discrete
                sample to consider it at the same time as that sample.
                Defaults to ``1e-7``.

        Returns:
            torch.Tensor | None: temporally indexed and interpolated state.

        .. admonition:: Shape
            :class: tensorshape

            ``time``:

            :math:`S_0 \times \cdots \times [D]`

            ``return``:

            :math:`S_0 \times \cdots \times [D]`

            Where:
                * :math:`S_0, \ldots` are the dimensions of each observation, given
                  by the shape of the data.
                * :math:`D` are the number of distinct observations to select.

        """
        if not self._initial:
            return self.data_.select(time, self.interpolate, tolerance=tolerance)

    def dump(self, **kwargs) -> torch.Tensor | None:
        r"""Returns the reducer's state over all observations.

        Returns:
            torch.Tensor | None: state over all observations, if state exists.

        Note:
            Before any samples have been added and before the data tensor has been
            initialized, this will return ``None``.

        Note:
            Results are temporally ordered from most recent to oldest, along the
            first dimension.
        """
        if not self._initial:
            self.data_.align(0)
            return self.data_.value.flip(0)

    def peek(self, **kwargs) -> torch.Tensor | None:
        r"""Returns the reducer's current state.

        Before any samples have been added and before the data tensor has been
        initialized, this will return ``None``.

        Returns:
            torch.Tensor | None: current state, if state exists.
        """
        if not self._initial:
            return self.data_.peek()

    def push(self, inputs: torch.Tensor, **kwargs) -> None:
        r"""Incorporates inputs into the reducer's state.

        Args:
            inputs (torch.Tensor): new observation to incorporate into state.
        """
        self.data_.push(inputs, inplace=self.inplace)

    def forward(self, *inputs: torch.Tensor, **kwargs) -> None:
        r"""Initializes state and incorporates inputs into the reducer's state.

        This performs any required initialization steps, maps the inputs,
        and pushes the new data.

        Args:
            *inputs (torch.Tensor): inputs to be mapped, then pushed.
        """
        # non-initial
        if not self._initial:
            self.push(self.fold(*inputs, self.peek()))

        # initial
        else:
            res = self.fold(*inputs, None)

            if self.data_.ignored:
                self.data_.initialize(res.shape, fill=self.__fill)

            self.push(res)
            self._initial = False
