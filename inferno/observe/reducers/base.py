from __future__ import annotations
from abc import ABC, abstractmethod
import inferno
from ... import HistoryModule
from inferno.infernotypes import OneToOne
import torch
from typing import Callable


class Reducer(HistoryModule, ABC):
    r"""Abstract base class for the recording of inputs over time."""

    def __init__(self, step_time, history_len):
        HistoryModule.__init__(self, step_time, history_len)

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

    def forward(self, inputs: torch.Tensor, **kwargs) -> None:
        """Initializes state and incorporates inputs into the reducer's state."""
        raise NotImplementedError(
            f"'Reducer.forward()' is abstract, {type(self).__name__} "
            "must implement the 'forward' method"
        )


class FoldReducer(Reducer):
    r"""Applies a function between the most recent previously stored data and a new observation.

    Args:
        fold_fn (Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor]): relation
            between the new input (left) and current state (right),
            returning the new state.
        step_time (float): length of time between observations.
        history_len (float): length of time for which observations should be stored.
        interpolation (inferno.Interpolation, optional): interpolation function to use
            when retrieving data between observations.
            Defaults to :py:func:`inferno.interp_nearest`.
        initializer (OneToOne[torch.Tensor] | None, optional): function to set the
            initial state, zeroes when None. Defaults to None.

    Note:
        The left-hand argument of ``fold_fn`` is the new input, and the right-hand
        argument is the current state. The right-hand argument will be None when
        no observations have been recorded.

    Note:
        The default ``map_fn`` implemented assumes only a single input, and will
        therefore fail if multiple input values are passed into :py:meth:`forward`.
    """

    def __init__(
        self,
        fold_fn: Callable[[torch.Tensor | None, torch.Tensor], torch.Tensor],
        step_time: float,
        history_len: float,
        *,
        interpolation: inferno.Interpolation = inferno.interp_nearest,
        initializer: OneToOne[torch.Tensor] | None = None,
    ):
        # call superclass constructor
        Reducer.__init__(self, step_time, history_len)

        # set non-persistant functions
        self.fold_ = fold_fn
        self.interpolate_ = interpolation
        self.initialize_ = initializer if initializer else lambda x: x.fill_(0)

        # register data buffer and helpers
        self.register_buffer("_data", torch.empty(0))
        self.register_constrained("_data")
        self.register_extra("_initial", True)

    @property
    def data(self):
        r"""Length of the simulation time step, in milliseconds.

        Args:
            value (torch.Tensor): new data storage tensor.

        Returns:
            torch.Tensor: data storage tensor.

        Note:
            The shape must be equivalent to the original, this allows for calls to
            be made to methods such as :py:meth:`~torch.Tensor.to`.

        Note:
            The order of the data tensor is not equivalent to the historical order.
            Use :py:meth:`dump` for this.
        """
        return self._data

    @data.setter
    def data(self, value: torch.Tensor):
        if value.shape != self._data.shape:
            raise RuntimeError(
                "shape of data cannot be changed, received value of shape "
                f"{tuple(value.shape)}, required value of "
                f"shape {tuple(self._data.shape)}"
            )
        self._data = value

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
            In the same units as :py:attr:`self.hlen`.
        """
        return Reducer.dt.fget(self)

    @dt.setter
    def dt(self, value: float):
        Reducer.dt.fset(self, value)
        self.clear(keepshape=True)

    @property
    def hlen(self) -> float:
        r"""Length of time over which prior values are stored.

        Args:
            value (float): new length of the history to store.

        Returns:
            float: length of the history.

        Note:
            Altering this property will reset the reducer.

        Note:
            In the same units as :py:attr:`self.dt`.
        """
        return Reducer.hlen.fget(self)

    @hlen.setter
    def hlen(self, value: float):
        Reducer.hlen.fset(self, value)
        self.clear(keepshape=True)

    def clear(self, keepshape=False, **kwargs) -> None:
        r"""Reinitializes the reducer's state.

        Args:
            keepshape (bool, optional): if the underlying storage shape should be
                preserved. Defaults to False.
        """
        if keepshape:
            self.reset("_data")
            self.data = self.initialize_(self.data)
        else:
            self.deregister_constrained("_data")
            self._data = inferno.empty(self.data, shape=(0,))
            self.register_constrained("_data")
        self._initial = True

    def view(
        self,
        time: float | torch.Tensor,
        tolerance: float = 1e-7,
    ) -> torch.Tensor | None:
        r"""Returns the reducer's state at a given time.

        Args:
            time (float | torch.Tensor): times, measured before present, at which
                to select from.
            tolerance (float, optional): maximum difference in time from a discrete
                sample to consider it at the same time as that sample. Defaults to 1e-7.

        Returns:
            torch.Tensor | None: temporally indexed and interpolated state.

        .. admonition:: Shape
            :class: tensorshape

            ``time``:

            :math:`N_0 \times \cdots \times [D]`

            ``return``:

            :math:`N_0 \times \cdots \times [D]`

            Where:
                * :math:`N_0, \ldots` are the dimensions of the reducer, including any underlying batch.
                * :math:`D` is the number of times for each value to select.

        Note:
            Before any samples have been added and before the data tensor has been
            initialized, this will return None.

        Note:
            The constraints on the shape of ``time`` are not enforced, and follows
            the underlying logic of :py:func:`torch.gather`.

        Note:
            This will fail if any values in ``time`` fall outside of the possible range.
        """
        if not self._initial:
            return self.select("_data", time, self.interpolate_, tolerance=tolerance)

    def dump(self, **kwargs) -> torch.Tensor | None:
        r"""Returns the reducer's state over all observations.

        Returns:
            torch.Tensor | None: state over all observations, if state exists.

        Note:
            Before any samples have been added and before the data tensor has been
            initialized, this will return None.

        Note:
            Results are temporally ordered from most recent to oldest, along the
            last dimension.
        """
        if not self._initial:
            return self.history("_data", latest_first=True)

    def peek(self, **kwargs) -> torch.Tensor | None:
        r"""Returns the reducer's current state.

        Returns:
            torch.Tensor | None: current state, if state exists.

        Note:
            Before any samples have been added and before the data tensor has been
            initialized, this will return None.
        """
        if not self._initial:
            return self.latest("_data")

    def push(self, inputs: torch.Tensor, **kwargs) -> None:
        r"""Incorporates inputs into the reducer's state.

        Args:
            inputs (torch.Tensor): new observation to incorporate into state.
        """
        self.pushto("_data", inputs)

    def forward(self, inputs: torch.Tensor, **kwargs) -> None:
        """Initializes state and incorporates inputs into the reducer's state.

        This performs any required initialization steps, maps the inputs,
        and pushes the new data.

        Args:
            *inputs (torch.Tensor): inputs to be mapped, then pushed.
        """
        # initialize data iff uninitialized
        if self.data.numel() == 0:
            self._data = inferno.empty(self._data, shape=(*inputs.shape, self.hsize))
            self.data = self.initialize_(self.data)

        # integrate inputs
        if not self._initial:
            self.push(self.fold_(inputs, self.latest("_data")))
        else:
            self.push(self.fold_(inputs, None))
            self._initial = False


class FoldingReducer(FoldReducer, ABC):
    """Subclassable reducer performing a fold operation between previous state and an observation.

    Args:
        step_time (float): length of time between observations.
        history_len (float): length of time for which observations should be stored.
    """

    def __init__(self, step_time: float, history_len: float):
        FoldReducer.__init__(
            self,
            fold_fn=self.fold,
            step_time=step_time,
            history_len=history_len,
            interpolation=self.interpolate,
            initializer=self.initialize,
        )

    @abstractmethod
    def fold(self, obs: torch.Tensor, state: torch.Tensor | None) -> torch.Tensor:
        r"""Calculation of the next state given an observation and prior state.

        Args:
            obs (torch.Tensor): observation to incorporate into state.
            state (torch.Tensor | None): state from the prior time step,
                None if no prior observations.

        Raises:
            NotImplementedError: abstract methods must be implemented by subclass.

        Returns:
            torch.Tensor: state for the current time step.
        """
        raise NotImplementedError(
            f"'FoldingReducer.fold()' is abstract, {type(self).__name__} "
            "must implement the 'fold' method"
        )

    @abstractmethod
    def initialize(self, inputs: torch.Tensor) -> torch.Tensor:
        r"""Initialization of state history before any observations are incorporated.

        Args:
            inputs (torch.Tensor): empty tensor of state.

        Raises:
            NotImplementedError: abstract methods must be implemented by subclass.

        Returns:
            torch.Tensor: filled state tensor.
        """
        raise NotImplementedError(
            f"'FoldingReducer.initialize()' is abstract, {type(self).__name__} "
            "must implement the 'initialize' method"
        )

    @abstractmethod
    def interpolate(
        self,
        prev_data: torch.Tensor,
        next_data: torch.Tensor,
        sample_at: torch.Tensor,
        step_time: float,
    ) -> torch.Tensor:
        r"""Manner of sampling state between observations.

        Args:
            prev_data (torch.Tensor): most recent observation prior to sample time.
            next_data (torch.Tensor): most recent observation subsequent to sample time.
            sample_at (torch.Tensor): relative time at which to sample data.
            step_data (float): length of time between the prior and
                subsequent observations.

        Raises:
            NotImplementedError: abstract methods must be implemented by subclass.

        Returns:
            torch.Tensor: interpolated data at sample time.
        """
        raise NotImplementedError(
            f"'FoldingReducer.interpolate()' is abstract, {type(self).__name__} "
            "must implement the 'interpolate' method"
        )
