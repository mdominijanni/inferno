from __future__ import annotations
from abc import ABC, abstractmethod
import inferno
from inferno import HistoryModule
from inferno.typing import OneToOne, ManyToOne
import torch
from typing import Callable


class Reducer(HistoryModule, ABC):
    r"""Abstract base class for the recording of inputs over time."""

    def __init__(self, step_time, history_len):
        HistoryModule.__init__(self, step_time, history_len)

    @abstractmethod
    def clear(self, *args, **kwargs) -> None:
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
        r"""Returns the reducer's entire state."""
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
    def push(self, inputs: torch.Tensor) -> None:
        r"""Incorporates inputs into the reducer's state."""
        raise NotImplementedError(
            f"'Reducer.push()' is abstract, {type(self).__name__} "
            "must implement the 'push' method"
        )

    def forward(self, *inputs: torch.Tensor, **kwargs) -> None:
        """Incorporates inputs into the reducer's state."""
        raise NotImplementedError(
            f"'Reducer.forward()' is abstract, {type(self).__name__} "
            "must implement the 'forward' method"
        )


class FoldReducer(Reducer):
    r"""Applies a function between the previously stored data and a new observation.

    Args:
        fold_fn (Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor]): recurrence relation between
        the new input (left) and current state (right), returning the new state.
        step_time (float): length of time between observations.
        history_len (float): length of time for which observations should be stored.
        interpolation (inferno.Interpolation, optional): interpolation function to use when retrieving
            data between observations. Defaults to inferno.interp_nearest.
        map_fn (OneToOne[torch.Tensor] | ManyToOne[torch.Tensor] | None, optional): transformation
            to apply to inputs, no transformation if None. Defaults to None.
        init_fn (OneToOne[torch.Tensor] | None, optional): function to set the initial state,
            zeroes when None. Defaults to None.

    Note:
        The left-hand argument of ``fold_fn`` is the new input, and the right-hand argument
        is the current state. The right-hand argument will be None when no observations have been recorded.

    Note:
        The default ``map_fn`` implemented assumes only a single input, and will therefore fail if multiple
        input values are passed into :py:meth:`forward`.
    """

    def __init__(
        self,
        fold_fn: Callable[[torch.Tensor | None, torch.Tensor], torch.Tensor],
        step_time: float,
        history_len: float,
        *,
        interpolation: inferno.Interpolation = inferno.interp_nearest,
        map_fn: OneToOne[torch.Tensor] | ManyToOne[torch.Tensor] | None = None,
        init_fn: OneToOne[torch.Tensor] | None = None,
    ):
        # call superclass constructor
        Reducer.__init__(self, step_time, history_len)

        # set non-persistant functions
        self.fold_ = fold_fn
        self.interpolate_ = interpolation
        self.map_ = map_fn if map_fn else lambda x: x
        self.initialize_ = init_fn if init_fn else lambda x: x.fill_(0)

        # register data buffer and helpers
        self.register_buffer("_data", torch.empty(0))
        self.register_constrained("_data")
        self.register_extra("_initial", True)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value: torch.Tensor):
        if value.shape != self._data.shape:
            raise RuntimeError(
                "shape of data cannot be changed, received value of shape "
                f"{tuple(value.shape)}, required value of shape {tuple(self._data.shape)}"
            )
        self._data = value

    @property
    def dt(self) -> float:
        r"""Length of the simulation time step, in milliseconds.

        Args:
            value (float): new simulation time step length.

        Returns:
            float: length of the simulation time step.
        """
        return Reducer.dt.fget(self)

    @dt.setter
    def dt(self, value: float):
        Reducer.dt.fset(self, value)
        self.clear(keepshape=True)

    def clear(self, keepshape=False) -> None:
        if keepshape:
            self.reset("_data")
            self.data = self.initialize_(self.data)
        else:
            self.deregister_constrained("_data")
            self._data = inferno.empty(self.data, shape=(0,))
            self.register_constrained("_data")
        self._initial = True

    def view(
        self, time: float | torch.Tensor, tolerance: float = 1e-7
    ) -> torch.Tensor | None:
        r"""Returns the reducer's state at a given time."""
        if self.data.numel() != 0:
            return self.select("_data", time, self.interpolate_, tolerance=tolerance)

    def dump(self, *args, **kwargs) -> torch.Tensor | None:
        r"""Returns the reducer's entire state."""
        if self.data.numel() != 0:
            return self.record("_data", latest_first=True)

    def peek(self) -> torch.Tensor | None:
        r"""Returns the reducer's current state."""
        if self.data.numel() != 0:
            return self.latest("_data")

    def push(self, inputs: torch.Tensor) -> None:
        r"""Incorporates inputs into the reducer's state."""
        self.pushto("_data", inputs)

    def forward(self, *inputs, **kwargs) -> None:
        # apply map to inputs
        inputs = self.map_(*inputs)

        # initialize data iff uninitialized
        if self.data.numel() == 0:
            self._data = inferno.empty(self.data, shape=(inputs.shape + (self.hlen,)))
            self.data = self.initialize_(self.data)
        # integrate inputs
        if not self._initial:
            self.push(self.fold_(self.latest("_data"), inputs))
        else:
            self.push(self.fold_(self.latest("_data"), None))
            self._initial = False


class FoldingReducer(FoldReducer, ABC):
    def __init__(self, step_time, history_len):
        FoldReducer.__init__(
            fold_fn=self.fold,
            step_time=step_time,
            history_len=history_len,
            interpolation=self.interpolate,
            map_fn=self.map,
            init_fn=self.initialize,
        )

    @abstractmethod
    def fold(self, obs: torch.Tensor, state: torch.Tensor | None) -> torch.Tensor:
        raise NotImplementedError(
            f"'FoldingReducer.fold()' is abstract, {type(self).__name__} "
            "must implement the 'fold' method"
        )

    @abstractmethod
    def map(self, *inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            f"'FoldingReducer.map()' is abstract, {type(self).__name__} "
            "must implement the 'map' method"
        )

    @abstractmethod
    def initialize(self, inputs: torch.Tensor) -> torch.Tensor:
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
        raise NotImplementedError(
            f"'FoldingReducer.interpolate()' is abstract, {type(self).__name__} "
            "must implement the 'interpolate' method"
        )
