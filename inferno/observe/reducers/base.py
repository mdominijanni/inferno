from __future__ import annotations
from abc import ABC, abstractmethod
import inferno
from inferno import Module
from inferno.typing import OneToOne, ManyToOne, OneToOneMethod, ManyToOneMethod
import torch
import types
from typing import Callable


class Reducer(Module, ABC):
    r"""Abstract base class for the recording of inputs over time."""

    def __init__(self):
        Module.__init__(self)

    @abstractmethod
    def clear(self, *args, **kwargs) -> None:
        r"""Reinitializes the reducer's state."""
        raise NotImplementedError(
            f"'Reducer.clear()' is abstract, {type(self).__name__} "
            "must implement the 'clear' method"
        )

    @abstractmethod
    def peek(self, *args, **kwargs) -> torch.Tensor | None:
        r"""Returns the reducer's current state."""
        raise NotImplementedError(
            f"'Reducer.peek()' is abstract, {type(self).__name__} "
            "must implement the 'peek' method"
        )

    @abstractmethod
    def pop(self, *args, **kwargs) -> torch.Tensor | None:
        r"""Resets the reducer's future state and returns its current state."""
        raise NotImplementedError(
            f"'Reducer.pop()' is abstract, {type(self).__name__} "
            "must implement the 'pop' method"
        )

    @abstractmethod
    def push(self, *inputs: torch.Tensor) -> None:
        r"""Incorporates inputs into the reducer's state."""
        raise NotImplementedError(
            f"'Reducer.push()' is abstract, {type(self).__name__} "
            "must implement the 'push' method"
        )

    def forward(self, *inputs: torch.Tensor) -> None:
        """Incorporates inputs into the reducer's state."""
        self.push(*inputs)


class MapReducer(Reducer):
    r"""Stores values in a rolling window which meet specified criteria, after applying a transformation.

    Integration of new input follows the steps below.

    1. ``mapfn`` is applied to the inputs, returning a single tensor.
    2. ``filterfn`` is applied to the output of ``mapfn``, returning a boolean value.
    3. if and only if the output of ``filterfn`` is true, the output of ``mapfn`` replaces the data at the index specified by the pointer, and the pointer is incremented.

    Args:
        window (int): size of the window over which observations should be recorded.
        mapfn (OneToOne[torch.Tensor] | ManyToOne[torch.Tensor] | None, optional): transformation
            to apply to inputs, no transformation if None. Defaults to None.
        filterfn (Callable[[torch.Tensor], bool] | None, optional): conditional test if transformed
            input should be integrated, always will if None. Defaults to None.

    Raises:
        ValueError: ``window`` must be a positive integer.

    Note:
        The default ``mapfn`` implemented assumes only a single input, and will therefore fail
        if multiple input values are passed into :py:meth:`forward`.

    Note:
        The :py:func:`~torch.nn.Module.to` of PyTorch modules does not support dtypes which are neither floating point
        nor complex. In its place, :py:meth:`datato` can be used.
    """

    def __init__(
        self,
        window: int,
        *,
        mapfn: OneToOne[torch.Tensor] | ManyToOne[torch.Tensor] | None = None,
        filterfn: Callable[[torch.Tensor], bool] | None = None,
    ):
        # call superclass constructor
        Reducer.__init__(self)

        # check that the window size is valid
        if int(window) < 1:
            raise ValueError(f"`window` must be a positive integer, received {window}")

        # set non-persistant functions
        self._mapfn = mapfn if mapfn else lambda x: x
        self._filterfn = filterfn if filterfn else lambda x: True

        # register persistant state
        self.register_extra("_window", window)
        self.register_extra("_pointer", -1)
        self.register_extra("_isfull", False)
        self.register_buffer("_data", torch.empty(0))

    def datato(self, *args, **kwargs):
        """Applies PyTorch ``to()`` function to storage tensor.

        This directly calles :py:func:`torch.Tensor.to`` on the underlying tensor and
        replacing the pre-call tensor with this. It can be used to set non-float/non-complex
        datatypes.
        """
        self._data = self._data.to(*args, **kwargs)

    def clear(self, keepshape=True) -> None:
        r"""Reinitializes the reducer's state.

        Args:
            keepshape (bool, optional): if the data shape should be preserved. Defaults to True.

        Note:
            Setting ``keepshape`` to True is faster as storage doesn't need to be reallocated,
            but this assumes the shape of the monitored data will stay the same. This additionally
            preserves datatype and device.
        """
        if not keepshape:
            newdata = inferno.zeros(self._data, shape=(0,))
            del self._data
            self.register_buffer("_data", newdata)
        self._pointer = -1
        self._isfull = False

    def peek(self, dim: int = -1) -> torch.Tensor | None:
        r"""Returns the tensor of observed states since the reducer was last cleared.

        Args:
            dim (int, optional): dimension along which observations are stacked. Defaults to -1.

        Returns:
            torch.Tensor | None: recorded observations if any have been made, otherwise None.

        Note:
            The number of samples along the time dimension will not exceed the number of observations recorded.
            The oldest sample will be at the start of the tensor, and the latest will be at the end. If padding
            is required, consider applying :py:func:`torch.nn.functional.pad` to the output.
        """
        if self._pointer != -1:
            if self._isfull:
                indices = (
                    torch.arange(0, self._window, dtype=torch.int64) + self._pointer + 1
                ) % self._window
            else:
                indices = torch.arange(0, self._pointer + 1, dtype=torch.int64)
            return self._data[indices].movedim(0, dim)

    def pop(self, dim: int = -1, keepshape=True) -> torch.Tensor | None:
        r"""Returns the current state from observations and resets the reducer.

        Args:
            dim (int, optional): dimension along which observations are stacked. Defaults to -1.
            keepshape (bool, optional): if the data shape should be preserved. Defaults to True.

        Returns:
            torch.Tensor | None: recorded observations if any have been made, otherwise None.

        Note:
            Preservation of type/device from :py:meth:`clear` applies, as does the shape of
            the output from :py:meth:`peek`.
        """
        res = self.peek(dim)
        self.clear(keepshape)
        return res

    def push(self, *inputs: torch.Tensor) -> None:
        r"""Incorporates inputs into the cumulative state.

        Args:
            inputs (torch.Tensor...): data to incorporate into the current state.
        """
        # apply map function to the inputs
        inputs = self._mapfn(*inputs)

        # only proceed if the observation passes the filter
        if self._filterfn(inputs):
            # allocate data tensor if it is not yet allocated
            if self._data.numel() == 0:
                self._data = inferno.empty(
                    self._data, shape=(self._window, *inputs.shape)
                )
            # set the pointer to the next element to overwrite
            self._pointer = (self._pointer + 1) % self._window
            # overwrite existing data
            self._data[self._pointer] = inputs
            # check if the buffer is full and overwrites are being performed
            self._isfull = (self._pointer == self._window - 1) or self._isfull


class MappingReducer(MapReducer, ABC):
    r"""Abstract base class for reducers like MapReducer.

    Uses bound methods as functions passed to its parent, :py:class:`MapReducer`. Accepts arguments which, if not None,
    will be bound to the named class methods, for the instanciated instance only..

    Args:
        window (int): size of the window over which observations should be recorded.
        mapmeth (OneToOneMethod[MappingReducer, torch.Tensor] | ManyToOneMethod[MappingReducer, torch.Tensor] | None, optional): if not None,
            overwrites the class-defined transformation to apply to inputs. Defaults to None.
        filtermeth (Callable[[MappingReducer, torch.Tensor], bool] | None, optional): if not None, overwrites the class-defined
            conditional test used to determine if transformed input should be integrated. Defaults to None.
    Note:
        For details on the underlying logic, see :py:class:`MapReducer`."""

    def __init__(
        self,
        window: int,
        *,
        mapmeth: OneToOneMethod[MappingReducer, torch.Tensor]
        | ManyToOneMethod[MappingReducer, torch.Tensor]
        | None = None,
        filtermeth: Callable[[MappingReducer, torch.Tensor], bool] | None = None,
    ):
        # dynamically replace instance method definitions
        if mapmeth:
            self.map_fn = types.MethodType(mapmeth, self)
        if filtermeth:
            self.filter_fn = types.MethodType(filtermeth, self)

        # call superclass constructor
        MapReducer.__init__(
            self,
            window,
            mapfn=self.map_fn,
            filterfn=self.filter_fn,
        )

    @abstractmethod
    def map_fn(self, *inputs: torch.Tensor) -> torch.Tensor:
        r"""Transformation to apply to inputs.

        Args:
            inputs (torch.Tensor...): inputs to the reducer's :py:meth:`forward` call.

        Returns:
            torch.Tensor: transformed inputs.
        """
        raise NotImplementedError(
            f"'MappingReducer.map_fn()' is abstract, {type(self).__name__} "
            "must implement the 'map_fn' method"
        )

    @abstractmethod
    def filter_fn(self, inputs: torch.Tensor) -> bool:
        r"""Conditional test if transformed input should be integrated.

        Args:
            inputs (torch.Tensor): inputs to the reducer's :py:meth:`forward` call after :py:meth:`map_fn`.

        Returns:
            bool: if the given input should be integrated into the reducer's state.
        """
        raise NotImplementedError(
            f"'MappingReducer.filter_fn()' is abstract, {type(self).__name__} "
            "must implement the 'filter_fn' method"
        )


class FoldReducer(Reducer):
    r"""Applies a function between the previously stored data and the new observation, merging the two.

    Integration of new input follows the steps below.

    1. ``mapfn`` is applied to the inputs, returning a single tensor.
    2. ``filterfn`` is applied to the output of ``mapfn``, returning a boolean value.
    3. ``foldfn`` is applied to the current state and the output of ``mapfn``, if and only if the output of ``filterfn`` is true
    4. the current state is replaced with the output of ``foldfn``, if and only if the output of ``filterfn`` is true

    Args:
        foldfn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): recurrence relation between
            the new input (left) and current state (right), returning the new state.
        mapfn (OneToOne[torch.Tensor] | ManyToOne[torch.Tensor] | None, optional): transformation
            to apply to inputs, no transformation if None. Defaults to None.
        filterfn (Callable[[torch.Tensor], bool] | None, optional): conditional test if transformed
            input should be integrated, always will if None. Defaults to None.
        initfn (OneToOne[torch.Tensor] | None, optional): function to return the initial state, when given
            the input tensor transformed by mapfn. Defaults to None.

    Note:
        The lefthand argument of ``foldfn`` is the new input, and the righthand argument is the current state.

    Note:
        The default ``mapfn`` implemented assumes only a single input, and will therefore fail
        if multiple input values are passed into :py:meth:`forward`.

    Note:
        Output of ``foldfn`` is not passed through ``fitlerfn``. Implementing a post-filter should be
        performed in ``foldfn`` itself.
    """

    def __init__(
        self,
        foldfn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        *,
        mapfn: OneToOne[torch.Tensor] | ManyToOne[torch.Tensor] | None = None,
        filterfn: Callable[[torch.Tensor], bool] | None = None,
        initfn: OneToOne[torch.Tensor] | None = None,
    ):
        # call superclass constructor
        Reducer.__init__(self)

        # set non-persistant functions
        self._foldfn = foldfn
        self._mapfn = mapfn if mapfn else lambda x: x
        self._filterfn = filterfn if filterfn else lambda x: True
        self._initfn = (
            initfn if initfn else lambda x: inferno.zeros(self._data, shape=x.shape)
        )

        # register persistant state
        self.register_buffer("_data", torch.empty(0))

    def datato(self, *args, **kwargs):
        """Applies PyTorch ``to()`` function to storage tensor.

        This directly calles :py:func:`torch.Tensor.to`` on the underlying tensor and
        replacing the pre-call tensor with this. It can be used to set non-float/non-complex
        datatypes.
        """
        self._data = self._data.to(*args, **kwargs)

    def clear(self) -> None:
        r"""Reinitializes the reducer's state."""
        newdata = inferno.zeros(self._data, shape=(0,))
        del self._data
        self.register_buffer("_data", newdata)

    def peek(self) -> torch.Tensor | None:
        r"""Returns the cumulative state from observations since the reducer was last cleared.

        Returns:
            torch.Tensor | None: accumulation of observations if any have been made, otherwise None.
        """
        if self._data.numel() != 0:
            return self._data

    def pop(self) -> torch.Tensor | None:
        r"""Returns the cumulative state from observations and resets the reducer.

        Returns:
            torch.Tensor | None: cumulative state of recorded observations if any have been made,
                otherwise None.
        """
        res = self.peek()
        self.clear()
        return res

    def push(self, *inputs: torch.Tensor) -> None:
        r"""Incorporates inputs into the cumulative state.

        Args:
            inputs (torch.Tensor...): data to incorporate into the current state.
        """
        # apply map function to the inputs
        inputs = self._mapfn(*inputs)

        # only proceed if the observation passes the filter
        if self._filterfn(inputs):
            # allocate data tensor if it is not yet allocated
            if self._data.numel() == 0:
                self._data = self._initfn(inputs)
            # compute the revised state
            self._data = self._foldfn(inputs, self._data)


class FoldingReducer(FoldReducer, ABC):
    r"""Abstract base class for reducers like FoldReducer.

    Uses bound methods as functions passed to its parent, :py:class:`FoldReducer`. Accepts arguments which, if not None,
    will be bound to the named class methods, for the instanciated instance only.

    Args:
        foldmeth (Callable[[FoldingReducer, torch.Tensor, torch.Tensor], torch.Tensor]): if not None, overwrites the
            class-defined recurrence relation between the new input (left) and current state (right). Defaults to None.
        mapmeth (OneToOneMethod[FoldingReducer, torch.Tensor] | ManyToOneMethod[FoldingReducer, torch.Tensor] | None, optional): if not None,
            overwrites the class-defined transformation to apply to inputs. Defaults to None.
        filtermeth (Callable[[FoldingReducer, torch.Tensor], bool] | None, optional): if not None, overwrites the class-defined
            conditional test used to determine if transformed input should be integrated. Defaults to None.
        initmeth (OneToOneMethod[FoldingReducer, torch.Tensor] | None, optional): if not None, overwrites the class-defined method
            to return the initial state, when giventhe input tensor transformed by mapmeth. Defaults to None.

    Note:
        For details on the underlying logic, see :py:class:`FoldReducer`.
    """

    def __init__(
        self,
        *,
        foldmeth: Callable[[FoldingReducer, torch.Tensor, torch.Tensor], torch.Tensor]
        | None = None,
        mapmeth: OneToOneMethod[FoldingReducer, torch.Tensor]
        | ManyToOneMethod[FoldingReducer, torch.Tensor]
        | None = None,
        filtermeth: Callable[[FoldingReducer, torch.Tensor], bool] | None = None,
        initmeth: OneToOneMethod[FoldingReducer, torch.Tensor] | None = None,
    ):
        # dynamically replace instance method definitions
        if foldmeth:
            self.fold_fn = types.MethodType(foldmeth, self)
        if mapmeth:
            self.map_fn = types.MethodType(mapmeth, self)
        if filtermeth:
            self.filter_fn = types.MethodType(filtermeth, self)
        if initmeth:
            self.init_fn = types.MethodType(initmeth, self)

        # call superclass constructor
        FoldReducer.__init__(
            self,
            foldfn=self.fold_fn,
            mapfn=self.map_fn,
            filterfn=self.filter_fn,
            initfn=self.init_fn,
        )

    @abstractmethod
    def fold_fn(self, obs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        r"""Recurrence relation between the new inputs and the current state.

        Args:
            obs (torch.Tensor): new observation.
            state (torch.Tensor): current state.

        Returns:
            torch.Tensor: revised state, with the new observation integrated.
        """
        raise NotImplementedError(
            f"'FoldingReducer.fold_fn()' is abstract, {type(self).__name__} "
            "must implement the 'fold_fn' method"
        )

    @abstractmethod
    def map_fn(self, *inputs: torch.Tensor) -> torch.Tensor:
        r"""Transformation to apply to inputs.

        Args:
            inputs (torch.Tensor...): inputs to the reducer's :py:meth:`forward` call.

        Returns:
            torch.Tensor: transformed inputs.
        """
        raise NotImplementedError(
            f"'FoldingReducer.map_fn()' is abstract, {type(self).__name__} "
            "must implement the 'map_fn' method"
        )

    @abstractmethod
    def filter_fn(self, inputs: torch.Tensor) -> bool:
        r"""Conditional test if transformed input should be integrated.

        Args:
            inputs (torch.Tensor): inputs to the reducer's :py:meth:`forward` call after :py:meth:`map_fn`.

        Returns:
            bool: if the given input should be integrated into the reducer's state.
        """
        raise NotImplementedError(
            f"'FoldingReducer.filter_fn()' is abstract, {type(self).__name__} "
            "must implement the 'filter_fn' method"
        )

    @abstractmethod
    def init_fn(self, inputs: torch.Tensor) -> torch.Tensor:
        r"""Function to return the initial state, when given a like tensor.

        Args:
            inputs (torch.Tensor): inputs to the reducer's :py:meth:`forward` call after :py:meth:`map_fn`.

        Returns:
            torch.Tensor: initial state based on the shape, dtype, and device of the input.
        """
        raise NotImplementedError(
            f"'FoldingReducer.init_fn()' is abstract, {type(self).__name__} "
            "must implement the 'init_fn' method"
        )


class CompositeReducer(Reducer):
    def __init__(self, window: int, inner: FoldReducer):
        # call superclass constructor
        Reducer.__init__(self)

        # register inner reducers
        self.register_module("_fold", inner)
        self.register_module("_map", MapReducer(window))

    def datato(self, *args, **kwargs):
        """Applies PyTorch ``to()`` function to storage tensor.

        This directly calles :py:func:`torch.Tensor.to`` on the underlying tensor and
        replacing the pre-call tensor with this. It can be used to set non-float/non-complex
        datatypes.
        """
        self._fold.datato(self, *args, **kwargs)
        self._map.datato(self, *args, **kwargs)

    def clear(self, keepshape=True) -> None:
        r"""Reinitializes the reducer's state.

        Args:
            keepshape (bool, optional): if the data shape should be preserved. Defaults to True.

        Note:
            Setting ``keepshape`` to True is faster as storage doesn't need to be reallocated,
            but this assumes the shape of the monitored data will stay the same. This additionally
            preserves datatype and device.
        """
        self._fold.clear()
        self._map.clear(keepshape=keepshape)

    def peek(self, dim: int = -1) -> torch.Tensor | None:
        r"""Returns the tensor of observed states since the reducer was last cleared.

        Args:
            dim (int, optional): dimension along which observations are stacked. Defaults to -1.

        Returns:
            torch.Tensor | None: recorded observations if any have been made, otherwise None.

        Note:
            The number of samples along the time dimension will not exceed the number of observations recorded.
            The oldest sample will be at the start of the tensor, and the latest will be at the end. If padding
            is required, consider applying :py:func:`torch.nn.functional.pad` to the output.
        """
        _ = self._fold.peek()
        return self._map.peek(dim=dim)

    def pop(self, dim: int = -1, keepshape=True) -> torch.Tensor | None:
        r"""Returns the current state from observations and resets the reducer.

        Args:
            dim (int, optional): dimension along which observations are stacked. Defaults to -1.
            keepshape (bool, optional): if the data shape should be preserved. Defaults to True.

        Returns:
            torch.Tensor | None: recorded observations if any have been made, otherwise None.

        Note:
            Preservation of type/device from :py:meth:`clear` applies, as does the shape of
            the output from :py:meth:`peek`.
        """
        _ = self._fold.pop()
        return self._map.pop(dim=dim, keepshape=keepshape)

    def push(self, *inputs: torch.Tensor) -> None:
        r"""Incorporates inputs into the cumulative state.

        Args:
            inputs (torch.Tensor...): data to incorporate into the current state.
        """
        self._fold(*inputs)
        self._map(self._fold.peek())
