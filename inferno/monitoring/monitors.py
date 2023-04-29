from abc import abstractmethod
from typing import Any
import warnings

import torch
import torch.nn as nn

from inferno._internal import get_nested_attr
from inferno.common.hookables import PreHookable, PostHookable
from inferno.monitoring.reducers.abstract import AbstractReducer
from inferno.monitoring.reducers.passthrough import SinglePassthroughReducer


class AbstractMonitor(PostHookable):
    """Abstract class for passing data to a reducer from a module on call after `forward()` execution.

    Args:
        reducer (AbstractReducer, optional): reducer used to process and store monitored parameter. Defaults to :py:class:`SinglePassthroughReducer`.
        train_update (bool, optional): if monitoring should occur during training. Defaults to `True`.
        eval_update (bool, optional): if monitoring should occur during evaluation. Defaults to `True`.
        module (nn.Module | None, optional): module being monitored. Defaults to `None`.
    """
    def __init__(
        self,
        reducer: AbstractReducer = SinglePassthroughReducer(),
        train_update: bool = True,
        eval_update: bool = True,
        module: nn.Module | None = None
    ):
        # call superclass constructor
        PostHookable.__init__(self, module)

        # warn if monitor will never occur
        if not train_update and not eval_update:
            warnings.warn(f"in {type(self).__name__}, 'train_update' and 'eval_update' set to False, {type(self).__name__} will never fire", category=RuntimeWarning)

        # set attributes
        self.reducer = reducer
        self.train_update = train_update
        self.eval_update = eval_update

    def clear(self, **kwargs) -> None:
        """Reinitializes the underlying reducer's state.
        """
        return self.reducer.clear(**kwargs)

    def peak(self, **kwargs) -> torch.Tensor | None:
        """Returns the current output state stored by the underlying reducer.

        Returns:
            torch.Tensor | None: output state stored by the underlying reducer.
        """
        return self.reducer.peak(**kwargs)

    def pop(self, **kwargs) -> torch.Tensor | None:
        """Returns the current output state stored by the underlying reducer and clears its state.

        Returns:
            torch.Tensor | None: output state stored by the underlying reducer.
        """
        return self.reducer.pop(**kwargs)

    @abstractmethod
    def forward(self, module: nn.Module, inputs: Any, outputs: Any) -> None:
        """Incorporates module forward input into the specified reducer.

        This is automatically called during :py:meth:`nn.Module.__call__` execution as a forward hook.

        Args:
            module (nn.Module): the module being monitored.
            inputs (Any): inputs passed to the module's `forward()` method.
            outputs (Any): outputs returned from the module's `forward()` method.

        Raises:
            NotImplementedError: :py:meth:`forward` is abstract and must be implemented by the subclass.
        """
        raise NotImplementedError


class AbstractPreMonitor(PreHookable):
    """Abstract class for passing data to a reducer from a module on call before `forward()` execution.

    Args:
        reducer (AbstractReducer, optional): reducer used to process and store monitored parameter. Defaults to :py:class:`SinglePassthroughReducer`.
        train_update (bool, optional): if monitoring should occur during training. Defaults to `True`.
        eval_update (bool, optional): if monitoring should occur during evaluation. Defaults to `True`.
        module (nn.Module | None, optional): module being monitored. Defaults to `None`.
    """
    def __init__(
        self,
        reducer: AbstractReducer = SinglePassthroughReducer(),
        train_update: bool = True,
        eval_update: bool = True,
        module: nn.Module | None = None
    ):
        # call superclass constructor
        PreHookable.__init__(self, module)

        # warn if monitor will never occur
        if not train_update and not eval_update:
            warnings.warn(f"in {type(self).__name__}, 'train_update' and 'eval_update' set to False, {type(self).__name__} will never fire", category=RuntimeWarning)

        # set attributes
        self.reducer = reducer
        self.train_update = train_update
        self.eval_update = eval_update

    def clear(self, **kwargs) -> None:
        """Reinitializes the underlying reducer's state.
        """
        return self.reducer.clear(**kwargs)

    def peak(self, **kwargs) -> torch.Tensor | None:
        """Returns the current output state stored by the underlying reducer.

        Returns:
            torch.Tensor | None: output state stored by the underlying reducer.
        """
        return self.reducer.peak(**kwargs)

    def pop(self, **kwargs) -> torch.Tensor | None:
        """Returns the current output state stored by the underlying reducer and clears its state.

        Returns:
            torch.Tensor | None: output state stored by the underlying reducer.
        """
        return self.reducer.pop(**kwargs)

    @abstractmethod
    def forward(self, module: nn.Module, inputs: Any) -> None:
        """Incorporates module forward input into the specified reducer.

        This is automatically called during :py:meth:`nn.Module.__call__` execution as a forward pre-hook.

        Args:
            module (nn.Module): the module being monitored.
            inputs (Any): inputs passed to the module's `forward()` method.
            outputs (Any): outputs returned from the module's `forward()` method.

        Raises:
            NotImplementedError: :py:meth:`forward` is abstract and must be implemented by the subclass.
        """
        raise NotImplementedError


class InputMonitor(AbstractMonitor):
    """Monitors the inputs given to a module on call.

    Args:
        reducer (AbstractReducer, optional): reducer used to process and store monitored parameter. Defaults to :py:class:`SinglePassthroughReducer`.
        index (int, optional): index of input to `forward()` to monitor. Defaults to `0`.
        train_update (bool, optional): if monitoring should occur during training. Defaults to `True`.
        eval_update (bool, optional): if monitoring should occur during evaluation. Defaults to `True`.
        module (nn.Module | None, optional): module being monitored. Defaults to `None`.
    """
    def __init__(
        self,
        reducer: AbstractReducer = SinglePassthroughReducer(),
        index: int = 0,
        train_update: bool = True,
        eval_update: bool = True,
        module: nn.Module | None = None
    ):
        # call superclass constructor
        AbstractMonitor.__init__(self, reducer, train_update, eval_update, module)

        # set index
        self.index = index

        # set forward function
        self.forward_fn = lambda inputs, index, reducer: reducer(inputs[index])

    def forward(self, module: nn.Module, inputs: Any, outputs: Any) -> None:
        """Incorporates module forward input into the specified reducer.

        This is automatically called during :py:meth:`nn.Module.__call__` execution as a forward hook.

        Args:
            module (nn.Module): the module being monitored.
            inputs (Any): inputs passed to the module's `forward()` method.
            outputs (Any): outputs returned from the module's `forward()` method.
        """
        if (self.train_update and module.training) or (self.eval_update and not module.training):
            self.forward_fn(inputs, self.index, self.reducer)


class OutputMonitor(AbstractMonitor):
    """Monitors the outputs from a module on call.

    Args:
        reducer (AbstractReducer, optional): reducer used to process and store monitored parameter. Defaults to :py:class:`SinglePassthroughReducer`.
        index (int, optional): index of output from `forward()` to monitor. Defaults to `None`.
        train_update (bool, optional): if monitoring should occur during training. Defaults to `True`.
        eval_update (bool, optional): if monitoring should occur during evaluation. Defaults to `True`.
        module (nn.Module | None, optional): module being monitored. Defaults to `None`.
    """
    def __init__(
        self,
        reducer: AbstractReducer = SinglePassthroughReducer(),
        index: int | None = None,
        train_update: bool = True,
        eval_update: bool = True,
        module: nn.Module | None = None
    ):
        # call superclass constructor
        AbstractMonitor.__init__(self, reducer, train_update, eval_update, module)

        # set index
        self.index = index

        # set forward function
        if self.index is not None:
            self.forward_fn = lambda outputs, index, reducer: reducer(outputs[index])
        else:
            self.forward_fn = lambda outputs, index, reducer: reducer(outputs)

    def forward(self, module: nn.Module, inputs: Any, outputs: Any) -> None:
        """Incorporates module forward output into the specified reducer.

        This is automatically called during :py:meth:`nn.Module.__call__` execution as a forward hook.

        Args:
            module (nn.Module): the module being monitored.
            inputs (Any): inputs passed to the module's `forward()` method.
            outputs (Any): outputs returned from the module's `forward()` method.
        """
        if (self.train_update and module.training) or (self.eval_update and not module.training):
            self.forward_fn(outputs, self.index, self.reducer)


class StateMonitor(AbstractMonitor):
    """Monitors an attribute of a module on call, after `forward()` execution.

    Args:
        attr (str): name of the module's attribute to be monitored, compatible with dot notation.
        reducer (AbstractReducer, optional): reducer used to process and store monitored parameter. Defaults to :py:class:`SinglePassthroughReducer`.
        train_update (bool, optional): if monitoring should occur during training. Defaults to `True`.
        eval_update (bool, optional): if monitoring should occur during evaluation. Defaults to `True`.
        module (nn.Module | None, optional): module being monitored. Defaults to `None`.
    """
    def __init__(
        self,
        attr: str,
        reducer: AbstractReducer = SinglePassthroughReducer(),
        train_update: bool = True,
        eval_update: bool = True,
        module: nn.Module | None = None
    ):
        # call superclass constructor
        AbstractMonitor.__init__(self, reducer, train_update, eval_update, module)

        # set attributes
        self.attr = attr

    def forward(self, module: nn.Module, inputs: Any, outputs: Any) -> None:
        """Incorporates a module's specified parameter attribute input into the specified reducer.

        This is automatically called during :py:meth:`nn.Module.__call__` execution as a forward hook.

        Args:
            module (nn.Module): the module being monitored.
            inputs (Any): inputs passed to the module's `forward()` method.
            outputs (Any): outputs returned from the module's `forward()` method.
        """
        if (self.train_update and module.training) or (self.eval_update and not module.training):
            self.reducer(get_nested_attr(module, self.attr))


class StatePreMonitor(AbstractPreMonitor):
    """Monitors an attribute of a module on call, before `forward()` execution.

    Args:
        attr (str): name of the module's attribute to be monitored, compatible with dot notation.
        reducer (AbstractReducer, optional): reducer used to process and store monitored parameter. Defaults to :py:class:`SinglePassthroughReducer`.
        train_update (bool, optional): if monitoring should occur during training. Defaults to `True`.
        eval_update (bool, optional): if monitoring should occur during evaluation. Defaults to `True`.
        module (nn.Module | None, optional): module being monitored. Defaults to `None`.
    """
    def __init__(
        self,
        attr: str,
        reducer: AbstractReducer = SinglePassthroughReducer(),
        train_update: bool = True,
        eval_update: bool = True,
        module: nn.Module | None = None
    ):
        # call superclass constructor
        AbstractPreMonitor.__init__(self, reducer, train_update, eval_update, module)

        # set attributes
        self.attr = attr

    def forward(self, module: nn.Module, inputs: Any) -> None:
        """Incorporates a module's specified parameter attribute input into the specified reducer.

        This is automatically called during :py:meth:`nn.Module.__call__` execution as a forward pre-hook.

        Args:
            module (nn.Module): the module being monitored.
            inputs (Any): inputs passed to the module's `forward()` method.
            outputs (Any): outputs returned from the module's `forward()` method.
        """
        if (self.train_update and module.training) or (self.eval_update and not module.training):
            monitored_attr = get_nested_attr(module, self.attr)
            if monitored_attr is not None:
                self.reducer(monitored_attr)
