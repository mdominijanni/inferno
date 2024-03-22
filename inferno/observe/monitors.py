from __future__ import annotations
from . import Reducer
from .. import Module, Hook
from .._internal import rgetattr
import torch
from typing import Any, Callable, Protocol
import weakref


class MonitorConstructor(Protocol):
    r"""Common constructor for monitors, used in updaters.

    Args:
        attr (str): attribute or nested attribute to target.
        module (Module): module to use as register base for monitoring.

    Returns:
        Monitor: newly constructed monitor.

    Important:
        The monitor returned must be registered.

    Note:
        If it makes sense to, the module to which the monitor is registered should be
        the same as the module given. Where not sensible, it should be registered
        in a submodule along the attribute path ``attr``.
    """

    def __call__(
        self,
        attr: str,
        module: Module,
    ) -> Monitor:
        r"""Callback protocol function"""
        ...


class Monitor(Module, Hook):
    r"""Base class for recording input, output, or state of a Module.

    Args:
        reducer (Reducer): underlying means for reducing samples over time
            and storing them.
        module (Module | None, optional): module to register as the target for monitoring,
            can be modified after construction. Defaults to None.
        prehook (Callable | None, optional): function to call before hooked's
            :py:meth:`~torch.nn.Module.forward`. Defaults to None.
        posthook (Callable | None, optional): function to call after hooked's
            :py:meth:`~torch.nn.Module.forward`. Defaults to None.
        prehook_kwargs (dict[str, Any] | None, optional): keyword arguments passed to
            :py:meth:`~torch.nn.Module.register_forward_pre_hook`. Defaults to None.
        posthook_kwargs (dict[str, Any] | None, optional): keyword arguments passed to
            :py:meth:`~torch.nn.Module.register_forward_hook`. Defaults to None.
        train_update (bool, optional): if this monitor should be called when the module
            being monitored is in train mode. Defaults to True.
        eval_update (bool, optional): if this monitor should be called when the module
            being monitored is in eval mode. Defaults to True.

    Important:
        While ``Monitor`` can be used directly, it must be subclassed to be used in
        cases where monitors are constructed automatically.

    See Also:
        See :py:meth:`~torch.nn.Module.register_forward_pre_hook` and
        :py:meth:`~torch.nn.Module.register_forward_hook` for keyword arguments that
        can be passed with ``prehook_kwargs`` and ``posthook_kwargs`` respectively.
    """

    def __init__(
        self,
        reducer: Reducer,
        module: Module | None = None,
        prehook: Callable | None = None,
        posthook: Callable | None = None,
        prehook_kwargs: dict[str, Any] | None = None,
        posthook_kwargs: dict[str, Any] | None = None,
        train_update: bool = True,
        eval_update: bool = True,
    ):
        # construct module superclasses
        Module.__init__(self)
        Hook.__init__(
            self,
            prehook=prehook,
            posthook=posthook,
            prehook_kwargs=prehook_kwargs,
            posthook_kwargs=posthook_kwargs,
            train_update=train_update,
            eval_update=eval_update,
        )

        # placeholder for weak reference (created on register)
        self._observed = None

        # register if as module is provided
        if module is not None:
            self.register(module)

        # register submodule
        self.reducer_ = reducer

    @classmethod
    def partialconstructor(cls, *args, **kwargs) -> MonitorConstructor:
        r"""Returns a function with a common signature for monitor construction.

        Raises:
            NotImplementedError: ``partialconstructor`` must be implemented
                by the subclass.

        Returns:
            MonitorConstructor: partial constructor for monitor.
        """
        raise NotImplementedError(
            f"{cls.__name__}(Monitor) must implement "
            "the method `partialconstructor`."
        )

    def clear(self, **kwargs) -> None:
        r"""Reinitializes the reducer's state."""
        return self.reducer_.clear(**kwargs)

    def view(self, *args, **kwargs) -> torch.Tensor | None:
        r"""Returns the reducer's state at a given time."""
        return self.reducer_.view(*args, **kwargs)

    def dump(self, *args, **kwargs) -> torch.Tensor | None:
        r"""Returns the reducer's state over all observations."""
        return self.reducer_.dump(*args, **kwargs)

    def peek(self, *args, **kwargs) -> torch.Tensor | None:
        r"""Returns the reducer's current state."""
        return self.reducer_.peek(*args, **kwargs)

    @property
    def reducer(self) -> Reducer:
        r"""Underlying reducer used by the monitor.

        Returns:
            Reducer: underlying reducer.
        """
        return self.reducer_

    def register(self, module: Module | None = None) -> None:
        r"""Registers the monitor as a forward hook/prehook.

        Args:
            module (Module | None, optional): module with which to register, last
                registered if None. Defaults to None.

        Raises:
            RuntimeError: weak reference to the last referenced module is no longer
                valid or did not exist.
        """
        # module from function arguments
        if module:
            try:
                Hook.register(self, module)
            except RuntimeError:
                raise RuntimeError(
                    f"{type(self).__name__}(Monitor) is already registered to a module "
                    "so register() was ignored"
                )
            else:
                self._observed = weakref.ref(module)

        # module from weakref and is unregistered
        elif not self.registered:
            # try to get the referenced module
            if self._observed and self._observed():
                module = self._observed()
                Hook.register(self, module)
            else:
                raise RuntimeError(
                    "weak reference to monitored module does not exist, "
                    "cannot infer argument 'module'"
                )


class InputMonitor(Monitor):
    r"""Records the inputs passed to a Module.

    Args:
        reducer (Reducer): underlying means for reducing samples over time
            and storing them.
        module (Module, optional): module to register as the target for monitoring,
            can be modified after construction. Defaults to None.
        train_update (bool, optional): if this monitor should be called when the module
            being monitored is in train mode. Defaults to True.
        eval_update (bool, optional): if this monitor should be called when the module
            being monitored is in eval mode. Defaults to True.
        prepend (bool, optional): if this monitor should be called before other
            registered forward prehooks. Defaults to False.
        filter_ (Callable[[tuple[Any, ...]], bool] | None, optional): test if the input
            should be passed to the reducer, ignores empty when None. Defaults to None.
        map_ (Callable[[tuple[Any, ...]], tuple[torch.Tensor, ...]] | None, optional):
            modifies the input before being passed to the reducer, identity when None.
            Defaults to None.

    Note:
        The inputs, which are received as a tuple, will be sent to ``filter_``. If this
        evaluates to ``True``, they are forwarded to ``map_``, then unpacked into
        ``reducer``.
    """

    def __init__(
        self,
        reducer: Reducer,
        module: Module = None,
        train_update: bool = True,
        eval_update: bool = True,
        prepend: bool = False,
        filter_: Callable[[tuple[Any, ...]], bool] | None = None,
        map_: Callable[[tuple[Any, ...]], tuple[torch.Tensor, ...]] | None = None,
    ):
        # set filter and map functions
        filter_ = filter_ if filter_ else lambda x: bool(x)
        map_ = map_ if map_ else lambda x: x

        # determine arguments for superclass constructor
        def prehook(module, args, *_):
            if filter_(args):
                reducer(*map_(args))

        # construct superclass
        Monitor.__init__(
            self,
            reducer=reducer,
            module=module,
            prehook=prehook,
            prehook_kwargs={"prepend": prepend},
            train_update=train_update,
            eval_update=eval_update,
        )

    @classmethod
    def partialconstructor(
        cls,
        reducer: Reducer,
        train_update: bool = True,
        eval_update: bool = True,
        prepend: bool = False,
        filter_: Callable[[tuple[Any, ...]], bool] | None = None,
        map_: Callable[[tuple[Any, ...]], torch.Tensor] | None = None,
    ) -> MonitorConstructor:
        r"""Returns a function with a common signature for monitor construction.

        Args:
            reducer (Reducer): underlying means for reducing samples over time
                and storing them.
            train_update (bool, optional): if this monitor should be called when the
                module being monitored is in train mode. Defaults to True.
            eval_update (bool, optional): if this monitor should be called when the
                module being monitored is in eval mode. Defaults to True.
            prepend (bool, optional): if this monitor should be called before other
                registered forward prehooks or posthooks. Defaults to False.
            filter_ (Callable[[tuple[Any, ...]], bool] | None, optional): test if the input
                should be passed to the reducer, ignores empty when None. Defaults to None.
            map_ (Callable[[tuple[Any, ...]], torch.Tensor] | None, optional):
                modifies the input before being passed to the reducer, 0th input if None.
                Defaults to None.

        Returns:
            MonitorConstructor: partial constructor for monitor.
        """

        def constructor(attr: str, module: Module):
            return cls(
                reducer=reducer,
                module=rgetattr(module, attr),
                train_update=train_update,
                eval_update=eval_update,
                prepend=prepend,
                filter_=filter_,
                map_=map_,
            )

        return constructor


class OutputMonitor(Monitor):
    r"""Records the outputs returned from a Module.

    Args:
        reducer (Reducer): underlying means for reducing samples over time
            and storing them.
        module (Module, optional): module to register as the target for monitoring,
            can be modified after construction. Defaults to None.
        train_update (bool, optional): if this monitor should be called when the module
            being monitored is in train mode. Defaults to True.
        eval_update (bool, optional): if this monitor should be called when the module
            being monitored is in eval mode. Defaults to True.
        prepend (bool, optional): if this monitor should be called before other
            registered forward posthooks. Defaults to False.
        filter_ (Callable[[Any], bool] | None, optional): test if the output should be
            passed to the reducer, ignores None values when None. Defaults to None.
        map_ (Callable[[Any], tuple[torch.Tensor, ...]] | None, optional): modifies the
            output before being passed to the reduer, wraps with a tuple if not already
            a tuple if None. Defaults to None.

    Note:
        The output depends on the :py:meth:`~torch.nn.Module.forward` of the
        :py:class:`~torch.nn.Module` being called. If it a single tensor, it will
        work as expected. Otherwise a ``map_`` must be specified which takes the
        output and returns a single tensor.
    """

    def __init__(
        self,
        reducer: Reducer,
        module: Module = None,
        train_update: bool = True,
        eval_update: bool = True,
        prepend: bool = False,
        filter_: Callable[[Any], bool] | None = None,
        map_: Callable[[Any], tuple[torch.Tensor, ...]] | None = None,
    ):
        # set filter and map functions
        filter_ = filter_ if filter_ else lambda x: x is not None
        map_ = map_ if map_ else lambda x: x if isinstance(x, tuple) else (x,)

        # determine arguments for superclass constructor
        def posthook(module, args, output, *_):
            if filter_(output):
                reducer(*map_(output))

        # construct superclass
        Monitor.__init__(
            self,
            reducer=reducer,
            module=module,
            posthook=posthook,
            posthook_kwargs={"prepend": prepend},
            train_update=train_update,
            eval_update=eval_update,
        )

    @classmethod
    def partialconstructor(
        cls,
        reducer: Reducer,
        train_update: bool = True,
        eval_update: bool = True,
        prepend: bool = False,
        filter_: Callable[[Any], bool] | None = None,
        map_: Callable[[Any], tuple[torch.Tensor, ...]] | None = None,
    ) -> MonitorConstructor:
        r"""Returns a function with a common signature for monitor construction.

        Args:
            reducer (Reducer): underlying means for reducing samples over time
                and storing them.
            train_update (bool, optional): if this monitor should be called when the
                module being monitored is in train mode. Defaults to True.
            eval_update (bool, optional): if this monitor should be called when the
                module being monitored is in eval mode. Defaults to True.
            prepend (bool, optional): if this monitor should be called before other
                registered forward prehooks or posthooks. Defaults to False.
            filter_ (Callable[[Any], bool] | None, optional): test if the output should be
                passed to the reducer, ignores None values when None. Defaults to None.
            map_ (Callable[[Any], tuple[torch.Tensor, ...]] | None, optional): modifies
                the output before being passed to the reduer, wraps with a tuple if not
                already a tuple if None. Defaults to None.

        Returns:
            MonitorConstructor: partial constructor for monitor.
        """

        def constructor(attr: str, module: Module):
            return cls(
                reducer=reducer,
                module=rgetattr(module, attr),
                train_update=train_update,
                eval_update=eval_update,
                prepend=prepend,
                filter_=filter_,
                map_=map_,
            )

        return constructor


class StateMonitor(Monitor):
    r"""Records the state of an attribute in a Module.

    Args:
        reducer (Reducer): underlying means for reducing samples over time
            and storing them.
        attr (str): attribute or nested attribute to target.
        module (Module, optional): module to register as the target for monitoring,
            can be modified after construction. Defaults to None.
        as_prehook (bool, optional): if this monitor should be called before the forward
            call of the module being monitored. Defaults to False.
        train_update (bool, optional): if this monitor should be called when the module
            being monitored is in train mode. Defaults to True.
        eval_update (bool, optional): if this monitor should be called when the module
            being monitored is in eval mode. Defaults to True.
        prepend (bool, optional): if this monitor should be called before other
            registered forward prehooks or posthooks. Defaults to False.
        filter_ (Callable[[Any], bool] | None, optional): test if the input should be
            passed to the reducer, ignores None values when None. Defaults to None.
        map_ (Callable[[Any], tuple[torch.Tensor, ...]] | None, optional): modifies the
            input before being passed to the reduer, wraps with a tuple if not already
            a tuple if None. Defaults to None.

    Note:
        The nested attribute should be specified with dot notation. For instance,
        if the observed module has an attribute ``a`` which in turn has an
        attribute ``b`` that should be monitored, then ``attr`` should be
        `'a.b'``. Even with nested attributes, the monitor's hook will be tied to
        the module with which it is registered.
    """

    def __init__(
        self,
        reducer: Reducer,
        attr: str,
        module: Module = None,
        as_prehook: bool = False,
        train_update: bool = True,
        eval_update: bool = True,
        prepend: bool = False,
        filter_: Callable[[Any], bool] | None = None,
        map_: Callable[[Any], tuple[torch.Tensor, ...]] | None = None,
    ):
        # set filter and map functions
        filter_ = filter_ if filter_ else lambda x: x is not None
        map_ = map_ if map_ else lambda x: x if isinstance(x, tuple) else (x,)

        # determine arguments for superclass constructor
        if as_prehook:

            def prehook(module, *_):
                res = rgetattr(module, attr)
                if filter_(res):
                    reducer(*map_(res))

            prehook_kwargs = {"prepend": prepend}
            posthook, posthook_kwargs = None, None

        else:

            def posthook(module, *_):
                res = rgetattr(module, attr)
                if filter_(res):
                    reducer(*map_(res))

            posthook_kwargs = {"prepend": prepend}
            prehook, prehook_kwargs = None, None

        # construct superclass
        Monitor.__init__(
            self,
            reducer=reducer,
            module=module,
            prehook=prehook,
            posthook=posthook,
            prehook_kwargs=prehook_kwargs,
            posthook_kwargs=posthook_kwargs,
            train_update=train_update,
            eval_update=eval_update,
        )

    @classmethod
    def partialconstructor(
        cls,
        reducer: Reducer,
        as_prehook: bool = False,
        train_update: bool = True,
        eval_update: bool = True,
        prepend: bool = False,
        filter_: Callable[[Any], bool] | None = None,
        map_: Callable[[Any], tuple[torch.Tensor, ...]] | None = None,
    ) -> MonitorConstructor:
        r"""Returns a function with a common signature for monitor construction.

        Args:
            reducer (Reducer): underlying means for reducing samples over time
                and storing them.
            as_prehook (bool, optional): if this monitor should be called before the
                forward call of the module being monitored. Defaults to False.
            train_update (bool, optional): if this monitor should be called when the
                module being monitored is in train mode. Defaults to True.
            eval_update (bool, optional): if this monitor should be called when the
                module being monitored is in eval mode. Defaults to True.
            prepend (bool, optional): if this monitor should be called before other
                registered forward prehooks or posthooks. Defaults to False.
            filter_ (Callable[[Any], bool] | None, optional): test if the input should be
                passed to the reducer, ignores None values when None. Defaults to None.
            map_ (Callable[[Any], tuple[torch.Tensor, ...]] | None, optional): modifies
                the input before being passed to the reduer, wraps with a tuple if not
                already a tuple if None. Defaults to None.

        Returns:
            MonitorConstructor: partial constructor for monitor.
        """

        def constructor(attr: str, module: Module):
            return cls(
                reducer=reducer,
                attr=attr,
                module=module,
                as_prehook=as_prehook,
                train_update=train_update,
                eval_update=eval_update,
                prepend=prepend,
                filter_=filter_,
                map_=map_,
            )

        constructor.monitor = cls
        constructor.reducer = type(reducer)

        return constructor


class DifferenceMonitor(Monitor):
    """Records the difference of an attribute in a Module before and after its forward call.

    Args:
        reducer (Reducer): underlying means for reducing samples over time
            and storing them.
        attr (str): attribute or nested attribute to target.
        module (Module, optional): module to register as the target for monitoring,
            can be modified after construction. Defaults to None.
        train_update (bool, optional): if this monitor should be called when the module
            being monitored is in train mode. Defaults to True.
        eval_update (bool, optional): if this monitor should be called when the module
            being monitored is in eval mode. Defaults to True.
        prepend (bool, optional): if this monitor should be called before other
            registered forward prehooks or posthooks. Defaults to False.
        filter_ (Callable[[Any, Any], bool] | None, optional): test if the input should
            be passed to the reducer, ignores None values when None. Defaults to None.
        map_ (Callable[[Any, Any], tuple[torch.Tensor, ...]] | None, optional): modifies
            the input before being passed to the reducer, post-forward value minus
            pre-forward value wrapped in a tuple if ``None``. Defaults to ``None``.
        op_ (Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None, optional): operation
            to calculate the difference between post-forward and pre-forward, only used
            when ``map_`` is ``None``, subtraction when ``None``. Defaults to ``None``.

    Note:
        The nested attribute should be specified with dot notation. For instance,
        if the observed module has an attribute ``a`` which in turn has an
        attribute ``b`` that should be monitored, then ``attr`` should be
        `'a.b'``. Even with nested attributes, the monitor's hook will be tied to
        the module with which it is registered.

    Note:
        The left-hand argument of ``filter_``, ``map_``, and ``op_`` is the attribute
        after the :py:meth:`~torch.nn.Module.forward` call of ``module`` is run, and
        the right-hand argument is before it is run.

        By default, ``filter_`` will only reject an input if both the pre and post
        states are ``None``. By default, ``map_`` will use ``op_`` to compare the
        pre-forward value from the post-forward value. If either is ``None`` (but not
        both), ``map_`` will assume the ``None`` value was composed of all-zeros.
    """

    def __init__(
        self,
        reducer: Reducer,
        attr: str,
        module: Module = None,
        train_update: bool = True,
        eval_update: bool = True,
        prepend: bool = False,
        filter_: Callable[[Any, Any], bool] | None = None,
        map_: Callable[[Any, Any], tuple[torch.Tensor, ...]] | None = None,
        op_: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ):
        # set filter and map functions
        def _default_filter(final, initial):
            return not (final is None and initial is None)

        def _default_map(final, initial, op=(op_ if op_ else lambda f, i: f - i)):
            return tuple(
                op(fv, iv)
                for fv, iv in zip(
                    final if isinstance(final, tuple) else (final,),
                    initial if isinstance(initial, tuple) else (initial,),
                )
            )

        filter_ = filter_ if filter_ else _default_filter
        map_ = map_ if map_ else _default_map

        # monitor state
        self.data = None

        # hook functions
        def prehook(module, *args):
            self.data = rgetattr(module, attr)

        def posthook(module, *args):
            res = rgetattr(module, attr)
            if filter_(res, self.data):
                reducer(*map_(res, self.data))
            self.data = None

        # construct superclass
        Monitor.__init__(
            self,
            reducer=reducer,
            module=module,
            prehook=prehook,
            posthook=posthook,
            prehook_kwargs={"prepend": prepend},
            posthook_kwargs={"prepend": prepend},
            train_update=train_update,
            eval_update=eval_update,
        )

    @classmethod
    def partialconstructor(
        cls,
        reducer: Reducer,
        train_update: bool = True,
        eval_update: bool = True,
        prepend: bool = False,
        filter_: Callable[[Any, Any], bool] | None = None,
        map_: Callable[[Any, Any], tuple[torch.Tensor, ...]] | None = None,
        op_: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> MonitorConstructor:
        r"""Returns a function with a common signature for monitor construction.

        Args:
            reducer (Reducer): underlying means for reducing samples over time
                and storing them.
            train_update (bool, optional): if this monitor should be called when the
                module being monitored is in train mode. Defaults to True.
            eval_update (bool, optional): if this monitor should be called when the
                module being monitored is in eval mode. Defaults to True.
            prepend (bool, optional): if this monitor should be called before other
                registered forward prehooks or posthooks. Defaults to False.
            filter_ (Callable[[Any, Any], bool] | None, optional): test if the input should
                be passed to the reducer, ignores None values when None. Defaults to None.
            map_ (Callable[[Any, Any], tuple[torch.Tensor, ...]] | None, optional): modifies
                the input before being passed to the reducer, post-forward value minus
                pre-forward value wrapped in a tuple if ``None``. Defaults to ``None``.
            op_ (Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None, optional): operation
                to calculate the difference between post-forward and pre-forward, only used
                when ``map_`` is ``None``, subtraction when ``None``. Defaults to ``None``.

        Returns:
            MonitorConstructor: partial constructor for monitor.
        """

        def constructor(attr: str, module: Module):
            return cls(
                reducer=reducer,
                attr=attr,
                module=module,
                train_update=train_update,
                eval_update=eval_update,
                prepend=prepend,
                filter_=filter_,
                map_=map_,
            )

        constructor.monitor = cls
        constructor.reducer = type(reducer)

        return constructor

    def clear(self, **kwargs) -> None:
        r"""Clears monitor state and reinitializes the reducer's state."""
        self.data = None
        return self.reducer_.clear(**kwargs)
