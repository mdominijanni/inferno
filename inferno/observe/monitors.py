from __future__ import annotations
from . import Reducer
from .. import Module, ContextualHook
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


class Monitor(Module, ContextualHook):
    r"""Base class for recording input, output, or state of a Module.

    Args:
        reducer (Reducer): underlying means for reducing samples over time
            and storing them.
        module (Module | None, optional): module to register as the target for monitoring,
            can be modified after construction. Defaults to ``None``.
        prehook (str | None, optional): name of the prehook method, if any, to execute,
            no prehook when ``None``. Defaults to ``None``.
        posthook (str | None, optional): name of the posthook method, if any, to execute,
            no posthook when ``None``. Defaults to ``None``.
        prehook_kwargs (dict[str, Any] | None, optional): keyword arguments passed to
            :py:meth:`~torch.nn.Module.register_forward_pre_hook`. Defaults to ``None``.
        posthook_kwargs (dict[str, Any] | None, optional): keyword arguments passed to
            :py:meth:`~torch.nn.Module.register_forward_hook`. Defaults to ``None``.
        train_update (bool, optional): if this monitor should be called when the module
            being monitored is in train mode. Defaults to ``True``.
        eval_update (bool, optional): if this monitor should be called when the module
            being monitored is in eval mode. Defaults to ``True``.

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
        prehook: str | None = None,
        posthook: str | None = None,
        prehook_kwargs: dict[str, Any] | None = None,
        posthook_kwargs: dict[str, Any] | None = None,
        train_update: bool = True,
        eval_update: bool = True,
    ):
        # construct module superclasses
        Module.__init__(self)
        ContextualHook.__init__(
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

    @property
    def latest(self) -> torch.Tensor:
        r"""Return's the reducer's current state.

        If :py:meth:`peek` has multiple options, this should be considered as the
        default. Unless overridden, :py:meth:`peek` is called without arguments.

        Returns:
            torch.Tensor: reducer's current state.
        """
        return self.reducer_.latest

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
                registered if ``None``. Defaults to ``None``.

        Raises:
            RuntimeError: weak reference to the last referenced module is no longer
                valid or did not exist.
        """
        # module from function arguments
        if module:
            try:
                ContextualHook.register(self, module)
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
                ContextualHook.register(self, module)
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
            can be modified after construction. Defaults to ``None``.
        train_update (bool, optional): if this monitor should be called when the module
            being monitored is in train mode. Defaults to ``True``.
        eval_update (bool, optional): if this monitor should be called when the module
            being monitored is in eval mode. Defaults to ``True``.
        prepend (bool, optional): if this monitor should be called before other
            registered forward prehooks. Defaults to ``False``.
        filter_ (Callable[[tuple[Any, ...]], bool] | None, optional): test if the input
            should be passed to the reducer, ignores empty when ``None``. Defaults to ``None``.
        map_ (Callable[[tuple[Any, ...]], tuple[torch.Tensor, ...]] | None, optional):
            modifies the input before being passed to the reducer, identity when ``None``.
            Defaults to ``None``.

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
        self.filter_ = filter_ if filter_ else lambda x: bool(x)
        self.map_ = map_ if map_ else lambda x: x

        # construct superclass
        Monitor.__init__(
            self,
            reducer=reducer,
            module=module,
            prehook="_monitor_call",
            prehook_kwargs={"prepend": prepend},
            train_update=train_update,
            eval_update=eval_update,
        )

    def _monitor_call(self, module, args, *_):
        if self.filter_(args):
            self.reducer_(*self.map_(args))

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
                module being monitored is in train mode. Defaults to ``True``.
            eval_update (bool, optional): if this monitor should be called when the
                module being monitored is in eval mode. Defaults to ``True``.
            prepend (bool, optional): if this monitor should be called before other
                registered forward prehooks or posthooks. Defaults to ``False``.
            filter_ (Callable[[tuple[Any, ...]], bool] | None, optional): test if the input
                should be passed to the reducer, ignores empty when ``None``. Defaults to ``None``.
            map_ (Callable[[tuple[Any, ...]], torch.Tensor] | None, optional):
                modifies the input before being passed to the reducer,
                :math:`0^\text{th}` input if ``None``. Defaults to ``None``.

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
            can be modified after construction. Defaults to ``None``.
        train_update (bool, optional): if this monitor should be called when the module
            being monitored is in train mode. Defaults to ``True``.
        eval_update (bool, optional): if this monitor should be called when the module
            being monitored is in eval mode. Defaults to ``True``.
        prepend (bool, optional): if this monitor should be called before other
            registered forward posthooks. Defaults to ``False``.
        filter_ (Callable[[Any], bool] | None, optional): test if the output should be
            passed to the reducer, ignores None values when ``None``. Defaults to ``None``.
        map_ (Callable[[Any], tuple[torch.Tensor, ...]] | None, optional): modifies the
            output before being passed to the reducer, wraps with a tuple if not already
            a tuple if ``None``. Defaults to ``None``.

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
        self.filter_ = filter_ if filter_ else lambda x: x is not None
        self.map_ = map_ if map_ else lambda x: x if isinstance(x, tuple) else (x,)

        # construct superclass
        Monitor.__init__(
            self,
            reducer=reducer,
            module=module,
            posthook="_monitor_call",
            posthook_kwargs={"prepend": prepend},
            train_update=train_update,
            eval_update=eval_update,
        )

    def _monitor_call(self, module, args, output, *_):
        if self.filter_(output):
            self.reducer(*self.map_(output))

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
                module being monitored is in train mode. Defaults to ``True``.
            eval_update (bool, optional): if this monitor should be called when the
                module being monitored is in eval mode. Defaults to ``True``.
            prepend (bool, optional): if this monitor should be called before other
                registered forward prehooks or posthooks. Defaults to ``False``.
            filter_ (Callable[[Any], bool] | None, optional): test if the output should be
                passed to the reducer, ignores ``None`` values when ``None``.
                Defaults to ``None``.
            map_ (Callable[[Any], tuple[torch.Tensor, ...]] | None, optional): modifies
                the output before being passed to the reducer, wraps with a tuple if not
                already a tuple if ``None``. Defaults to ``None``.

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
            can be modified after construction. Defaults to ``None``.
        as_prehook (bool, optional): if this monitor should be called before the forward
            call of the module being monitored. Defaults to ``False``.
        train_update (bool, optional): if this monitor should be called when the module
            being monitored is in train mode. Defaults to ``True``.
        eval_update (bool, optional): if this monitor should be called when the module
            being monitored is in eval mode. Defaults to ``True``.
        prepend (bool, optional): if this monitor should be called before other
            registered forward prehooks or posthooks. Defaults to ``False``.
        filter_ (Callable[[Any], bool] | None, optional): test if the input should be
            passed to the reducer, ignores ``None`` values when ``None``.
            Defaults to ``None``.
        map_ (Callable[[Any], tuple[torch.Tensor, ...]] | None, optional): modifies the
            input before being passed to the reducer, wraps with a tuple if not already
            a tuple if ``None``. Defaults to ``None``.

    Note:
        The end target of this can be a method name, however ``map_`` will need to be
        specified in such a way as to call the method with desired arguments.

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
        self.filter_ = filter_ if filter_ else lambda x: x is not None
        self.map_ = map_ if map_ else lambda x: x if isinstance(x, tuple) else (x,)
        self.__observed_attr = attr

        # construct superclass
        Monitor.__init__(
            self,
            reducer=reducer,
            module=module,
            prehook="_monitor_call" if as_prehook else None,
            posthook="_monitor_call" if not as_prehook else None,
            prehook_kwargs={"prepend": prepend} if as_prehook else None,
            posthook_kwargs={"prepend": prepend} if not as_prehook else None,
            train_update=train_update,
            eval_update=eval_update,
        )

    def _monitor_call(self, module, args, *_):
        res = rgetattr(module, self.__observed_attr)
        if self.filter_(res):
            self.reducer_(*self.map_(res))

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
                forward call of the module being monitored. Defaults to ``False``.
            train_update (bool, optional): if this monitor should be called when the
                module being monitored is in train mode. Defaults to ``True``.
            eval_update (bool, optional): if this monitor should be called when the
                module being monitored is in eval mode. Defaults to ``True``.
            prepend (bool, optional): if this monitor should be called before other
                registered forward prehooks or posthooks. Defaults to ``False``.
            filter_ (Callable[[Any], bool] | None, optional): test if the input should be
                passed to the reducer, ignores ``None`` values when ``None``.
                Defaults to ``None``.
            map_ (Callable[[Any], tuple[torch.Tensor, ...]] | None, optional): modifies
                the input before being passed to the reducer, wraps with a tuple if not
                already a tuple if ``None``. Defaults to ``None``.

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

        return constructor


class DifferenceMonitor(Monitor):
    r"""Records the difference of an attribute in a Module before and after its forward call.

    Args:
        reducer (Reducer): underlying means for reducing samples over time
            and storing them.
        attr (str): attribute or nested attribute to target.
        module (Module, optional): module to register as the target for monitoring,
            can be modified after construction. Defaults to ``None``.
        train_update (bool, optional): if this monitor should be called when the module
            being monitored is in train mode. Defaults to ``True``.
        eval_update (bool, optional): if this monitor should be called when the module
            being monitored is in eval mode. Defaults to ``True``.
        prepend (bool, optional): if this monitor should be called before other
            registered forward prehooks or posthooks. Defaults to ``False``.
        filter_ (Callable[[Any, Any], bool] | None, optional): test if the input should
            be passed to the reducer, ignores ``None`` values when ``None``.
            Defaults to ``None``.
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

        self.filter_ = filter_ if filter_ else _default_filter
        self.map_ = map_ if map_ else _default_map
        self.__observed_attr = attr

        # monitor state
        self.__data = None

        # construct superclass
        Monitor.__init__(
            self,
            reducer=reducer,
            module=module,
            prehook="_monitor_pre_call",
            posthook="_monitor_post_call",
            prehook_kwargs={"prepend": prepend},
            posthook_kwargs={"prepend": prepend},
            train_update=train_update,
            eval_update=eval_update,
        )

    def _monitor_pre_call(self, module, args, *_):
        self.__data = rgetattr(module, self.__observed_attr)

    def _monitor_post_call(self, module, args, *_):
        res = rgetattr(module, self.__observed_attr)
        if self.filter_(res, self.__data):
            self.reducer_(*self.map_(res, self.__data))

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
                module being monitored is in train mode. Defaults to ``True``.
            eval_update (bool, optional): if this monitor should be called when the
                module being monitored is in eval mode. Defaults to ``True``.
            prepend (bool, optional): if this monitor should be called before other
                registered forward prehooks or posthooks. Defaults to ``False``.
            filter_ (Callable[[Any, Any], bool] | None, optional): test if the input should
                be passed to the reducer, ignores ``None`` values when ``None``.
                Defaults to ``None``.
            map_ (Callable[[Any, Any], tuple[torch.Tensor, ...]] | None, optional): modifies
                the input before being passed to the reducer, post-forward value minus
                pre-forward value wrapped in a tuple if ``None``. Defaults to ``None``.
            op_ (Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None, optional): operation
                to calculate the difference between post-forward and pre-forward,
                only used when ``map_`` is ``None``, subtraction when ``None``.
                Defaults to ``None``.

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

        return constructor

    def clear(self, **kwargs) -> None:
        r"""Clears monitor state and reinitializes the reducer's state."""
        self.__data = None
        return self.reducer_.clear(**kwargs)


class MultiStateMonitor(Monitor):
    r"""Records a combination of the state of multiple attributes in a Module.

    Attributes are passed to the reducer in-order.

    Args:
        reducer (Reducer): underlying means for reducing samples over time
            and storing them.
        attr (str): attribute or nested attribute to target.
        subattrs (tuple[str, ...]): attributes, relative to ``attr``, to target.
        module (Module, optional): module to register as the target for monitoring,
            can be modified after construction. Defaults to ``None``.
        as_prehook (bool, optional): if this monitor should be called before the forward
            call of the module being monitored. Defaults to ``False``.
        train_update (bool, optional): if this monitor should be called when the module
            being monitored is in train mode. Defaults to ``True``.
        eval_update (bool, optional): if this monitor should be called when the module
            being monitored is in eval mode. Defaults to ``True``.
        prepend (bool, optional): if this monitor should be called before other
            registered forward prehooks or posthooks. Defaults to ``False``.
        filter_ (Callable[[tuple[Any, ...]], bool] | None, optional): test if the input
            should be passed to the reducer, ignores ``None`` values when ``None``.
            Defaults to ``None``.
        map_ (Callable[[tuple[Any, ...]], tuple[torch.Tensor, ...]] | None, optional):
            modifies the input before being passed to the reducer, identity if ``None``.
            Defaults to ``None``.

    Note:
        The end targets of this can be a method name, however ``map_`` will need to be
        specified in such a way as to call the method with desired arguments.

    Note:
        The nested attributes should be specified with dot notation. For instance,
        if the observed module has an attribute ``a`` which in turn has an
        attribute ``b`` that should be monitored, then ``attr`` should be
        `'a.b'``. Even with nested attributes, the monitor's hook will be tied to
        the module with which it is registered.
    """

    def __init__(
        self,
        reducer: Reducer,
        attr: str,
        subattrs: tuple[str, ...],
        module: Module = None,
        as_prehook: bool = False,
        train_update: bool = True,
        eval_update: bool = True,
        prepend: bool = False,
        filter_: Callable[[tuple[Any, ...]], bool] | None = None,
        map_: Callable[[tuple[Any, ...]], tuple[torch.Tensor, ...]] | None = None,
    ):
        # set filter and map functions
        self.filter_ = filter_ if filter_ else lambda x: x is not None
        self.map_ = map_ if map_ else lambda x: x
        if attr:
            self.__observed_attrs = tuple(f"{attr}.{satr}" for satr in subattrs)
        else:
            self.__observed_attrs = tuple(satr for satr in subattrs)

        # construct superclass
        Monitor.__init__(
            self,
            reducer=reducer,
            module=module,
            prehook="_monitor_call" if as_prehook else None,
            posthook="_monitor_call" if not as_prehook else None,
            prehook_kwargs={"prepend": prepend} if as_prehook else None,
            posthook_kwargs={"prepend": prepend} if not as_prehook else None,
            train_update=train_update,
            eval_update=eval_update,
        )

    def _monitor_call(self, module, args, *_):
        res = tuple(rgetattr(module, oa) for oa in self.__observed_attrs)
        if self.filter_(res):
            self.reducer_(*self.map_(res))

    @classmethod
    def partialconstructor(
        cls,
        reducer: Reducer,
        subattrs: tuple[str, ...],
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
            subattrs (tuple[str, ...]): attributes, relative to ``attr``, to target.
            as_prehook (bool, optional): if this monitor should be called before the
                forward call of the module being monitored. Defaults to ``False``.
            train_update (bool, optional): if this monitor should be called when the
                module being monitored is in train mode. Defaults to ``True``.
            eval_update (bool, optional): if this monitor should be called when the
                module being monitored is in eval mode. Defaults to ``True``.
            prepend (bool, optional): if this monitor should be called before other
                registered forward prehooks or posthooks. Defaults to ``False``.
            filter_ (Callable[[Any], bool] | None, optional): test if the input should be
                passed to the reducer, ignores ``None`` values when ``None``.
                Defaults to ``None``.
            map_ (Callable[[Any], tuple[torch.Tensor, ...]] | None, optional): modifies
                the input before being passed to the reducer, wraps with a tuple if not
                already a tuple if ``None``. Defaults to ``None``.

        Returns:
            MonitorConstructor: partial constructor for monitor.
        """

        def constructor(attr: str, module: Module):
            return cls(
                reducer=reducer,
                attr=attr,
                subattrs=subattrs,
                module=module,
                as_prehook=as_prehook,
                train_update=train_update,
                eval_update=eval_update,
                prepend=prepend,
                filter_=filter_,
                map_=map_,
            )

        return constructor
