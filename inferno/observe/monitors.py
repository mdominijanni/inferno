from __future__ import annotations
from . import Reducer
from .. import Module, Hook
from abc import ABC, abstractmethod
from inferno._internal import rgetattr
from inferno.infernotypes import ManyToOne
import torch
from typing import Any, Callable, Protocol, Type
import weakref


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

        # register if as module is provided
        if module is not None:
            self.register(module)

        # register submodule
        self.reducer_ = reducer

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
        mapping (ManyToOne[torch.Tensor] | None, optional): modifies/selects
            which inputs to forward to reducer. Defaults to None.


    Caution:
        When implementing custom updaters, :py:class:`InputMonitor` should never be
        used. Use either :py:class:`StateMonitor` or :py:class:`DifferenceMonitor`.

    Note:
        The inputs, which are received as a tuple, will be unpacked and sent to
        the reducer. If there is only one input tensor, this will work as expected.
        Otherwise a ``mapping`` must be specified which takes the unpacked inputs
        and returns a single tensor.
    """

    def __init__(
        self,
        reducer: Reducer,
        module: Module = None,
        train_update: bool = True,
        eval_update: bool = True,
        prepend: bool = False,
        mapping: ManyToOne[torch.Tensor] | None = None,
    ):
        # determine arguments for superclass constructor
        if mapping:

            def prehook(module, fwdargs, *args):
                reducer(mapping(*fwdargs))

        else:

            def prehook(module, fwdargs, *args):
                reducer(*fwdargs)

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
        mapping (Callable[[Any], torch.Tensor] | None, optional): modifies/selects
            which outputs to forward to the reducer. Defaults to None.

    Caution:
        When implementing custom updaters, :py:class:`OutputMonitor` should never be
        used. Use either :py:class:`StateMonitor` or :py:class:`DifferenceMonitor`.

    Note:
        The output depends on the :py:meth:`~torch.nn.Module.forward` of the
        :py:class:`~torch.nn.Module` being called. If it a single tensor, it will
        work as expected. Otherwise a ``mapping`` must be specified which takes the
        output and returns a single tensor.
    """

    def __init__(
        self,
        reducer: Reducer,
        module: Module = None,
        train_update: bool = True,
        eval_update: bool = True,
        prepend: bool = False,
        mapping: Callable[[Any], torch.Tensor] | None = None,
    ):
        # determine arguments for superclass constructor
        if mapping:

            def posthook(module, fwdargs, output, *args):
                reducer(mapping(output))

        else:

            def posthook(module, fwdargs, output, *args):
                reducer(output)

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


class MonitorConstructor(Protocol):
    r"""Common constructor for managed monitors, used in updaters.

    Args:
        regattr (str): attribute or nested attribute to target.
        regmodule (Module): module to register as the target for monitoring.

    Returns:
        ManagedMonitor: newly constructed monitor.

    Important:
        The monitor returned must be registered.
    """

    monitor: Type[ManagedMonitor]
    reducer: Type[Reducer]

    def __call__(
        self,
        attr: str,
        module: Module,
    ) -> ManagedMonitor:
        r"""Callback protocol function"""
        ...


class ManagedMonitor(Monitor, ABC):
    r"""Monitors with construction which can be managed by other modules.

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

    Note:
        This keeps the same constructor signature as :py:class:`Monitor` and when the
        constructor is called directly will behave like a :py:class:`Monitor`.

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

        if module:
            self._monitored_ref = weakref.ref(module)

        # add tag data
        self._tags = {}

    @classmethod
    @abstractmethod
    def partialconstructor(cls, *args, **kwargs) -> MonitorConstructor:
        r"""Returns a function with a common signature for monitor construction.

        Raises:
            NotImplementedError: ``partialconstructor`` must be implemented
                by the subclass.
        """
        raise NotImplementedError(
            f"{cls.__name__}(ManagedMonitor) must implement "
            "the method `partialconstructor`."
        )

    def register(self, module: Module | None = None):
        r"""Registers the monitor as a forward hook/prehook.

        Args:
            module (Module | None, optional): module with which to register, last
                registered if None. Defaults to None.

        Raises:
            RuntimeError: weak reference to the last referenced module is no longer
                valid or did not exist
        """
        # get module from weakref if required
        if not module:
            # don't duplicate register call if module is unspecified
            if not self.registered:
                if self._monitored_ref and self._monitored_ref():
                    module = self._monitored_ref()
                else:
                    if module is None:
                        raise RuntimeError(
                            "weak reference to monitored module does not exist, "
                            "cannot infer argument 'module'"
                        )

        # register using superclass
        Monitor.register(self, module)

        # update stored weak reference
        self._monitored_ref = weakref.ref(module)

    @property
    def tags(self) -> dict[str, Any]:
        r"""Saved monitor tags.

        This should be used to determine which monitors are unique for the purpose of
        creating aliases. By default, the class of monitor and reducer is used with keys
        ``"monitor"`` and ``"reducer"`` respectively.

        Returns:
            dict[str, Any]: _description_
        """
        return {"monitor": type(self), "reducer": type(self.reducer), **self._tags}

    def tag(self, **tags: Any) -> None:
        r"""Set non-default tags.

        Keyword Args:
            **kwargs (Any): tags to set on the monitor, overriding existing non-default
                tags.

        """
        self._tags = tags


class StateMonitor(ManagedMonitor):
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
        map_ (Callable[[Any], torch.Tensor] | None, optional): modifies the input before
            being passed to the reduer, identity if None. Defaults to None.

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
        map_: Callable[[Any], torch.Tensor] | None = None,
    ):
        # set filter and map functions
        filter_ = filter_ if filter_ else lambda x: x is not None
        map_ = map_ if map_ else lambda x: x

        # determine arguments for superclass constructor
        if as_prehook:

            def prehook(module, *args):
                res = rgetattr(module, attr)
                if filter_(res):
                    reducer(map_(res))

            prehook_kwargs = {"prepend": prepend}
            posthook, posthook_kwargs = None, None

        else:

            def posthook(module, *args):
                res = rgetattr(module, attr)
                if filter_(res):
                    reducer(map_(res))

            posthook_kwargs = {"prepend": prepend}
            prehook, prehook_kwargs = None, None

        # construct superclass
        ManagedMonitor.__init__(
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
        map_: Callable[[Any], torch.Tensor] | None = None,
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
            filter_ (Callable[[Any], bool] | None, optional): test if the input should
                be passed to the reducer, ignores None values when None.
                Defaults to None.
            map_ (Callable[[Any], torch.Tensor] | None, optional): modifies the input
                before being passed to the reduer, identity if None.
                Defaults to None.

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


class DifferenceMonitor(ManagedMonitor):
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
            be passed to the reducer, ignores None pre or post values when None.
            Defaults to None.
        map_ (Callable[[Any, Any], torch.Tensor] | None, optional): modifies the input
            before being passed to the reduer, post minus pre if None.
            Defaults to None.

    Note:
        The nested attribute should be specified with dot notation. For instance,
        if the observed module has an attribute ``a`` which in turn has an
        attribute ``b`` that should be monitored, then ``attr`` should be
        `'a.b'``. Even with nested attributes, the monitor's hook will be tied to
        the module with which it is registered.

    Note:
        The left-hand argument of ``filter_`` and ``map_`` is the attribute after
        the :py:meth:`~torch.nn.Module.forward` of ``module`` is run, and the right-hand
        argument is before it is run.

        By default, ``filter_`` will only reject an input if both the pre and post
        states are ``None``. By default, ``map_`` will subtract the pre-forward value
        from the post-forward value. If either is ``None``, it will assume the ``None``
        value was composed of all-zeros.
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
        map_: Callable[[Any, Any], torch.Tensor] | None = None,
    ):
        # set filter and map functions
        filter_ = filter_ if filter_ else lambda f, i: not (f is None and i is None)
        map_ = (
            map_
            if map_
            else lambda f, i: (0 if f is None else f) - (0 if i is None else i)
        )

        # monitor state
        self.data = None

        # hook functions
        def prehook(module, *args):
            self.data = rgetattr(module, attr)

        def posthook(module, *args):
            res = rgetattr(module, attr)
            if filter_(res, self.data):
                reducer(map_(res, self.data))
            self.data = None

        # construct superclass
        ManagedMonitor.__init__(
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
        map_: Callable[[Any, Any], torch.Tensor] | None = None,
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
            filter_ (Callable[[Any, Any], bool] | None, optional): test if the input
                should be passed to the reducer, ignores None pre or post values when
                None. Defaults to None.
            map_ (Callable[[Any, Any], torch.Tensor] | None, optional): modifies the
                input before being passed to the reduer, post minus pre if None.
                Defaults to None.

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


class PreMonitor(Monitor):
    r"""Applies a function to prehook arguments and passes the results to the reducer.

    Args:
        reducer (Reducer): underlying means for reducing samples over time
            and storing them.
        mapping (Callable[[Module, tuple[Any, ...]], torch.Tensor]): function
            applied to the hook arguments with output passed to the reducer.
        module (Module, optional): module to register as the target for monitoring,
            can be modified after construction. Defaults to None.
        train_update (bool, optional): if this monitor should be called when the module
            being monitored is in train mode. Defaults to True.
        eval_update (bool, optional): if this monitor should be called when the module
            being monitored is in eval mode. Defaults to True.
        prepend (bool, optional): if this monitor should be called before other
            registered forward hooks. Defaults to False.
        with_kwargs (bool, optional): if keyword arguments to the forward function
            should be included in the hook call. Defaults to False.

    See Also:
        See :py:meth:`~torch.nn.Module.register_forward_pre_hook` for more information
        on the contents of the prehook arguments.
    """

    def __init__(
        self,
        reducer: Reducer,
        mapping: Callable[[Module, tuple[Any, ...]], torch.Tensor],
        module: Module = None,
        train_update: bool = True,
        eval_update: bool = True,
        prepend: bool = False,
        with_kwargs: bool = False,
    ):
        # hook function
        if with_kwargs:

            def prehook(module, args, kwargs):
                reducer(mapping(module, args, kwargs))

        else:

            def prehook(module, args):
                reducer(mapping(module, args))

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


class PostMonitor(Monitor):
    r"""Applies a function to posthook arguments and passes the results to the reducer.

    Args:
        reducer (Reducer): underlying means for reducing samples over time
            and storing them.
        mapping (Callable[[Module, tuple[Any, ...], Any], torch.Tensor]): function
            applied to the hook arguments with output passed to the reducer.
        module (Module, optional): module to register as the target for monitoring,
            can be modified after construction.Defaults to None.
        train_update (bool, optional): if this monitor should be called when the module
            being monitored is in train mode. Defaults to True.
        eval_update (bool, optional): if this monitor should be called when the module
            being monitored is in eval mode. Defaults to True.
        prepend (bool, optional): if this monitor should be called before other
            registered forward hooks. Defaults to False.
        with_kwargs (bool, optional): if keyword arguments to the forward function
            should be included in the hook call. Defaults to False.

    See Also:
        See :py:meth:`~torch.nn.Module.register_forward_hook` for more information
        on the contents of the posthook arguments.
    """

    def __init__(
        self,
        reducer: Reducer,
        mapping: Callable[[Module, tuple[Any, ...]], torch.Tensor],
        module: Module = None,
        train_update: bool = True,
        eval_update: bool = True,
        prepend: bool = False,
        with_kwargs: bool = False,
    ):
        # hook function
        if with_kwargs:

            def posthook(module, args, kwargs, output):
                reducer(mapping(module, args, kwargs, output))

        else:

            def posthook(module, args, output):
                reducer(mapping(module, args, output))

        # construct superclass
        Monitor.__init__(
            self,
            reducer=reducer,
            module=module,
            prehook=posthook,
            prehook_kwargs={"prepend": prepend},
            train_update=train_update,
            eval_update=eval_update,
        )
