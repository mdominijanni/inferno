from inferno import Module, WrapperModule, Hook
from inferno._internal import rgetattr
import torch
from typing import Any, Callable
from . import Reducer


class Monitor(WrapperModule, Hook):
    r"""Base class for recording input, output, or state of a Module.

    Args:
        reducer (Reducer): underlying means for reducing samples over time and storing them.
        module (Module, optional): module to register as the target for monitoring, can be modified after construction. Defaults to None.
        prehook (Callable | None, optional): function to call before registrant's :py:meth:`~torch.nn.Module.forward`. Defaults to None.
        posthook (Callable | None, optional): function to call after registrant's :py:meth:`~torch.nn.Module.forward`. Defaults to None.
        prehook_kwargs (dict[str, Any] | None, optional): keyword arguments passed to :py:meth:`~torch.nn.Module.register_forward_pre_hook`. Defaults to None.
        posthook_kwargs (dict[str, Any] | None, optional): keyword arguments passed to :py:meth:`~torch.nn.Module.register_forward_hook`. Defaults to None.
        train_update (bool, optional): if this monitor should be called when the module being monitored is in train mode. Defaults to True.
        eval_update (bool, optional): if this monitor should be called when the module being monitored is in eval mode. Defaults to True.
    """

    def __init__(
        self,
        reducer: Reducer,
        module: Module = None,
        prehook: Callable | None = None,
        posthook: Callable | None = None,
        prehook_kwargs: dict[str, Any] | None = None,
        posthook_kwargs: dict[str, Any] | None = None,
        train_update: bool = True,
        eval_update: bool = True,
    ):
        # construct module superclasses
        WrapperModule.__init__(self, reducer)
        Hook.__init__(
            self,
            prehook=prehook,
            posthook=posthook,
            prehook_kwargs=prehook_kwargs,
            posthook_kwargs=posthook_kwargs,
            train_update=train_update,
            eval_update=eval_update,
            module=module,
        )

    @property
    def reducer(self) -> Reducer:
        """Reducer associated with the monitor.

        Returns:
            Reducer: reducer used for storing samples over time.
        """
        return self.submodule


class InputMonitor(Monitor):
    r"""Records the inputs passed to a Module.

    Args:
        reducer (Reducer): underlying means for reducing samples over time and storing them.
        module (Module, optional): module to register as the target for monitoring, can be modified after construction. Defaults to None.
        train_update (bool, optional): if this monitor should be called when the module being monitored is in train mode. Defaults to True.
        eval_update (bool, optional): if this monitor should be called when the module being monitored is in eval mode. Defaults to True.
        prepend (bool, optional): if this monitor should be called before other registered forward pre-hooks. Defaults to False.
        mapfn(Callable[[tuple[torch.Tensor]], tuple[torch.Tensor]] | None, optional): modifies/selects inputs to forward to reducer. Defaults to None.

    Note:
        The inputs, which are received as a tuple, will be unpacked and sent to the reducer.
        Most built-in reducers will select the input at index 0. Custom behavior can be defined
        by specifying a custom ``mapfn`` or ``mapmeth`` there. Defining a custom ``mapfn`` here
        will behave similarly and is expected to produce a tuple which will be unpacked by
        the reducer.
    """

    def __init__(
        self,
        reducer: Reducer,
        module: Module = None,
        train_update: bool = True,
        eval_update: bool = True,
        prepend: bool = False,
        mapfn: Callable[[tuple[torch.Tensor]], tuple[torch.Tensor]] | None = None,
    ):
        # determine arguments for superclass constructor
        if mapfn:
            prehook = lambda m, a: reducer(*mapfn(a))
        else:
            prehook = lambda m, a: reducer(*a)

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


class OutputMonitor(WrapperModule, Hook):
    r"""Records the outputs returned from a Module.

    Args:
        reducer (Reducer): underlying means for reducing samples over time and storing them.
        module (Module, optional): module to register as the target for monitoring, can be modified after construction. Defaults to None.
        train_update (bool, optional): if this monitor should be called when the module being monitored is in train mode. Defaults to True.
        eval_update (bool, optional): if this monitor should be called when the module being monitored is in eval mode. Defaults to True.
        prepend (bool, optional): if this monitor should be called before other registered forward hooks. Defaults to False.
        mapfn(Callable[[torch.Tensor | tuple[torch.Tensor]], tuple[torch.Tensor]] | None, optional): modifies/selects outputs to forward to reducer. Defaults to None.

    Note:
        The manner in which outputs are received depends on the module being called. Because reducers
        receive unpacked inputs (which will treat a single tensor as a 1-tuple), they are by default
        unmodified. Most built-in reducers will select the input at index 0. Custom behavior can be defined
        by specifying a custom ``mapfn`` or ``mapmeth`` there. Defining a custom ``mapfn`` here
        will behave similarly and is expected to produce a tuple which will be unpacked by
        the reducer. For consistency, the default ``mapfn`` will wrap the output in a tuple if it not already one.
    """

    def __init__(
        self,
        reducer: Reducer,
        module: Module = None,
        train_update: bool = True,
        eval_update: bool = True,
        prepend: bool = False,
        mapfn: Callable[[torch.Tensor | tuple[torch.Tensor]], tuple[torch.Tensor]]
        | None = None,
    ):
        # determine arguments for superclass constructor
        if not mapfn:
            mapfn = lambda x: x if isinstance(x, tuple | list) else (x,)

        # construct superclass
        Monitor.__init__(
            self,
            reducer=reducer,
            module=module,
            posthook=lambda m, a, o: reducer(*mapfn(o)),
            posthook_kwargs={"prepend": prepend},
            train_update=train_update,
            eval_update=eval_update,
        )


class StateMonitor(WrapperModule, Hook):
    r"""Records the state of an attribute in a Module.

    Args:
        reducer (Reducer): underlying means for reducing samples over time and storing them.
        attr (str): attribute or nested attribute to target.
        module (Module, optional): module to register as the target for monitoring, can be modified after construction. Defaults to None.
        as_prehook (bool, optional): if the monitor should be called before the registrant's :py:meth:`~torch.nn.Module.forward` call. Defaults to False.
        train_update (bool, optional): if this monitor should be called when the module being monitored is in train mode. Defaults to True.
        eval_update (bool, optional): if this monitor should be called when the module being monitored is in eval mode. Defaults to True.
        prepend (bool, optional): if this monitor should be called before other registered forward hooks. Defaults to False.

    Note:
        The nested attribute should be specified with dot notation. For instance, if the registrant has an attribute
        ``a`` which in turn has an attribute ``b`` that should be monitored, then ``attr`` should be ``'a.b'``. Even
        with nested attributes, the monitor's hook will be tied to the registrant.
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
    ):
        # determine arguments for superclass constructor
        if as_prehook:
            prehook = lambda m, a: reducer(rgetattr(m, attr))
            prehook_kwargs = {"prepend": prepend}
            posthook = None
            posthook_kwargs = None
        else:
            prehook = None
            prehook_kwargs = lambda m, a, o: reducer(rgetattr(m, attr))
            posthook = None
            posthook_kwargs = {"prepend": prepend}

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
