from inferno import Module, WrapperModule, Hook
from inferno._internal import rgetattr
from inferno.typing import ManyToMany
import torch
from typing import Any, Callable
from . import Reducer


class Monitor(Module, Hook):
    r"""Base class for recording input, output, or state of a Module.

    Args:
        reducer (Reducer): underlying means for reducing samples over time and storing them.
        module (Module, optional): module to register as the target for monitoring,
            can be modified after construction. Defaults to None.
        prehook (Callable | None, optional): function to call before registrant's
            :py:meth:`~torch.nn.Module.forward`. Defaults to None.
        posthook (Callable | None, optional): function to call after registrant's
            :py:meth:`~torch.nn.Module.forward`. Defaults to None.
        prehook_kwargs (dict[str, Any] | None, optional): keyword arguments passed to
            :py:meth:`~torch.nn.Module.register_forward_pre_hook`. Defaults to None.
        posthook_kwargs (dict[str, Any] | None, optional): keyword arguments passed to
            :py:meth:`~torch.nn.Module.register_forward_hook`. Defaults to None.
        train_update (bool, optional): if this monitor should be called when the module
            being monitored is in train mode. Defaults to True.
        eval_update (bool, optional): if this monitor should be called when the module
            being monitored is in eval mode. Defaults to True.
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
        Module.__init__(self)
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

        # register submodule
        self.reducer_ = reducer

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
        reducer (Reducer): underlying means for reducing samples over time and storing them.
        module (Module, optional): module to register as the target for monitoring,
            can be modified after construction.Defaults to None.
        train_update (bool, optional): if this monitor should be called when the module
            being monitored is in train mode. Defaults to True.
        eval_update (bool, optional): if this monitor should be called when the module
            being monitored is in eval mode. Defaults to True.
        prepend (bool, optional): if this monitor should be called before other
            registered forward hooks. Defaults to False.
        map_fn(Callable[[tuple[torch.Tensor]], tuple[torch.Tensor]] | None, optional): modifies/selects
            which inputs to forward to reducer. Defaults to None.

    Note:
        The inputs, which are received as a tuple, will be unpacked and sent to the reducer.
        Most built-in reducers will select the input at index 0. Custom behavior can be defined
        by specifying a custom ``map_fn``. Defining a custom ``map_fn`` here will behave similarly
        and is expected to produce a tuple which will be unpacked into the reducer.
    """

    def __init__(
        self,
        reducer: Reducer,
        module: Module = None,
        train_update: bool = True,
        eval_update: bool = True,
        prepend: bool = False,
        map_fn: ManyToMany[torch.Tensor] | None = None,
    ):
        # determine arguments for superclass constructor
        if map_fn:
            def prehook(module, args):
                return reducer(*map_fn(*args))
        else:
            def prehook(module, args):
                return reducer(*args)

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
        module (Module, optional): module to register as the target for monitoring,
            can be modified after construction.Defaults to None.
        train_update (bool, optional): if this monitor should be called when the module
            being monitored is in train mode. Defaults to True.
        eval_update (bool, optional): if this monitor should be called when the module
            being monitored is in eval mode. Defaults to True.
        prepend (bool, optional): if this monitor should be called before other
            registered forward hooks. Defaults to False.
        map_fn(Callable[[torch.Tensor | tuple[torch.Tensor]], tuple[torch.Tensor]] | None, optional): modifies/selects
            outputs to forward to reducer. Defaults to None.

    Note:
        The manner in which outputs are received depends on the module being called. Because reducers
        receive unpacked inputs. If the output is a tuple, it will automatically be unpacked
        Custom behavior can be defined by specifying a custom ``map_fn``. Defining a custom
        ``map_fn`` here will behave similarly and is expected to produce a tuple which will
        be unpacked into the reducer.

    Note:
        If the output is an instance of :py:class:`tuple`, it will be unpacked automatically.
    """

    def __init__(
        self,
        reducer: Reducer,
        module: Module = None,
        train_update: bool = True,
        eval_update: bool = True,
        prepend: bool = False,
        map_fn: Callable[[Any], tuple[torch.Tensor, ...]]
        | None = None,
    ):
        # determine arguments for superclass constructor
        if map_fn:
            def posthook(module, args, output):
                if isinstance(output, tuple):
                    return reducer(*map_fn(*output))
                else:
                    return reducer(*map_fn(output))
        else:
            def posthook(module, args, output):
                if isinstance(output, tuple):
                    return reducer(*output)
                else:
                    return reducer(output)

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


class StateMonitor(WrapperModule, Hook):
    r"""Records the state of an attribute in a Module.

    Args:
        reducer (Reducer): underlying means for reducing samples over time and storing them.
        attr (str): attribute or nested attribute to target.
        module (Module, optional): module to register as the target for monitoring,
            can be modified after construction.Defaults to None.
        train_update (bool, optional): if this monitor should be called when the module
            being monitored is in train mode. Defaults to True.
        eval_update (bool, optional): if this monitor should be called when the module
            being monitored is in eval mode. Defaults to True.
        prepend (bool, optional): if this monitor should be called before other
            registered forward hooks. Defaults to False.

    Note:
        The nested attribute should be specified with dot notation. For instance, if the registrant has an attribute
        ``a`` which in turn has an attribute ``b`` that should be monitored, then ``attr`` should be ``'a.b'``. Even
        with nested attributes, the monitor's hook will be tied to the registrant.

    Note:
        If the monitored attribute is an instance of :py:class:`tuple`, it will be unpacked automatically.
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
            def prehook(module, args):
                value = rgetattr(module, attr)
                if isinstance(value, tuple):
                    return reducer(*value)
                else:
                    return reducer(value)
            prehook_kwargs = {"prepend": prepend}

            posthook = None
            posthook_kwargs = None

        else:
            prehook = None
            prehook_kwargs = None

            def posthook(module, args, output):
                value = rgetattr(module, attr)
                if isinstance(value, tuple):
                    return reducer(*value)
                else:
                    return reducer(value)
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
