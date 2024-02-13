import attrs
from .math import Interpolation
from collections import OrderedDict
from collections.abc import Mapping
from functools import cached_property
from inferno._internal import rsetattr, instance_of, numeric_limit
import itertools
import math
import torch
import torch.nn as nn
from typing import Any, Callable
import warnings


class Module(nn.Module):
    r"""An extension of PyTorch's `Module
    <https://pytorch.org/docs/master/generated/torch.nn.Module.html#torch.nn.Module>`_ class.

    This extends :py:class:`torch.nn.Module` so that "extra state" is handled in a way
    similar to regular tensor state (e.g. buffers and parameters). This enables simple
    export/import to/from a state dictionary.

    Note:
        Like with :py:class:`torch.nn.Module`, an :py:meth:`__init__` call must be made
        to the parent class before assignment on the child. This class's constructor
        will automatically call PyTorch's.
    """

    def __init__(self, *args, **kwargs):
        nn.Module.__init__(self, *args, **kwargs)
        self._extras = OrderedDict()

    def register_extra(self, name: str, value: Any):
        r"""Adds an extra variable to the module.

        This is typically used in a manner to
        :py:meth:`~torch.nn.Module.register_buffer`, except that the value being
        registered is not limited to being a :py:class:`~torch.Tensor`.

        Args:
            name (str): name of the extra, which can be accessed from this module
                using the provided name.
            value (Any): extra to be registered.

        Raises:
            TypeError: if the extra variable being registered is an instance of
                :py:class:`torch.Tensor` or :py:class:`torch.nn.Module`.

        Note:
            :py:class:`~torch.Tensor` and :py:class:`~torch.nn.Module` objects cannot
            be registered as extras and should be registered using existing methods.
        """
        if not isinstance(name, str):
            raise TypeError(
                f"extra name must be a string, received {type(name).__name__}"
            )
        elif "." in name:
            raise KeyError('extra name cannot contain "."')
        elif name == "":
            raise KeyError('extra name cannot be empty string ""')
        elif hasattr(self, name) and name not in self._extras:
            raise KeyError(f"attribute '{name}' already exists")
        elif isinstance(value, torch.Tensor | nn.Module):
            raise TypeError(
                f"cannot assign '{type(value).__name__}' object to '{name}'"
            )
        else:
            self._extras[name] = value

    def get_extra(self, target: str) -> Any:
        r"""Returns the extra given by ``target`` if it exists, otherwise throws an error.

        This functions similarly to, and has the same specification of ``target`` as
        :py:meth:`~torch.nn.Module.get_submodule`.

        Args:
            target (str): fully-qualified string name of the extra for which to look.

        Returns:
            Any: the extra referenced ``target``.

        Raises:
            AttributeError: if the target string references an invalid path, the
                terminal module is an instance of :py:class:`torch.nn.Module` but not
                :py:class:`Module`, or resolves to something that is not an extra.
        """
        module_path, _, extra_name = target.rpartition(".")

        module = self.get_submodule(module_path)

        if not isinstance(module, Module):
            raise AttributeError(
                f"{module.__class__.__name__} " "is not an instance of inferno.Module"
            )
        if not hasattr(module, extra_name):
            raise AttributeError(
                f"{module.__class__.__name__} " f"has no attribute `{extra_name}`"
            )

        extra = getattr(module, extra_name)
        if extra_name not in module._extras:
            raise AttributeError(f"`{extra_name}` is not an extra")

        return extra

    def get_extra_state(self) -> dict[str, Any]:
        return self._extras

    def set_extra_state(self, state: dict[str, Any]):
        self._extras.update(state)

    def __setstate__(self, state):
        self.__dict__.update(state)
        if "_extras" not in self.__dict__:
            self._extras = OrderedDict()
        return super().__setstate__(state)

    def __getattr__(self, name: str) -> Any:
        if "_extras" in self.__dict__:
            _extras = self.__dict__["_extras"]
            if name in _extras:
                return _extras[name]
        return super().__getattr__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        _extras = self.__dict__.get("_extras")
        if _extras is not None and name in _extras:
            _extras[name] = value
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in self._extras:
            del self._extras[name]
        else:
            return super().__delattr__(name)

    def __dir__(self):
        module_attrs = dir(self.__class__)
        attrs = list(self.__dict__.keys())
        parameters = list(self._parameters.keys())
        modules = list(self._modules.keys())
        buffers = list(self._buffers.keys())
        extras = list(self._extras.keys())
        keys = module_attrs + attrs + parameters + modules + buffers + extras

        # Eliminate attrs that are not legal Python variable names
        keys = [key for key in keys if not key[0].isdigit()]

        return sorted(keys)


class Configuration(Mapping):
    r"""Class which provides unpacking functionality when used in conjunction with
    the `attrs library <https://www.attrs.org/en/stable/>`_.

    When defining configuration classes which are to be wrapped by
    :py:func:`attrs.define`, if this is subclassed, then it can be unpacked with ``**``.

    .. automethod:: _asadict_
    """

    def _asadict_(self) -> dict[str, Any]:
        r"""Controls how the fields of this class are convereted into a dictionary.

        This will flatten any nested :py:class:`Configuration` objects using their own
        :py:meth:`_asadict_` method. If there are naming conflicts (i.e. if a nested
        configuration has) a field with the same name, only one will be preserved.
        This can be overridden to change its behavior.

        Returns:
            dict[str, Any]: dictionary of field names to the objects they represent.

        Note:
            This only packages those attributes which were registered via
            :py:func:`attrs.field`.
        """
        d = []
        for k in attrs.fields_dict(type(self)):
            v = getattr(self, k)
            if isinstance(v, Configuration):
                d.extend(v._asadict_().items())
            else:
                d.append((k, v))
        return dict(d)


class Hook:
    r"""Provides forward hook/prehook functionality for subclasses.

    `Hook` provides functionality to register and deregister itself as
    forward hook with a :py:class:`torch.nn.Module` object. This is performed using
    :py:meth:`~torch.nn.Module.register_forward_hook` to register itself as a forward
    hook and it manages the returned :py:class:`~torch.utils.hooks.RemovableHandle`
    to deregister itself.

    Args:
        prehook (Callable | None, optional): function to call before hooked module's
            :py:meth:`~torch.nn.Module.forward`. Defaults to None.
        posthook (Callable | None, optional): function to call after hooked module's
            :py:meth:`~torch.nn.Module.forward`. Defaults to None.
        prehook_kwargs (dict[str, Any] | None, optional): keyword arguments passed to
            :py:meth:`~torch.nn.Module.register_forward_pre_hook`. Defaults to None.
        posthook_kwargs (dict[str, Any] | None, optional): keyword arguments passed to
            :py:meth:`~torch.nn.Module.register_forward_hook`. Defaults to None.
        train_update (bool, optional): if the hooks should be run when hooked module is
            in train mode. Defaults to True.
        eval_update (bool, optional): if the hooks should be run when hooked module is
            in eval mode. Defaults to True.
    Note:
        If not None, the signature of the prehook must be of the following form.

        .. code-block:: python

            hook(module, args) -> None or modified input

        Or, if ``with_kwargs`` is passed as a keyword argument.

        .. code-block:: python

            hook(module, args, kwargs) -> None or modified input

        See :py:meth:`torch.nn.Module.register_forward_pre_hook` for
        further information.

    Note:
        If not None, the signature of the posthook must be of the following form.

        .. code-block:: python

            hook(module, args, output) -> None or modified output

        Or, if ``with_kwargs`` is passed as a keyword argument.

        .. code-block:: python

            hook(module, args, kwargs, output) -> None or modified output

        See :py:meth:`torch.nn.Module.register_forward_hook` for further information.


    Raises:
        RuntimeError: at least one of ``prehook`` and ``posthook`` must not be None.
        RuntimeError: at least one of ``train_update`` and ``eval_update`` must be True.
        TypeError: if parameter ``module`` is not ``None``, then it must be an instance
            of :py:class:`torch.nn.Module`.
    """

    def __init__(
        self,
        prehook: Callable | None = None,
        posthook: Callable | None = None,
        *,
        prehook_kwargs: dict[str, Any] | None = None,
        posthook_kwargs: dict[str, Any] | None = None,
        train_update: bool = True,
        eval_update: bool = True,
    ):
        # check if at least one callable is defined
        if not prehook and not posthook:
            raise RuntimeError(
                "at least one of `prehook` and `posthook` must not be None."
            )

        # prehook and posthook functions
        self._prefunc = prehook
        self._postfunc = posthook

        # set returned handle
        self._prehandle = None
        self._posthandle = None

        # set hook registering kwargs
        self._prekwargs = prehook_kwargs if prehook_kwargs else {}
        self._postkwargs = posthook_kwargs if posthook_kwargs else {}

        # set training conditionals
        self._trainupdate = train_update
        self._evalupdate = eval_update

    @property
    def on_train(self) -> bool:
        """If the hook is called when the module passed in is in training mode.

        Args:
            value (bool): if the hook should be called when the module is training.

        Returns:
            bool: if the hook is called when the module is training.
        """
        return self._trainupdate

    @on_train.setter
    def on_train(self, value: bool) -> None:
        self._trainupdate = value

    @property
    def on_eval(self) -> bool:
        """If the hook is called when the module passed in is in evaluation mode.

        Args:
            value (bool): if the hook should be called when the module is evaluating.

        Returns:
            bool: if the hook is called when the module is evaluating.
        """
        return self._evalupdate

    @on_eval.setter
    def on_eval(self, value: bool) -> None:
        self._evalupdate = value

    @property
    def registered(self) -> bool:
        r"""If there is a module to which this hook is registered

        Returns:
            bool: if a module to which this hook is registred.
        """
        return self._prehandle or self._posthandle

    def register(self, module: nn.Module) -> None:
        r"""Registers the hook as a forward hook/prehook with specified
        :py:class:`~torch.nn.Module`.

        Args:
            module (nn.Module): PyTorch module to which the forward hook
                will be registered.

        Raises:
            TypeError: parameter ``module`` must be an instance of
                :py:class:`torch.nn.Module`.

        Warns:
            RuntimeWarning: each :py:class:`Hook` can only be registered to one
                :py:class:`~torch.nn.Module` and will ignore :py:meth:`register`
                if already registered.
        """
        if not self.registered:
            e = instance_of("module", module, nn.Module)
            if e:
                raise e

            if self._prefunc:
                self._prehandle = module.register_forward_pre_hook(
                    self._prefunc, **self._prekwargs
                )

            if self._postfunc:
                self._posthandle = module.register_forward_hook(
                    self._postfunc, **self._postkwargs
                )
        else:
            warnings.warn(
                f"this {type(self).__name__} object is already registered to an object "
                "so new `register()` was ignored",
                category=RuntimeWarning,
            )

    def deregister(self) -> None:
        r"""Deregisters the hook as a forward hook/prehook from registered
        :py:class:`~torch.nn.Module`, if it is already registered."""
        if self._prehandle:
            self._prehandle.remove()
            self._prehandle = None
        if self._posthandle:
            self._posthandle.remove()
            self._posthandle = None

    def __del__(self) -> None:
        r"""Automatically deregister on deconstruction."""
        return self.deregister()

    def prehook(self, *args) -> Any:
        r"""Calls the prehook function if updating in the module's mode.

        Either two or three arguments will be passed in depending on if ``with_kwargs``
        was set to True in ``prehook_kwargs`` on construction.

        Returns:
            Any: modified input or None, depending on the prehook passed on init.
        """
        if (self._evalupdate, self._trainupdate)[args[0].training]:
            return self._prefunc(*args)

    def posthook(self, *args) -> Any:
        r"""Calls the posthook function if updating in the module's mode.

        Either three or four arguments will be passed in depending on if ``with_kwargs``
        was set to True in ``posthook_kwargs`` on construction.

        Returns:
            Any: modified output or None, depending on the posthook passed on init.
        """
        if (self._evalupdate, self._trainupdate)[args[0].training]:
            return self._postfunc(*args)


class StateHook(Hook):
    r"""Interactable hook which only acts on module state.

    Args:
        hook (Callable[[nn.Module], None]): function to call on hooked module's
            :py:meth:`~torch.nn.Module.__call__`.
        module (nn.Module): module to which the hook should be registered.
        train_update (bool, optional): if the hook should be run when hooked module is
            in train mode. Defaults to True.
        eval_update (bool, optional): if the hook should be run when hooked module is
            in eval mode. Defaults to True.
        as_prehook (bool, optional): if the hook should be run prior to the hooked
            module's :py:meth:`~torch.nn.Module.forward` call. Defaults to False.
        prepend (bool, optional): if the hook should be run prior to the hooked
            module's previously registered forward hooks. Defaults to False.
        always_call (bool, optional): if the hook should be run even if an exception
            occurs, only applies when ``as_prehook`` is False. Defaults to False.

    Note:
        To trigger the hook regardless of the hooked module's training state,
        call the ``StateHook`` object. The hook will not run if it is not registered.

    Note:
        Unlike with :py:class:`Hook`, the ``hook`` here will only be passed a single
        argument (the registered module itself) and any output will be ignored.
    """

    def __init__(
        self,
        hook: Callable[[nn.Module], None],
        module: nn.Module,
        train_update: bool = True,
        eval_update: bool = True,
        *,
        as_prehook: bool = False,
        prepend: bool = False,
        always_call: bool = False,
    ):
        # core state
        self._hook = hook
        self._module = module

        # create hook wrapper
        def hookwrapper(*args):
            self._hook(args[0])

        # construct superclass
        if as_prehook:
            Hook.__init__(
                self,
                prehook=hookwrapper,
                prehook_kwargs={"prepend": prepend},
                train_update=train_update,
                eval_update=eval_update,
            )
        else:
            Hook.__init__(
                self,
                posthook=hookwrapper,
                posthook_kwargs={"prepend": prepend, "always_call": always_call},
                train_update=train_update,
                eval_update=eval_update,
            )

    @property
    def module(self) -> nn.Module:
        r"""Module to which the hook is applied

        Returns:
            nn.Module: module to which the hook is applied.
        """
        return self._module

    def register(self) -> None:
        r"""Registers the hook as a forward hook/prehook."""
        Hook.register(self, self.module)

    def __call__(self):
        r"""Executes the hook at any time, only if it is registered."""
        if self.registered:
            self._hook(self.module)


class RemoteHook(StateHook):
    r"""A state hook which acts on modules other than the registered module.

    Unlike a regular :py:class:`StateHook` where the module itself is passed on
    its call, here an "inner module" is passed, along with associated keyword
    arguments.

    Subclass this for cases where updates should be triggered by a module on
    other modules. For example, normalizing layer weights after an updater call.

    Args:
        hook (Callable[[nn.Module, dict[str, Any]], None]): function to call on hooked
            module's :py:meth:`~torch.nn.Module.__call__`.
        module (nn.Module): module to which the hook should be registered.
        inner_train_update (bool, optional): if the hook should be run on the inner
            module when the inner module is in training mode. Defaults to True.
        inner_eval_update (bool, optional): if the hook should be run on the inner
            module when the inner module is in evaluation mode. Defaults to False.
        as_prehook (bool, optional): if the hook should be run prior to the hooked
            module's :py:meth:`~torch.nn.Module.forward` call. Defaults to False.
        prepend (bool, optional): if the hook should be run prior to the hooked
            module's previously registered forward hooks. Defaults to False.
        always_call (bool, optional): if the hook should be run even if an exception
            occurs, only applies when ``as_prehook`` is False. Defaults to False.

    Note:
        By default, the hook will be triggered on the the registered module's call
        whether its in training or evaluation mode (since generally this should) be
        controlled by the ``inner_train_update`` and ``inner_eval_update`` arguments.
        The same properties as in :py:class:`StateHook` can be used to alter this.
    """

    def __init__(
        self,
        hook: Callable[[nn.Module, dict[str, Any]], None],
        module: nn.Module,
        inner_train_update: bool = True,
        inner_eval_update: bool = False,
        *,
        as_prehook: bool = False,
        prepend: bool = False,
        always_call: bool = False,
    ):
        # layer state
        self.inner = nn.ModuleDict()
        self.innerparams = {}

        # internally called hook
        self._innerhook = hook

        # update triggers
        self._innertrainupdate = inner_train_update
        self._innerevalupdate = inner_eval_update

        # call superclass constructor
        StateHook.__init__(
            self,
            self.innerhook,
            module,
            True,
            True,
            as_prehook=as_prehook,
            prepend=prepend,
            always_call=always_call,
        )

    def innerhook(self, *args) -> None:
        r"""Takes hook call and conditionally performs hook on inner modules."""
        for key, module in self.inner.items():
            if (self._innerevalupdate, self._innertrainupdate)[module.training]:
                self._innerhook(module, **self.innerparams[key])

    @staticmethod
    def _keyof(module: nn.Module) -> str:
        """Gets the key (hexidecimal ID) of an inner module.

        Args:
            module (nn.Module): module for which to generate the key.

        Returns:
            str: key for the given module.
        """
        return hex(id(module))

    @property
    def on_train_inner(self) -> bool:
        """If the hook is called when the inner module is in training mode.

        Args:
            value (bool): if the hook should be called when the
                inner module is training.

        Returns:
            bool: if the hook is called when the inner module is training.
        """
        return self._innertrainupdate

    @on_train_inner.setter
    def on_train_inner(self, value: bool) -> None:
        self._innertrainupdate = value

    @property
    def on_eval_inner(self) -> bool:
        """If the hook is called when the inner module passed in is in evaluation mode.

        Args:
            value (bool): if the hook should be called when the
                inner module is evaluating.

        Returns:
            bool: if the hook is called when the inner module is evaluating.
        """
        return self._innerevalupdate

    @on_eval_inner.setter
    def on_eval_inner(self, value: bool) -> None:
        self._innerevalupdate = value

    def add_inner(self, *modules: nn.Module, **kwargs: dict[str, Any]) -> None:
        r"""Adds inner modules to the hook.

        Args:
            *modules (~torch.nn.Module): inner modules to add.
            **kwargs (dict[str, Any]): keyword arguments for the hook call.
        """
        for module in modules:
            key = self._keyof(module)
            if key not in self.inner:
                self.inner[key] = module
                self.innerparams[key] = kwargs

    def remove_inner(self, *modules: nn.Module) -> None:
        r"""Removes inner modules from the hook.

        Args:
            *modules (~torch.nn.Module): inner modules to remove.
        """
        for module in modules:
            key = self._keyof(module)
            if key in self.inner:
                del self.inner[key]
            if key in self.innerparams:
                del self.innerparams[key]


class DimensionalModule(Module):
    r"""Module with support for dimensionally constrained buffers and parameters.

    Args:
        constraints (tuple[int, int]): tuple of (dim, size) dimensional constraints for
            constrained buffers and parameters.

    Raises:
        ValueError: constraints must specify a positive number of elements.
        RuntimeError: no two constraints may share a dimension.

    Note:
        Each argument must be a 2-tuple of integers, where the first element is the
        dimension to which a constraint is applied and the second is the size of that
        dimension. Dimensions can be negative.

    Important:
        Constraints are not checked on every assignment and as such may be invalidated.
        This can be tested with :py:meth:`reconstrain`.

    """

    def __init__(
        self,
        *constraints: tuple[int, int],
    ):
        # call superclass constructor
        Module.__init__(self)

        # register extras
        self.register_extra("_constraints", dict())
        self.register_extra("_constrained_buffers", set())
        self.register_extra("_constrained_parameters", set())

        # check for consistent constraints
        for dim, size in constraints:
            dim, size = int(dim), int(size)
            if size < 1:
                raise ValueError(
                    f"constraint {(dim, size)} specifies an invalid (nonpositive) "
                    "number of elements."
                )
            if dim in self._constraints:
                raise RuntimeError(
                    f"constraint {(dim, size)} conflicts with constraint "
                    f"{dim, self._constraints[dim]}."
                )
            self._constraints[dim] = size

    @staticmethod
    def mindims_(constraints: dict[int, int]) -> int:
        """Computes minimum number of required dimensions for a constrained tensor.

        Args:
            constraints (dict[int, int]): constraint dictionary of (dim, size).

        Returns:
            int: minimum required number of dimensions.
        """
        if not constraints:
            return 0

        maxc = max(constraints)
        minc = min(constraints)

        return (maxc + 1 if maxc >= 0 else 0) - (minc if minc <= -1 else 0)

    @classmethod
    def compatible_(cls, value: torch.Tensor, constraints: dict[int, int]) -> bool:
        """Test if a tensor is compatible with a set of constraints.

        Args:
            value (torch.Tensor): tensor to test.
            constraints (dict[int, int]): constraint dictionary to test with.

        Returns:
            bool: if the tensor is compatible.
        """
        # check if value has fewer than minimum required number of dimensions
        if value.ndim < cls.mindims_(constraints):
            return False

        # check if constraints are met
        for dim, size in constraints.items():
            if value.shape[dim] != size:
                return False

        return True

    @classmethod
    def compatible_like_(
        cls,
        shape: tuple[int],
        constraints: dict[int, int],
    ) -> tuple[int]:
        """Generates a shape like the input, but compatible with the constraints.

        Args:
            shape (tuple[int]): shape to make compatible
            constraints (dict[int, int]): constraint dictionary to test against.

        Raises:
            RuntimeError: dimensionality of shape is insufficient.
            RuntimeError: constraints contains nonpositive sized dimensions.

        Returns:
            tuple[int]: compatiblized shape.
        """
        # ensure shape is of sufficient dimensionality
        req_ndims = cls.mindims_(constraints)

        if len(shape) < req_ndims:
            raise RuntimeError(
                f"`shape` {shape} with dimensionality {len(shape)} cannot be made "
                f"compatible, requires a minimum dimensionality of {req_ndims}."
            )

        # create new shape
        new_shape = list(shape)
        for dim, size in constraints.items():
            if size < 1:
                raise RuntimeError(
                    f"`shape` {shape} cannot contain nonpositive sized dimensions."
                )
            else:
                new_shape[dim] = size
        return tuple(new_shape)

    @cached_property
    def constraints(self) -> dict[int, int]:
        r"""Returns the constraint dictionary, sorted by dimension.

        Returns:
            dict[int, int]: active constraints, represented as a dictionary.

        Note:
            The results will be sorted by dimension, from low to high. Therefore,
            positive dimensions are presented first, in increasing order, then negative
            dimensions also in increasing order.
        """
        fwd, rev = [], []
        for dim, size in sorted(self._constraints.items()):
            rev.append((dim, size)) if dim < 0 else fwd.append((dim, size))

        return dict(fwd + rev)

    @cached_property
    def constraints_repr(self) -> str:
        r"""Returns a string representation of constraints.

        Returns:
            str: active constraints, represented as a string.

        Note:
            Like with :py:meth:`constraints`, dimensions are sorted from low to high.
            Underscores represent dimensions which must be present in the constrained
            tensor but with an unspecified value.
        """
        # split constraints into forward and reverse (negative) indices, sorted
        fwd, rev = [], []
        for dim, size in self.constraints.items():
            rev.append((dim, size)) if dim < 0 else fwd.append((dim, size))

        # representation elements
        elems = []

        # forward indexed constraints
        # expect dimension 0
        expc = 0
        for dim, size in fwd:
            # add unconstrained placeholders
            elems.extend(["_" for _ in range(dim - expc)])
            # add contraint value
            elems.append(f"{size}")
            # set expected next dimension
            expc = dim + 1

        # aribtrary separation
        elems.append("...")

        # reverse indexed constraints
        # no expected dimension
        expc = None
        for dim, size in rev:
            # add unconstrained placeholders
            if expc is not None:
                elems.extend(["_" for _ in range(dim - expc)])
            # add contraint value
            elems.append(f"{size}")
            # set expected next dimension
            expc = dim + 1
        # final cases
        if expc is not None:
            elems.extend(["_" for _ in range(expc, 0)])

        return f"({', '.join(elems)})"

    @property
    def mindims(self) -> int:
        r"""Minimum number of constrained dimensions for constrained tensor.

        Returns:
            int: minimum required number of dimensions.
        """
        return self.mindims_(self.constraints)

    def compatible(self, value: torch.Tensor) -> bool:
        """Test if a tensor is compatible with the module's constraints.

        Args:
            value (torch.Tensor): tensor to test.

        Returns:
            bool: if the tensor is compatible.
        """
        return self.compatible_(value, self._constraints)

    def compatible_like(self, shape: tuple[int]) -> tuple[int]:
        """Generates a shape like the input, but compatible with the constraints.

        Args:
            shape (tuple[int]): shape to make compatible
            constraints (dict[int, int] | None, optional): constraint dictionary
                to test with, uses current constraints if None. Defaults to None.

        Raises:
            RuntimeError: dimensionality of shape is insufficient.
            RuntimeError: constraints contains nonpositive sized dimensions.

        Returns:
            tuple[int]: compatiblized shape.
        """
        return self.compatible_like_(shape, self._constraints)

    def reconstrain(self, dim: int | None, size: int | None) -> None:
        """Edits existing constraints and reshapes constrained buffers and parameters
        accordingly.

        Args:
            dim (int | None): dimension to which a constraint should be added, removed,
                or modified, or if None, only existing constraints are tested.
            size (int | None): size of the new constraint, or None if the constraint
                should be removed.

        Raises:
            RuntimeError: constrained buffer or parameter had its shape modified
                externally and is no longer compatible.
            ValueError: size must specify a positive number of elements.
            ValueError: added constraint is incompatible with existing buffer or
                parameter.
        """
        # delete cache for constraint properties
        try:
            del self.constraints
        except AttributeError:
            pass
        try:
            del self.constraints_repr
        except AttributeError:
            pass

        # remove deleted buffers and parameters as constrained
        for name in tuple(self._constrained_buffers):
            try:
                _ = self.get_buffer(name)
            except AttributeError:
                self.deregister_constrained(name)

        for name in tuple(self._constrained_parameters):
            try:
                _ = self.get_parameter(name)
            except AttributeError:
                self.deregister_constrained(name)

        # ensure buffers and parameters are still properly constrained
        for name in self._constrained_buffers:
            buffer = self.get_buffer(name)
            if (
                buffer is not None
                and buffer.numel() > 0
                and not self.compatible(buffer)
            ):
                raise RuntimeError(f"constrained buffer {name} has been invalidated.")
        for name in self._constrained_parameters:
            param = self.get_parameter(name)
            if param is not None and param.numel() > 0 and not self.compatible(param):
                raise RuntimeError(
                    f"constrained parameter {name} has been invalidated."
                )

        # end early if no dimensions (check mode)
        if dim is None:
            return

        # convert arguments to integers
        dim, size = int(dim), None if size is None else int(size)

        # check for valid size constraint
        if size is not None and size < 1:
            raise ValueError(
                f"`size` {size} specifies an invalid (nonpositive) number of elements."
            )

        # addition of constraint
        if dim not in self._constraints and size is not None:
            # create constraints with new addition
            constraints = itertools.chain(self._constraints.items(), ((dim, size),))
            constraints = {d: s for d, s in constraints}

            # ensure constrained buffers and parameters are compatible
            for name in self._constrained_buffers:
                buffer = self.get_buffer(name)
                if (
                    buffer is not None
                    and buffer.numel() > 0
                    and not self.compatible_(buffer, constraints)
                ):
                    raise ValueError(
                        f"constraint is incompatible with buffer '{name}'."
                    )

            for name in self._constrained_parameters:
                param = self.get_parameter(name)
                if (
                    param is not None
                    and param.numel() > 0
                    and not self.compatible_(param, constraints)
                ):
                    raise ValueError(
                        f"constraint is incompatible with parameter '{name}'."
                    )

            # add new constraint
            self._constraints[dim] = size
            return

        # removal of constraint
        if dim in self._constraints and size is None:
            del self._constraints[dim]
            return

        # alteration of constraint
        if dim in self._constraints and size is not None:
            # alter stored constraint value
            self._constraints[dim] = size

            # reallocate buffers
            for name in self._constrained_buffers:
                buffer = self.get_buffer(name)
                if buffer is not None and buffer.numel() > 0:
                    rsetattr(
                        self,
                        name,
                        torch.zeros(
                            self.compatible_like(buffer.shape),
                            dtype=buffer.dtype,
                            layout=buffer.layout,
                            device=buffer.device,
                            requires_grad=buffer.requires_grad,
                        ),
                    )

            # reallocate parameters
            for name in self._constrained_parameters:
                param = self.get_parameter(name)

                if param is not None and param.numel() > 0:
                    rsetattr(
                        self,
                        name,
                        nn.Parameter(
                            torch.zeros(
                                self.compatible_like(param.shape),
                                dtype=param.dtype,
                                layout=param.layout,
                                device=param.device,
                                requires_grad=param.data.requires_grad,
                            ),
                            requires_grad=param.requires_grad,
                        ),
                    )
            return

    def register_constrained(self, name: str):
        """Registers an existing buffer or parameter as constrained.

        Args:
            name (str): fully-qualified string name of the buffer or
                parameter to register.

        Raises:
            RuntimeError: dimension of attribute does not match constraint.
            AttributeError: attribute is not a registered buffer or parameter.

        Note:
            Registered parameters of type None cannot be constrained.
        """
        # calculate required number of dimensions
        req_ndims = self.mindims

        # attempts to register buffer
        try:
            buffer = self.get_buffer(name)
        except AttributeError:
            pass
        else:
            if (
                buffer is not None
                and buffer.numel() > 0
                and not self.compatible(buffer)
            ):
                raise RuntimeError(
                    f"buffer '{name}' has shape of {tuple(buffer.shape)} "
                    f"incompatible with constrained shape {self.constraints_repr}, "
                    "dimensions must match and must have at least "
                    f"{req_ndims} dimensions"
                )
            else:
                self._constrained_buffers.add(name)
            return

        # attempts to register parameter
        try:
            param = self.get_parameter(name)
        except AttributeError:
            pass
        else:
            if param is not None and param.numel() > 0 and not self.compatible(param):
                raise RuntimeError(
                    f"parameter '{name}' has shape of {tuple(param.shape)} "
                    f"incompatible with constrained shape {self.constraints_repr}, "
                    "dimensions must match and must have at least "
                    f"{req_ndims} dimensions"
                )
            else:
                self._constrained_parameters.add(name)
            return

        raise AttributeError(
            f"`name` '{name}' does not specify a registered buffer or parameter."
        )

    def deregister_constrained(self, name: str):
        """Deregisters a buffer or parameter as constrained.

        Args:
            name (str): fully-qualified string name of the buffer or
                parameter to register.
        """
        # remove if in buffers
        if name in self._constrained_buffers:
            self._constrained_buffers.remove(name)

        # remove if in parameters
        if name in self._constrained_parameters:
            self._constrained_parameters.remove(name)


class HistoryModule(DimensionalModule):
    """Module which keeps track of previous attribute values.

    Args:
        step_time (float): length of time between stored values in history.
        history_len (float): length of time over which prior values are stored.

    Raises:
        ValueError: step time must be positive.
        ValueError: history length must be nonnegative.
    """

    def __init__(
        self,
        step_time: float,
        history_len: float,
    ):
        # ensure valid step time and history length parameters
        step_time, e = numeric_limit("step_time", step_time, 0, "gt", float)
        if e:
            raise e
        history_len, e = numeric_limit("history_len", history_len, 0, "gte", float)
        if e:
            raise e

        # calculate size of time dimension
        history_size = math.ceil(history_len / step_time) + 1

        # call superclass constructor
        DimensionalModule.__init__(self, (-1, history_size))

        # register extras
        self.register_extra("_pointer", dict())
        self.register_extra("_step_time", step_time)
        self.register_extra("_history_len", history_len)

    @property
    def dt(self) -> float:
        r"""Length of time between stored values in history.

        Args:
            value (float): new time step length.

        Returns:
            float: length of the time step.

        Note:
            In the same units as :py:attr:`self.hlen`.
        """
        return self._step_time

    @dt.setter
    def dt(self, value: float) -> None:
        # cast value as float
        value = float(value)

        # ensure valid step time
        if value <= 0:
            raise RuntimeError(f"step time must be positive, received {value}")

        # compute revised time dimension size
        hsize = math.ceil(self.hlen / value) + 1

        # reconstrain if required
        if hsize != self.hsize:
            self.reconstrain(-1, hsize)

        # set revised step time
        self._step_time = value

    @property
    def hlen(self) -> float:
        r"""Length of time over which prior values are stored.

        Args:
            value (float): new length of the history to store.

        Returns:
            float: length of the history.

        Note:
            In the same units as :py:attr:`self.dt`.
        """
        return self._history_len

    @hlen.setter
    def hlen(self, value: float) -> None:
        # cast value as float
        value = float(value)

        # ensure valid history length
        if value < 0:
            raise RuntimeError(f"history length must be nonnegative, received {value}")

        # compute revised time dimension size
        hsize = math.ceil(value / self.dt) + 1

        # reconstrain if required
        if hsize != self.hsize:
            self.reconstrain(-1, hsize)

        # set revised history length
        self._history_len = value

    @property
    def hsize(self) -> int:
        r"""Number of stored time slices for each history tensor.

        Returns:
            float: length of the history, in units of time.
        """
        return self.constraints.get(-1)

    def tick(self, name: str) -> None:
        r"""Manually increment the time, by the step time, for a specified attribute.

        Args:
            name (str): attribute for which time should be incremented.
        """
        if name in self._pointer:
            self._pointer[name] = (self._pointer[name] + 1) % self.hsize
        else:
            warnings.warn(
                f"'{name}' has not correctly been registered as a constrained "
                "attribute, call ignored.",
                category=RuntimeWarning,
            )

    def register_constrained(self, name: str):
        r"""Sets a registered buffer or parameter as constrained.

        Args:
            name (str): attribute to constrain.

        Note:
            This implies the attribute will have its history tracked, and therefore
            must have a final dimension of size :py:attr:`hsize`. Because this is
            implemented using :py:class:`DimensionalModule` constraints, it must meet
            any other constraints that have been added.
        """
        DimensionalModule.register_constrained(self, name)
        if name not in self._pointer:
            self._pointer[name] = 0

    def deregister_constrained(self, name: str):
        r"""Sets a registered buffer or parameter as unconstrained.

        Args:
            name (str): attribute to constrain.

        Note:
            This implies the attribute will not have its history tracked. Because
            this is implemented using :py:class:`DimensionalModule` constraints, it
            will also be freed of any other constraints.
        """
        DimensionalModule.deregister_constrained(self, name)
        if name in self._pointer:
            del self._pointer[name]

    def select(
        self,
        name: str,
        time: float | torch.Tensor,
        interpolation: Interpolation,
        *,
        tolerance: float = 1e-7,
        offset: int = 1,
    ) -> torch.Tensor:
        r"""Selects elements of a constrained attribute based on prior time.

        Args:
            name (str): name of the attribute to target.
            time (float | torch.Tensor): time before present to select from.
            interpolation (Interpolate): method to interpolate between discrete
                time steps.
            tolerance (float, optional): maximum difference in time from a discrete
                sample to onsider it at the same time as that sample. Defaults to 1e-7.
            offset (int, optional): window index offset, number of :py:meth:`tick`
                calls back. Defaults to 1.

        Returns:
            torch.Tensor: interpolated tensor selected at a prior time.

        .. admonition:: Shape
            :class: tensorshape

            ``time``:

            :math:`N_0 \times \cdots \times [D]`

            ``return``:

            :math:`N_0 \times \cdots \times [D]`

            Where:
                * :math:`N_0, \ldots` are the dimensions of the constrained tensor.
                * :math:`D` is the number of times for each value to select.

        Note:
            The constraints on the shape of ``time`` are not enforced, and follows
            the underlying logic of :py:func:`torch.gather`.

        Note:
            By default, `offset` is set to `1`. This is the correct configuration to
            use under normal circumstances where :py:meth:`pushto` is used for element
            insertion. Also this is useful for when :py:meth:`tick` is called after a
            call to :py:meth:`insert` and before :py:meth:`select`. This should be set
            to `0` if :py:meth:`tick` has not been called since the last
            :py:meth:`insert`.

        Note:
            The argument `interpolate` is a function which takes in four
            arguments, as follows.

                * tensor: nearest observed state before the selected time.
                * tensor: nearest observed state after the selected time.
                * tensor: time after the "before state" for which results should be produced.
                * float: difference in time between the before and after state.

            It must return a tensor of values interpolated between the samples
            at the two times. Some functions which meet the :py:class:`Interpolation`
            type are included in the library.

        Note:
            If ``time`` is a scalar and is within tolerance of an integer index, then
            a slice will be returned without ever attempting interpolation.

            If ``time`` is a tensor, interpolation will be called regardless, and
            the time passed into the interpolation call will be set to either ``0``
            or :py:attr:`self.dt`. Interpolation results are then overwritten with
            exact values before returning.
        """
        # access underlying buffer or parameter
        if name in self._constrained_buffers:
            data = self.get_buffer(name)
        elif name in self._constrained_parameters:
            data = self.get_parameter(name)
        else:
            raise AttributeError(
                f"name {name} does not specify a constrained buffer or parameter."
            )

        # check that constrained is registered via HistoryModule
        if name not in self._pointer:
            raise RuntimeError(
                f"name {name} references an improperly registered constrained attribute."
            )

        # check that constrained is initialized
        if data is None or data.numel() == 0:
            raise AttributeError(
                f"name {name} references an uninitialized constrained attribute."
            )

        # computed values
        tmax = self.dt * (self.hsize - 1)

        # scalar time
        if not isinstance(time, torch.Tensor):
            # cast values
            time, offset = float(time), int(offset)

            # check that time is in valid range
            if time + tolerance < 0:
                raise ValueError(f"time must be nonnegative, received {time}.")
            if time - tolerance > tmax:
                raise ValueError(f"time must not exceed {tmax}, received {time}.")

            # convert time into continuous index
            index = time / self.dt
            r_index = round(index)
            if abs(self.dt * r_index - time) < tolerance:
                index = r_index

            # apply offset to pointer
            pointer = (self._pointer[name] - offset) % self.hsize

            # access data by index and interpolate
            if isinstance(index, int):
                return data[(pointer - index) % self.hsize]
            else:
                prev_data = data[..., (pointer - int(index + 1)) % self.hsize]
                next_data = data[..., (pointer - int(index)) % self.hsize]
                return interpolation(
                    prev_data.unsqueeze(-1),
                    next_data.unsqueeze(-1),
                    torch.full(
                        data.shape[:-1],
                        self.dt * (index % 1),
                        dtype=data.dtype,
                        device=data.device,
                    ).unsqueeze(-1),
                    self.dt,
                ).squeeze(-1)

        # tensor time
        else:
            # ensure time is of correct datatype and on correct device
            time = time.to(device=data.device)
            if not time.is_floating_point():
                time = time.to(dtype=torch.float32)

            # determine if output dimension should be squeezed
            if time.ndim == data.ndim - 1:
                squeeze_res = True
                time = time.unsqueeze(-1)
            elif time.ndim == data.ndim:
                squeeze_res = False
            else:
                raise RuntimeError(
                    f"`time` has incompatible number of dimensions {time.ndim}, "
                    "must have number of dimensions equal to "
                    f"{data.ndim} or {data.ndim - 1}."
                )

            # check that time values are in valid range
            if torch.any(time + tolerance < 0):
                raise ValueError(
                    f"`time` must only be nonnegative, received {time.amin().item()}."
                )
            if torch.any(time - tolerance > tmax):
                raise ValueError(
                    f"`time` must never exceed {tmax}, received {time.amax().item()}."
                )

            # convert time into continuous indices
            indices = time / self.dt
            r_indices = indices.round()
            indices = torch.where(
                (self.dt * r_indices - time).abs() < tolerance,
                r_indices,
                indices,
            )

            # apply offset to pointer
            pointer = (self._pointer[name] - offset) % self.hsize

            # access data by index and interpolate
            prev_idx = (pointer - indices.ceil().long()) % self.hsize
            prev_data = torch.gather(
                data,
                -1,
                prev_idx,
            )

            next_idx = (pointer - indices.floor().long()) % self.hsize
            next_data = torch.gather(
                data,
                -1,
                next_idx,
            )

            res = interpolation(prev_data, next_data, self.dt * (indices % 1), self.dt)

            # use direct values where indices are exact
            res = torch.where(prev_idx == next_idx, prev_data, res)

            # unsqueeze result if necessary to match input
            if squeeze_res:
                return res.squeeze(-1)
            else:
                return res

    def reset(self, name: str, data: Any) -> None:
        r"""Resets a constrained attribute to some value or values.

        Args:
            name (str): name of the attribute to target.
            data (Any): data to insert.

        Raises:
            RuntimeError: cannot reset an uninitialized buffer or parameter.
        """
        if name in self._constrained_buffers:
            buffer = self.get_buffer(name)
            if buffer is None or buffer.numel() == 0:
                raise RuntimeError(f"cannot reset '{name}', buffer is uninitialized.")
            buffer[:] = data
            self._pointer[name] = 0
        elif name in self._constrained_parameters:
            param = self.get_parameter(name)
            if param is None or param.numel() == 0:
                raise RuntimeError(
                    f"cannot reset '{name}', parameter is uninitialized."
                )
            param[:] = data
            self._pointer[name] = 0
        else:
            raise AttributeError(
                f"`name` {name} does not specify a constrained buffer or parameter."
            )

    def pushto(self, name: str, data: torch.Tensor) -> None:
        r"""Inserts a slice at the current time into a constrained attribute and
        advances to the next time.

        Args:
            name (str): name of the attribute to target.
            data (torch.Tensor): data to insert.

        Raises:
            RuntimeError: cannot push to an uninitialized buffer or parameter.
            RuntimeError: data has an incompatible shape with buffer or parameter.
            AttributeError: specified name is not a constrained buffer or parameter.
        """
        if name in self._constrained_buffers:
            buffer = self.get_buffer(name)
            if buffer is None:
                raise RuntimeError(f"cannot push to {name}, buffer is uninitialized.")
            if data.shape != buffer.shape[:-1]:
                raise RuntimeError(
                    f"`data` has a shape of {tuple(data.shape)}, "
                    f"required shape is {tuple(buffer.shape[:-1])}."
                )
            buffer[..., self._pointer[name]] = data
            self.tick(name)
        elif name in self._constrained_parameters:
            param = self.get_parameter(name)
            if param is None:
                raise RuntimeError(
                    f"cannot push to '{name}', parameter is uninitialized."
                )
            if data.shape != param.shape[:-1]:
                raise RuntimeError(
                    f"`data` has a shape of {tuple(data.shape)}, "
                    f"required shape is {tuple(param.shape[:-1])}."
                )
            param[..., self._pointer[name]] = data
            self.tick(name)
        else:
            raise AttributeError(
                f"`name` {name} does not specify a constrained buffer or parameter."
            )

    def update(self, name: str, data: torch.Tensor, offset: int = 0) -> None:
        r"""Updates time slice of a tensor at a specific index.

        Args:
            name (str): name of the attribute to target.
            data (torch.Tensor): data to insert into history.
            offset (int, optional): number of steps before present to update.
                Defaults to 0.
        """
        # access underlying buffer or parameter
        if name in self._constrained_buffers:
            data = self.get_buffer(name)
        elif name in self._constrained_parameters:
            data = self.get_parameter(name)
        else:
            raise AttributeError(
                f"`name` {name} does not specify a constrained buffer or parameter."
            )

        # insert new time slice
        pointer = (self._pointer[name] - int(offset)) % self.hsize
        data[..., pointer] = data

    def latest(self, name: str, offset: int = 1) -> torch.Tensor:
        r"""Retrieves the most recent slice of a constrained attribute.

        Args:
            name (str): name of the attribute to target.
            offset (int, optional): window index offset, number of :py:meth:`tick`
                calls back. Defaults to 1.

        Raises:
            AttributeError: specified name is not a constrained buffer or parameter.

        Returns:
            torch.Tensor: most recent slice of the tensor selected.

        Note:
            By default, `offset` is set to `1`. This is the correct configuration to
            use under normal circumstances where :py:meth:`pushto` is used for element
            insertion. Also this is useful for when :py:meth:`tick` is called after a
            call to :py:meth:`insert` and before :py:meth:`select`. This should be set
            to `0` if :py:meth:`tick` has not been called since the last
            :py:meth:`insert`.
        """
        # access underlying buffer or parameter
        if name in self._constrained_buffers:
            data = self.get_buffer(name)
        elif name in self._constrained_parameters:
            data = self.get_parameter(name)
        else:
            raise AttributeError(
                f"`name` {name} does not specify a constrained buffer or parameter."
            )

        return data[..., (self._pointer[name] - offset) % self.hsize]

    def history(self, name: str, offset: int = 1, latest_first=True) -> torch.Tensor:
        r"""Retrieves the recorded history of a constrained attribute.

        Args:
            name (str): name of the attribute to target.
            offset (int, optional): window index offset, number of :py:meth:`tick`
                calls back. Defaults to 1.
            latest_first (bool, optional): if the most recent sample should be at the
                zeroth index. Defaults to False.

        Returns:
            torch.Tensor: value of the attribute at every observed time.
        """
        # access underlying buffer or parameter
        if name in self._constrained_buffers:
            data = self.get_buffer(name)
        elif name in self._constrained_parameters:
            data = self.get_parameter(name)
        else:
            raise AttributeError(
                f"`name` {name} does not specify a constrained buffer or parameter."
            )

        # sorted latest last
        data = torch.roll(data, offset - self._pointer[name] - 1, -1)

        # reverse if required
        if latest_first:
            data = data.flip(-1)

        return data
