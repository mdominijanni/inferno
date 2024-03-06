from __future__ import annotations
from .._internal import argtest
from abc import ABC, abstractmethod
import attrs
from collections import OrderedDict
from collections.abc import Mapping
from functools import cache
from itertools import chain
import torch
import torch.nn as nn
from typing import Any, Callable
import warnings
import weakref


class Module(nn.Module):
    r"""An extension of PyTorch's Module class.

    This extends :py:class:`torch.nn.Module` so that "extra state" is handled in a way
    similar to regular tensor state (e.g. buffers and parameters). This enables simple
    export to and import from a state dictionary.

    Unlike PyTorch's native :py:class:`~torch.nn.Module`, the overrides for
    :py:meth:`__getattr__`, :py:meth:`__setattr__`, and :py:meth:`__delattr__` will
    check if an attribute is a property on the class and if it is, will use the base
    :py:class:`object` methods.

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
            TypeError: extra name has to be a string.
            KeyError: extra name cannot be the empty string and cannot contain ".".
            TypeError: extras cannot be instances of :py:class:`~torch.Tensor` or
                :py:class:`~torch.nn.Module`.

        Important:
            In order to be accessed with dot notation, the name must be a valid
            Python identifier.

        Note:
            :py:class:`~torch.Tensor`, :py:class:`~torch.nn.Parameter`, and
            :py:class:`~torch.nn.Module` objects cannot be registered as extras and
            should be registered using existing methods.
        """
        if not isinstance(name, str):
            raise TypeError(
                f"extra name must be a string, received {type(name).__name__}"
            )
        elif "." in name:
            raise KeyError("extra name cannot contain '.'")
        elif name == "":
            raise KeyError("extra name cannot be empty string ''")
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
                f"{module.__class__.__name__} " "is not an inferno Module"
            )
        if not hasattr(module, extra_name):
            raise AttributeError(
                f"{module.__class__.__name__} " f"has no attribute '{extra_name}'"
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
        if isinstance(getattr(type(self), name, None), property):
            return object.__getattr__(self, name)
        else:
            if "_extras" in self.__dict__:
                _extras = self.__dict__["_extras"]
                if name in _extras:
                    return _extras[name]

            return super().__getattr__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(getattr(type(self), name, None), property):
            object.__setattr__(self, name, value)
        else:
            _extras = self.__dict__.get("_extras")
            if _extras is not None and name in _extras:
                _extras[name] = value
            else:
                super().__setattr__(name, value)

    def __delattr__(self, name) -> None:
        if isinstance(getattr(type(self), name, None), property):
            object.__delattr__(self, name)
        else:
            if name in self._extras:
                del self._extras[name]
            else:
                super().__delattr__(name)

    def __dir__(self):
        module_attrs = dir(self.__class__)
        attrs = list(self.__dict__.keys())
        parameters = list(self._parameters.keys())
        modules = list(self._modules.keys())
        buffers = list(self._buffers.keys())
        extras = list(self._extras.keys())
        keys = module_attrs + attrs + parameters + modules + buffers + extras

        keys = [key for key in keys if key.isidentifier()]

        return sorted(keys)


class DimensionalModule(Module):
    def __init__(self, *constraints: tuple[int, int], live: bool = False):
        # call superclass constructor
        Module.__init__(self)

        # register state
        self.__constraints = dict()
        self.__constrained_buffers = set()
        self.__constrained_params = set()
        self.__live_assert = True

        # cached values
        self.__constraint_cache = cache(self.__calc_constraints)
        self.__extra_repr_cache = cache(self.__calc_extra_repr)

        # check for consistent constraints
        for d, s in constraints:
            dim, size = int(d), int(s)
            if size < 1:
                raise ValueError(
                    f"constraint {(d, s)} specifies an invalid (nonpositive) "
                    "number of elements"
                )
            if dim in self.__constraints:
                raise RuntimeError(
                    f"constraint {(dim, size)} conflicts with constraint "
                    f"{(dim, self.__constraints[dim])}."
                )
            self.__constraints[dim] = size

    def __setattr__(self, name: str, value: Any) -> None:
        if self.live and name in self.__constrained:
            pass
        else:
            super().__setattr__(name, value)

    @classmethod
    def dims_(cls, constraints: dict[int, int]) -> int:
        r"""Computes minimum number of required dimensions for a constrained tensor.

        Args:
            constraints (dict[int, int]): constraint dictionary of (dim, size).

        Returns:
            int: minimum required number of dimensions.
        """
        if not constraints:
            return 0

        return max(max(constraints) + 1, 0) - min(min(constraints), 0)

    @classmethod
    def compatible_(cls, tensor: torch.Tensor, constraints: dict[int, int]) -> bool:
        r"""Test if a tensor is compatible with a set of constraints.

        Args:
            tensor (torch.Tensor): value to test.
            constraints (dict[int, int]): constraint dictionary of (dim, size).

        Returns:
            bool: if the tensor is compatible.
        """
        # check if the tensor has the minimum dimensionality
        if tensor.ndim < cls.dims_(constraints):
            return False

        # test all constraints
        for dim, size in constraints.items():
            if tensor.shape[dim] != size:
                return False

        return True

    @property
    def constraints(self) -> dict[int, int]:
        r"""Returns the constraint dictionary, sorted by dimension.

        The results will be sorted by dimension, from first to last. Therefore,
        positive dimensions are presented first, in increasing order, then negative
        dimensions also in increasing order.

        Returns:
            dict[int, int]: active constraints, represented as a dictionary.
        """
        return self.__constraint_cache()

    @property
    def dims(self) -> int:
        r"""Minimum number of required dimensions for a constrained tensor.

        Returns:
            int: minimum required number of dimensions.
        """
        return self.dims_(self.__constraints)

    @property
    def live(self) -> bool:
        r"""If constraints should be enforced on attribute assignment.

        Args:
            value (bool): if constraints should be enforced on attribute assignment.

        Returns:
            bool: if constraints should be enforced on attribute assignment.
        """
        return self.__live_assert

    @live.setter
    def live(self, value: bool) -> None:
        self.__live_assert = bool(value)

    def extra_repr(self) -> str:
        return "\n".join(
            (
                f"constraints=({self.__extra_repr_cache()})",
                f"constrained=({','.join(self.__constrained)})",
            )
        )

    def __calc_constraints(self) -> dict[int, int]:
        r"""Calculates sorted constraints for the cache.

        Returns:
            dict[int, int]: active constraints, represented as a dictionary.
        """
        fwd, rev = [], []
        for dim, size in sorted(self.__constraints.items()):
            rev.append((dim, size)) if dim < 0 else fwd.append((dim, size))

        return dict(fwd + rev)

    def __calc_extra_repr(self) -> str:
        r"""Calculates the extra representation layout of constraints.

        Returns:
            str: extra representation format of constraints.
        """
        # split constraints into forward and reverse (negative) indices, sorted
        fwd, rev = [], []
        for dim, size in sorted(self.__constraints.items()):
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

        return ", ".join(elems)

    def compatible(self, tensor: torch.Tensor) -> bool:
        r"""Test if a tensor is compatible with the constraints.

        Args:
            tensor (torch.Tensor): value to test.

        Returns:
            bool: if the tensor is compatible.
        """
        return self.compatible_(tensor, self.__constraints)

    def reconstrain(self, dim: int, size: int | None) -> DimensionalModule:
        # clear cached values
        self.__constraint_cache.cache_clear()
        self.__extra_repr_cache.cache_clear()

        # remove deleted buffers from constrained set
        for name in tuple(self.__constrained_buffers):
            try:
                _ = self.get_buffer(name)
            except AttributeError:
                self.deregister_constrained(name)

        # remove deleted parameters from constrained set
        for name in tuple(self.__constrained_params):
            try:
                _ = self.get_parameter(name)
            except AttributeError:
                self.deregister_constrained(name)

        # cast and validate arguments
        dim = int(dim)
        size = None if size is None else argtest.gt("size", size, 0, int)

        # removes constraint
        if dim in self.__constraints and not size:
            # deletes constraint (always safe to do so)
            del self.__constraints[dim]

            # ensures constrained values are still valid
            self.validate()

            return self

        # creates constraint
        if dim not in self.__constraints and size:
            # ensures constrained values are still valid
            self.validate()




    def register_constrained(self, name: str) -> None:
        r"""Registers an existing buffer or parameter as constrained.

        Args:
            name (str): fully-qualified string name of the buffer or
                parameter to register.

        Raises:
            RuntimeError: shape of the buffer or parameter is invalid.
            AttributeError: attribute is not a registered buffer or parameter.

        Caution:
            A registered :py:class:`~torch.nn.Parameter` with a value of ``None``
            cannot be constrained as it is not returned by
            :py:meth:`~torch.nn.Module.get_parameter`.
        """
        # attempts to register buffer
        try:
            b = self.get_buffer(name)
        except AttributeError:
            pass
        else:
            if b is not None and b.numel() > 0 and not self.compatible(b):
                raise RuntimeError(
                    f"buffer '{name}' has shape of {tuple(b.shape)} "
                    f"incompatible with constrained shape ({self.__extra_repr_cache()}), "
                    f"dimensions must match and '{name}' must have at least "
                    f"{self.dims} dimensions"
                )
            else:
                self.__constrained_buffers.add(name)
            return

        # attempts to register parameter
        try:
            p = self.get_parameter(name)
        except AttributeError:
            pass
        else:
            if p is not None and p.numel() > 0 and not self.compatible(p):
                raise RuntimeError(
                    f"parameter '{name}' has shape of {tuple(p.shape)} "
                    f"incompatible with constrained shape ({self.__extra_repr_cache()}), "
                    f"dimensions must match and '{name}' must have at least "
                    f"{self.dims} dimensions"
                )
            else:
                self.__constrained_params.add(name)
            return

        # invalid name
        raise AttributeError(
            f"'nam'` ('{name}') does not specify a registered buffer or parameter"
        )

    def deregister_constrained(self, name: str):
        r"""Deregisters a buffer or parameter as constrained.

        If the name given isn't a constrained buffer or parameter, calling this does
        nothing.

        Args:
            name (str): fully-qualified string name of the buffer or
                parameter to register.
        """
        # remove if in buffers
        if name in self.__constrained_buffers:
            self.__constrained_buffers.remove(name)

        # remove if in parameters
        if name in self.__constrained_params:
            self.__constrained_params.remove(name)

    def validate(self) -> None:
        r"""Validates constraints.

        Along with testing constrained buffers and parameters, if a registered
        constrained name no longer points at a buffer or parameter, that name is removed.

        Raises:
            RuntimeError: constrained buffer or parameter is no longer valid.
        """
        # remove deleted buffers from constrained set
        for name in tuple(self.__constrained_buffers):
            try:
                _ = self.get_buffer(name)
            except AttributeError:
                self.deregister_constrained(name)

        # remove deleted parameters from constrained set
        for name in tuple(self.__constrained_params):
            try:
                _ = self.get_parameter(name)
            except AttributeError:
                self.deregister_constrained(name)

        # ensure constrained buffers have valid shape
        for name, b in map(
            lambda n: (n, self.get_buffer(n)), self.__constrained_buffers
        ):
            if b is not None and b.numel() > 0 and not self.compatible(b):
                raise RuntimeError(f"constrained buffer '{name}' is invalid")

        # ensure constrained parameters have valid shape
        for name, p in map(
            lambda n: (n, self.get_parameter(n)), self.__constrained_params
        ):
            if p is not None and p.numel() > 0 and not self.compatible(p):
                raise RuntimeError(f"constrained parameter '{name}' is invalid")


class Hook:
    r"""Provides forward hook and prehook functionality for subclasses.

    `Hook` provides functionality to register and deregister itself as
    forward hook with a :py:class:`~torch.nn.Module` object. This is performed using
    :py:meth:`~torch.nn.Module.register_forward_hook` to register itself as a forward
    hook and it manages the returned :py:class:`~torch.utils.hooks.RemovableHandle`
    to deregister itself.

    Args:
        prehook (Callable | None, optional): function to call before hooked module's
            :py:meth:`~torch.nn.Module.forward`. Defaults to ``None``.
        posthook (Callable | None, optional): function to call after hooked module's
            :py:meth:`~torch.nn.Module.forward`. Defaults to ``None``.
        prehook_kwargs (dict[str, Any] | None, optional): keyword arguments passed to
            :py:meth:`~torch.nn.Module.register_forward_pre_hook`. Defaults to ``None``.
        posthook_kwargs (dict[str, Any] | None, optional): keyword arguments passed to
            :py:meth:`~torch.nn.Module.register_forward_hook`. Defaults to ``None``.
        train_update (bool, optional): if the hooks should be run when hooked module is
            in train mode. Defaults to ``True``.
        eval_update (bool, optional): if the hooks should be run when hooked module is
            in eval mode. Defaults to ``True``.

    Note:
        If not ``None``, the signature of the prehook must be of the following form.

        .. code-block:: python

            hook(module, args) -> None or modified input

        Or, if ``with_kwargs`` is passed as a keyword argument.

        .. code-block:: python

            hook(module, args, kwargs) -> None or modified input

        See :py:meth:`~torch.nn.Module.register_forward_pre_hook` for
        further information.

    Note:
        If not ``None``, the signature of the posthook must be of the following form.

        .. code-block:: python

            hook(module, args, output) -> None or modified output

        Or, if ``with_kwargs`` is passed as a keyword argument.

        .. code-block:: python

            hook(module, args, kwargs, output) -> None or modified output

        See :py:meth:`~torch.nn.Module.register_forward_hook` for further information.

    Raises:
        RuntimeError: at least one of ``prehook`` and ``posthook`` must not be None.
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
        _ = argtest.onedefined(("prehook", prehook), ("posthook", posthook))

        # prehook and posthook functions
        if isinstance(prehook, Callable):
            self.__prehook = prehook
        else:
            self.__prehook = None

        if isinstance(posthook, Callable):
            self.__posthook = posthook
        else:
            self.__posthook = None

        # set returned handle
        self.__prehook_handle = None
        self.__posthook_handle = None

        # set hook registering kwargs
        self.__prehook_kwargs = prehook_kwargs if prehook_kwargs else {}
        self.__posthook_kwargs = posthook_kwargs if posthook_kwargs else {}

        # set training conditionals
        self.__call_train = train_update
        self.__call_eval = eval_update

        # finalizer
        self.__finalizer = weakref.finalize(self, self.deregister)

    def __wrapped_prehook(self, module, *args, **kwargs):
        if self.trainexec and module.training:
            return self.__prehook(module, *args, **kwargs)

        if self.evalexec and not module.training:
            return self.__prehook(module, *args, **kwargs)

    def __wrapped_posthook(self, module, *args, **kwargs):
        if self.trainexec and module.training:
            return self.__posthook(module, *args, **kwargs)

        if self.evalexec and not module.training:
            return self.__posthook(module, *args, **kwargs)

    @property
    def trainexec(self) -> bool:
        """If the hook is called when the module passed in is in training mode.

        Args:
            value (bool): if the hook should be called when the module is training.

        Returns:
            bool: if the hook is called when the module is training.
        """
        return self.__call_train

    @trainexec.setter
    def trainexec(self, value: bool) -> None:
        self.__call_train = value

    @property
    def evalexec(self) -> bool:
        """If the hook is called when the module passed in is in evaluation mode.

        Args:
            value (bool): if the hook should be called when the module is evaluating.

        Returns:
            bool: if the hook is called when the module is evaluating.
        """
        return self.__call_eval

    @evalexec.setter
    def evalexec(self, value: bool) -> None:
        self.__call_eval = value

    @property
    def registered(self) -> bool:
        r"""If there is a module to which this hook is registered

        Returns:
            bool: if a module to which this hook is registred.
        """
        return self.__prehook_handle or self.__posthook_handle

    def register(self, module: nn.Module) -> None:
        r"""Registers the hook as a forward hook and/or prehook.

        Args:
            module (~torch.nn.Module): PyTorch module to which the forward hook
                will be registered.

        Raises:
            TypeError: parameter ``module`` must be an instance of
                :py:class:`~torch.nn.Module`.

        Warns:
            RuntimeWarning: each :py:class:`Hook` can only be registered to one
                :py:class:`~torch.nn.Module` and will ignore :py:meth:`register`
                if already registered.
        """
        if not self.registered:
            _ = argtest.instance("module", module, nn.Module)

            if self.__prehook:
                self.__prehook_handle = module.register_forward_pre_hook(
                    self.__prehook, **self.__prehook_kwargs
                )

            if self.__posthook:
                self.__posthook_handle = module.register_forward_hook(
                    self.__posthook, **self.__posthook_kwargs
                )
        else:
            warnings.warn(
                f"this {type(self).__name__} is already registered to a module "
                "so new `register()` was ignored",
                category=RuntimeWarning,
            )

    def deregister(self) -> None:
        r"""Deregisters the hook as a forward hook and/or prehook.

        If the :py:class:`Hook` is not registered, this is still safe to call.
        """
        if self.__prehook_handle:
            self.__prehook_handle.remove()
            self.__prehook_handle = None

        if self.__posthook_handle:
            self.__posthook_handle.remove()
            self.__posthook_handle = None


class StateHook(Module, Hook, ABC):
    r"""Interactable hook which only acts on module state.

    Args:
        module (nn.Module): module to which the hook should be registered.
        train_update (bool, optional): if the hook should be run when hooked module is
            in train mode. Defaults to ``True``.
        eval_update (bool, optional): if the hook should be run when hooked module is
            in eval mode. Defaults to ``True``.
        as_prehook (bool, optional): if the hook should be run prior to the hooked
            module's :py:meth:`~torch.nn.Module.forward` call. Defaults to ``False``.
        prepend (bool, optional): if the hook should be run prior to the hooked
            module's previously registered forward hooks. Defaults to ``False``.
        always_call (bool, optional): if the hook should be run even if an exception
            occurs, only applies when ``as_prehook`` is ``False``. Defaults to ``False``.

    Note:
        To trigger the hook regardless of the hooked module's training state,
        call the ``StateHook`` object. The hook will not run if it is not registered.

    Note:
        Unlike with :py:class:`Hook`, the ``hook`` here will only be passed a single
        argument (the registered module itself) and any output will be ignored.
    """

    def __init__(
        self,
        module: nn.Module,
        train_update: bool = True,
        eval_update: bool = True,
        *,
        as_prehook: bool = False,
        prepend: bool = False,
        always_call: bool = False,
    ):
        # construct module superclass
        Module.__init__(self)

        # subclass state
        self.__hooked_module = argtest.instance("module", module, nn.Module)

        # construct hook superclass
        if as_prehook:
            Hook.__init__(
                self,
                prehook=self.__wrapped_statehook,
                prehook_kwargs={"prepend": prepend},
                train_update=train_update,
                eval_update=eval_update,
            )
        else:
            Hook.__init__(
                self,
                posthook=self.__wrapped_statehook,
                posthook_kwargs={"prepend": prepend, "always_call": always_call},
                train_update=train_update,
                eval_update=eval_update,
            )

    def __wrapped_statehook(self, module, *args, **kwargs):
        self.hook(module)

    @abstractmethod
    def hook(self, module: nn.Module) -> None:
        r"""Function to be called on the registered module's call.

        Args:
            module (nn.Module): registered module.

        Raises:
            NotImplementedError: ``hook`` must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__}(StateHook) must implement the method 'hook'"
        )

    @property
    def module(self) -> nn.Module:
        r"""Module to which the hook is applied.

        Returns:
            ~torch.nn.Module: module to which the hook is applied.
        """
        return self.__hooked_module

    def register(self) -> None:
        r"""Registers state the hook as a forward hook or prehook."""
        Hook.register(self, self.module)

    def forward(self, force: bool = False, ignore_mode: bool = False) -> None:
        """Executes the hook at any time, by default only when registered.

        Args:
            force (bool, optional): run the hook even if it is unregistered.
                Defaults to ``False``.
            ignore_mode (bool, optional): run the hook even if it the current mode
                would normally prevent execution. Defaults to ``False``.

        Note:
            This will respect if the hooked module, registered or not, is in
            training or evaluation mode (only relevant if manually configured).
        """
        if self.registered or force:
            if ignore_mode:
                self.hook(self.module)
            elif self.trainexec and self.module.training:
                self.hook(self.module)
            elif self.evalexec and not self.module.training:
                self.hook(self.module)


class Configuration(Mapping):
    r"""Class which provides unpacking functionality when used in conjunction with the
    attrs library.

    When defining configuration classes which are to be wrapped by
    :py:func:`attrs.define`, if this is subclassed, then it can be unpacked with ``**``.

    .. automethod:: _asadict_
    """

    def _asadict_(self) -> dict[str, Any]:
        r"""Controls how the fields of this class are convereted into a dictionary.

        This will flatten any nested :py:class:`Configuration` objects using their own
        :py:meth:`_asadict_` method. If there are naming conflicts, i.e. if a nested
        configuration has a field with the same name, only one will be preserved.
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
