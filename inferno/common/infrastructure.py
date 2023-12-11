import attrs
from collections import OrderedDict
from collections.abc import Mapping
import torch
import torch.nn as nn
from typing import Any, Callable
import warnings


class Module(nn.Module):
    r"""An extension of `PyTorch's Module
    <https://pytorch.org/docs/master/generated/torch.nn.Module.html#torch.nn.Module>`_ class.

    This extends :py:class:`torch.nn.Module` so that "extra state" is handled in a way similar to regular
    tensor state (e.g. buffers and parameters). This enables simple export/import to/from a state dictionary.

    Note:
        Like with :py:class:`torch.nn.Module`, an :py:meth:`__init__` call must be made to the parent
        class before assignment on the child. This class's constructor will automatically call PyTorch's.
    """

    def __init__(self, *args, **kwargs):
        nn.Module.__init__(self, *args, **kwargs)
        self._extras = OrderedDict()

    def register_extra(self, name: str, value: Any):
        r"""Adds an extra variable to the module.

        This is typically used in a manner to :py:meth:`~torch.nn.Module.register_buffer`, except that the value being
        registered is not limited to being a :py:class:`~torch.Tensor`.

        Args:
            name (str): name of the extra, which can be accessed from this module using the provided name.
            value (Any): extra to be registered.

        Raises:
            TypeError: if the extra variable being registered is an instance of :py:class:`torch.Tensor`
                or :py:class:`torch.nn.Module`.
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
            AttributeError: if the target string references an invalid path, the terminal module is
                an instance of :py:class:`torch.nn.Module` but not :py:class:`Module`, or resolves to
                something that is not an extra.
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


class WrapperModule(Module):
    r"""Module which translucently wraps a submodule.

    Attributes or methods which have no definition on the parent will instead be called
    on the primary submodule.

    Args:
        submodule (Module): module which is being wrapped.

    Raises:
        TypeError: ``submodule`` cannot be an instance of :py:class:`WrapperModule`.

    Note:
        Some methods, such as :py:class:`~torch.nn.Module.forward`, are defined on the parent even when
        not overridden, and as such are not made transparent. Additionally assignment cannot be performed
        on the submodule.
    """

    def __init__(self, submodule):
        nn.Module.__init__(self)
        if isinstance(submodule, WrapperModule):
            raise TypeError(
                f"`submodule` of type {type(submodule)} "
                "cannot be an instance of WrapperModule"
            )
        self.submodule = submodule

    def __getattr__(self, name):
        if name != "submodule":
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.submodule, name)
        return super().__getattr__(name)


class Configuration(Mapping):
    r"""Class which provides unpacking functionality when used in conjunction with
    the `attrs library <https://www.attrs.org/en/stable/>`_.

    When defining configuration classes which are to be wrapped by :py:func:`attrs.define`, if
    this is subclassed, then it can be unpacked with ``**``.

    .. automethod:: _asadict_
    """

    def _asadict_(self) -> dict[str, Any]:
        r"""Controls how the fields of this class are convereted into a dictionary.

        This will flatten any nested :py:class:`Configuration` objects using their own
        :py:meth:`_asadict_` method. If there are naming conflicts (i.e. if a nested configuration has)
        a field with the same name, only one will be preserved. This can be overridden to change its behavior.

        Returns:
            dict[str, Any]: dictionary of field names to the objects they represent.

        Note:
            This only packages those attributes which were registered via :py:func:`attrs.field`.
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
        prehook (Callable | None, optional): function to call before registrant's
            :py:meth:`~torch.nn.Module.forward`. Defaults to None.
        posthook (Callable | None, optional): function to call after registrant's
            :py:meth:`~torch.nn.Module.forward`. Defaults to None.
        prehook_kwargs (dict[str, Any] | None, optional): keyword arguments passed to
            :py:meth:`~torch.nn.Module.register_forward_pre_hook`. Defaults to None.
        posthook_kwargs (dict[str, Any] | None, optional): keyword arguments passed to
            :py:meth:`~torch.nn.Module.register_forward_hook`. Defaults to None.
        train_update (bool, optional): if the hooks should be run when registrant is
            in train mode. Defaults to True.
        eval_update (bool, optional): if the hooks should be run when registrant is
            in eval mode. Defaults to True.
        module (nn.Module, optional): PyTorch module to which the forward hook will
            be registered. Defaults to `None`.

    Note:
        If not None, the signature of the prehook must be of the following form.
        .. signature::
            hook(module, args) -> None or modified input

        See :py:meth:`torch.nn.Module.register_forward_pre_hook` for further information.

    Note:
        If not None, the signature of the posthook must be of the following form.
        .. signature::
            hook(module, args, output) -> None or modified output

        See :py:meth:`torch.nn.Module.register_forward_hook` for further information.


    Raises:
        RuntimeError: at least one of ``prehook`` and ``posthook`` must not be None.
        RuntimeError: at least one of ``train_update`` and ``eval_update`` must not be True.
        TypeError: if parameter `module` is not `None`, then it must be an instance of :py:class:`torch.nn.Module`.
    """

    def __init__(
        self,
        prehook: Callable | None = None,
        posthook: Callable | None = None,
        prehook_kwargs: dict[str, Any] | None = None,
        posthook_kwargs: dict[str, Any] | None = None,
        train_update: bool = True,
        eval_update: bool = True,
        module: nn.Module | None = None,
    ):
        # check if at least one callable is defined
        if not prehook and not posthook:
            raise RuntimeError(
                "at least one of `prehook` and `posthook` must not be None"
            )

        # check if at least one update is true
        if not train_update and not eval_update:
            raise RuntimeError(
                "at least one of `train_update` and `eval_update` must not be True"
            )

        # set returned handle
        self._prehook_handle = None
        self._posthook_handle = None

        # set hook registering kwargs
        self._prehook_kwargs = prehook_kwargs if prehook_kwargs else {}
        self._posthook_kwargs = posthook_kwargs if posthook_kwargs else {}

        # set hooks
        if not train_update and eval_update:
            if self._prehook_kwargs.get("with_kwargs"):
                if train_update:
                    self._prehook_fn = (
                        lambda module, args, kwargs: None
                        if not module.training
                        else prehook(module, args, kwargs)
                    )
                if eval_update:
                    self._prehook_fn = (
                        lambda module, args, kwargs: None
                        if module.training
                        else prehook(module, args, kwargs)
                    )
            else:
                if train_update:
                    self._prehook_fn = (
                        lambda module, args: None
                        if not module.training
                        else prehook(module, args)
                    )
                if eval_update:
                    self._prehook_fn = (
                        lambda module, args: None
                        if module.training
                        else prehook(module, args)
                    )
        else:
            self._prehook_fn = prehook

        if not train_update and eval_update:
            if self._posthook_kwargs.get("with_kwargs"):
                if train_update:
                    self._posthook_fn = (
                        lambda module, args, kwargs: None
                        if not module.training
                        else posthook(module, args, kwargs)
                    )
                if eval_update:
                    self._posthook_fn = (
                        lambda module, args, kwargs: None
                        if module.training
                        else posthook(module, args, kwargs)
                    )
            else:
                if train_update:
                    self._posthook_fn = (
                        lambda module, args: None
                        if not module.training
                        else posthook(module, args)
                    )
                if eval_update:
                    self._posthook_fn = (
                        lambda module, args: None
                        if module.training
                        else posthook(module, args)
                    )
        else:
            self._posthook_fn = posthook

        # register with module if provided
        if module is not None:
            self.register(module)

    def register(self, module: nn.Module) -> None:
        """Registers the `Hook` as a forward hook/prehook with specified :py:class:`torch.nn.Module`.

        Args:
            module (nn.Module): PyTorch module to which the forward hook will be registered.

        Raises:
            TypeError: parameter `module` must be an instance of :py:class:`nn.Module`

        Warns:
            RuntimeWarning: each `Hook` can only be registered to one :py:class:`torch.nn.Module`
            and will ignore :py:meth:`register` if already registered.
        """
        if not self._prehook_handle or not self._posthook_handle:
            if not isinstance(module, nn.Module):
                raise TypeError(
                    f"'module' parameter of type {type(self).__name__} must be an instance of {nn.Module}"
                )
            if self._prehook_fn:
                self._prehook_handle = module.register_forward_pre_hook(
                    self, **self._prehook_kwargs
                )
            if self._posthook_fn:
                self._posthook_handle = module.register_forward_hook(
                    self, **self._posthook_kwargs
                )
        else:
            warnings.warn(
                f"this {type(self).__name__} object is already registered to an object so new register() was ignored",
                category=RuntimeWarning,
            )

    def deregister(self) -> None:
        """Deregisters the `Hook` as a forward hook/prehook from registered :py:class:`torch.nn.Module`,
        iff it is already registered."""
        if self._prehook_handle:
            self._prehook_handle.remove()
            self._prehook_handle = None
        if self._posthook_handle:
            self._posthook_handle.remove()
            self._posthook_handle = None
