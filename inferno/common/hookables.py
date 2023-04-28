from abc import ABC, abstractmethod
from typing import Optional, Any
import warnings

import torch.nn as nn


class PreHookable(nn.Module, ABC):
    """Abstract class which provides forward pre-hook functionality for subclasses.

    `PreHookable` is an abstract class which provides functionality to register and deregister itself as
    forward pre-hook with a :py:class:`torch.nn.Module` object. This is performed using :py:meth:`torch.nn.Module.register_forward_pre_hook`
    to register itself as a forward pre-hook and it manages the returned :py:class:`torch.utils.hooks.RemovableHandle` to deregister itself.

    .. note:
        A forward pre-hook, and consequently `PreHookable`, is invoked automatically when the :py:class:`torch.nn.Module` is called and
        *before* :py:meth:`torch.nn.Module.forward` is executed.

    Args:
        module (Optional[torch.nn.Module], optional): PyTorch module to which the forward pre-hook will be registered. Defaults to `None`.

    Raises:
        TypeError: if parameter `module` is not `None`, then it must be an instance of :py:class:`torch.nn.Module`.
    """
    def __init__(self, module: Optional[nn.Module] = None):
        # call superclass constructor
        nn.Module.__init__(self)

        # set returned handle
        self.handle = None

        # register with module if provided
        if module is not None:
            self.register(module)

    @abstractmethod
    def forward(self, module: nn.Module, inputs: Any) -> Optional[Any]:
        """Defines the computation performed every time associated :py:class:`torch.nn.Module` is called.

        Args:
            module (torch.nn.Module): PyTorch module to which the forward pre-hook is registered.
            inputs (Any): inputs passed to the :py:class:`torch.nn.Module` call.

        Raises:
            NotImplementedError: py:meth:`forward` is abstract and must be implemented by the subclass.

        Returns:
            Optional[Any]: Modified inputs to be passed to :py:meth:`torch.nn.Module.forward` iff not `None`.
        """
        raise NotImplementedError(f"'PreHookable.forward()' is abstract, {type(self).__name__} must implement the 'forward()' method")

    def register(self, module: nn.Module) -> None:
        """Registers the `PreHookable` as a forward pre-hook with specified :py:class:`torch.nn.Module`.

        Args:
            module (torch.nn.Module): PyTorch module to which the forward pre-hook will be registered.

        Raises:
            TypeError: parameter `module` must be an instance of :py:class:`torch.nn.Module`

        Warns:
            RuntimeWarning: each `PreHookable` can only be registered to one :py:class:`torch.nn.Module` and will ignore :py:meth:`register` if already registered.
        """
        if not self.handle:
            if not isinstance(module, nn.Module):
                raise TypeError(f"'module' parameter of type {type(self).__name__} must be an instance of {nn.Module}")
            self.handle = module.register_forward_pre_hook(self)
        else:
            warnings.warn(f"this {type(self).__name__} object is already registered to an object so new register() was ignored", category=RuntimeWarning)

    def deregister(self) -> None:
        """Deregisters the `PreHookable` as a forward pre-hook from registered :py:class:`torch.nn.Module`, iff it is already registered.
        """
        if self.handle:
            self.handle.remove()
            self.handle = None


class PostHookable(nn.Module, ABC):
    """Abstract class which provides forward hook functionality for subclasses.

    `PostHookable` is an abstract class which provides functionality to register and deregister itself as
    forward hook with a :py:class:`torch.nn.Module` object. This is performed using :py:meth:`torch.nn.Module.register_forward_hook`
    to register itself as a forward hook and it manages the returned :py:class:`torch.utils.hooks.RemovableHandle` to deregister itself.

    .. note:
        A forward hook, and consequently `PostHookable`, is invoked automatically when the :py:class:`torch.nn.Module` is called but
        *after* :py:meth:`torch.nn.Module.forward` is executed.

    Args:
        module (Optional[torch.nn.Module], optional): PyTorch module to which the forward hook will be registered. Defaults to `None`.

    Raises:
        TypeError: if parameter `module` is not `None`, then it must be an instance of :py:class:`torch.nn.Module`.
    """
    def __init__(self, module: Optional[nn.Module] = None):

        # call superclass constructor
        nn.Module.__init__(self)

        # set returned handle
        self.handle = None

        # register with module if provided
        if module is not None:
            self.register(module)

    @abstractmethod
    def forward(self, module: nn.Module, inputs: Any, outputs: Any) -> Optional[Any]:
        """Defines the computation performed every time associated :py:class:`torch.nn.Module` is called.

        Args:
            module (nn.Module): PyTorch module to which the forward hook is registered.
            inputs (Any): inputs passed to the :py:class:`torch.nn.Module` call.
            outputs (Any): results from the :py:meth:`torch.nn.Module.forward` call.

        Raises:
            NotImplementedError: py:meth:`forward` is abstract and must be implemented by the subclass.

        Returns:
            Optional[Any]: Modified outputs to be returned from the :py:meth:`torch.nn.Module` call iff not `None`.
        """
        raise NotImplementedError(f"'PostHookable.forward()' is abstract, {type(self).__name__} must implement the 'forward()' method")

    def register(self, module: nn.Module) -> None:
        """Registers the `PostHookable` as a forward hook with specified :py:class:`torch.nn.Module`.

        Args:
            module (torch.nn.Module): PyTorch module to which the forward hook will be registered.

        Raises:
            TypeError: parameter `module` must be an instance of :py:class:`torch.nn.Module`

        Warns:
            RuntimeWarning: each `PostHookable` can only be registered to one :py:class:`torch.nn.Module` and will ignore :py:meth:`register` if already registered.
        """
        if not self.handle:
            if not isinstance(module, nn.Module):
                raise TypeError(f"'module' parameter of type {type(self).__name__} must be an instance of {nn.Module}")
            self.handle = module.register_forward_hook(self)
        else:
            warnings.warn(f"this {type(self).__name__} object is already registered to an object so new register() was ignored", category=RuntimeWarning)

    def deregister(self) -> None:
        """Deregisters the `PostHookable` as a forward hook from registered :py:class:`torch.nn.Module`, iff it is already registered.
        """
        if self.handle:
            self.handle.remove()
            self.handle = None
