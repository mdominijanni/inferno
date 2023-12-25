from inferno._internal import rsetattr
from .math import Interpolation
import attrs
from collections import deque, OrderedDict
from collections.abc import Mapping
from functools import cached_property
import math
import torch
import torch.nn as nn
from typing import Any, Callable
import warnings


class Module(nn.Module):
    r"""An extension of PyTorch's `Module
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

        .. code-block:: python

            hook(module, args) -> None or modified input

        See :py:meth:`torch.nn.Module.register_forward_pre_hook` for further information.

    Note:
        If not None, the signature of the posthook must be of the following form.

        .. code-block:: python

            hook(module, args, output) -> None or modified output

        See :py:meth:`torch.nn.Module.register_forward_hook` for further information.


    Raises:
        RuntimeError: at least one of ``prehook`` and ``posthook`` must not be None.
        RuntimeError: at least one of ``train_update`` and ``eval_update`` must not be True.
        TypeError: if parameter ``module`` is not ``None``, then it must be an instance of :py:class:`torch.nn.Module`.
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
        """Registers the hook as a forward hook/prehook with specified :py:class:`torch.nn.Module`.

        Args:
            module (nn.Module): PyTorch module to which the forward hook will be registered.

        Raises:
            TypeError: parameter ``module`` must be an instance of :py:class:`nn.Module`

        Warns:
            RuntimeWarning: each :py:class:`Hook` can only be registered to one :py:class:`~torch.nn.Module`
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
        """Deregisters the hook as a forward hook/prehook from registered :py:class:`~torch.nn.Module`,
        if it is already registered."""
        if self._prehook_handle:
            self._prehook_handle.remove()
            self._prehook_handle = None
        if self._posthook_handle:
            self._posthook_handle.remove()
            self._posthook_handle = None


class DimensionalModule(Module):
    """Module with support for dimensionally constrained buffers and parameters.

    Args:
        constraints (tuple[int, int]): tuple of (dim, size) dimensional constraints for
            constrained buffers and parameters.

    Raises:
        ValueError: constraints must specify a positive number of elements.
        RuntimeError: no two constraints may share a dimension.

    Note:
        Each argument must be a 2-tuple of integers, where the first element is the dimension to
        which a constraint is applied and the second is the size of that dimension. Dimensions can
        be negative.
    """

    def __init__(
        self,
        *constraints: tuple[int, int],
    ):
        # call superclass constructor
        Module.__init__(self)

        # disallow empty constraints
        if not len(constraints):
            raise RuntimeError("no constraints were specified")

        # register extras
        self.register_extra("_constraints", dict())
        self.register_extra("_constrained_buffers", set())
        self.register_extra("_constrained_parameters", set())

        # check for consistent constraints
        for dim, size in constraints:
            dim, size = int(dim), int(size)
            if size < 1:
                raise ValueError(
                    f"constraint {(dim, size)} specifies an invalid (non-positive) number of elements."
                )
            if dim in self._constraints:
                raise RuntimeError(
                    f"constraint {(dim, size)} conflicts with constraint {dim, self._constraints[dim]}."
                )
            self._constraints[dim] = size

    @cached_property
    def constraints(self) -> dict[int, int]:
        """Returns the constraint dictionary, sorted by dimension.

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
        """Returns a string representation of constraints.

        Returns:
            str: active constraints, represented as a string.

        Note:
            Like with :py:meth:`constraints`, dimensions are sorted from low to high.
            Underscores represent dimensions which must be present in the constrained
            tensor but with an unspecified value.
        """
        # split constraints into forward and reverse (negative) indices, sorted
        fwd, rev = [], []
        for dim, size in self.constraints:
            rev.append((dim, size)) if dim < 0 else fwd.append((dim, size))

        # representation elements
        elems = []

        # forward indexed constraints
        # expect dimension 0
        expc = 0
        for dim in fwd:
            # add unconstrained placeholders
            elems.extend(["_" for _ in range(dim - expc)])
            # add contraint value
            elems.append(f"{self._contraints[dim]}")
            # set expected next dimension
            expc = dim + 1

        # aribtrary separation
        elems.append("...")

        # reverse indexed constraints
        # no expected dimension
        expc = None
        for dim in rev:
            # add unconstrained placeholders
            if expc is not None:
                elems.extend(["_" for _ in range(dim - expc)])
            # add contraint value
            elems.append(f"{self._contraints[dim]}")
            # set expected next dimension
            expc = dim + 1

        return f"({', '.join(elems)})"

    def compatible(
        self, value: torch.Tensor, constraints: dict[int, int] | None = None
    ) -> bool:
        """Test if a tensor is compatible with the module's constraints.

        Args:
            value (torch.Tensor): tensor to test.
            constraints (dict[int, int] | None, optional): constraint dictionary to test with,
                uses current constraints if None. Defaults to None.

        Returns:
            bool: if the tensor is compatible.
        """
        # select constraints
        if constraints is None:
            constraints = self._constraints

        # compute necessary minimum number of dimensions
        maxc = max(constraints)
        minc = min(constraints)

        # check if value has fewer than minimum required number of dimensions
        if value.ndim < (maxc + 1 if maxc >= 0 else 0) - (minc if minc <= -1 else 0):
            return False

        # check if constraints are met
        for dim, size in constraints:
            if value.shape[dim] != size:
                return False

        return True

    def compatible_like(
        self,
        shape: tuple[int],
        add_dims: bool = False,
        constraints: dict[int, int] | None = None,
    ) -> tuple[int]:
        """Generates a shape like the input, but compatible with constraints.

        Args:
            shape (tuple[int]): shape to make compatible
            add_dims (bool, optional): if constrained dimensions should be added. Defaults to False.
            constraints (dict[int, int] | None, optional): constraint dictionary to test with,
                uses current constraints if None. Defaults to None.

        Raises:
            RuntimeError: without adding dimensions, the dimensionality of shape is insufficient.
            RuntimeError: even with adding dimensions, the dimensionality of shape is insufficient.

        Returns:
            tuple[int]: compatiblized shape.
        """
        # select constraints
        if constraints is None:
            constraints = self._constraints

        # modify existing dimensions
        if not add_dims:
            # ensure shape is of sufficient dimensionality
            maxc = max(constraints)
            minc = min(constraints)
            req_ndims = (maxc + 1 if maxc >= 0 else 0) - (minc if minc <= -1 else 0)

            if len(shape) < req_ndims:
                raise RuntimeError(
                    f"shape {shape} with dimensionality {len(shape)} cannot be made compatible, "
                    f"requires a minimum dimensionality of {req_ndims}"
                )

            # create new shape
            return tuple(
                size if dim not in constraints else constraints[dim]
                for dim, size in enumerate(shape)
            )

        # create new dimensions
        else:
            # construct minimal forward shape with placeholders
            fwd, expc = deque(), 0
            for dim, size in sorted(self._constraints.items()):
                if dim >= 0:
                    fwd.extend([None for _ in range(expc, dim)] + [size])
                    expc = dim + 1

            # construct minimal reverse shape with placeholders
            rev, expc = deque(), -1
            for dim, size in sorted(self._constraints.items(), reverse=True):
                if dim <= -1:
                    rev.extendleft([None for _ in range(dim, expc)] + [size])
                    expc = dim - 1

            # ensure shape is of sufficient dimensionality
            req_ndims = fwd.count(None) + rev.count(None)

            if len(shape) < req_ndims:
                raise RuntimeError(
                    f"shape {shape} with dimensionality {len(shape)} cannot be made compatible, "
                    f"requires a minimum dimensionality of {req_ndims}"
                )

            # create new shape
            shape = deque(shape)
            fwd = [size if size is not None else shape.popleft() for size in fwd]
            rev = reversed(
                [size if size is not None else shape.pop() for size in reversed(rev)]
            )
            return fwd + list(shape) + rev

    def reconstrain(self, dim: int, size: int | None):
        """Edits existing constraints and reshapes constrained buffers and parameters accordingly.

        Args:
            dim (int): dimension to which a constraint should be added, removed, or modified.
            size (int | None): size of the new constraint, or None if the constraint should be removed.

        Raises:
            RuntimeError: constrained buffer or parameter had its shape modified externally and is no longer compatible.
            ValueError: size must specify a positive number of elements.
            RuntimeError: added constraint is incompatible with existing buffer or parameter.
        """
        # delete cache for constraint properties
        del self.constraints
        del self.constraints_repr

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

        # convert to integers
        dim, size = int(dim), None if size is None else int(size)

        # check for valid size constraint
        if size is not None and size < 1:
            raise ValueError(
                f"size {size} specifies an invalid (non-positive) number of elements."
            )

        # addition of constraint
        if dim not in self._constraints and size is not None:
            # create constraints with new addition
            constraints = {
                d: s for d, s in tuple(self._constraints.items()) + ((dim, size),)
            }

            # ensure constrained buffers and parameters are compatible
            for name in self._constrained_buffers:
                buffer = self.get_buffer(name)
                if (
                    buffer is not None
                    and buffer.numel() > 0
                    and not self.compatible(buffer, constraints=constraints)
                ):
                    raise RuntimeError(
                        f"constraint cannot be added, incompatible with buffer {name}."
                    )

            for name in self._constrained_parameters:
                param = self.get_parameter(name)
                if (
                    param is not None
                    and param.numel() > 0
                    and not self.compatible(param, constraints=constraints)
                ):
                    raise RuntimeError(
                        f"constraint cannot be added, incompatible with parameter {name}."
                    )

            # add new constraint
            self._constraints[dim] = size

        else:
            # removal of constraint
            if size is None:
                del self._constraints[dim]

            # alteration of constraint
            else:
                # create constraints with new addition
                constraints = {
                    d: s for d, s in tuple(self._constraints.items()) + ((dim, size),)
                }

                # reallocate buffers
                for name in self._constrained_buffers:
                    buffer = self.get_buffer(name)

                    if buffer is not None and buffer.numel() > 0:
                        rsetattr(
                            self,
                            name,
                            torch.zeros(
                                self.compatible_like(
                                    buffer.shape,
                                    add_dims=False,
                                    constraints=constraints,
                                ),
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
                                    self.compatible_like(
                                        param.shape,
                                        add_dims=False,
                                        constraints=constraints,
                                    ),
                                    dtype=param.dtype,
                                    layout=param.layout,
                                    device=param.device,
                                    requires_grad=param.data.requires_grad,
                                ),
                                requires_grad=param.requires_grad,
                            ),
                        )

                self._constraints[dim] = size

    def register_constrained(self, name: str):
        """Registers an existing buffer or parameter as constrained.

        Args:
            name (str): fully-qualified string name of the buffer or parameter to register.

        Raises:
            RuntimeError: dimension of attribute does not match constraint.
            AttributeError: attribute is not a registered buffer or parameter.
        """
        # calculate required number of dimensions
        maxc = max(self._constraints)
        minc = min(self._constraints)
        req_ndims = (maxc + 1 if maxc >= 0 else 0) - (minc if minc <= -1 else 0)

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
                    f"buffer {name} has shape of {tuple(buffer.shape)} "
                    f"incompatible with constrained shape {self.constraints_repr}, "
                    f"dimensions must match and must have at least {req_ndims} dimensions"
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
                    f"parameter {name} has shape of {tuple(param.shape)} "
                    f"incompatible with constrained shape {self.constraints_repr}, "
                    f"dimensions must match and must have at least {req_ndims} dimensions"
                )
            else:
                self._constrained_parameters.add(name)
            return

        raise AttributeError(
            f"name {name} does not specify a registered buffer or parameter."
        )

    def deregister_constrained(self, name: str):
        """Deregisters a buffer or parameter as constrained.

        Args:
            name (str): fully-qualified string name of the buffer or parameter to register.
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
        ValueError: history length must be non-negative.
    """
    def __init__(
        self,
        step_time: float,
        history_len: float,
    ):
        # ensure valid step time and history length parameters
        step_time, history_len = float(step_time), float(history_len)
        if step_time <= 0:
            raise ValueError(f"step time must be positive, received {step_time}")

        if history_len < 0:
            raise ValueError(
                f"history length must be non-negative, received {history_len}"
            )

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
            raise RuntimeError(f"history length must be non-negative, received {value}")

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
                f"{name} has not correctly been registered as a constrained attribute, call ignored.",
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
        if name in self._pointer:
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
            interpolation (Interpolate): method to interpolate between discrete time steps.
            tolerance (float, optional): maximum difference in time from a discrete sample to
                consider it at the same time as that sample. Defaults to 1e-7.
            offset (int, optional): window index offset, number of :py:meth:`tick` calls back. Defaults to 1.

        Returns:
            torch.Tensor: interpolated tensor selected at a prior time.

        Shape:
            ``time``: `broadcastable <https://pytorch.org/docs/stable/notes/broadcasting.html>`_
            with :math:`N_0 \times \cdots \times [D]`, where :math:`N_0 \times \cdots` is the
            underlying shape, and :math:`D` is the number of delay selectors.

            **outputs**:
            :math:`N_0 \times \cdots` \times [D]`, where :math:`D` is only included if
            it was in ``time``.

        Note:
            By default, `offset` is set to `1`. This is the correct configuration to use under normal
            circumstances where :py:meth:`pushto` is used for element insertion. Also this is useful for
            when :py:meth:`tick` is called after a call to :py:meth:`insert` and before :py:meth:`select`.
            This should be set to `0` if :py:meth:`tick` has not been called since the last :py:meth:`insert`.

        Note:
            The argument `interpolate` is a function which takes in five arguments, as follows.

                - tensor: nearest observed state before the selected time.
                - tensor: nearest observed state after the selected time.
                - tensor: time after the "before state" for which results should be produced.
                - float: difference in time between the before and after state.

            It must return a tensor of values interpolated between the samples at the two times.
            Some functions which meet the :py:class:`Interpolation` type are included in the library.

        Note:
            In cases where the times selected are a match for an observed index within the tolerance,
            the interpolation function is still called and its results still used. The ``select_at``
            values will be altered to either ``0`` or :py:attr:`self.dt`.
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

        # convert time to a tensor iff necessary and ensure device matches data
        if not isinstance(data, torch.Tensor):
            time = torch.full(data.shape[:-1], float(time), device=data.device)
            if not time.is_floating_point():
                time = time.to(dtype=torch.float32)
        else:
            time = time.to(device=data.device)

        # determine if output dimension should be squeezed
        if time.ndim == data.ndim - 1:
            squeeze_res = True
            time = time.unsqueeze(-1)
        elif time.ndim == data.ndim:
            squeeze_res = False
        else:
            raise RuntimeError(f"time has incompatible number of dimensions {time.ndim}, "
                               f"must have number of dimensions equal to {data.ndim} or {data.ndim - 1}")

        # validate that time values are correct
        if torch.any(time < 0):
            raise ValueError("time must contain only non-negative values.")
        if torch.any(time - tolerance > (self.hsize - 1) * self.dt):
            raise ValueError(
                f"time contains a value which exceeds maximum of {(self.hsize - 1) * self.dt}."
            )

        # compute indicies, corrected for tolerance
        indices = time / self.dt
        indices = torch.where(
            torch.abs(indices.round() - indices) * self.dt < tolerance,
            indices.round(),
            indices,
        )

        # correct time based on indices
        time = indices * self.dt

        # offset pointer
        pointer = (self._pointer[name] - offset) % self.hsize

        # observation before sample, index ceiling
        prev_data = torch.gather(
            data,
            -1,
            (pointer - indices.ceil().long()) % self.hsize,
        )

        # observation after sample, index floor
        next_data = torch.gather(
            data,
            -1,
            (pointer - indices.floor().long()) % self.hsize,
        )

        # interpolate and reshape if necessary
        res = interpolation(
            prev_data,
            next_data,
            torch.clamp((indices.ceil() / self.dt) - time, min=0, max=self.dt),
            self.dt,
        )
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
                raise RuntimeError(
                    f"cannot reset {name}, buffer is uninitialized."
                )
            buffer[:] = data
            self._pointer[name] = 0
        elif name in self._constrained_parameters:
            param = self.get_parameter(name)
            if param is None or param.numel() == 0:
                raise RuntimeError(
                    f"cannot reset {name}, parameter is uninitialized."
                )
            param[:] = data
        else:
            raise AttributeError(
                f"name {name} does not specify a constrained buffer or parameter."
            )

    def pushto(self, name: str, data: torch.Tensor) -> None:
        r"""Inserts a slice at the current time into a constrained attribute and advances to the next time.

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
                raise RuntimeError(
                    f"cannot push to {name}, buffer is uninitialized."
                )
            if tuple(data.shape) + (self.hlen,) != buffer.shape:
                raise RuntimeError(
                    f"data has a shape of {tuple(data.shape)}, "
                    f"required shape is {tuple(buffer.shape[:-1])}"
                )
            buffer[..., self._pointer[name]] = data
            self.tick(name)
        elif name in self._constrained_parameters:
            param = self.get_parameter(name)
            if param is None:
                raise RuntimeError(
                    f"cannot push to {name}, parameter is uninitialized."
                )
            if tuple(data.shape) + (self.hlen,) != param.shape:
                raise RuntimeError(
                    f"data has a shape of {tuple(data.shape)}, "
                    f"required shape is {tuple(param.shape[:-1])}"
                )
            param[..., self._pointer[name]] = data
            self.tick(name)
        else:
            raise AttributeError(
                f"name {name} does not specify a constrained buffer or parameter."
            )

    def latest(self, name: str, offset: int = 1) -> torch.Tensor:
        r"""Retrieves the most recent slice of a constrained attribute.

        Args:
            name (str): name of the attribute to target.
            offset (int, optional): window index offset, number of :py:meth:`tick` calls back. Defaults to 1.

        Raises:
            AttributeError: specified name is not a constrained buffer or parameter.

        Returns:
            torch.Tensor: most recent slice of the tensor selected.

        Note:
            By default, `offset` is set to `1`. This is the correct configuration to use under normal
            circumstances where :py:meth:`pushto` is used for element insertion. Also this is useful for
            when :py:meth:`tick` is called after a call to :py:meth:`insert` and before :py:meth:`select`.
            This should be set to `0` if :py:meth:`tick` has not been called since the last :py:meth:`insert`.
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

        return data[..., (self._pointer[name] - offset) % self.hlen]

    def history(self, name: str, offset: int = 1, latest_first=True) -> torch.Tensor:
        r"""Retrieves the recorded history of a constrained attribute.

        Args:
            name (str): name of the attribute to target.
            offset (int, optional): window index offset, number of :py:meth:`tick` calls back. Defaults to 1.
            latest_first (bool, optional): if the most recent sample should be at the zeroth index. Defaults to False.

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
                f"name {name} does not specify a constrained buffer or parameter."
            )

        # sorted latest last
        data = torch.roll(data, offset - self._pointer[name] - 1, -1)

        # reverse if required
        if latest_first:
            data = data.flip(-1)

        return data
