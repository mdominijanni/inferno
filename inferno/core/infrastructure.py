from __future__ import annotations
from ..interpolation import Interpolation
from .._internal import argtest, rsetattr
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Iterable
from functools import cache
from itertools import chain
import math
import torch
import torch.nn as nn
from typing import Any, Callable
import warnings
import weakref


class Module(nn.Module):
    r"""An extension of PyTorch's Module class.

    This extends :py:class:`torch.nn.Module` so that "extra state" is handled in a way
    similar to regular tensor state (e.g. buffers and parameters). This enables simple
    export to and import from a state dictionary. This does not enforce exact matching
    keys, and will insert new keys as required.

    Additionally, attribute assignment will check if the name refers to a property in
    the class and if so, uses ``object.__setattr__``.

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
    r"""Module with support for dimensionally constrained buffers and parameters.

    Args:
        *constraints (tuple[int, int]): tuple of (dim, size) dimensional constraints
            for constrained buffers and parameters.
        live (bool, optional): if constraints should be evaluated on constrained
            attribute set. Defaults to False.

    Raises:
        ValueError: size specified by each constraint must be positive.
        RuntimeError: two or more constraints have the same dimension.

    Caution:
        The names of constrained buffers and parameters will persist via the
        state dictionary but the constraints themselves will not.

    Important:
        The constraints given must refer to unique elements. For example, if a
        constraint is placed on the dim ``1`` and dim ``-1``, a tensor must be at
        least three-dimensional, since in a tensor with two dimensions, ``1`` and
        ``-1`` refer to the same dimension.

    Important:
        Constrained values which are either ``None``, scalar (i.e. have zero
        dimensions), or have no elements (i.e. have a zero-dimension) are
        automatically ignored.
    """

    def __init__(self, *constraints: tuple[int, int], live: bool = False):
        # setter-dependent attribute
        object.__setattr__(self, "_live_assert", live)

        # call superclass constructor
        Module.__init__(self)

        # transient state
        self._constraints = dict()

        # persistent state
        self.register_extra("_constrained_buffers", set())
        self.register_extra("_constrained_params", set())

        # cached values
        self._constraint_cache = cache(self._calc_constraints)
        self._extra_repr_cache = cache(self._calc_extra_repr)

        # check for consistent constraints
        for d, s in constraints:
            dim, size = int(d), int(s)
            if size < 1:
                raise ValueError(
                    f"constraint {(d, s)} specifies an invalid (nonpositive) "
                    "number of elements"
                )
            if dim in self._constraints:
                raise RuntimeError(
                    f"constraint {(dim, size)} conflicts with constraint "
                    f"{(dim, self._constraints[dim])}."
                )
            self._constraints[dim] = size

    @staticmethod
    def _ignore_tensor(tensor: torch.Tensor | nn.Parameter | None) -> bool:
        r"""Checks if a tensor should be ignored for constraints.

        Args:
            tensor (torch.Tensor | nn.Parameter | None): tensor to check.

        Returns:
            bool: if the tensor should be ignored.
        """
        return tensor is None or not tensor.ndim or not tensor.numel()

    def __setattr__(self, name: str, value: Any) -> None:
        if (
            self.liveconstrain
            and (name in self._constrained_buffers or name in self._constrained_params)
            and isinstance(value, torch.Tensor)
        ):
            if self._ignore_tensor(value) or self.compatible(value):
                super().__setattr__(name, value)
            else:
                raise RuntimeError(
                    f"tensor of shape {tuple(value.shape)} being assigned to '{name}' "
                    f"is not compatible with constraints ({self._extra_repr_cache()})"
                )
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

    @classmethod
    def compatible_like_(
        cls,
        value: torch.Tensor | nn.Parameter | Iterable[int],
        constraints: dict[int, int],
    ) -> torch.Tensor | nn.Parameter | tuple[int, ...]:
        r"""Creates a shape or new tensor like the input compatible with a set of constraints.

        Args:
            value (torch.Tensor | nn.Parameter | Iterable[int]): _description_
            constraints (dict[int, int]): constraint dictionary of (dim, size).

        Raises:
            RuntimeError: given shape does not have sufficient dimensionality to be
                made compatible with constraints.
            ValueError: size specified by each constraint must be positive.

        Returns:
            torch.Tensor | nn.Parameter | tuple[int, ...]: new tensor or parameter like
            the input, or the new compatible shape if not given a tensor.
        """
        # put original shape into a mutable
        shape = list(value.shape) if isinstance(value, torch.Tensor) else list(value)

        # ensure minimum dimensionality
        if len(shape) < cls.dims_(constraints):
            raise RuntimeError(
                f"'value' of shape {tuple(shape)} with {len(shape)} dims cannot be made "
                f"compatible, requires a minimum dimensionality of {cls.dims_(constraints)}"
            )

        # set new sizes via constraints
        for d, s in constraints.items():
            if s < 1:
                raise ValueError(
                    f"'constraints' specifies nonpositive sized constraint {(d, s)}"
                )
            else:
                shape[d] = s

        # create a zero-valued tensor or parameter like the input if given a tensor
        if isinstance(value, nn.Parameter):
            return nn.Parameter(
                torch.zeros(
                    shape,
                    dtype=value.dtype,
                    layout=value.layout,
                    device=value.device,
                    requires_grad=value.data.requires_grad,
                ),
                requires_grad=value.requires_grad,
            )
        elif isinstance(value, torch.Tensor):
            return torch.zeros(
                shape,
                dtype=value.dtype,
                layout=value.layout,
                device=value.device,
                requires_grad=value.requires_grad,
            )
        else:
            return tuple(shape)

    @property
    def constraints(self) -> dict[int, int]:
        r"""Returns the constraint dictionary, sorted by dimension.

        The results will be sorted by dimension, from first to last. Therefore,
        positive dimensions are presented first, in increasing order, then negative
        dimensions also in increasing order.

        Returns:
            dict[int, int]: active constraints, represented as a dictionary.
        """
        return self._constraint_cache()

    @property
    def dims(self) -> int:
        r"""Minimum number of required dimensions for a constrained tensor.

        Returns:
            int: minimum required number of dimensions.
        """
        return self.dims_(self._constraints)

    @property
    def liveconstrain(self) -> bool:
        r"""If constraints should be enforced on attribute assignment.

        Args:
            value (bool): if constraints should be enforced on attribute assignment.

        Returns:
            bool: if constraints should be enforced on attribute assignment.
        """
        return self._live_assert

    @liveconstrain.setter
    def liveconstrain(self, value: bool) -> None:
        self._live_assert = bool(value)

    def extra_repr(self) -> str:
        return "\n".join(
            (
                f"constraints=({self._extra_repr_cache()})",
                f"constrained_buffers=({','.join(self._constrained_buffers)})",
                f"constrained_parameters=({','.join(self._constrained_params)})",
            )
        )

    def _calc_constraints(self) -> dict[int, int]:
        r"""Calculates sorted constraints for the cache.

        Returns:
            dict[int, int]: active constraints, represented as a dictionary.
        """
        fwd, rev = [], []
        for dim, size in sorted(self._constraints.items()):
            rev.append((dim, size)) if dim < 0 else fwd.append((dim, size))

        return dict(fwd + rev)

    def _calc_extra_repr(self) -> str:
        r"""Calculates the extra representation layout of constraints.

        Returns:
            str: extra representation format of constraints.
        """
        # split constraints into forward and reverse (negative) indices, sorted
        fwd, rev = [], []
        for dim, size in sorted(self._constraints.items()):
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

    def _trim_constrained(self) -> None:
        r"""Deletes constrained buffers and parameters which no longer exist."""
        # remove deleted buffers from constrained set
        for name in tuple(self._constrained_buffers):
            try:
                _ = self.get_buffer(name)
            except AttributeError:
                self.deregister_constrained(name)

        # remove deleted parameters from constrained set
        for name in tuple(self._constrained_params):
            try:
                _ = self.get_parameter(name)
            except AttributeError:
                self.deregister_constrained(name)

    def _test_constrained(self) -> None:
        r"""Validates constraints."""
        # ensure constrained buffers have valid shape
        for name, b in map(
            lambda n: (n, self.get_buffer(n)), self._constrained_buffers
        ):
            if not (self._ignore_tensor(b) or self.compatible(b)):
                raise RuntimeError(f"constrained buffer '{name}' is invalid")

        # ensure constrained parameters have valid shape
        for name, p in map(
            lambda n: (n, self.get_parameter(n)), self._constrained_params
        ):
            if not (self._ignore_tensor(p) or self.compatible(p)):
                raise RuntimeError(f"constrained parameter '{name}' is invalid")

    def compatible(self, tensor: torch.Tensor) -> bool:
        r"""Test if a tensor is compatible with the constraints.

        Args:
            tensor (torch.Tensor): value to test.

        Returns:
            bool: if the tensor is compatible.
        """
        return self.compatible_(tensor, self._constraints)

    def compatible_like(
        self,
        value: torch.Tensor | nn.Parameter | Iterable[int],
    ) -> torch.Tensor | nn.Parameter | tuple[int, ...]:
        r"""Creates a shape or new tensor like the input compatible with the constraints.

        Args:
            value (torch.Tensor | nn.Parameter | Iterable[int]): _description_

        Returns:
            torch.Tensor | nn.Parameter | tuple[int, ...]: new tensor or parameter like
            the input, or the new compatible shape if not given a tensor.
        """
        return self.compatible_like_(value, self._constraints)

    def reconstrain(self, dim: int, size: int | None) -> DimensionalModule:
        r"""Modifies constraints.

        Adding constraints will not modify the constrained tensors, whereas modifying
        an existing constraint will create a new zero-tensor with the shape of that
        dimension modified.

        If the tensor was modified to be compatible with the new constraint ahead
        of time (i.e. if :py:attr:`liveconstrain` is ``False`` and was set with its new value),
        then reallocation will not occur.

        Args:
            dim (int): dimension to which a constraint should be added, removed,
                or modified.
            size (int | None): size of the new constraint, None if the constraint is
                to be removed.

        Raises:
            RuntimeError: constrained buffer or parameter is no longer a compatible shape.
            RuntimeError: constrained buffer or parameter would not be valid after the
                change in constraints.

        Returns:
            DimensionalModule: self.
        """
        # clear cached values
        self._constraint_cache.cache_clear()
        self._extra_repr_cache.cache_clear()

        # remove deleted buffers and parameters from constrained set
        self._trim_constrained()

        # cast and validate arguments
        dim = int(dim)
        size = None if size is None else argtest.gt("size", size, 0, int)

        # removes constraint
        if dim in self._constraints and not size:
            # removes constraint
            del self._constraints[dim]

            # tests if constrained buffers and parameters are still valid
            self._test_constrained()

            # returns self
            return self

        # creates constraint
        if dim not in self._constraints and size:
            # ensures constrained values are still valid
            self._test_constrained()

            # tests if buffers and parameters will be valid with new constraints
            cns = {**self._constraints, dim: size}

            for name, b in map(
                lambda n: (n, self.get_buffer(n)), self._constrained_buffers
            ):
                if not (self._ignore_tensor(b) or self.compatible_(b, cns)):
                    raise RuntimeError(
                        f"constrained buffer '{name}' would be invalidated by the "
                        f"addition of constraint {(dim, size)}"
                    )

            for name, p in map(
                lambda n: (n, self.get_parameter(n)), self._constrained_params
            ):
                if not (self._ignore_tensor(p) or self.compatible_(p, cns)):
                    raise RuntimeError(
                        f"constrained parameter '{name}' would be invalidated by the "
                        f"addition of constraint {(dim, size)}"
                    )

            # adds constraint
            self._constraints[dim] = size

            # returns self
            return self

        # alters constraint
        if dim in self._constraints and size:
            # check that all tensors have minimum required dimensionality
            ndim = self.dims

            for name, b in map(
                lambda n: (n, self.get_buffer(n)), self._constrained_buffers
            ):
                if not (self._ignore_tensor(b) or b.ndim >= ndim):
                    raise RuntimeError(
                        f"constrained buffer '{name}' with {b.ndim} dims cannot be made "
                        f"compatible, requires a minimum dimensionality of {ndim}"
                    )

            for name, p in map(
                lambda n: (n, self.get_parameter(n)), self._constrained_params
            ):
                if not (self._ignore_tensor(p) or p.ndim >= ndim):
                    raise RuntimeError(
                        f"constrained parameter '{name}' with {p.ndim} dims cannot be made "
                        f"compatible, requires a minimum dimensionality of {ndim}"
                    )

            # edit constraint
            self._constraints[dim] = size

            # reassign incompatible parameters
            for name, value in chain(
                map(lambda n: (n, self.get_buffer(n)), self._constrained_buffers),
                map(lambda n: (n, self.get_parameter(n)), self._constrained_params),
            ):
                if not (self._ignore_tensor(value) or self.compatible(value)):
                    rsetattr(self, name, self.compatible_like(value))

            # returns self
            return self

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
            if not (self._ignore_tensor(b) or self.compatible(b)):
                raise RuntimeError(
                    f"buffer '{name}' has shape of {tuple(b.shape)} "
                    f"incompatible with constrained shape ({self._extra_repr_cache()}), "
                    f"dimensions must match and '{name}' must have at least "
                    f"{self.dims} dimensions"
                )
            else:
                self._constrained_buffers.add(name)
            return

        # attempts to register parameter
        try:
            p = self.get_parameter(name)
        except AttributeError:
            pass
        else:
            if not (self._ignore_tensor(p) or self.compatible(p)):
                raise RuntimeError(
                    f"parameter '{name}' has shape of {tuple(p.shape)} "
                    f"incompatible with constrained shape ({self._extra_repr_cache()}), "
                    f"dimensions must match and '{name}' must have at least "
                    f"{self.dims} dimensions"
                )
            else:
                self._constrained_params.add(name)
            return

        # invalid name
        raise AttributeError(
            f"'nam'` ('{name}') does not specify a registered buffer or parameter"
        )

    def deregister_constrained(self, name: str) -> None:
        r"""Deregisters a buffer or parameter as constrained.

        If the name given isn't a constrained buffer or parameter, calling this does
        nothing.

        Args:
            name (str): fully-qualified string name of the buffer or
                parameter to register.
        """
        # remove if in buffers
        if name in self._constrained_buffers:
            self._constrained_buffers.remove(name)

        # remove if in parameters
        if name in self._constrained_params:
            self._constrained_params.remove(name)

    def get_constrained(self, name: str) -> torch.Tensor | nn.Parameter:
        # retrieve from buffers
        if name in self._constrained_buffers:
            return self.get_buffer(name)

        # retrieve from parameters
        if name in self._constrained_params:
            return self.get_parameter(name)

        raise AttributeError(
            f"'name' ('{name}') is not a constrained buffer or parameter"
        )

    def validate(self) -> None:
        r"""Validates constraints.

        Along with testing constrained buffers and parameters, if a registered
        constrained name no longer points at a buffer or parameter, that name is removed.

        Raises:
            RuntimeError: constrained buffer or parameter is no longer valid.
        """
        self._trim_constrained()
        self._test_constrained()


class RecordModule(DimensionalModule):
    r"""Module with support for buffers and parameters with time-based indexing.

    Args:
        step_time (float): length of time between stored values in the record.
        duration (float): length of time over which prior values are stored.

    Caution:
        When restoring from a state dictionary, the "pointer" to the next time slice to
        overwrite is preserved along with the names of added constrained buffers and
        parameters, but the step time and duration are not.
    """

    def __init__(
        self,
        step_time: float,
        duration: float,
    ):
        # argument validation
        step_time = argtest.gt("step_time", step_time, 0, float)
        duration = argtest.gte("duration", duration, 0, float)

        # size of the history dimension
        size = math.ceil(duration / step_time) + 1

        # call superclass constructor
        DimensionalModule.__init__(self, (-1, size))

        # transient state
        self._step_time = step_time
        self._duration = duration

        # persistent state
        self.register_extra("_pointers", dict())

    @property
    def dt(self) -> float:
        r"""Length of time between stored values in history.

        In the same units as :py:attr:`self.duration`.

        Args:
            value (float): new time step length.

        Returns:
            float: length of the time step.

        Note:
            If a :py:meth:`reconstrain` operation needs to be performed, all state will
            be overwritten with zeros.
        """
        return self._step_time

    @dt.setter
    def dt(self, value: float) -> None:
        # cast value as float and validate
        value = argtest.gt("value", value, 0, float)

        # recompute size of the history dimension
        size = math.ceil(self._duration / value) + 1

        # reconstrain if required
        if size != self.recordsz:
            DimensionalModule.reconstrain(self, -1, size)

        # set revised step time
        self._step_time = value

    @property
    def duration(self) -> float:
        r"""Length of time over which prior values are stored.

        In the same units as :py:attr:`self.dt`.

        Args:
            value (float): new length of the history to store.

        Returns:
            float: length of the record.

        Note:
            If a :py:meth:`reconstrain` operation needs to be performed, all state will
            be overwritten with zeros.
        """
        return self._duration

    @duration.setter
    def duration(self, value: float) -> None:
        # cast value as float and validate
        value = argtest.gte("value", value, 0, float)

        # recompute size of the history dimension
        size = math.ceil(value / self._step_time) + 1

        # reconstrain if required
        if size != self.recordsz:
            DimensionalModule.reconstrain(self, -1, size)

        # set revised history length
        self._duration = value

    @property
    def recordsz(self) -> int:
        r"""Number of stored time slices for each record tensor.

        Returns:
            int: length of the record, in number of slices.
        """
        return self.constraints.get(-1)

    def _get_constrained_record(self, name: str) -> torch.Tensor | nn.Parameter:
        r"""Gets the value of a constraint which is a record.

        Args:
            name (str): name of the buffer or parameter.

        Raises:
            RuntimeError: the name specifies a buffer or parameter which was not
                constrained by :py:class:`RecordModule`.
            RuntimeError: the name specifies an uninitialized attribute, i.e. one which
                is ``None``, has no elements, or is a scalar (a 0-dimensional tensor).

        Returns:
            torch.Tensor | nn.Parameter: constrained record tensor.
        """
        data = self.get_constrained(name)

        if name not in self._pointers:
            raise RuntimeError(
                f"'name' ('{name}') specifies an improperly constrained attribute"
            )

        if self._ignore_tensor(data):
            raise RuntimeError(
                f"'name' ('{name}') specifies an uninitialized attribute"
            )

        return data

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
        DimensionalModule.register_constrained(self, name)
        if name not in self._pointers:
            self._pointers[name] = 0

    def deregister_constrained(self, name: str):
        r"""Deregisters a buffer or parameter as constrained.

        If the name given isn't a constrained buffer or parameter, calling this does
        nothing.

        Args:
            name (str): fully-qualified string name of the buffer or
                parameter to register.
        """
        DimensionalModule.deregister_constrained(self, name)
        if name in self._pointers:
            del self._pointers[name]

    def reconstrain(self, dim: int, size: int | None) -> RecordModule:
        r"""Modifies constraints.

        Adding constraints will not modify the constrained tensors, whereas modifying
        an existing constraint will create a new zero-tensor with the shape of that
        dimension modified.

        If the tensor was modified to be compatible with the new constraint ahead
        of time (i.e. if :py:attr:`liveconstrain` is ``False`` and was set with its new value),
        then reallocation will not occur.

        Args:
            dim (int): dimension to which a constraint should be added, removed,
                or modified.
            size (int | None): size of the new constraint, None if the constraint is
                to be removed.

        Raises:
            RuntimeError: constrained buffer or parameter is no longer a compatible shape.
            RuntimeError: constrained buffer or parameter would not be valid after the
                change in constraints.

        Returns:
            RecordModule: self.
        """
        if dim == -1:
            raise RuntimeError(
                f"{type(self).__name__}(RecordModule) cannot reconstrain the record "
                "dimension (-1)"
            )
        else:
            return DimensionalModule.reconstrain(self, dim, size)

    def reset(self, name: str, data: Any) -> None:
        r"""Resets a constrained attribute to some value or values.

        Args:
            name (str): name of the attribute to target.
            data (Any): data to insert.
        """
        self._get_constrained_record(name)[:] = data
        self._pointers[name] = 0

    def latest(self, name: str, offset: int = 1) -> torch.Tensor:
        r"""Retrieves the most recent slice of a constrained attribute.

        Args:
            name (str): name of the attribute to target.
            offset (int, optional): number of steps before present to select from.
                Defaults to 1.

        Returns:
            torch.Tensor: most recent slice of the tensor selected.
        """
        data = self._get_constrained_record(name)
        return data[..., (self._pointers[name] - int(offset)) % self.recordsz]

    def record(self, name: str, value: torch.Tensor, offset: int = 0) -> None:
        """Overwrites the record at the current slice and increments the pointer.

        Args:
            name (str): name of the attribute to target.
            value (torch.Tensor): value to write into the current time step.
            offset (int, optional): number of steps before present to update.
                Defaults to 0.
        """
        data = self._get_constrained_record(name)
        if value.shape != data.shape[:-1]:
            raise ValueError(
                f"'value' has shape of {tuple(value.shape)} which does not match the "
                f"required shape of {tuple(data.shape[:-1])}"
            )
        data[..., (self._pointers[name] - int(offset)) % self.recordsz] = value
        self._pointers[name] = (self._pointers[name] + 1) % self.recordsz

    def select(
        self,
        name: str,
        time: float | torch.Tensor,
        interp: Interpolation,
        *,
        tolerance: float = 1e-7,
        offset: int = 1,
    ) -> torch.Tensor:
        r"""Selects elements of a constrained attribute based on prior time.

        If ``time`` is a scalar and is within tolerance of an integer index, then
        a slice will be returned without ever attempting interpolation.

        If ``time`` is a tensor, interpolation will be called regardless, and the time
        passed into the interpolation call will be set to either ``0`` or
        :py:attr:`self.dt`. Interpolation results are then overwritten with exact values
        before returning.

        Args:
            name (str): name of the attribute from which to select.
            time (float | torch.Tensor): time(s) before present to select from.
            interp (Interpolation): method to interpolate between discrete time steps.
            tolerance (float, optional): maximum difference in time from a discrete
                sample to consider a time co-occuring with the sample. Defaults to 1e-7.
            offset (int, optional): window index offset as number of steps prior to the
                location of the next time slice to overwrite. Defaults to 1.

        Returns:
            torch.Tensor: interpolated values selected at a prior time(s).

        .. admonition:: Shape
            :class: tensorshape

            ``time``:

            :math:`N_0 \times \cdots \times [D]`

            ``return``:

            :math:`N_0 \times \cdots \times [D]`

            Where:
                * :math:`N_0, \ldots` are the dimensions of the constrained tensor.
                * :math:`D` is the number of times for each value to select.

        Tip:
            Mimicing the behavior of :py:func:`torch.gather`, if ``time`` is out of the
            valid range, a :py:class:`ValueError` will be thrown. To avoid this, clamp
            ``time`` like ``time.clamp(0, module.duration)``.
        """
        # get underlying data
        data = self._get_constrained_record(name)

        # apply offset to pointer
        pointer = (self._pointers[name] - int(offset)) % self.recordsz

        # tensor selector
        if isinstance(time, torch.Tensor):
            # cast values and test
            time = time.to(device=data.device)
            if not time.is_floating_point():
                time = time.to(dtype=torch.float32)
            _ = argtest.gte("time", time.amin(), -tolerance)
            _ = argtest.lte(
                "time", time.amax(), self.dt * (self.recordsz - 1) + tolerance
            )

            # check if the output should be squeezed
            if time.ndim == data.ndim - 1:
                squeeze = True
                time = time.unsqueeze(-1)
            elif time.ndim == data.ndim:
                squeeze = False
            else:
                raise RuntimeError(
                    f"'time' has incompatible number of dimensions {time.ndim}, "
                    f"must have either {data.ndim} or {data.ndim - 1} dimensions"
                )

            # compute continuous index
            index = time / self.dt
            rindex = index.round()
            index = torch.where(
                (self.dt * rindex - time).abs() < tolerance, rindex, index
            )

            # access data by index and interpolate
            prev_idx = (pointer - index.ceil().long()) % self.recordsz
            prev_data = torch.gather(data, -1, prev_idx)

            next_idx = (pointer - index.floor().long()) % self.recordsz
            next_data = torch.gather(data, -1, next_idx)

            res = interp(prev_data, next_data, self.dt * (index % 1), self.dt)

            # bypass interpolation for exact indices
            res = torch.where(prev_idx == next_idx, prev_data, res)

            # conditionally squeeze
            return res.squeeze(-1) if squeeze else res

        # scalar selector
        else:
            # cast values and test
            time = argtest.minmax_incl(
                "time",
                time,
                -tolerance,
                self.dt * (self.recordsz - 1) + tolerance,
                float,
            )

            # compute continuous index
            index = time / self.dt
            rindex = round(index)
            if abs(self.dt * rindex - time) < tolerance:
                index = rindex

            # integer index (no interpolation)
            if isinstance(index, int):
                return data[..., (pointer - index) % self.recordsz]

            # float index (interpolation)
            else:
                return interp(
                    data[..., (pointer - int(index + 1)) % self.recordsz].unsqueeze(-1),
                    data[..., (pointer - int(index)) % self.recordsz].unsqueeze(-1),
                    torch.full(
                        (*data.shape[:-1], 1),
                        self.dt * (index % 1),
                        dtype=data.dtype,
                        device=data.device,
                    ),
                    self.dt,
                ).squeeze(-1)

    def aligned(
        self, name: str, latest_first: bool = True, offset: int = 1
    ) -> torch.Tensor:
        r"""Full aligned record of a recorded tensor.

        The native storage order is latest data at the last index (after being aligned
        with :py:func:`torch.roll`). By default this will return the latest data at
        the first index.

        Args:
            name (str): name of the attribute to target.
            latest_first (bool, optional): if the most recent slice should be at the
                zeroth index. Defaults to True.
            offset (int, optional): window index offset as number of steps prior to the
                location of the next time slice to overwrite. Defaults to 1.

        Returns:
            torch.Tensor: _description_
        """
        # access raw data
        data = self._get_constrained_record(name)

        # align based on pointer and offset (latest is last, native ordering)
        data = torch.roll(data, offset - self._pointers[name] - 1, -1)

        # reverse if latest is first
        if latest_first:
            data = data.flip(-1)

        return data


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
        return self.__prehook_handle is not None or self.__posthook_handle is not None

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
            raise RuntimeError()
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
