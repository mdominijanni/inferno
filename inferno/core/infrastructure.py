from __future__ import annotations
from .tensor import empty, full, fullc, zeros
from ..functional import Interpolation, Extrapolation, interp_nearest, extrap_nearest
from .._internal import argtest
from abc import ABC, abstractmethod
from collections import namedtuple, OrderedDict
import einops as ein
from itertools import chain, repeat, starmap
import math
import torch
import torch.nn as nn
from types import MethodType
from typing import Any, Callable
import weakref


class Module(nn.Module):
    r"""An extension of PyTorch's Module class.

    This extends :py:class:`torch.nn.Module` so that "extra state" is handled in a way
    similar to regular tensor state (e.g. buffers and parameters). This enables simple
    export to and import from a state dictionary. This does not enforce exact matching
    keys, and will insert new keys as required.

    Additionally, attribute assignment will check if the name refers to a property or
    another descriptor and will use the descriptor's ``__set__`` behavior instead.

    Note:
        Like with :py:class:`torch.nn.Module`, an `__init__()` call must be made
        to the parent class before assignment on the child. This class's constructor
        will automatically call PyTorch's.
    """

    def __init__(self, *args, **kwargs):
        nn.Module.__init__(self, *args, **kwargs)
        self._extras = OrderedDict()

    def register_extra(self, name: str, value: Any) -> None:
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
            :py:class:`~torch.Tensor`, :py:class:`~torch.nn.parameter.Parameter`, and
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
        r"""Returns the extra state to include in the module's state_dict.

        Returns:
            dict[str, Any]: extra state to store in the module's state dictionary.
        """
        return self._extras

    def set_extra_state(self, state: dict[str, Any]) -> None:
        r"""Set extra state contained in the loaded state dictionary.

        This function is called from :py:meth:`~torch.nn.Module.load_state_dict` to
        handle any extra state found within the :py:meth:`~torch.nn.Module.state_dict`.

        Args:
            state (dict): extra state from the state dictionary.
        """
        self._extras.update(state)

    def __setstate__(self, state) -> None:
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
        # intercept parameter and module assignments
        descriptor = getattr(type(self), name, None)
        if (
            isinstance(descriptor, property)
            or hasattr(descriptor, "__get__")
            or hasattr(descriptor, "__set__")
            or hasattr(descriptor, "__delete__")
        ):
            descriptor.__set__(self, value)  # type: ignore
        else:
            _extras = self.__dict__.get("_extras")
            if _extras is not None and name in _extras:
                _extras[name] = value
            else:
                super().__setattr__(name, value)

    def __delattr__(self, name) -> None:
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


def _constraint_dimensionality(constraints: dict[int, int], strict: bool) -> int:
    r"""Computes the minimum dimensionality for a set of constraints.

    Args:
        constraints (dict[int, int]): dictionary of constraints.
        strict (bool): if constraints should be applied strictly.

    Returns:
        int: minimum number of dimensions for a constrained tensor.
    """
    if not constraints:
        return 0
    elif strict:
        return max(max(constraints) + 1, 0) - min(min(constraints), 0)
    else:
        return max(max(constraints) + 1, abs(min(constraints)))


def _constraints_compatible(
    tensor: torch.Tensor | nn.Parameter, constraints: dict[int, int], strict: bool
) -> bool:
    r"""Tests if a tensor is compatible with constraints.

    Args:
        tensor (torch.Tensor | nn.Parameter): tensor to test for compatibility.
        constraints (dict[int, int]): dictionary of constraints.
        strict (bool): if constraints should be applied strictly.

    Returns:
        bool: if the tensor is valid under the dimensional constraints.
    """
    if tensor.ndim < _constraint_dimensionality(constraints, strict):
        return False
    else:
        return all(
            starmap(lambda d, s, shape=tensor.shape: shape[d] == s, constraints.items())
        )


def _constraints_consistent(constraints: dict[int, int], ndims: int) -> bool:
    r"""Tests if constraints will not conflict.

    When constraints are applied strictly, the constraints must be consistent regardless
    of the exact sizes. Otherwise, this must check for consistency since specific
    sizes may create impossible constraints.

    Args:
        constraints (dict[int, int]): dictionary of constraints.
        ndims (int): dimensionality of the tensor to constrain.

    Returns:
        bool: if a tensor of given dimensionality can be constrained consistently.
    """
    hypoth: list[int | None] = list(repeat(None, times=ndims))

    for dim, size in constraints.items():
        if hypoth[dim] is None:
            hypoth[dim] = size
        elif hypoth[dim] == size:
            continue
        else:
            return False

    return True


def _shapedtensor_finalization(owner: weakref.ReferenceType, name: str) -> None:
    r"""Finalizer function for ShapedTensor."""
    owner = owner()
    if owner:
        if hasattr(owner, f"_{name}_data"):
            delattr(owner, f"_{name}_data")
        if hasattr(owner, f"_{name}_constraints"):
            delattr(owner, f"_{name}_constraints")


class ShapedTensor:
    r"""Tensor attribute with constrained shape.

    Some states for ``value`` are ignored for convenience. If it is ``None``,
    an instance of either :py:class:`~torch.nn.parameter.UninitializedBuffer` or
    :py:class:`~torch.nn.parameter.UninitializedParameter`, or has no elements and only a single
    dimension (such as if created with ``torch.empty(0)``).

    When ``value`` is ``None``, a registered buffer is created, otherwise a parameter
    will only be added if an :py:class:`~torch.nn.parameter.Parameter` is given.
    Assignment of a parameter to ``None`` is unsupported.

    Args:
        owner (Module): module to which this attribute will belong.
        name (str): name of the attribute.
        value (torch.Tensor | nn.Parameter | None): tensor-like data for the attribute.
        constraints (dict[int, int] | None, optional): constraints given as a
            dictionary of dimensions to their corresponding size. Defaults to ``None``.
        persist_data (bool, optional): if the data should persist across the
            state dictionary, only used with buffers. Defaults to ``True``.
        persist_constraints (bool, optional): if the constraints should persist
            across the state dictionary. Defaults to ``False``.
        strict (bool, optional): if each dimension must specify a unique dimension
            for the tensor. Defaults to ``True``.
        live (bool, optional): if constraint validity should be tested on
            assignment. Defaults to ``False``.

    Raises:
        RuntimeError: given data is neither ignored nor compatible with given
            constraints.

    Caution:
        This has a finalizer which will delete the attributes added to the module when
        its reference count goes to zero.
    """

    LinkedAttributes = namedtuple("ShapedTensorAttributes", ("data", "constraints"))  # type: ignore

    def __init__(
        self,
        owner: Module,
        name: str,
        value: torch.Tensor | nn.Parameter | None,
        constraints: dict[int, int] | None = None,
        persist_data: bool = True,
        persist_constraints: bool = False,
        strict: bool = True,
        live: bool = False,
    ):
        # ensure the name is a valid identifier
        _ = argtest.identifier("name", name)

        # internal state
        self.__owner = weakref.ref(owner)
        self.__name = name
        self.__strict = strict
        self.__live = live
        self.__finalizer = weakref.finalize(
            self, _shapedtensor_finalization, self.__owner, self.__name
        )

        # create constraints dictionary
        constraints = dict(constraints) if constraints else {}

        # invalid initial constraints
        if not self._ignore_or_compatible(value, constraints, strict):
            assert value is not None
            raise RuntimeError(
                f"initial value of shape {tuple(value.shape)} is not compatible with "
                f"constraints: {tuple(constraints.items())}"
            )

        # registered attribute names
        self.__attributes = ShapedTensor.LinkedAttributes(
            f"_{self.__name}_data", f"_{self.__name}_constraints"
        )

        # register data
        if isinstance(owner, nn.Module) and not isinstance(value, nn.Parameter):
            owner.register_buffer(
                self.__attributes.data, value, persistent=persist_data
            )
        else:
            setattr(owner, self.__attributes.data, value)

        # register constraints
        if persist_constraints and isinstance(owner, Module):
            owner.register_extra(self.__attributes.constraints, constraints)
        else:
            setattr(owner, self.__attributes.constraints, constraints)

    @classmethod
    def create(
        cls,
        owner: Module,
        name: str,
        value: torch.Tensor | nn.Parameter | None,
        constraints: dict[int, int] | None = None,
        persist_data: bool = True,
        persist_constraints: bool = False,
        strict: bool = True,
        live: bool = False,
    ) -> None:
        r"""Creates a shaped tensor and adds it as an attribute.

        The following two calls are equivalent.

        .. code-block:: python

            module.name = ShapedTensor(module, name, value)

        .. code-block:: python

            ShapedTensor.create(module, name, value)

        Args:
            owner (Module): module to which this attribute will belong.
            name (str): name of the attribute.
            value (torch.Tensor | nn.Parameter | None): tensor-like data for the
                attribute.
            constraints (dict[int, int] | None, optional): constraints given as a
                dictionary of dimensions to their corresponding size.
                Defaults to ``None``.
            persist_data (bool, optional): if the data should persist across the
                state dictionary, only used with buffers. Defaults to ``True``.
            persist_constraints (bool, optional): if the constraints should persist
                across the state dictionary. Defaults to ``False``.
            strict (bool, optional): if each dimension must specify a unique dimension
                for the tensor. Defaults to ``True``.
            live (bool, optional): if constraint validity should be tested on
                assignment. Defaults to ``False``.
        """
        constrained = cls(
            owner,
            name,
            value,
            constraints,
            persist_data=persist_data,
            persist_constraints=persist_constraints,
            strict=strict,
            live=live,
        )
        setattr(constrained.owner, constrained.name, constrained)

    @staticmethod
    def __make_compatible(
        tensor: torch.Tensor | nn.Parameter, dim: int, size: int
    ) -> torch.Tensor:
        r"""Corrects the tensor's shape."""
        if tensor.shape[dim] > size:
            slices = list(repeat(slice(None), times=tensor.ndim))
            slices[dim] = slice(tensor.shape[dim] - size, None)
            return tensor[*slices]

        elif tensor.shape[dim] < size:
            shape = list(tensor.shape)
            shape[dim] = size - tensor.shape[dim]
            return torch.cat((zeros(tensor, shape=shape), tensor), dim)

        elif isinstance(tensor, nn.Parameter):
            return tensor.data

        else:
            return tensor

    @staticmethod
    def _ignore(tensor: torch.Tensor | nn.Parameter | None) -> bool:
        r"""Tests if compatibility for the input should be ignored.

        Args:
            tensor (torch.Tensor | nn.Parameter | None): value to check if ignored.

        Returns:
            bool: if the value will be ignored by constraints.
        """
        return (
            tensor is None
            or isinstance(tensor, nn.UninitializedBuffer | nn.UninitializedParameter)
            or not (tensor.numel() or tensor.ndim > 1)
        )

    @staticmethod
    def _ignore_or_compatible(
        tensor: torch.Tensor | nn.Parameter | None,
        constraints: dict[int, int],
        strict: bool,
    ) -> bool:
        r"""Tests if compatibility for the input should be ignored or they are compatible.

        Args:
            tensor (torch.Tensor | nn.Parameter | None): value to check if ignored or compatible.
            constraints (dict[int, int]): dictionary of constraints.
            strict (bool): if constraints should be applied strictly.

        Returns:
            bool: if the value will be ignored by constraints or is compatible with constraints.
        """
        return (
            tensor is None
            or isinstance(tensor, nn.UninitializedBuffer | nn.UninitializedParameter)
            or not (tensor.numel() or tensor.ndim > 1)
            or _constraints_compatible(tensor, constraints, strict)
        )

    @staticmethod
    def resize(
        value: torch.Tensor | nn.Parameter,
        dim: int,
        size: int,
        preserve_tail: bool = True,
        fill: Any = 0,
    ) -> torch.Tensor | nn.Parameter:
        r"""Resizes a tensor's dimension.

        A more generalized version of the built-in automatic resizing, this can be
        used before calling :py:meth:`reconstrain` if more control is needed.

        If ``value`` is a tensor, then a tensor will be returned, a new tensor if
        the shape needed to be changed, otherwise the same tensor as was given.

        If ``value`` is a parameter, then a parameter will be returned. If it needed
        to be reshaped, the parameter's data will be set with the reshaped data prior
        to being returned.

        When ``preserve_tail`` is ``True``, the tail of the tensor will be kept as-is,
        otherwise the head of the tensor will be kept. This corresponds to where
        slices will be removed or appended. Assuming a dimension is not sized to zero,
        then if ``preserve_tail`` is ``True``, ``tensor[-1]`` will return the same
        values before and after, otherwise ``tensor[0]`` will.

        Args:
            value (torch.Tensor | nn.Parameter): tensor-like value to resize.
            dim (int): dimension to resize.
            size (int): new size for the dimension.
            preserve_tail (bool, optional): if the tail (higher indices) of a tensor
                should be unaltered. Defaults to ``True``.
            fill (Any, optional): value with which to fill expanded dimensions.
                Defaults to ``0``.

        Returns:
            torch.Tensor | nn.Parameter: resized tensor.
        """
        dim = int(dim)
        size = argtest.gte("size", size, 0, int)

        # shrink dimension
        if value.shape[dim] > size:
            slices = list(repeat(slice(None), times=value.ndim))

            if preserve_tail:
                slices[dim] = slice(value.shape[dim] - size, None)
            else:
                slices[dim] = slice(None, size)

            data = value[*slices]

        # expand dimension
        elif value.shape[dim] < size:
            shape = list(value.shape)
            shape[dim] = size - value.shape[dim]

            if preserve_tail:
                data = torch.cat((full(value, fill, shape=shape), value), dim)
            else:
                data = torch.cat((value, full(value, fill, shape=shape)), dim)

        # no change
        else:
            return value

        # set parameter data before returning if needed
        if isinstance(value, nn.Parameter):
            value.data = data
            return value
        else:
            return data

    @property
    def __constraints(self) -> dict[int, int]:
        r"""Module internal constraint getter."""
        return getattr(self.__owner(), self.__attributes.constraints)

    @property
    def __data(self) -> torch.Tensor | nn.Parameter | None:
        r"""Module internal data getter."""
        return getattr(self.__owner(), self.__attributes.data)

    @__data.setter
    def __data(self, value: torch.Tensor | nn.Parameter | None) -> None:
        r"""Module internal data setter."""
        data = self.__data
        if (
            isinstance(data, nn.Parameter)
            and isinstance(value, torch.Tensor)
            and not isinstance(value, nn.Parameter | nn.UninitializedBuffer)
        ):
            data.data = value
        else:
            setattr(self.__owner(), self.__attributes.data, value)

    @property
    def owner(self) -> Module | None:
        r"""Module which owns this attribute.

        Returns:
            Module | None: owner of the attribute if it exists.
        """
        return self.__owner()

    @property
    def name(self) -> str:
        r"""Name of the attribute.

        Two attributes with names derived from ``name`` are added to the owner.

        * ``_{name}_data``, the constrained tensor.
        * ``_{name}_constraints``, the dictionary of constraints.

        Returns:
            str: name of the attribute.
        """
        return self.__name

    @property
    def attributes(self) -> ShapedTensor.LinkedAttributes:
        r"""Names of the dependent attributes created.

        This is a named tuple with attributes ``data`` and ``constraints``.

        Returns:
            ShapedTensor.LinkedAttributes: names of the created attributes in the
            containing module.
        """
        return self.__attributes

    @property
    def value(self) -> torch.Tensor | nn.Parameter | None:
        r"""Value of the constrained tensor.

        If ``live`` was set on initialization, every setter call will ensure the tensor
        being set is valid (constrained or ignored).

        When created as a :py:class:`~torch.nn.parameter.Parameter`, assignment to
        ``None`` is prevented. If the current ``value`` is a
        :py:class:`~torch.nn.parameter.Parameter` but the assigned value is a
        :py:class:`~torch.Tensor`, it will automatically assign to the ``data``
        attribute of ``value``.

        Args:
            value (value: torch.Tensor | nn.Parameter | None): value to which the
                constrained attribute will be set.

        Returns:
            torch.Tensor | nn.Parameter | None: constrained attribute.
        """
        return self.__data

    @value.setter
    def value(self, value: torch.Tensor | nn.Parameter | None) -> None:
        # cannot assign none to parameter
        if isinstance(self.__data, nn.Parameter) and value is None:
            raise RuntimeError("cannot assign None to a constrained parameter")
        # live constrain
        if self.__live:
            if self._ignore_or_compatible(value, self.__constraints, self.__strict):
                self.__data = value
            else:
                assert value is not None
                raise ValueError(
                    f"cannot set a tensor with shape {tuple(value.shape)} with "
                    f"constraints: {tuple(self.__constraints.items())}"
                )
        else:
            self.__data = value

    @property
    def constraints(self) -> dict[int, int]:
        r"""Dictionary of registered constraints.

        Each key corresponds to a dimension of the tensor, and its associated value
        is the size of that dimension.

        Returns:
            dict[int, int]: dictionary of constraints.
        """
        return dict(self.__constraints)

    @property
    def ignored(self) -> bool:
        r"""If the current tensor is ignored.

        Returns:
            bool: if the current tensor is ignored.
        """
        return self._ignore(self.__data)

    @property
    def live(self) -> bool:
        r"""If assignments should be constraint tested.

        Args:
            value (bool): if assignments should be constraint tested.

        Returns:
            bool: if assignments should be constraint tested.
        """
        return self.__live

    @live.setter
    def live(self, value: bool) -> None:
        self.__live = bool(value)

    @property
    def strict(self) -> bool:
        r"""If constraints should be strictly tested.

        When strict constraints are used, each constrained dimension must refer to a
        unique tensor dimension. For example, if constraints are placed on dimensions
        ``0`` and ``-1``, then ``strict`` would require a tensor to have at least two
        dimensions. Otherwise, as long as the constraints are all met, regardless of
        uniqueness, a tensor is considered valid.

        Args:
            value (bool): if constraints should be strictly tested.

        Returns:
            bool: if constraints should be strictly tested.
        """
        return self.__strict

    @strict.setter
    def strict(self, value: bool) -> None:
        self.__strict = bool(value)

    @property
    def valid(self) -> bool:
        r"""If the shaped tensor is valid.

        A shaped tensor is considered valid if the value is either ignored or is
        compatible with the current constraints and its owner still exists.

        Returns:
            bool: if the shaped tensor is valid.
        """
        return bool(self.__owner()) and self._ignore_or_compatible(
            self.__data, self.__constraints, self.__strict
        )

    @property
    def dimensionality(self) -> int:
        r"""Minimum number of dimensions a tensor needs to satisfy constraints.

        Returns:
            int: minimum valid dimensionality.
        """
        return _constraint_dimensionality(self.__constraints, self.__strict)

    def compatible(self, tensor: torch.Tensor | nn.Parameter) -> bool:
        r"""Checks if a tensor is compatible.

        Args:
            tensor (torch.Tensor | nn.Parameter): tensor to test for compatibility.

        Returns:
            bool: if the given tensor is dimensionally compatible.
        """
        return _constraints_compatible(tensor, self.__constraints, self.__strict)

    def reconstrain(
        self, dim: int, size: int | None
    ) -> torch.Tensor | nn.Parameter | None:
        r"""Add, edit, or remove a constraint.

        When ``size`` is ``None``, the corresponding constraint will be removed. When
        ``dim`` is not in the current constraints, it will add that constraint. When
        ``dim`` is in the current constraints and ``size`` is not ``None``, that
        constraint will be altered. Automatic resizing only occurs on editing a
        constraint, not on adding a constraint.

        When editing a constraint, if a tensor is already compatible with that
        constraint, then it will not be altered. If the size of the dimension is
        reduced, then elements will be cut off, preserving those towards the end.
        If the size of the dimension is increased, then zero-valued elements will be
        added to the start.

        Args:
            dim (int): dimension on which to modify the constraint.
            size (int | None): new size for the specified dimension.

        Raises:
            ValueError: dimension specified for constraint removal is unconstrained.
            ValueError: constrained tensor would be invalidated by adding this constraint.
            RuntimeError: previous operation invalidated constrained tensor.
            RuntimeError: constrained tensor cannot be made valid with constraint.

        Returns:
            torch.Tensor | nn.Parameter | None: newly constrained value.
        """
        # cast and validate arguments
        dim = int(dim)
        size = None if size is None else argtest.gte("size", size, 0, int)

        # strongly reference data and constraints
        data, constraints = self.__data, self.__constraints

        # invalid input combination
        if dim not in constraints and size is None:
            raise ValueError(f"cannot remove constraint on unconstrained dim {dim}")

        # create constraint
        elif dim not in constraints:
            # size cannot be none
            assert size is not None

            # safe to add with an ignored tensor
            if self._ignore(data):
                constraints[dim] = size

            # check if the tensor has been invalidated
            else:
                # data cannot be none
                assert data is not None

                if _constraints_compatible(data, constraints, self.__strict):

                    # add if compatible
                    if _constraints_compatible(
                        data, constraints | {dim: size}, self.__strict
                    ):
                        constraints[dim] = size

                    else:
                        raise ValueError(
                            "constrained tensor would be invalidated by constraint of "
                            f"size {size} on dim {dim}"
                        )

                else:
                    raise RuntimeError("constrained tensor has been invalidated")

        # remove constraint
        elif size is None:
            # always safe to remove a constraint
            del constraints[dim]

            # check that tensor is still valid
            if not self._ignore_or_compatible(data, constraints, self.__strict):
                raise RuntimeError("constrained tensor has been invalidated")

        # alter constraint
        else:
            # safe to edit with an ignored tensor
            if self._ignore(data):
                constraints[dim] = size

            else:
                # data cannot be none
                assert data is not None

                # tensor has sufficient dimensionality
                if data.ndim >= _constraint_dimensionality(
                    constraints, self.__strict
                ) and _constraints_consistent(constraints | {dim: size}, data.ndim):
                    # edit the constraint
                    constraints[dim] = size

                    # alter if not already compatible
                    if not _constraints_compatible(data, constraints, self.__strict):
                        self.__data = self.__make_compatible(data, dim, size)
                        data = self.__data

                # tensor has insufficient dimensionality
                else:
                    raise RuntimeError(
                        "constrained tensor cannot be made valid with altered constraint"
                    )

        # return altered value
        return data


def _unwind_ptr(pointer: int, offset: int | float, size: int) -> int:
    return (pointer - int(offset)) % size


def _unwind_tensor_ptr(pointer: int, offset: torch.Tensor, size: int) -> torch.Tensor:
    return (pointer - offset.long()) % size


def _recordtensor_finalization(owner: weakref.ReferenceType, name: str) -> None:
    r"""Finalizer function for RecordTensor."""
    owner = owner()
    if owner:
        if hasattr(owner, f"_{name}_dt"):
            delattr(owner, f"_{name}_dt")
        if hasattr(owner, f"_{name}_duration"):
            delattr(owner, f"_{name}_duration")
        if hasattr(owner, f"_{name}_inclusive"):
            delattr(owner, f"_{name}_inclusive")
        if hasattr(owner, f"_{name}_pointer"):
            delattr(owner, f"_{name}_pointer")


class RecordTensor(ShapedTensor):
    r"""Tensor attribute with recorded history.

    Read Operations: :py:meth:`peek`, :py:meth:`pop`, :py:meth:`read`,
    :py:meth:`readrange`, :py:meth:`select`

    Write Operations: :py:meth:`push`, :py:meth:`write`, :py:meth:`writerange`,
    :py:meth:`insert`

    Args:
        owner (Module): module to which this attribute will belong.
        name (str): name of the attribute.
        step_time (float): length of time between stored values in the record.
        duration (float): length of time over which prior values are stored.
        value (torch.Tensor | nn.Parameter | None): tensor-like data for the attribute.
        constraints (dict[int, int] | None, optional): constraints given as a
            dictionary of dimensions to their corresponding size. Defaults to ``None``.
        persist_data (bool, optional): if the data should persist across the
            state dictionary, only used with buffers. Defaults to ``True``.
        persist_constraints (bool, optional): if the constraints should persist
            across the state dictionary. Defaults to ``False``.
        persist_temporal (bool, optional): if temporal information (step time and
            duration) should persist across the state dictionary. Defaults to ``False``.
        strict (bool, optional): if each dimension must specify a unique dimension
            for the tensor. Defaults to ``True``.
        live (bool, optional): if constraint validity should be tested on
            assignment. Defaults to ``False``.
        inclusive (bool, optional): if the duration should represent the maximum time which can
            be sampled. Defaults to ``False``.

    Note:
        While the last dimension of a selector tensor is used to index times, the
        underlying storage uses the first dimension such that each time slice is stored
        contiguously.
    """

    LinkedAttributes = namedtuple(  # type: ignore
        "RecordTensorAttributes",
        ("data", "constraints", "dt", "duration", "inclusive", "pointer"),
    )

    def __init__(
        self,
        owner: Module,
        name: str,
        step_time: float,
        duration: float,
        value: torch.Tensor | nn.Parameter | None,
        constraints: dict[int, int] | None = None,
        persist_data: bool = True,
        persist_constraints: bool = False,
        persist_temporal: bool = False,
        strict: bool = True,
        live: bool = False,
        inclusive: bool = False,
    ):
        # argument validation
        step_time = argtest.gt("step_time", step_time, 0, float)
        duration = argtest.gte("duration", duration, 0, float)

        # size of the history dimension
        size = max(math.ceil(duration / step_time) + bool(inclusive), 1)

        # alter constraints
        constraints = {
            (d + 1 if d >= 0 else d): s
            for d, s in (constraints if constraints else {}).items()
        } | {0: size}

        # reshape value if not ignored
        if not self._ignore(value):
            assert value is not None
            if isinstance(value, nn.Parameter):
                value.data = value.data.unsqueeze(0).repeat(
                    *chain((size,), repeat(1, times=value.ndim))
                )
            else:
                value = value.unsqueeze(0).repeat(
                    *chain((size,), repeat(1, times=value.ndim))
                )

        # call superclass constructor
        ShapedTensor.__init__(
            self,
            owner,
            name,
            value,
            constraints,
            persist_data=persist_data,
            persist_constraints=persist_constraints,
            strict=strict,
            live=live,
        )

        # finalizer state
        self.__owner = weakref.ref(owner)
        self.__finalizer = weakref.finalize(
            self,
            _recordtensor_finalization,
            self.__owner,
            self.name,
        )

        # registered attribute names
        self.__attributes: RecordTensor.LinkedAttributes = RecordTensor.LinkedAttributes(  # type: ignore
            ShapedTensor.attributes.fget(self).data,  # type: ignore
            ShapedTensor.attributes.fget(self).constraints,  # type: ignore
            f"_{self.name}_dt",
            f"_{self.name}_duration",
            f"_{self.name}_inclusive",
            f"_{self.name}_pointer",
        )

        # register temporal state
        if isinstance(owner, Module) and persist_temporal:
            owner.register_extra(self.__attributes.dt, step_time)
            owner.register_extra(self.__attributes.duration, duration)
            owner.register_extra(self.__attributes.inclusive, inclusive)
        else:
            setattr(owner, self.__attributes.dt, step_time)
            setattr(owner, self.__attributes.duration, duration)
            setattr(owner, self.__attributes.inclusive, inclusive)

        # register the pointer (persist if possible)
        if isinstance(owner, Module):
            owner.register_extra(self.__attributes.pointer, 0)
        else:
            setattr(owner, self.__attributes.pointer, 0)

    @classmethod
    def create(  # type: ignore
        cls,
        owner: Module,
        name: str,
        step_time: float,
        duration: float,
        value: torch.Tensor | nn.Parameter | None,
        constraints: dict[int, int] | None = None,
        persist_data: bool = True,
        persist_constraints: bool = False,
        persist_temporal: bool = False,
        strict: bool = True,
        live: bool = False,
        inclusive: bool = False,
    ) -> None:
        r"""Creates a record tensor and adds it as an attribute.

        The following two calls are equivalent.

        .. code-block:: python

            module.name = RecordTensor(owner, name, step_time, duration, value)

        .. code-block:: python

            RecordTensor.create(module, name, step_time, duration, value)

        Args:
            owner (Module): module to which this attribute will belong.
            name (str): name of the attribute.
            step_time (float): length of time between stored values in the record.
            duration (float): length of time over which prior values are stored.
            value (torch.Tensor | nn.Parameter | None): tensor-like data for the
                attribute.
            constraints (dict[int, int] | None, optional): constraints given as a
                dictionary of dimensions to their corresponding size.
                Defaults to ``None``.
            persist_data (bool, optional): if the data should persist across the
                state dictionary, only used with buffers. Defaults to ``True``.
            persist_constraints (bool, optional): if the constraints should persist
                across the state dictionary. Defaults to ``False``.
            persist_temporal (bool, optional): if temporal information (step time and
                duration) should persist across the state dictionary.
                Defaults to ``False``.
            strict (bool, optional): if each dimension must specify a unique dimension
                for the tensor. Defaults to ``True``.
            live (bool, optional): if constraint validity should be tested on
                assignment. Defaults to ``False``.
            inclusive (bool, optional): if the duration should represent the maximum time which
                can be sampled. Defaults to ``False``.
        """
        constrained = cls(
            owner,
            name,
            step_time,
            duration,
            value,
            constraints,
            persist_data=persist_data,
            persist_constraints=persist_constraints,
            persist_temporal=persist_temporal,
            strict=strict,
            live=live,
            inclusive=inclusive,
        )
        setattr(constrained.owner, constrained.name, constrained)

    @property
    def __constraints(self) -> dict[int, int]:
        r"""Module internal constraint getter."""
        return getattr(self.__owner(), self.__attributes.constraints)

    @property
    def __data(self) -> torch.Tensor | nn.Parameter | None:
        r"""Module internal data getter."""
        return getattr(self.__owner(), self.__attributes.data)

    @__data.setter
    def __data(self, value: torch.Tensor | nn.Parameter | None) -> None:
        r"""Module internal data setter."""
        data = self.__data
        if (
            isinstance(data, nn.Parameter)
            and isinstance(value, torch.Tensor)
            and not isinstance(value, nn.Parameter | nn.UninitializedBuffer)
        ):
            data.data = value
        else:
            setattr(self.__owner(), self.__attributes.data, value)

    @property
    def __dt(self) -> float:
        r"""Module internal step time getter."""
        return getattr(self.__owner(), self.__attributes.dt)

    @property
    def __duration(self) -> float:
        r"""Module internal duration getter."""
        return getattr(self.__owner(), self.__attributes.duration)

    @property
    def __inclusive(self) -> bool:
        r"""Module internal inclusivity getter."""
        return getattr(self.__owner(), self.__attributes.inclusive)

    @property
    def __pointer(self) -> int:
        r"""Module internal pointer getter."""
        return getattr(self.__owner(), self.__attributes.pointer)

    @__pointer.setter
    def __pointer(self, value: int) -> None:
        r"""Module internal pointer setter."""
        setattr(self.__owner(), self.__attributes.pointer, int(value))

    @property
    def __recordsz(self) -> int:
        r"""Module internal record size getter."""
        return self.__constraints[0]

    @property
    def attributes(self) -> RecordTensor.LinkedAttributes:  # type: ignore
        r"""Names of the dependent attributes created.

        This is a named tuple with attributes ``data``, ``constraints``, ``dt``,
        ``duration``, and ``pointer``.

        Returns:
            RecordTensor.LinkedAttributes: names of the created attributes in the
            containing module.
        """
        return self.__attributes

    @property
    def constraints(self) -> dict[int, int]:
        r"""Dictionary of registered constraints.

        Each key corresponds to a dimension of the tensor, and its associated value
        is the size of that dimension. This shifts the dimensions of negative
        constraints to make the record size transparent.

        Returns:
            dict[int, int]: dictionary of constraints.
        """
        return {
            (d - 1 if d >= 0 else d): s for d, s in self.__constraints.items() if d != 0
        }

    @property
    def dt(self) -> float:
        r"""Length of time between recorded observations.

        In the same units as :py:attr:`duration`.

        If the step time is changed such that the record size needs to change, a
        :py:meth:`reconstrain` operation will be performed automatically, preserving
        the newest entires. Stored values may no longer be logically valid, but will
        still be accessible.

        Args:
            value (float): new time step length.

        Returns:
            float: length of the time step.
        """
        return self.__dt

    @dt.setter
    def dt(self, value: float) -> None:
        # validate argument
        value = argtest.gt("dt", value, 0, float)

        # assign updated step time
        setattr(self.__owner(), self.__attributes.dt, value)

        # recompute size of the history dimension
        size = max(math.ceil(self.__duration / self.__dt) + self.__inclusive, 1)

        # reconstrain if required
        if size != self.__recordsz:
            with torch.no_grad():
                self.align(0)
                _ = ShapedTensor.reconstrain(self, 0, size)

    @property
    def duration(self) -> float:
        r"""Length of time over which prior values are stored.

        In the same units as :py:attr:`dt`.

        If the step time is changed such that the record size needs to change, a
        :py:meth:`reconstrain` operation will be performed automatically, preserving
        the newest entires.

        Args:
            value (float): new length of the record.

        Returns:
            float: length of the record as the length of time.

        Note:
            If ``duration`` is not evenly divided by :py:attr:`dt`, then the number
            of observations stored will be rounded up. This property will always return
            the duration set although the range of accessible values may be larger.
        """
        return self.__duration

    @duration.setter
    def duration(self, value: float) -> None:
        # validate argument
        value = argtest.gte("duration", value, 0, float)

        # assign updated duration
        setattr(self.__owner(), self.__attributes.duration, value)

        # recompute size of the history dimension
        size = max(math.ceil(self.__duration / self.__dt) + self.__inclusive, 1)

        # reconstrain if required
        if size != self.__recordsz:
            with torch.no_grad():
                self.align(0)
                _ = ShapedTensor.reconstrain(self, 0, size)

    @property
    def inclusive(self) -> bool:
        r"""If the duration represents the maximum accessible range.

        Args:
            value (bool): inclusivity of observations.

        Returns:
            bool: if the duration represents the maximum accessible range.

        Note:
            :py:attr:`duration` will remain the same but its interpretation may change.
            If :py:attr:`inclusive` is set to a different value, :py:attr:`recordsz` will
            change and the storage will be rebuilt.
        """
        return self.__inclusive

    @inclusive.setter
    def inclusive(self, value: bool) -> None:
        # copy current duration
        duration = self.__duration

        # assign updated inclusivity
        setattr(self.__owner(), self.__attributes.inclusive, value)

        # reassign duration via propety
        self.duration = duration

    @property
    def recordsz(self) -> int:
        r"""Number of observations stored.

        .. math::
            N = \max\left(\left\lceil \frac{T}{\Delta t} \right\rceil + [\text{incl}], 1\right)

        For :py:attr:`duration` :math:`T`, :py:attr:`dt` :math:`\Delta t`, and
        :py:attr:`inclusive` :math:`\text{incl}`.

        Returns:
            int: length of the record as the number of observations.
        """
        return self.__recordsz

    @property
    def shape(self) -> tuple[int, ...] | None:
        r"""Shape of the observations.

        Returns:
            tuple[int, ...] | None: shape of the observations, ``None`` when storage
            is uninitialized (ignored).
        """
        # strongly reference data
        data = self.__data

        if self._ignore(data):
            return None
        else:
            assert data is not None
            return (*data.shape[1:],)

    @property
    def pointer(self) -> int:
        r"""Current index of the pointer.

        The location of the pointer indicates the location of the next observation
        which will be overwritten.

        Returns:
            int: current index of the pointer.
        """
        return self.__pointer

    @property
    def latest(self) -> torch.Tensor | None:
        r"""Most recent stored observation.

        When used as a getter, this is an alias for :py:meth:`peek`.

        When used as a setter, this is an alias for :py:meth:`push` with ``inplace`` set
        to ``False``.

        When used as a deleter, this is an alias for :py:meth:`decr` with ``pos`` set
        to ``1``.

        Args:
            value (torch.Tensor): observation to push.

        Returns:
            torch.Tensor | None: value of the most recently recorded observation.
        """
        return self.peek()

    @latest.setter
    def latest(self, value: torch.Tensor) -> None:
        self.push(value, inplace=False)

    @latest.deleter
    def latest(self) -> None:
        self.decr(1)

    @property
    def value(self) -> torch.Tensor | nn.Parameter | None:
        r"""Record storage tensor.

        When created as a :py:class:`~torch.nn.parameter.Parameter`, assignment to
        ``None`` is prevented. If the current ``value`` is a
        :py:class:`~torch.nn.parameter.Parameter` but the assigned value is a
        :py:class:`~torch.Tensor`, it will automatically assign to the ``data``
        attribute of ``value``.

        When used as a deleter, this acts as an alias for
        ``self.deinitialize(use_uninitialized=False)``.

        Args:
            value (value: torch.Tensor | nn.Parameter | None): value to set the storage
                tensor to.
        Returns:
            torch.Tensor | nn.Parameter | None: storage tensor.

        .. admonition:: Shape
            :class: tensorshape

            ``value``, ``return``:

            :math:`N \times S_0 \times \cdots`

            Where:
                * :math:`N` is the number of observations the storage can hold,
                  equal to :py:attr:`recordsz`.
                * :math:`S_0, \ldots` are the dimensions of each observation, given
                  by :py:attr:`shape`.

        Caution:
            This should be used for setting properties of the underlying storage
            (device, data type, etc.) and for advanced initialization/deinitialization.
            General read/write access should generally be performed through the provided
            if possible.

        Note:
            If after assigning the new value, :py:attr:`~ShapedTensor.ignored` is
            ``True``, the pointer will be moved to ``0``, otherwise it will not be
            changed.
        """
        return ShapedTensor.value.fget(self)  # type: ignore

    @value.setter
    def value(self, value: torch.Tensor | nn.Parameter | None) -> None:
        _ = ShapedTensor.value.fset(self, value)  # type: ignore
        if self._ignore(self.__data):
            setattr(self.__owner(), f"_{self.__name}_pointer", 0)

    @value.deleter
    def value(self) -> None:
        self.deinitialize(False)

    def align(self, index: int = 0) -> None:
        r"""Aligns the storage such that the oldest observation is at a specified index.

        Args:
            index (int, optional): index to align to. Defaults to ``0``.

        Raises:
            RuntimeError: cannot align when storage is uninitialized (ignored).
        """
        # check for a valid index
        index = argtest.index("index", index, self.__recordsz, "recordsz")

        # strongly reference data
        data = self.__data

        # only constrained state can be aligned
        if not self._ignore(data):
            assert data is not None
            self.__data = data.roll(index - self.__pointer, 0)
            self.__pointer = index
        else:
            raise RuntimeError("cannot align uninitialized storage")

    def reset(self, fill: Any | None = 0) -> None:
        r"""Fills the storage with a given value and aligns it to zero.

        Args:
            fill (Any | None, optional): value with which to fill the storage, or if
                ``None`` no fill will be applied. Defaults to ``0``.
        """
        # strongly reference data
        data = self.__data

        if fill is not None:
            # perform fill if not ignored
            if not self._ignore(data):
                assert data is not None
                with torch.no_grad():
                    data.fill_(fill)

            # reset pointer to start
            self.__pointer = 0

        else:
            self.align(0)

    def initialize(
        self,
        shape: tuple[int, ...],
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        fill: Any = 0,
    ) -> torch.Tensor | nn.Parameter:
        r"""Initializes the storage tensor.

        Args:
            shape (tuple[int, ...]): shape, excluding the record dimension, of the
                storage tensor.
            device (torch.device | None, optional): overrides the device on which to
                place the tensor when not ``None``. Defaults to ``None``.
            dtype (torch.dtype | None, optional): overrides data tyoe of the tensor
                when not ``None``. Defaults to ``None``.
            fill (Any, optional): value with which to fill the tensor.
                Defaults to ``0``.

        Returns:
            torch.Tensor | nn.Parameter: initialized buffer or parameter.
        """
        # strongly reference data and get record size
        data, recordsz = self.__data, self.__recordsz

        # pytorch uninitialized buffer or parameter
        if isinstance(data, nn.UninitializedBuffer | nn.UninitializedParameter):
            # materialize, which alters in-place
            data.materialize((recordsz, *shape), device=device, dtype=dtype)

            # fill in-place
            with torch.no_grad():
                data.fill_(fill)

            assert isinstance(self.__data, torch.Tensor | nn.Parameter)

        # initialized but empty tensor or parameter
        elif isinstance(data, torch.Tensor):
            # reassign using value defaults, automatic parameter assignment test
            self.__data = full(
                data, fill, shape=(recordsz, *shape), dtype=dtype, device=device
            )

        # none value, always a buffer, simple overwrite
        else:
            self.__data = torch.full(
                (recordsz, *shape), fill, dtype=dtype, device=device
            )

        # set pointer to zero and return created or materialized attribute
        self.__pointer = 0
        return self.__data

    def deinitialize(
        self,
        use_uninitialized: bool = False,
    ) -> torch.Tensor | nn.Parameter:
        r"""Deinitializes the storage tensor.

        This either assigns an empty tensor with shape ``[0]`` as the value or either
        :py:class:`~torch.nn.parameter.UninitializedBuffer` or
        :py:class:`~torch.nn.parameter.UninitializedParameter`. The device, data type, and
        gradient requirement will be preserved.

        If the storage tensor is already not initialized, it will still be reassigned.
        If it is ``None``, the defaults of will be used and it will be reassigned either
        with ``UninitializedBuffer()`` or ``torch.empty(0)``.

        Args:
            use_uninitialized (bool, optional): if an uninitialized buffer or
                uninitialized parameter should be used. Defaults to ``False``.

        Returns:
            torch.Tensor | nn.Parameter: deinitialized storage.
        """
        # strongly reference data
        data = self.__data

        if isinstance(data, nn.Parameter):
            if use_uninitialized:
                self.__data = nn.UninitializedParameter(  # type: ignore
                    requires_grad=data.requires_grad,
                    device=data.device,
                    dtype=data.dtype,
                )
            elif isinstance(data, nn.UninitializedParameter):
                self.__data = nn.Parameter(
                    torch.empty(0, dtype=data.dtype, device=data.device),
                    data.requires_grad,
                )
            else:
                data.data = empty(data, shape=(0,))

        elif isinstance(data, torch.Tensor):
            if use_uninitialized:
                self.__data = nn.UninitializedBuffer(  # type: ignore
                    requires_grad=data.requires_grad,
                    device=data.device,
                    dtype=data.dtype,
                )
            elif isinstance(data, nn.UninitializedBuffer):
                self.__data = torch.empty(
                    0,
                    dtype=data.dtype,
                    device=data.device,
                    requires_grad=data.requires_grad,
                )
            else:
                self.__data = empty(data, shape=(0,))

        else:
            if use_uninitialized:
                self.__data = nn.UninitializedBuffer()
            else:
                self.__data = torch.empty(0)

        # set pointer to zero and return created tensor or parameter
        self.__pointer = 0
        return self.__data

    def incr(self, pos: int = 1) -> int:
        r"""Moves the pointer forward.

        Args:
            pos (int, optional): number of steps by which to move the pointer
                forward. Defaults to ``1``.

        Returns:
            int: new location of the pointer.

        Raises:
            RuntimeError: cannot modify the pointer when the storage is
                uninitialized (ignored).
        """
        if self._ignore(self.__data):
            raise RuntimeError("cannot modify pointer when storage is uninitialized")
        else:
            self.__pointer = _unwind_ptr(self.__pointer, -pos, self.__recordsz)

        return self.__pointer

    def decr(self, pos: int = 1) -> None:
        r"""Moves the pointer backward.

        Args:
            pos (int, optional): number of steps by which to move the pointer
                backward. Defaults to ``1``.

        Returns:
            int: new location of the pointer.

        Raises:
            RuntimeError: cannot modify the pointer when the storage is
                uninitialized (ignored).
        """
        if self._ignore(self.__data):
            raise RuntimeError("cannot modify pointer when storage is uninitialized")
        else:
            self.__pointer = _unwind_ptr(self.__pointer, pos, self.__recordsz)

        return self.__pointer

    def peek(self) -> torch.Tensor | None:
        r"""Retrieves the most recently pushed observation.

        If the state is uninitialized, then ``None`` will be returned. Otherwise this
        is an alias for ``self.read(offset=1)``.

        Returns:
            torch.Tensor | None: most recently pushed observation.
        """
        if self._ignore(self.__data):
            return None
        else:
            return self.read(1)

    def pop(self) -> torch.Tensor | None:
        r"""Retrieves the most recently pushed observation and decrements the pointer.

        If the state is uninitialized, then ``None`` will be returned and the pointer
        will be unaltered.

        Returns:
            torch.Tensor | None: most recently pushed observation.

        Important:
            Unlike a pop-operation on most data structures, this does not affect the
            underlying storage. It only moves the pointer back so the next
            :py:meth:`push` will overwrite that value.
        """
        if self._ignore(self.__data):
            return None
        else:
            self.decr(1)
            return self.read(0)

    def push(self, obs: torch.Tensor, inplace: bool = False) -> None:
        r"""Records an observation to the current location and advances the pointer.

        This is an alias for the following code.

        .. code-block:: python

            self.write(value, offset=0, inplace=inplace)
            self.incr(pos=1)

        If the storage is uninitialized, then it will be made automatically. The
        data type and gradient requirement will be preserved, but the storage
        will be put on the same device as ``obs``. If ``inplace`` is ``False``, the
        data type may be promoted to that of ``obs`` by PyTorch. Storage will be
        zero-filled.

        Args:
            obs (torch.Tensor): observation to write.
            inplace (bool, optional): if the operation should be performed in-place
                with :py:class:`torch.no_grad`. Defaults to ``False``.
        """
        if self._ignore(self.__data):
            self.initialize(obs.shape, device=obs.device, fill=0)
        self.write(obs, offset=0, inplace=inplace)
        self.incr(1)

    def read(self, offset: int = 1) -> torch.Tensor:
        r"""Reads the observation at an index relative to the pointer.

        The pointer specifies the next observation to overwrite, and the offset
        specifies the number of observations back from which to read. The default
        value ``offset=1`` will return the most recently pushed observation.

        Args:
            offset (int, optional): number of steps before the pointer.
                Defaults to ``1``.

        Raises:
            RuntimeError: cannot read from uninitialized (ignored) storage.

        Returns:
            torch.Tensor: observation at the specified index.

        .. admonition:: Shape
            :class: tensorshape

            ``return``:

            :math:`S_0 \times \cdots`

            Where:
                * :math:`S_0, \ldots` are the dimensions of each observation, given
                  by :py:attr:`shape`.
        """
        # strongly reference data and get from internal properties
        data, ptr, recordsz = self.__data, self.__pointer, self.__recordsz

        # cannot read from uninitialized storage
        if self._ignore(data):
            raise RuntimeError("cannot read from uninitialized storage")
        else:
            return data[_unwind_ptr(ptr, offset, recordsz), ...]

    def write(self, obs: torch.Tensor, offset: int = 0, inplace: bool = False) -> None:
        r"""Writes an observation at an index relative to the pointer.

        The pointer specifies the next observation to overwrite, and the offset
        specifies the number of observations back from which to write. The default
        value ``offset=0`` overwrite the oldest observation.

        Args:
            obs (torch.Tensor): observation to write at the specified offset.
            offset (int, optional): number of steps before the pointer.
                Defaults to ``0``.
            inplace (bool, optional): if the operation should be performed in-place
                with :py:class:`torch.no_grad`. Defaults to ``False``.

        Raises:
            RuntimeError: cannot write to uninitialized (ignored) storage.
            ValueError: shape of ``value`` must match the shape of an observation.

        .. admonition:: Shape
            :class: tensorshape

            ``obs``:

            :math:`S_0 \times \cdots`

            Where:
                * :math:`S_0, \ldots` are the dimensions of each observation, given
                  by :py:attr:`shape`.

        Important:
            The :py:class:`~torch.dtype` of ``obs`` is not changed, so when ``inplace``
            is set to ``False``, this may cause the data type of the stored tensor to
            change.
        """
        # strongly reference data and get from internal properties
        data, ptr, recordsz = self.__data, self.__pointer, self.__recordsz

        # cannot write to uninitialized storage
        if self._ignore(data):
            raise RuntimeError("cannot write to uninitialized storage")

        # shape must be the same as an observation
        elif (*obs.shape,) != (*data.shape[1:],):
            raise ValueError(
                f"shape of 'obs' {(*obs.shape,)} must have the shape "
                f"{(*data.shape[1:],)}, like a stored observation"
            )

        # in-place overwrite
        elif inplace:
            with torch.no_grad():
                index = _unwind_ptr(ptr, offset, recordsz)
                data[index, ...] = obs.to(dtype=data.dtype)

        # splice in
        else:
            index = _unwind_ptr(ptr, offset, recordsz)
            self.__data = torch.cat(
                (
                    data[slice(None, index), ...],
                    obs.to(dtype=data.dtype).unsqueeze(0),
                    data[slice(index + 1, None), ...],
                ),
                0,
            )

    def readrange(
        self,
        length: int,
        offset: int | torch.Tensor = 1,
        forward: bool = False,
    ) -> torch.Tensor:
        r"""Reads multiple sequential observations.

        When ``forward`` is ``False``, then ``length`` observations will be read
        from the following interval.

        .. code-block:: python

            [pointer - offset - length + 1, pointer - offset]

        When ``forward`` is ``True``, then ``length`` observations will be read
        from the following interval.

        .. code-block:: python

            [pointer - offset, pointer - offset + length - 1]

        Args:
            length (int): number of observations to read.
            offset (int | torch.Tensor, optional): number of steps before the pointer.
                Defaults to ``1``.
            forward (bool, optional): if the offset pointer indicates the index of the
                first observation. Defaults to ``False``.

        Raises:
            RuntimeError: cannot read from uninitialized (ignored) storage.
            ValueError: shape of ``offset`` must match the shape of an observation.

        Returns:
            torch.Tensor: observations at the specified indices.

        .. admonition:: Shape
            :class: tensorshape

            ``offset``:

            :math:`S_0 \times \cdots`

            ``return``:

            :math:`S_0 \times \cdots \times L`

            Where:
                * :math:`S_0, \ldots` are the dimensions of each observation, given
                  by :py:attr:`shape`.
                * :math:`L`, the number of observations, given by ``length``.
        """
        # strongly reference data and get from internal properties
        data, ptr, recordsz = self.__data, self.__pointer, self.__recordsz

        # shift offset backward if using noninitial offset
        if not forward:
            offset = offset + (length - 1)

        # cannot read from uninitialized storage
        if self._ignore(data):
            raise RuntimeError("cannot read from uninitialized storage")

        # scalar offset
        elif not isinstance(offset, torch.Tensor):
            start = _unwind_ptr(ptr, offset, recordsz)
            end = _unwind_ptr(ptr, offset - length, recordsz)

            if start > end:
                return ein.rearrange(
                    torch.cat((data[start:, ...], data[:end, ...]), 0), "t ... -> ... t"
                )
            else:
                return ein.rearrange(data[start:end, ...], "t ... -> ... t")

        # shape must be the same as an observation
        elif (*offset.shape,) != (*data.shape[1:],):
            raise ValueError(
                f"shape of 'offset' {(*offset.shape,)} must have the shape "
                f"{(*data.shape[1:],)}, like a stored observation"
            )

        # tensor offset
        else:
            offset = ein.rearrange(
                offset.unsqueeze(-1)
                - torch.arange(0, length, dtype=torch.int64, device=offset.device),
                "... t -> t ...",
            )
            return ein.rearrange(
                torch.gather(data, 0, _unwind_tensor_ptr(ptr, offset, recordsz)),
                "t ... -> ... t",
            )

    def writerange(
        self,
        obs: torch.Tensor,
        offset: int | torch.Tensor = 0,
        forward: bool = False,
        inplace: bool = False,
    ) -> None:
        r"""Writes multiple sequential observations.

        When ``forward`` is ``False``, then ``length`` observations will be written
        to the following interval.

        .. code-block:: python

            [pointer - offset - length + 1, pointer - offset]

        When ``forward`` is ``True``, then ``length`` observations will be written
        to the following interval.

        .. code-block:: python

            [pointer - offset, pointer - offset + length - 1]

        Args:
            obs (torch.Tensor): observation to write at the specified offsets.
            offset (int | torch.Tensor, optional): number of steps before the pointer.
                Defaults to ``0``.
            forward (bool, optional): if the offset pointer indicates the index of the
                first observation. Defaults to ``False``.
            inplace (bool, optional): if the operation should be performed in-place
                with :py:class:`torch.no_grad`. Defaults to ``False``.

        Raises:
            RuntimeError: cannot write to uninitialized (ignored) storage.
            ValueError: written observations need to have a shape compatible with
                storage.
            ValueError: cannot write more observations to storage than storage records.
            ValueError: shape of ``offset`` must match the shape of an observation.

        Important:
            The :py:class:`~torch.dtype` of ``obs`` is not changed, ``offset`` is not
            a tensor, and ``inplace`` is set to ``False``, this may cause the data type
            of the stored tensor to change. When ``offset`` is a tensor, a
            :py:func:`~torch.scatter` operation is performed so the data type will first
            be converted tot hat of the underlying storage.

        .. admonition:: Shape
            :class: tensorshape

            ``obs``:

            :math:`S_0 \times \cdots \times L`

            ``offset``:

            :math:`S_0 \times \cdots`

            ``return``:

            :math:`S_0 \times \cdots \times L`

            Where:
                * :math:`S_0, \ldots` are the dimensions of each observation, given
                  by :py:attr:`shape`.
                * :math:`L`, the number of observations.
        """
        # strongly reference data and get from internal properties
        data, ptr, recordsz = self.__data, self.__pointer, self.__recordsz
        length = obs.shape[-1]

        # shift offset backward if using noninitial offset
        if not forward:
            offset = offset + (obs.shape[-1] - 1)

        # cannot write to uninitialized storage
        if self._ignore(data):
            raise RuntimeError("cannot write to an uninitialized storage")

        # observations should be shaped like storage at all but final
        if obs.shape[:-1] != data.shape[1:]:
            raise ValueError(
                f"'obs' has shape {(*obs.shape,)} but must have the same shape as "
                f"multiple observations {(*data.shape[1:],)} stacked along the final dimension"
            )

        # cannot write beyond storage
        if obs.shape[-1] > data.shape[0]:
            raise ValueError(
                f"cannot write the ({obs.shape[-1]}) observations from 'obs' when "
                f"storage only holds ({data.shape[0]}) observations"
            )

        # scalar offset
        elif not isinstance(offset, torch.Tensor):
            ptr = _unwind_ptr(ptr, offset, recordsz)

            # reshape observations for compatibility
            obs = ein.rearrange(obs, "... t -> t ...")

            # in-place
            if inplace:
                with torch.no_grad():
                    offset = -torch.arange(
                        0, length, dtype=torch.int64, device=data.device
                    )
                    indices = _unwind_tensor_ptr(ptr, offset, recordsz)
                    data[indices, ...] = obs.to(dtype=data.dtype)

            # noncontiguous range
            elif ptr + length > recordsz:
                self.__data = torch.cat(
                    (
                        obs[slice(recordsz - ptr, None), ...].to(dtype=data.dtype),
                        data[slice(length - (recordsz - ptr), ptr), ...],
                        obs[slice(None, recordsz - ptr), ...].to(dtype=data.dtype),
                    ),
                    0,
                )

            # contiguous range
            else:
                self.__data = torch.cat(
                    (
                        data[slice(0, ptr), ...],
                        obs,
                        data[slice(ptr + length, None), ...],
                    ),
                    0,
                )

        # shape must be the same as an observation
        elif (*offset.shape,) != (*data.shape[1:],):
            raise ValueError(
                f"shape of 'offset' {(*offset.shape,)} must have the shape "
                f"{(*data.shape[1:],)}, like a stored observation"
            )

        # tensor offset
        else:
            offset = ein.rearrange(
                offset.unsqueeze(-1)
                - torch.arange(0, length, dtype=torch.int64, device=offset.device),
                "... t -> t ...",
            )
            indices = _unwind_tensor_ptr(ptr, offset, recordsz)

            # reshape observations for compatibility
            obs = ein.rearrange(obs, "... t -> t ...")

            # write to storage
            if inplace:
                with torch.no_grad():
                    data.scatter_(0, indices, obs)
            else:
                self.__data = torch.scatter(data, 0, indices, obs)

    def select(
        self,
        time: torch.Tensor | float,
        interp: Interpolation | None = None,
        *,
        tolerance: float = 1e-6,
        offset: int = 1,
        interp_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        r"""Selects previously observed elements of the record tensor by time.

        If ``time`` is a scalar and is within tolerance of an integer index, then
        a observation will be returned without ever attempting interpolation.

        If ``time`` is a tensor, interpolation will be called regardless, and the time
        passed into the interpolation call will be set to either ``0`` or
        :py:attr:`dt`. Interpolation results are then overwritten with exact values
        before returning.

        Args:
            time (torch.Tensor | float): time at which to read each element from the
                underlying storage.
            interp (Interpolation | None, optional): function to interpolate between
                neighboring observations, nearest when ``None``. Defaults to ``None``.
            tolerance (float, optional): maximum difference in time from a discrete
                sample to consider a time co-occurring with the sample.
                Defaults to ``1e-6``.
            offset (int, optional): number of steps before the pointer.
                Defaults to ``1``.
            interp_kwargs (dict[str, Any] | None, optional): dictionary of keyword
                arguments to pass to ``interp``. Defaults to ``None``.

        Raises:
            RuntimeError: cannot select from uninitialized (ignored) storage.
            ValueError: ``time`` must have the same number of dimensions as an
                observation or that number plus one.
            ValueError: all elements of ``time`` must be within the range of observation
                times plus the tolerance.

        Returns:
            torch.Tensor: interpolated values selected at a prior times.

        .. admonition:: Shape
            :class: tensorshape

            ``time``:

            :math:`S_0 \times \cdots \times [D]`

            ``return``:

            :math:`S_0 \times \cdots \times [D]`

            Where:
                * :math:`S_0, \ldots` are the dimensions of each observation, given
                  by :py:attr:`shape`.
                * :math:`D` are the number of distinct observations to select.
        """
        # use nearest interpolation by default
        if not interp:
            interp = interp_nearest

        # strongly reference data and get from internal properties
        data = self.__data
        ptr, recordsz, dt = self.__pointer, self.__recordsz, self.__dt

        # cannot select from uninitialized storage
        if self._ignore(data):
            raise RuntimeError("cannot select from uninitialized storage")

        # tensor time
        elif isinstance(time, torch.Tensor):
            # check if the output should be squeezed
            if time.ndim == data.ndim - 1:
                squeeze = True
                time = time.unsqueeze(-1)
            elif time.ndim == data.ndim:
                squeeze = False
            else:
                raise ValueError(
                    f"'time' has an incompatible number of dimensions ({time.ndim}), "
                    f"it must have either {data.ndim - 1} or {data.ndim} dimensions"
                )

            # check that times are in range
            tmin, tmax = time.amin(), time.amax()
            if tmin < -tolerance or tmax > dt * (recordsz - 1) + tolerance:
                raise ValueError(
                    f"all elements of 'time' (min={tmin}, max={tmax}) must be within "
                    f"the valid range of observations including tolerance, the "
                    f"interval [{-tolerance}, {dt * (recordsz - 1) + tolerance}]"
                )

            # compute continuous shft
            shift = time / dt
            shiftr = shift.round()
            shift = ein.rearrange(
                torch.where(torch.abs(dt * shiftr - time) <= tolerance, shiftr, shift),
                "... t -> t ...",
            )

            # update offset with shift
            offset = offset + shift

            # indices of the nearest observations
            prev_idx, next_idx = offset.ceil(), offset.floor()
            stacked_idx = _unwind_tensor_ptr(
                ptr, torch.cat((prev_idx, next_idx), 0), recordsz
            )

            # get stored observations at specified indices
            prev_data, next_data = torch.tensor_split(
                torch.gather(data, 0, stacked_idx),
                (offset.shape[0],),
                0,
            )

            # interpolate from neighboring observations
            res = interp(
                prev_data,
                next_data,
                dt - dt * (shift % 1),
                dt,
                **(interp_kwargs if interp_kwargs else {}),
            )

            # bypass interpolation for exact indices
            res = ein.rearrange(
                torch.where(prev_idx == next_idx, prev_data, res), "t ... -> ... t"
            )

            # conditionally squeeze and return
            return res.squeeze(-1) if squeeze else res

        # scalar time
        else:
            # cast time and check for a valid range
            disptime, time = time, float(time)
            if time < -tolerance or time > dt * (recordsz - 1) + tolerance:
                raise ValueError(
                    f"'time' ({disptime}) must be within the valid range of "
                    "observations, including tolerance, the interval "
                    f"[{-tolerance}, {dt * (recordsz - 1) + tolerance}]"
                )

            # compute continuous shift
            shift = time / dt

            # directly read when within tolerance
            if abs(dt * round(shift) - time) <= tolerance:
                return data[_unwind_ptr(ptr, offset + round(shift), recordsz), ...]

            # interpolate between observation
            else:
                # update offset with shift
                offset = offset + shift

                return interp(
                    data[_unwind_ptr(ptr, math.ceil(offset), recordsz), ...],
                    data[_unwind_ptr(ptr, math.floor(offset), recordsz), ...],
                    fullc(data, dt - dt * (shift % 1), shape=data.shape[1:]),
                    dt,
                    **(interp_kwargs if interp_kwargs else {}),
                )

    def insert(
        self,
        obs: torch.Tensor,
        time: torch.Tensor | float,
        extrap: Extrapolation | None = None,
        *,
        tolerance: float = 1e-6,
        offset: int = 0,
        inplace: bool = False,
        extrap_kwargs: dict[str, Any] | None = None,
    ) -> None:
        r"""Inserts new elements into the record tensor by time.

        If ``time`` is a scalar and is within tolerance of an integer index, then
        the observation will be written without ever attempting extrapolation.

        If ``time`` is a tensor, interpolation will be called regardless, and the time
        passed into the extrapolation call will be set to either ``0`` or
        :py:attr:`dt`. Extrapolation results are then overwritten with exact values
        before writing.

        The :py:class:`~torch.dtype` of elements inserted into the underlying storage
        will be cast back to the data type of the storage after extrapolation.

        Args:
            obs (torch.Tensor): observation to write at the specified times.
            time (torch.Tensor | float): time at which to write each element of the
                observation.
            extrap (Extrapolation | None, optional): function to interpolate from the
                observation to its neighbors, nearest when ``None``.
                Defaults to ``None``.
            tolerance (float, optional): maximum difference in time from a discrete
                sample to consider a time co-occurring with the sample.
                Defaults to ``1e-6``.
            offset (int, optional): number of steps before the pointer.
                Defaults to ``0``.
            inplace (bool, optional): if the operation should be performed in-place
                with :py:class:`torch.no_grad`. Defaults to ``False``.
            extrap_kwargs (dict[str, Any] | None, optional): dictionary of keyword
                arguments to pass to ``extrap``. Defaults to ``None``.

        Raises:
            RuntimeError: cannot insert into uninitialized (ignored) storage.
            ValueError: shape of ``obs`` must match the shape of an observation.
            ValueError: shape of ``time`` must match the shape of an observation.
            ValueError: all elements of ``time`` must be within the range of observation
                times plus the tolerance.

        .. admonition:: Shape
            :class: tensorshape

            ``obs``, ``time``:

            :math:`S_0 \times \cdots`

            Where:
                * :math:`S_0, \ldots` are the dimensions of each observation, given
                  by :py:attr:`shape`.

        Tip:
            Limitations on :py:func:`torch.scatter` make it risky to insert multiple
            observations at once and is therefore unsupported. To write multiple values
            at once, consider using :py:meth:`writerange`.
        """
        # use nearest extrapolation by default
        if not extrap:
            extrap = extrap_nearest

        # strongly reference data and get from internal properties
        data = self.__data
        ptr, recordsz, dt = self.__pointer, self.__recordsz, self.__dt

        # cannot insert into uninitialized storage
        if self._ignore(data):
            raise RuntimeError("cannot insert into uninitialized storage")

        # shape must be the same as an observation
        elif (*obs.shape,) != (*data.shape[1:],):
            raise ValueError(
                f"shape of 'obs' {(*obs.shape,)} must have the shape "
                f"{(*data.shape[1:],)}, like a stored observation"
            )

        # tensor time
        elif isinstance(time, torch.Tensor):
            # shape must be the same as an observation
            if (*time.shape,) != (*data.shape[1:],):
                raise ValueError(
                    f"shape of 'time' {(*time.shape,)} must have the shape "
                    f"{(*data.shape[1:],)}, like a stored observation"
                )

            # check that times are in range
            tmin, tmax = time.amin(), time.amax()
            if tmin < -tolerance or tmax > dt * (recordsz - 1) + tolerance:
                raise ValueError(
                    f"all elements of 'time' (min={tmin}, max={tmax}) must be within "
                    f"the valid range of observations including tolerance, the "
                    f"interval [{-tolerance}, {dt * (recordsz - 1) + tolerance}]"
                )

            # compute continuous shft
            shift = time / dt
            shiftr = shift.round()
            shift = torch.where(
                torch.abs(dt * shiftr - time) <= tolerance, shiftr, shift
            )

            # unsqueeze first dimension
            obs = obs.unsqueeze(0)
            shift = shift.unsqueeze(0)

            # update offset with shift
            offset = offset + shift

            # indices of the nearest observations
            prev_idx, next_idx = offset.ceil(), offset.floor()
            stacked_idx = _unwind_tensor_ptr(
                ptr, torch.cat((prev_idx, next_idx), 0), recordsz
            )

            # get stored observations at specified indices
            prev_data, next_data = torch.tensor_split(
                torch.gather(data, 0, stacked_idx), 2, 0
            )

            # extrapolate data to write
            prev_exobs, next_exobs = extrap(
                obs,
                dt - dt * (shift % 1),
                prev_data,
                next_data,
                dt,
                **(extrap_kwargs if extrap_kwargs else {}),
            )

            # bypass extrapolation for exact indices
            bypass = prev_idx == next_idx
            prev_exobs = torch.where(bypass, obs, prev_exobs)
            next_exobs = torch.where(bypass, obs, next_exobs)

            # write to storage
            if inplace:
                with torch.no_grad():
                    data.scatter_(
                        0,
                        stacked_idx,
                        torch.cat((prev_exobs, next_exobs), 0).to(dtype=data.dtype),
                    )

            else:
                self.__data = torch.scatter(
                    data,
                    0,
                    stacked_idx,
                    torch.cat((prev_exobs, next_exobs), 0).to(dtype=data.dtype),
                )

        # scalar time
        else:
            # cast time and check for a valid range
            disptime, time = time, float(time)
            if time < -tolerance or time > dt * (recordsz - 1) + tolerance:
                raise ValueError(
                    f"'time' ({disptime}) must be within the valid range of "
                    "observations, including tolerance, the interval "
                    f"[{-tolerance}, {dt * (recordsz - 1) + tolerance}]"
                )

            # compute continuous shift
            shift = time / dt

            # directly write when within tolerance
            if abs(dt * round(shift) - time) <= tolerance:
                self.write(obs, offset + round(shift), inplace=inplace)

            # extrapolate to neighbors
            else:
                # update offset with shift
                offset = offset + shift

                # indices of the nearest observations
                prev_idx = _unwind_ptr(ptr, math.ceil(offset), recordsz)
                next_idx = _unwind_ptr(ptr, math.floor(offset), recordsz)

                # extrapolate data to write
                prev_exobs, next_exobs = extrap(
                    obs,
                    fullc(data, dt - dt * (shift % 1), shape=data.shape[1:]),
                    data[prev_idx, ...],
                    data[next_idx, ...],
                    dt,
                    **(extrap_kwargs if extrap_kwargs else {}),
                )

                # in-place write
                if inplace:
                    with torch.no_grad():
                        data[prev_idx, ...] = prev_exobs
                        data[next_idx, ...] = next_exobs

                # write as contiguous range
                else:
                    self.writerange(
                        torch.stack((prev_exobs, next_exobs), -1).to(dtype=data.dtype),
                        math.ceil(offset),
                        forward=True,
                        inplace=False,
                    )

    def reconstrain(
        self, dim: int, size: int | None
    ) -> torch.Tensor | nn.Parameter | None:
        r"""Add, edit, or remove a constraint.

        Like :py:meth:`ShapedTensor.reconstrain`, except ``dim`` is modified
        to account for the record dimension. Negative values of ``dim`` will have
        ``1`` subtracted from them.

        Args:
            dim (int): dimension on which to modify the constraint.
            size (int | None): new size for the specified dimension.

        Returns:
            torch.Tensor | nn.Parameter | None: newly constrained value.
        """
        # align so oldest data are overwritten or prepended to
        if not self._ignore(self.__data):
            self.align()

        return ShapedTensor.reconstrain(self, dim + (dim >= 0), size)


def _virtualtensor_finalization(owner: weakref.ReferenceType, name: str) -> None:
    r"""Finalizer function for VirtualTensor."""
    owner = owner()
    if owner:
        if hasattr(owner, f"_{name}_ref"):
            delattr(owner, f"_{name}_ref")


class VirtualTensor:
    r"""Tensor attribute derived from other attributes.

    This wraps the functionality around creating a tensor derived from other
    attributes of an object while preserving the type and device conversion enabled by
    :py:meth:`~torch.nn.Module.to`.

    Args:
        owner (Module): module to which this attribute will belong.
        name (str): name of the attribute.
        materializer (str | Callable[[Module, torch.dtype, torch.device], torch.Tensor]): function
            to calculate the value of the virtual tensor.
        dtype (torch.dtype | None, optional): data type of the virtual tensor, PyTorch
            default when ``None``. Defaults to ``None``.
        device (torch.device | None, optional): device on which the virtual tensor is
            stored, PyTorch default when ``None``. Defaults to ``None``.
        persist (bool, optional): if the buffer which stores the dtype and device should
            persist across the state dictionary. Defaults to ``False``.

    Raises:
        AttributeError: string ``materializer`` must be an attribute of ``owner``.
        TypeError: attribute specified by ``materializer`` must be a method.

    Caution:
        This has a finalizer which will delete the attributes added to the module when
        its reference count goes to zero.

    Note:
        When ``materializer`` is a string, it will weakly reference the method in
        ``owner`` as a :py:class:`~weakref.WeakMethod`. Otherwise a strong reference
        is created to the function, and the weakref to ``owner`` is dereferenced and
        passed in on each call.
    """

    LinkedAttributes = namedtuple("VirtualTensorAttributes", ("ref"))

    def __init__(
        self,
        owner: Module,
        name: str,
        materializer: str | Callable[[Module, torch.dtype, torch.device], torch.Tensor],
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        persist: bool = False,
    ):
        # ensure the name is a valid identifier
        _ = argtest.identifier("name", name)

        # basic internal state
        self.__owner = weakref.ref(owner)
        self.__name = name
        self.__finalizer = weakref.finalize(
            self, _virtualtensor_finalization, self.__owner, self.__name
        )

        # materializer state
        if isinstance(materializer, str):
            if not hasattr(owner, materializer):
                raise AttributeError(f"'owner' has no attribute '{materializer}'")
            elif not isinstance(getattr(owner, materializer), MethodType):
                raise TypeError(
                    f"attribute '{materializer}' in 'owner' must be a method"
                )
            else:
                self.__materializer = weakref.WeakMethod(getattr(owner, materializer))
        else:
            self.__materializer = materializer

        # registered attribute names
        self.__attributes = VirtualTensor.LinkedAttributes(f"_{self.__name}_ref")

        # register reference
        if isinstance(owner, nn.Module):
            owner.register_buffer(
                self.__attributes.ref,
                torch.empty(0, dtype=dtype, device=device),
                persistent=persist,
            )
        else:
            setattr(
                owner, self.__attributes.ref, torch.empty(0, dtype=dtype, device=device)
            )

    @classmethod
    def create(
        cls,
        owner: Module,
        name: str,
        materializer: str | Callable[[Module, torch.dtype, torch.device], torch.Tensor],
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        persist: bool = False,
    ) -> None:
        r"""Creates a record tensor and adds it as an attribute.

        The following two calls are equivalent.

        .. code-block:: python

            module.name = VirtualTensor(owner, name, materializer)

        .. code-block:: python

            VirtualTensor.create(module, name, materializer)

        Args:
            owner (Module): module to which this attribute will belong.
            name (str): name of the attribute.
            materializer (str | Callable[[Module, torch.dtype, torch.device], torch.Tensor]): function
                to calculate the value of the virtual tensor.
            dtype (torch.dtype | None, optional): data type of the virtual tensor,
                PyTorch default when ``None``. Defaults to ``None``.
            device (torch.device | None, optional): device on which the virtual tensor
                is stored, PyTorch default when ``None``. Defaults to ``None``.
            persist (bool, optional): if the buffer which stores the dtype and device
                should persist across the state dictionary. Defaults to ``False``.
        """
        virtual = cls(
            owner,
            name,
            materializer,
            dtype=dtype,
            device=device,
            persist=persist,
        )
        setattr(virtual.owner, virtual.name, virtual)

    @property
    def __ref(self) -> torch.Tensor:
        r"""Module internal reference getter."""
        return getattr(self.__owner(), self.__attributes.ref)

    @__ref.setter
    def __ref(self, value: torch.Tensor) -> None:
        r"""Module internal reference setter."""
        return setattr(self.__owner(), self.__attributes.ref, value)

    @property
    def owner(self) -> Module | None:
        r"""Module which owns this attribute.

        Returns:
            Module | None: owner of the attribute if it exists.
        """
        return self.__owner()

    @property
    def name(self) -> str:
        r"""Name of the attribute.

        Two attributes with names derived from ``name`` are added to the owner.

        * ``_{name}_ref``, the data type and device reference tensor.

        Returns:
            str: name of the attribute.
        """
        return self.__name

    @property
    def attributes(self) -> ShapedTensor.LinkedAttributes:
        r"""Names of the dependent attributes created.

        This is a named tuple with attribute ``ref``.

        Returns:
            VirtualTensor.LinkedAttributes: names of the created attributes in the
            containing module.
        """
        return self.__attributes

    @property
    def dtype(self) -> torch.dtype:
        r"""Data type of the reference tensor.

        Args:
            value (torch.dtype): data type of the reference tensor.

        Returns:
            torch.dtype: data type of the reference tensor.
        """
        return self.__ref.dtype

    @dtype.setter
    def dtype(self, value: torch.dtype) -> None:
        self.__ref = self.__ref.to(dtype=value)

    @property
    def device(self) -> torch.device:
        r"""Compute device of the reference tensor.

        Args:
            value (torch.device): compute device of the reference tensor.

        Returns:
            torch.device: compute device of the reference tensor.
        """
        return self.__ref.device

    @device.setter
    def device(self, value: torch.device) -> None:
        self.__ref = self.__ref.to(device=value)

    @property
    def value(self) -> torch.Tensor:
        r"""Computed value of the virtual tensor.

        Although the reference data type and device will be passed into the
        ``materializer`` specified on initialization, it will also be cast with
        :py:meth:`~torch.Tensor.to` afterwards. This should be considered a fallback
        in the event the materializer fails to ensure the output is of the specified
        data type and located on the specified device.

        Returns:
            torch.Tensor: computed tensor.
        """
        # strongly reference underlying tensor
        ref = self.__ref

        # as a method if initialized with a string
        if isinstance(self.__materializer, weakref.WeakMethod):
            return self.__materializer()(ref.dtype, ref.device).to(
                dtype=ref.dtype, device=ref.device
            )
        else:
            return self.__materializer(self.__owner(), ref.dtype, ref.device).to(
                dtype=ref.dtype, device=ref.device
            )

    def to(self, *args, **kwargs) -> None:
        r"""Sets dtype and/or device for the reference tensor.

        This calls :py:meth:`~torch.Tensor.to` with the given positional arguments and
        keyword arguments and reassigns the reference tensor accordingly.
        """
        self.__ref = self.__ref.to(*args, **kwargs)


def _detach_handles(*handles) -> None:
    for h in handles:
        if h:
            h.remove()


class Hook:
    r"""Provides and manages forward hook and prehook functionality.

    `Hook` provides functionality to register and deregister itself as
    forward hook with a :py:class:`~torch.nn.Module` object. This is performed using
    :py:meth:`~torch.nn.Module.register_forward_hook` to register itself as a forward
    hook and it manages the returned `RemovableHandle` to deregister itself.

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
            self._prehook_call = prehook
        else:
            self._prehook_call = None

        if isinstance(posthook, Callable):
            self._posthook_call = posthook
        else:
            self._posthook_call = None

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
        self.__finalizer = None

    def __wrapped_prehook(self, module, *args, **kwargs) -> Any | None:
        if self.trainexec and module.training:
            return self._prehook_call(module, *args, **kwargs)

        if self.evalexec and not module.training:
            return self._prehook_call(module, *args, **kwargs)

    def __wrapped_posthook(self, module, *args, **kwargs) -> Any | None:
        if self.trainexec and module.training:
            return self._posthook_call(module, *args, **kwargs)

        if self.evalexec and not module.training:
            return self._posthook_call(module, *args, **kwargs)

    @property
    def trainexec(self) -> bool:
        r"""If the hook is called when the module passed in is in training mode.

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
        r"""If the hook is called when the module passed in is in evaluation mode.

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
            if self._prehook_call:
                weakself = weakref.ref(self)
                self.__prehook_handle = module.register_forward_pre_hook(
                    lambda module, *args, **kwargs: weakself().__wrapped_prehook(
                        module, *args, **kwargs
                    ),
                    **self.__prehook_kwargs,
                )

            if self._posthook_call:
                weakself = weakref.ref(self)
                self.__posthook_handle = module.register_forward_hook(
                    lambda module, *args, **kwargs: weakself().__wrapped_posthook(
                        module, *args, **kwargs
                    ),
                    **self.__posthook_kwargs,
                )

            if self.__finalizer:
                self.__finalizer.detach()
            self.__finalizer = weakref.finalize(
                self, _detach_handles, self.__prehook_handle, self.__posthook_handle
            )

        else:
            raise RuntimeError(
                f"this {type(self).__name__} is already registered to a module "
                "so new register() was ignored"
            )

    def deregister(self) -> None:
        r"""Deregisters the hook as a forward hook and/or prehook.

        If the :py:class:`Hook` is not registered, this is still safe to call.
        """
        _detach_handles(self.__prehook_handle, self.__posthook_handle)
        self.__prehook_handle = None
        self.__posthook_handle = None
        if self.__finalizer:
            self.__finalizer.detach()
        self.__finalizer = None


class ContextualHook(Hook):
    r"""Provides forward hook and prehook functionality for subclasses.

    This is used to manage references to the ``ContextualHook`` in a safe way for the
    garbage collector (i.e. without cyclic references).

    Args:
        prehook (str | None, optional): name of the prehook method, if any, to execute,
            no prehook when ``None``. Defaults to ``None``.
        posthook (str | None, optional): name of the posthook method, if any, to execute,
            no posthook when ``None``. Defaults to ``None``.
        prehook_kwargs (dict[str, Any] | None, optional): keyword arguments passed to
            :py:meth:`~torch.nn.Module.register_forward_pre_hook`. Defaults to ``None``.
        posthook_kwargs (dict[str, Any] | None, optional): keyword arguments passed to
            :py:meth:`~torch.nn.Module.register_forward_hook`. Defaults to ``None``.
        train_update (bool, optional): if the hooks should be run when hooked module is
            in train mode. Defaults to ``True``.
        eval_update (bool, optional): if the hooks should be run when hooked module is
            in eval mode. Defaults to ``True``.

    Raises:
        RuntimeError: at least one of ``prehook`` and ``posthook`` must not be None.

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
    """

    def __init__(
        self,
        prehook: str | None = None,
        posthook: str | None = None,
        *,
        prehook_kwargs: dict[str, Any] | None = None,
        posthook_kwargs: dict[str, Any] | None = None,
        train_update: bool = True,
        eval_update: bool = True,
    ):
        # check that something will occur on call
        _ = argtest.onedefined(("prehook", prehook), ("posthook", posthook))

        # weakly reference self for prehook
        weakself_bfc = weakref.ref(self)

        def context_prehook(*args, **kwargs):
            getattr(weakself_bfc(), prehook)(*args, **kwargs)

        # weakly reference self for posthook
        weakself_afc = weakref.ref(self)

        def context_posthook(*args, **kwargs):
            getattr(weakself_afc(), posthook)(*args, **kwargs)

        # call superclass constructor
        Hook.__init__(
            self,
            prehook=context_prehook if prehook else None,
            posthook=context_posthook if posthook else None,
            prehook_kwargs=prehook_kwargs if prehook else None,
            posthook_kwargs=posthook_kwargs if posthook else None,
            train_update=train_update,
            eval_update=eval_update,
        )


class StateHook(Module, ContextualHook, ABC):
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
        self._hooked_module = argtest.instance("module", module, nn.Module)

        # construct hook superclass
        ContextualHook.__init__(
            self,
            prehook="_StateHook__wrapped_hook" if as_prehook else None,
            posthook="_StateHook__wrapped_hook" if not as_prehook else None,
            prehook_kwargs={"prepend": prepend},
            posthook_kwargs={"prepend": prepend, "always_call": always_call},
            train_update=train_update,
            eval_update=eval_update,
        )

    def __wrapped_hook(self, module, *args, **kwargs) -> None:
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
        return self._hooked_module

    def register(self) -> None:
        r"""Registers state the hook as a forward hook or prehook."""
        if not self.registered:
            Hook.register(self, self.module)

    def forward(self, force: bool = False, ignore_mode: bool = False) -> None:
        r"""Executes the hook at any time, by default only when registered.

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
