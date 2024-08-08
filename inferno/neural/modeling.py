from __future__ import annotations
from .. import Module
from .._internal import argtest
from ..functional import HalfBounding, FullBounding
from abc import ABC, abstractmethod
from functools import cache, partial
import torch
import torch.nn as nn
from typing import Any, Callable
import weakref


class Accumulator(Module):
    r"""Used to accumulate updates for a parameter."""

    def __init__(self):
        # call superclass constructor
        Module.__init__(self)

        # state
        self._pos = nn.ParameterList()
        self._neg = nn.ParameterList()

        # parameters
        self.reduce = torch.sum
        self.bind = lambda x, p, n: p - n

        # cached state access
        def calc_pos():
            if len(self._pos):
                return self.reduce(torch.stack([*self._pos], 0), 0)
            else:
                return None

        def calc_neg():
            if len(self._neg):
                return self.reduce(torch.stack([*self._neg], 0), 0)
            else:
                return None

        self._pos_cache = cache(calc_pos)
        self._neg_cache = cache(calc_neg)

    @property
    def pos(self) -> torch.Tensor | None:
        r"""Positive update component.

        Args:
            value (torch.Tensor | None): appends to update component.

        Returns:
            torch.Tensor | None: accumulated update component.
        """
        return self._pos_cache()

    @pos.setter
    def pos(self, value: torch.Tensor | None) -> None:
        if value is not None:
            self._pos.append(value)
            self._pos_cache.cache_clear()

    @pos.deleter
    def pos(self) -> None:
        self._pos = nn.ParameterList()
        self._pos_cache.cache_clear()

    @property
    def neg(self) -> torch.Tensor | None:
        r"""Negative update component.

        Args:
            value (torch.Tensor | None): appends to update component.

        Returns:
            torch.Tensor | None: accumulated update component.
        """
        return self._neg_cache()

    @neg.setter
    def neg(self, value: torch.Tensor | None) -> None:
        if value is not None:
            self._neg.append(value)
            self._neg_cache.cache_clear()

    @neg.deleter
    def neg(self) -> None:
        self._neg = nn.ParameterList()
        self._neg_cache.cache_clear()

    def reduction(
        self, fn: Callable[[torch.Tensor, int], torch.Tensor] | None = None
    ) -> None:
        r"""Sets the function used for reducing multiple updates.

        When ``fn`` is ``None``, it sets the default reducer, :py:func:`torch.sum`.

        Args:
            fn (Callable[[torch.Tensor, int], torch.Tensor] | None, optional):
                function for reducing updates. Defaults to ``None``.
        """
        if fn:
            self.reduce = fn
        else:
            self.reduce = torch.sum

    def upperbound(
        self,
        bound: HalfBounding | None,
        max: float | None = None,
        /,
        **kwargs: Any,
    ) -> None:
        r"""Sets the function used for parameter bounding on the upper limit.

        When ``bound`` is ``None``, no upper bound will be applied (and will remove
        any full bound present). When ``bound`` is not ``None``, them ``max`` cannot
        be ``None``.

        Args:
            bound (HalfBounding | None): bounding function.
            max (float | None, optional): upper bound. Defaults to ``None``.
            **kwargs (Any): keyword arguments for the bounding function.
        """
        # convert bounds to tuple
        if not isinstance(self.bind, list):
            self.bind = [lambda x, p: p, lambda x, n: n]

        # determine bounding function
        if bound:
            self.bind[0] = lambda x, p, ub=max, k=kwargs: bound(x, p, ub, **k)
        else:
            self.bind[0] = lambda x, p: p

    def lowerbound(
        self,
        bound: HalfBounding | None,
        min: float | None = None,
        /,
        **kwargs: Any,
    ) -> None:
        r"""Sets the function used for parameter bounding on the lower limit.

        When ``bound`` is ``None``, no lower bound will be applied (and will remove
        any full bound present). When ``bound`` is not ``None``, them ``min`` cannot
        be ``None``.

        Args:
            bound (HalfBounding | None): bounding function.
            min (float | None, optional): lower bound. Defaults to ``None``.
            **kwargs (Any): keyword arguments for the bounding function.
        """
        # convert bounds to tuple
        if not isinstance(self.bind, list):
            self.bind = [lambda x, p: p, lambda x, n: n]

        # determine bounding function
        if bound:
            self.bind[1] = lambda x, n, lb=min, k=kwargs: bound(x, n, lb, **k)
        else:
            self.bind[1] = lambda x, n: n

    def fullbound(
        self,
        bound: FullBounding | None,
        max: float | None = None,
        min: float | None = None,
        /,
        **kwargs: Any,
    ) -> None:
        r"""Sets the function used for parameter bounding on the upper and lower limits.

        When ``bound`` is ``None``, no full bound will be applied (and will remove
        any upper or lower bound present). When ``bound`` is not ``None``, then
        ``max`` or ``min`` cannot be ``None``.

        Args:
            bound (FullBounding | None): bounding function.
            max (float | None, optional): upper bound. Defaults to ``None``.
            min (float | None, optional): lower bound. Defaults to ``None``.
            **kwargs (Any): keyword arguments for the bounding function.
        """

        # determine bounding function
        if bound:
            self.bind = lambda x, p, n, ub=max, lb=min, k=kwargs: bound(
                x, p, n, ub, lb, **k
            )
        else:
            self.bind = lambda x, p, n: p - n

    def clear(self, **kwargs) -> None:
        r"""Clears the accumulator's state."""
        del self.pos
        del self.neg

    def update(self, param: torch.Tensor, **kwargs) -> torch.Tensor | None:
        r"""Computes the update.

        Args:
            param (torch.Tensor): parameter being updated.

        Returns:
            torch.Tensor | None: value of the update.
        """
        # get partial updates
        pos, neg = self.pos, self.neg

        # ltp and ltd
        if pos is not None and neg is not None:
            if isinstance(self.bind, list):
                return self.bind[0](param, pos) - self.bind[1](param, neg)
            else:
                return self.bind(param, pos, neg)

        # ltp only
        elif pos is not None:
            if isinstance(self.bind, list):
                return self.bind[0](param, pos)
            else:
                return self.bind(param, pos, torch.zeros_like(pos))

        # ltd only
        elif neg is not None:
            if isinstance(self.bind, list):
                return -self.bind[1](param, neg)
            else:
                return self.bind(param, torch.zeros_like(neg), neg)

        # no update
        else:
            return None

    def forward(self, param: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""Computes the update and returns a tensor with it applied.

        Args:
            param (torch.Tensor): parameter being updated.

        Returns:
            torch.Tensor: parameter with the update applied.
        """
        update = self.update(param, **kwargs)
        if update is not None:
            return param + update
        else:
            return param


class Updater(Module):
    r"""Managed accumulated updates for module parameters.

    The added parameters are all set as properties which return an
    :py:class:`Accumulator` corresponding to that parameter. Care must be taken to
    avoid naming collisions, although the number of attributes in ``Updater`` not in
    ``Module`` are small. See the methods :py:meth:`_getacc_`, :py:meth:`_setacc_`, and
    :py:meth:`_delacc_` for more information.

    When a ``reduction`` is not specified, the default from
    :py:attr:`Accumulator.reduction` is used.

    Args:
        module (Updatable): module with updatable parameters.
        *params (str): parameters to set as trainable.
        reduction (Callable[[torch.Tensor, int], torch.Tensor] | None, optional):
                function for reducing updates. Defaults to ``None``.

    Caution:
        An ``Updater`` only weakly references its parent module, if its parent is
        deleted this updater will be made invalid.

    Note:
        The initializer creates an object of a dynamically created type with a base
        type of ``Updater``.
    """

    def __init__(
        self,
        module: Updatable,
        *params: str,
        reduction: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
        **kwargs,
    ):
        # define dynamic class
        self.__class__ = type(
            f"{type(module).__name__}{type(self).__name__}",
            (type(self),),
            {
                p: property(
                    partial(self._getacc_, attr=p),
                    partial(self._setacc_, attr=p),
                    partial(self._delacc_, attr=p),
                )
                for p in params
            },
        )

        # call superclass constructor
        Module.__init__(self, **kwargs)

        # check that the module has required parameters
        _ = argtest.members("module", module, *params)

        # set internal module (weakly referenced)
        self._parent_module = weakref.ref(module)

        # set update states and associated functions
        self.updates_ = nn.ModuleDict({p: Accumulator() for p in params})
        if reduction:
            for acc in self.updates_.values:
                acc.reduction = reduction

    @staticmethod
    def _getacc_(self: Updater, attr: str) -> Accumulator:
        r"""Gets the accumulator for a given attribute.

        Args:
            self (Updater): updater, self via the associated property.
            attr (str): parameter name to target.

        Returns:
            Accumulator: associated accumulator for the given parameter.
        """
        return self.updates_[attr]

    @staticmethod
    def _setacc_(
        self: Updater,
        value: tuple[torch.Tensor | None, torch.Tensor | None] | torch.Tensor | None,
        attr: str,
    ) -> None:
        r"""Updates the accumulator values for a given attribute.

        As a property, setting with a 2-tuple assumes the first term is the positive
        portion of the update and the second term is the negative portion. If instead
        a tensor is given, it assumes this update is only the positive portion.
        Any None values are ignored. The following blocks shows equivalent statements.

        .. code-block:: python

            updater.attr = pos_update, neg_update
            updater.attr.pos, updater.attr.neg = pos_update, neg_update

        .. code-block:: python

            updater.attr = pos_update
            updater.attr.pos = pos_update

        Args:
            self (Updater): updater, self via the associated property.
            value (tuple[torch.Tensor | None, torch.Tensor | None] | torch.Tensor | None):
                value of the update to assign.
            attr (str): parameter name to target.

        Important:
            The negative portions of updates should still be positively valued as they
            will be subtracted from the positive portion.
        """
        if isinstance(value, torch.Tensor | None):
            self.updates_[attr].pos = value
        else:
            self.updates_[attr].pos, self.updates_[attr].neg = value

    @staticmethod
    def _delacc_(self: Updater, attr: str) -> None:
        r"""Clears the accumulator state for a given attribute.

        As a property, this is equivalent to using ``del`` on the
        :py:attr:`Accumulator.pos` and :py:attr:`Accumulator.neg` properties directly,
        which itself resets them back to their empty states.

        Args:
            self (Updater): updater, self via the associated property.
            attr (str): parameter name to target.
        """
        del self.updates_[attr].pos
        del self.updates_[attr].neg

    @property
    def parent(self) -> Module | None:
        r"""Parent module, if valid.

        Returns:
            Module | None: parent module if the reference to it still exists.
        """
        return self._parent_module()

    @property
    def names(self) -> tuple[str, ...]:
        r"""Names of updatable attributes.

        Returns:
            tuple[str, ...]: names of updatable parameters.
        """
        return tuple(v for v in self.updates_.keys())

    def clear(self, **kwargs) -> None:
        r"""Clears all of the accumulators' states."""
        for acc in self.updates_.values():
            acc.clear(**kwargs)

    def forward(self, *params: str, **kwargs) -> None:
        r"""Applies accumulated updates.

        Args:
            *params (str): parameters to update, all parameters when ``None`` are specified.
        """
        if not params:
            params = self.updates_.keys()

        module = self._parent_module()

        if not module:
            raise RuntimeError("'parent' module is no longer a valid reference")
        else:
            for p in params:
                setattr(module, p, self.updates_[p](getattr(module, p), **kwargs))


class Updatable(ABC):
    r"""Adds parameter updating functionality to a module."""

    def __init__(self):
        self.updater_: Updater | None = None

    @property
    def updatable(self) -> bool:
        r"""If the module is updatable.

        Returns:
            bool: if the module is updatable.
        """
        return self.updater is not None

    @property
    def updater(self) -> Updater | None:
        r"""Updater for the module.

        Deleting this attribute deletes the associated updater.

        Args:
            Updater: new updater to set.

        Returns:
            Updater | None: current updater if it exists, otherwise None.
        """
        return self.updater_

    @updater.setter
    def updater(self, value: Updater) -> None:
        self.updater_ = value

    @updater.deleter
    def updater(self) -> None:
        self.updater_ = None

    @abstractmethod
    def defaultupdater(self, *includes: str, **kwargs) -> Updater:
        r"""Default updater for this object.

        Args:
            *includes (str): additional instance-specific parameters to include.

        Raises:
            RuntimeError: ``defaultupdater`` must be implemented by the subclass.

        Returns:
            Updater: the default updater.
        """
        raise RuntimeError(
            f"'{type(self).__name__}(Updatable) must implement "
            "the method 'defaultupdater'"
        )

    def clear(self, **kwargs) -> None:
        r"""Clears the updater's state."""
        if self.updatable:
            self.updater.clear(**kwargs)

    def update(self, clear: bool = True, **kwargs) -> None:
        r"""Applies all accumulated updates.

        Args:
            clear (bool, optional): if accumulators should be cleared after updating.
                Defaults to ``True``.
        """
        if self.updatable:
            self.updater(**kwargs)
            if clear:
                self.updater.clear(**kwargs)

    def updatesome(self, *params, clear: bool = True, **kwargs) -> None:
        r"""Applies accumulated updates to specific parameters.

        Args:
            *params (str): parameters to update.
            clear (bool, optional): if accumulators should be cleared after updating.
                Defaults to ``True``.
        """
        for p in params:
            self.updater(p, **kwargs)
            if clear:
                getattr(self.updater, p).clear(**kwargs)
