from __future__ import annotations
from .. import Module
from ..bounding import HalfBounding, FullBounding
from abc import ABC, abstractmethod
from functools import cache, partial
from inferno._internal import argtest
import torch
import torch.nn as nn
from typing import Any, Callable


class Accumulator(Module):
    r"""Used by :py:class:`Updater`, accumulated updates for a parameter."""

    def __init__(self):
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
        """Positive update component.

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
        """Negative update component.

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
                function for reducing updates. Defaults to None.
        """
        if fn:
            self.reduce = fn
        else:
            self.reduce = torch.sum

    def upperbound(
        self,
        bound: HalfBounding | None,
        max: float | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        r"""Sets the function used for parameter bounding on the upper limit.

        When ``bound`` is ``None``, no upper bound will be applied (and will remove
        any full bound present). When ``bound`` is not ``None``, them ``max`` cannot
        be ``None``.

        Args:
            bound (HalfBounding | None): bounding function.
            max (float | None, optional): upper bound. Defaults to None.
            kwargs (dict[str, Any] | None, optional): keyword arguments for the
                bounding function. Defaults to None.
        """
        # convert kwargs if required
        kw = kwargs if kwargs else {}

        # convert bounds to tuple
        if not isinstance(self.bind, tuple):
            self.bind = (lambda x, p: p, lambda x, n: n)

        # determine bounding function
        if bound:
            self.bind[0] = lambda x, p, ub=max, k=kw: bound(x, p, ub, **k)
        else:
            self.bind[0] = lambda x, p: p

    def lowerbound(
        self,
        bound: HalfBounding | None,
        min: float | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        r"""Sets the function used for parameter bounding on the lower limit.

        When ``bound`` is ``None``, no lower bound will be applied (and will remove
        any full bound present). When ``bound`` is not ``None``, them ``min`` cannot
        be ``None``.

        Args:
            bound (HalfBounding | None): bounding function.
            min (float | None, optional): lower bound. Defaults to None.
            kwargs (dict[str, Any] | None, optional): keyword arguments for the
                bounding function. Defaults to None.
        """
        # convert kwargs if required
        kw = kwargs if kwargs else {}

        # convert bounds to tuple
        if not isinstance(self.bind, tuple):
            self.bind = (lambda x, p: p, lambda x, n: n)

        # determine bounding function
        if bound:
            self.bind[1] = lambda x, n, lb=min, k=kw: bound(x, n, lb, **k)
        else:
            self.bind[1] = lambda x, n: n

    def fullbound(
        self,
        bound: FullBounding | None,
        max: float | None = None,
        min: float | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        r"""Sets the function used for parameter bounding on the upper and lower limits.

        When ``bound`` is ``None``, no full bound will be applied (and will remove
        any upper or lower bound present). When ``bound`` is not ``None``, then
        ``max`` or ``min`` cannot be ``None``.

        Args:
            bound (FullBounding | None): bounding function.
            max (float | None, optional): upper bound. Defaults to None.
            min (float | None, optional): lower bound. Defaults to None.
            kwargs (dict[str, Any] | None, optional): keyword arguments for the
                bounding function. Defaults to None.
        """
        # convert kwargs if required
        kw = kwargs if kwargs else {}

        # determine bounding function
        if bound:
            self.bind = lambda x, p, n, ub=max, lb=min, k=kw: bound(
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
            if isinstance(self.bind, tuple):
                return self.bind[0](param, pos) - self.bind[1](param, neg)
            else:
                return self.bind(param, pos, neg)

        # ltp only
        elif pos is not None:
            if isinstance(self.bind, tuple):
                return self.bind[0](param, pos)
            else:
                return self.bind(param, pos, torch.zeros_like(pos))

        # ltd only
        elif neg is not None:
            if isinstance(self.bind, tuple):
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
    """Managed accumulated updates for module parameters.

    The added parameters are all set as properties which return an
    :py:class:`Accumulator` corresponding to that parameter. Care must be taken to
    avoid naming collisions, although the number of attributes in ``Updater`` not in
    ``Module`` are small. See the methods :py:meth:`_get_`, :py:meth:`_set_`, and
    :py:meth:`_del_` for more information.

    Args:
        module (Updatable): module with updatable parameters.
        *params (str): parameters to set as trainable.

    Note:
        The initializer creates an object of a dynamically created type with a base
        type of ``Updater``.
    """

    def __init__(self, module: Updatable, *params: str, **kwargs):
        # define dynamic class
        self.__class__ = type(
            f"{type(module).__name__}{type(self).__name__}",
            (type(self),),
            {
                p: property(
                    partial(self._get_, attr=p),
                    partial(self._set_, attr=p),
                    partial(self._del_, attr=p),
                )
                for p in params
            },
        )

        # call superclass constructor
        Module.__init__(self, **kwargs)

        # check that the module has required parameters
        _ = argtest.members("module", module, *params)

        # set update states and associated functions
        self.updates_ = nn.ModuleDict({p: self.Accumulator() for p in params})

    @staticmethod
    def _get_(self: Updater, attr: str) -> Accumulator:
        r"""Gets the accumlator for a given attribute.

        Args:
            self (Updater): updater, self via the associated property.
            attr (str): parameter name to target.

        Returns:
            Accumulator: associated accumulator for the given parameter.
        """
        self.updates_[attr]

    @staticmethod
    def _set_(
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
            self (Updater): updater, self via the associated sproperty.
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
    def _del_(self: Updater, attr: str) -> None:
        r"""Clears the accumulator state for a given attribute.

        As a property, this is equivalent to using ``del`` on the
        :py:attr:`~Accumulator.pos` and :py:attr:`~Accumulator.neg` properties directly.

        Args:
            self (Updater): updater, self via the associated property.
            attr (str): parameter name to target.
        """
        del self.updates_[attr].pos
        del self.updates_[attr].neg

    @property
    def names(self) -> tuple[str, ...]:
        r"""Names of updatable attributes.

        Returns:
            tuple[str, ...]: names of updatable parameters.
        """
        return tuple(v for v in self.updates_.keys())

    def clear(self, **kwargs) -> None:
        r"""Clears all of the accumulators' states."""
        for p in self.updates_:
            delattr(self, p)

    def forward(self, module: Updatable, *params: str, **kwargs) -> None:
        r"""Applies accumulated updates.

        Args:
            module (Updatable): module to which updates will be applied.
            *params (str): parameters to update, all parameters when none are specified.
        """
        if not params:
            params = self.updates_.keys()

        for p in params:
            setattr(module, p, self.updates_[p](getattr(module, p), **kwargs))


class Updatable(ABC):
    r"""Adds parameter updating functionality to a module."""

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

        Args:
            value (Updater): new updater.

        Returns:
            Updater | None: current updater if it exists, otherwise None.
        """
        if hasattr(self, "updater_"):
            return self.updater_

    @updater.setter
    def updater(self, value: Updater) -> None:
        self.updater_ = value

    @updater.deleter
    def updater(self) -> None:
        if hasattr(self, "updater_"):
            del self.updater_

    @abstractmethod
    def defaultupdater(self) -> Updater:
        r"""Sets the current updater to the default and returns it.

        Raises:
            RuntimeError: ``defaultupdater`` must be implemented by the subclass.

        Returns:
            Updater: newly set default updater.
        """
        raise RuntimeError(
            f"'{type(self).__name__}(Updatable) must implement "
            "the method 'defaultupdater'"
        )

    def clear(self, **kwargs) -> None:
        r"""Clears the updater's state."""
        if self.updatable:
            self.updater.clear(**kwargs)

    def update(self, *params: str, clear: bool = True, **kwargs) -> None:
        r"""Applies all accumulated updates.

        Args:
            *params (str): name of the paraameters to update, updates all parameters
                when none are specified.
            clear (bool, optional): if accumulators should be cleared after updating.
                Defaults to True.
        """
        if self.updatable:
            self.updater(self)
        for acc in self.updates_.values():
            acc.update(self.parent, **kwargs)
            if clear:
                acc.clear(**kwargs)


class UpdaterV1(Module):
    r"""Wraps a connection for updates and on-update hooks.

    This encloses a connection object and provides some top-level properties, for others
    the enclosed connection can be called through :py:attr:`connection`.

    Specifically, this is used to accumulate multiple updates through different
    trainers, perform any additional logic on the updates, the apply them. It also
    allows for hooks like :py:class:`Normalization` and :py:class:`Clamping` to be
    tied to updates rather than inference steps.

    Updaters should call the :py:meth:`update_weight`, :py:meth:`update_bias`, and
    :py:meth:`update_delay` functions to add their own updates. Hooks which target
    updates (such as bounding hooks) should target the properties ending in ``_update``.
    Once these are altered, no more updates can be accumulated until after the next call.

    The arguments ending in ``_reduction`` should have a signature as follows:

    .. code-block:: python

        reduction(input: torch.Tensor, dim: int) -> torch.Tensor

    and examples include :py:func:`torch.mean`, :py:func:`torch.sum`,
    :py:func`torch.amin`, and :py:func`torch.amax`.

    The reduced potentiative updates are added to the value and the depressive updates
    are subtracted from the value.

    Args:
        connection (Connection): wrapped connection.

    Keyword Args:
        weight_reduction (Callable[[torch.Tensor, int], torch.Tensor] | None, optional):
            function to reduce weights from multiple trainers. Defaults to None.
        bias_reduction (Callable[[torch.Tensor, int], torch.Tensor] | None, optional):
            function to reduce biases from multiple trainers. Defaults to None.
        delay_reduction (Callable[[torch.Tensor, int], torch.Tensor] | None, optional):
            function to reduce delays from multiple trainers. Defaults to None.
    """

    def __init__(
        self,
        connection: Connection,
        *,
        weight_reduction: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
        bias_reduction: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
        delay_reduction: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
    ):
        # call superclass constructor
        Module.__init__(self)

        # composed connection
        self.connection_ = connection

        # internal reduction functions
        if weight_reduction:
            self.weight_reduction_ = weight_reduction
        else:
            self.weight_reduction_ = torch.sum

        if bias_reduction:
            self.bias_reduction_ = bias_reduction
        else:
            self.bias_reduction_ = torch.sum

        if delay_reduction:
            self.delay_reduction_ = delay_reduction
        else:
            self.delay_reduction_ = torch.sum

        # internal update values
        self._pos_weight_updates = nn.ParameterList()
        self.register_buffer("_pos_w_up", None)
        self._neg_weight_updates = nn.ParameterList()
        self.register_buffer("_neg_w_up", None)

        self._pos_bias_updates = nn.ParameterList()
        self.register_buffer("_pos_b_up", None)
        self._neg_bias_updates = nn.ParameterList()
        self.register_buffer("_neg_b_up", None)

        self._pos_delay_updates = nn.ParameterList()
        self.register_buffer("_pos_d_up", None)
        self._neg_delay_updates = nn.ParameterList()
        self.register_buffer("_neg_d_up", None)

        # parameter dependence
        self._weight_bounding_upper = None
        self._weight_bounding_lower = None
        self._weight_bounding = None

    @property
    def weight_reduction(self) -> Callable[[torch.Tensor, int], torch.Tensor]:
        r"""Reduction used for multiple weight updates, positive and negative.

        This function should have the following signature.

        .. code-block:: python

            reduction(input: torch.Tensor, dim: int) -> torch.Tensor

        The argument ``dim`` can also take a tuple of integers as with most appropriate
        functions in PyTorch but is never given a tuple here. Dimensions reduced along
        should not be kept by default. Valid PyTorch functions include byt are not
        limited to :py:func:`torch.mean`, :py:func:`torch.sum`, :py:func`torch.amin`,
        and :py:func`torch.amax`.

        Args:
            value (Callable[[torch.Tensor, int], torch.Tensor]): function for
                reducing weight updates.

        Returns:
            Callable[[torch.Tensor, int], torch.Tensor]: function for reducing
            weight updates.
        """
        return self.weight_reduction_

    @weight_reduction.setter
    def weight_reduction(
        self, value: Callable[[torch.Tensor, int], torch.Tensor]
    ) -> None:
        self.weight_reduction_ = value

    @property
    def bias_reduction(self) -> Callable[[torch.Tensor, int], torch.Tensor]:
        r"""Reduction used for multiple bias updates, positive and negative.

        This function should have the following signature.

        .. code-block:: python

            reduction(input: torch.Tensor, dim: int) -> torch.Tensor

        The argument ``dim`` can also take a tuple of integers as with most appropriate
        functions in PyTorch but is never given a tuple here. Dimensions reduced along
        should not be kept by default. Valid PyTorch functions include byt are not
        limited to :py:func:`torch.mean`, :py:func:`torch.sum`, :py:func`torch.amin`,
        and :py:func`torch.amax`.

        Args:
            value (Callable[[torch.Tensor, int], torch.Tensor]): function for
                reducing bias updates.

        Returns:
            Callable[[torch.Tensor, int], torch.Tensor]: function for reducing
            bias updates.
        """
        return self.bias_reduction_

    @bias_reduction.setter
    def bias_reduction(
        self, value: Callable[[torch.Tensor, int], torch.Tensor]
    ) -> None:
        self.bias_reduction_ = value

    @property
    def delay_reduction(self) -> Callable[[torch.Tensor, int], torch.Tensor]:
        r"""Reduction used for multiple delay updates, positive and negative.

        This function should have the following signature.

        .. code-block:: python

            reduction(input: torch.Tensor, dim: int) -> torch.Tensor

        The argument ``dim`` can also take a tuple of integers as with most appropriate
        functions in PyTorch but is never given a tuple here. Dimensions reduced along
        should not be kept by default. Valid PyTorch functions include byt are not
        limited to :py:func:`torch.mean`, :py:func:`torch.sum`, :py:func`torch.amin`,
        and :py:func`torch.amax`.

        Args:
            value (Callable[[torch.Tensor, int], torch.Tensor]): function for
                reducing delay updates.

        Returns:
            Callable[[torch.Tensor, int], torch.Tensor]: function for reducing
            delay updates.
        """
        return self.delay_reduction_

    @delay_reduction.setter
    def delay_reduction(
        self, value: Callable[[torch.Tensor, int], torch.Tensor]
    ) -> None:
        self.delay_reduction_ = value

    def weight_upper_bounding(
        self,
        bounding: HalfBounding,
        max: float,
        **kwargs: Any,
    ) -> None:
        r"""Adds upper bounding to weights.

        Args:
            bounding (HalfBounding): bounding function to use.
            max (float): upper bound of weights
            **kwargs (Any): passed as keyword arguments to ``bounding``.

        Note:
            This clears away anything set by :py:meth:`weight_bounding`.
        """

        def weight_ub(param, update, limit=max, kwargs=kwargs):
            return bounding(param, update, limit, **kwargs)

        self._weight_bounding = None
        self._weight_bounding_upper = weight_ub

    def weight_lower_bounding(
        self,
        bounding: HalfBounding,
        min: float,
        **kwargs: Any,
    ) -> None:
        r"""Adds lower bounding to weights.

        Args:
            bounding (HalfBounding): bounding function to use.
            min (float): lower bound of weights
            **kwargs (Any): passed as keyword arguments to ``bounding``.

        Note:
            This clears away anything set by :py:meth:`weight_bounding`.
        """

        def weight_lb(param, update, limit=min, kwargs=kwargs):
            return bounding(param, update, limit, **kwargs)

        self._weight_bounding = None
        self._weight_bounding_lower = weight_lb

    def weight_bounding(
        self,
        bounding: FullBounding,
        max: float,
        min: float,
        **kwargs: Any,
    ) -> None:
        r"""Adds full bounding to weights.

        Args:
            bounding (FullBounding): bounding function to use.
            max (float): upper bound of weights.
            min (float): lower bound of weights
            **kwargs (Any): passed as keyword arguments to ``bounding``.

        Note:
            This clears away anything set by :py:meth:`weight_upper_bounding` and
            :py:meth:`weight_lower_bounding`.
        """

        def weight_fb(
            param, pos_update, neg_update, max_limit=max, min_limit=min, kwargs=kwargs
        ):
            return bounding(
                param, pos_update, neg_update, max_limit, min_limit, **kwargs
            )

        self._weight_bounding_lower = None
        self._weight_bounding_upper = None
        self._weight_bounding = weight_fb

    def bias_upper_bounding(
        self,
        bounding: HalfBounding,
        max: float,
        **kwargs: Any,
    ) -> None:
        r"""Adds upper bounding to biases.

        Args:
            bounding (HalfBounding): bounding function to use.
            max (float): upper bound of biases
            **kwargs (Any): passed as keyword arguments to ``bounding``.

        Note:
            This clears away anything set by :py:meth:`bias_bounding`.
        """

        def bias_ub(param, update, limit=max, kwargs=kwargs):
            return bounding(param, update, limit, **kwargs)

        self._bias_bounding = None
        self._bias_bounding_upper = bias_ub

    def bias_lower_bounding(
        self,
        bounding: HalfBounding,
        min: float,
        **kwargs: Any,
    ) -> None:
        r"""Adds lower bounding to biases.

        Args:
            bounding (HalfBounding): bounding function to use.
            min (float): lower bound of biases
            **kwargs (Any): passed as keyword arguments to ``bounding``.

        Note:
            This clears away anything set by :py:meth:`bias_bounding`.
        """

        def bias_lb(param, update, limit=min, kwargs=kwargs):
            return bounding(param, update, limit, **kwargs)

        self._bias_bounding = None
        self._bias_bounding_lower = bias_lb

    def bias_bounding(
        self,
        bounding: FullBounding,
        max: float,
        min: float,
        **kwargs: Any,
    ) -> None:
        r"""Adds full bounding to biases.

        Args:
            bounding (FullBounding): bounding function to use.
            max (float): upper bound of biases.
            min (float): lower bound of biases
            **kwargs (Any): passed as keyword arguments to ``bounding``.

        Note:
            This clears away anything set by :py:meth:`bias_upper_bounding` and
            :py:meth:`bias_lower_bounding`.
        """

        def bias_fb(
            param, pos_update, neg_update, max_limit=max, min_limit=min, kwargs=kwargs
        ):
            return bounding(
                param, pos_update, neg_update, max_limit, min_limit, **kwargs
            )

        self._bias_bounding_lower = None
        self._bias_bounding_upper = None
        self._bias_bounding = bias_fb

    def delay_upper_bounding(
        self,
        bounding: HalfBounding,
        max: float,
        **kwargs: Any,
    ) -> None:
        r"""Adds upper bounding to delays.

        Args:
            bounding (HalfBounding): bounding function to use.
            max (float): upper bound of delays
            **kwargs (Any): passed as keyword arguments to ``bounding``.

        Note:
            This clears away anything set by :py:meth:`delay_bounding`.
        """

        def delay_ub(param, update, limit=max, kwargs=kwargs):
            return bounding(param, update, limit, **kwargs)

        self._delay_bounding = None
        self._delay_bounding_upper = delay_ub

    def delay_lower_bounding(
        self,
        bounding: HalfBounding,
        min: float,
        **kwargs: Any,
    ) -> None:
        r"""Adds lower bounding to delays.

        Args:
            bounding (HalfBounding): bounding function to use.
            min (float): lower bound of delays
            **kwargs (Any): passed as keyword arguments to ``bounding``.

        Note:
            This clears away anything set by :py:meth:`delay_bounding`.
        """

        def delay_lb(param, update, limit=min, kwargs=kwargs):
            return bounding(param, update, limit, **kwargs)

        self._delay_bounding = None
        self._delay_bounding_lower = delay_lb

    def delay_bounding(
        self,
        bounding: FullBounding,
        max: float,
        min: float,
        **kwargs: Any,
    ) -> None:
        r"""Adds full bounding to delays.

        Args:
            bounding (FullBounding): bounding function to use.
            max (float): upper bound of delays.
            min (float): lower bound of delays
            **kwargs (Any): passed as keyword arguments to ``bounding``.

        Note:
            This clears away anything set by :py:meth:`delay_upper_bounding` and
            :py:meth:`delay_lower_bounding`.
        """

        def delay_fb(
            param, pos_update, neg_update, max_limit=max, min_limit=min, kwargs=kwargs
        ):
            return bounding(
                param, pos_update, neg_update, max_limit, min_limit, **kwargs
            )

        self._delay_bounding_lower = None
        self._delay_bounding_upper = None
        self._delay_bounding = delay_fb

    @property
    def pos_weight_update(self) -> torch.Tensor | None:
        r"""Positive component of weight updates.

        If this value hasn't been overridden with a setter call, it will reduce the
        stored updates and return that. Otherwise it will return the value used to
        override. This is reset upon applying updates.

        Args:
            value (torch.Tensor): overwrite value of positive weight updates.

        Returns:
            torch.Tensor | None: current positive weight updates.
        """
        if self._pos_w_up is None:
            return self._get_update(self.weight_reduction, self._pos_weight_updates)
        else:
            return self._pos_w_up

    @pos_weight_update.setter
    def pos_weight_update(self, value: torch.Tensor) -> torch.Tensor:
        self._pos_w_up = value

    @property
    def neg_weight_update(self) -> torch.Tensor | None:
        r"""Negative component of weight updates.

        If this value hasn't been overridden with a setter call, it will reduce the
        stored updates and return that. Otherwise it will return the value used to
        override. This is reset upon applying updates.

        Args:
            value (torch.Tensor): overwrite value of negative weight updates.

        Returns:
            torch.Tensor | None: current positive negative updates.
        """
        if self._neg_w_up is None:
            return self._get_update(self.weight_reduction, self._neg_weight_updates)
        else:
            return self._neg_w_up

    @neg_weight_update.setter
    def neg_weight_update(self, value: torch.Tensor) -> torch.Tensor:
        self._neg_w_up = value

    @property
    def pos_bias_update(self) -> torch.Tensor | None:
        r"""Positive component of bias updates.

        If this value hasn't been overridden with a setter call, it will reduce the
        stored updates and return that. Otherwise it will return the value used to
        override. This is reset upon applying updates.

        Args:
            value (torch.Tensor): overwrite value of positive bias updates.

        Returns:
            torch.Tensor | None: current positive bias updates.
        """
        if self._pos_b_up is None:
            return self._get_update(self.bias_reduction, self._pos_bias_updates)
        else:
            return self._pos_b_up

    @pos_bias_update.setter
    def pos_bias_update(self, value: torch.Tensor) -> torch.Tensor:
        self._pos_b_up = value

    @property
    def neg_bias_update(self) -> torch.Tensor | None:
        r"""Negative component of bias updates.

        If this value hasn't been overridden with a setter call, it will reduce the
        stored updates and return that. Otherwise it will return the value used to
        override. This is reset upon applying updates.

        Args:
            value (torch.Tensor): overwrite value of negative bias updates.

        Returns:
            torch.Tensor | None: current positive negative updates.
        """
        if self._neg_b_up is None:
            return self._get_update(self.bias_reduction, self._neg_bias_updates)
        else:
            return self._neg_b_up

    @neg_bias_update.setter
    def neg_bias_update(self, value: torch.Tensor) -> torch.Tensor:
        self._neg_b_up = value

    @property
    def pos_delay_update(self) -> torch.Tensor | None:
        r"""Positive component of delay updates.

        If this value hasn't been overridden with a setter call, it will reduce the
        stored updates and return that. Otherwise it will return the value used to
        override. This is reset upon applying updates.

        Args:
            value (torch.Tensor): overwrite value of positive delay updates.

        Returns:
            torch.Tensor | None: current positive delay updates.
        """
        if self._pos_d_up is None:
            return self._get_update(self.delay_reduction, self._pos_delay_updates)
        else:
            return self._pos_d_up

    @pos_delay_update.setter
    def pos_delay_update(self, value: torch.Tensor) -> torch.Tensor:
        self._pos_d_up = value

    @property
    def neg_delay_update(self) -> torch.Tensor | None:
        r"""Negative component of delay updates.

        If this value hasn't been overridden with a setter call, it will reduce the
        stored updates and return that. Otherwise it will return the value used to
        override. This is reset upon applying updates.

        Args:
            value (torch.Tensor): overwrite value of negative delay updates.

        Returns:
            torch.Tensor | None: current positive negative updates.
        """
        if self._neg_d_up is None:
            return self._get_update(self.delay_reduction, self._neg_delay_updates)
        else:
            return self._neg_d_up

    @neg_delay_update.setter
    def neg_delay_update(self, value: torch.Tensor) -> torch.Tensor:
        self._neg_d_up = value

    @property
    def connection(self) -> Connection:
        r"""Connection submodule.

        Returns:
            Connection: existing connection.
        """
        return self.connection_

    def _add_update(
        self, update: torch.Tensor | nn.Parameter, store: nn.ParameterList
    ) -> None:
        r"""Module Internal: adds update tensor to an update store.

        Args:
            update (torch.Tensor | nn.Parameter): tensor containing the update.
            store (nn.ParameterList): store the update is added to.
        """
        if isinstance(update, nn.Parameter):
            store.append(update)
        else:
            store.append(nn.Parameter(update, requires_grad=update.requires_grad))

    def _get_update(
        self,
        reduction: Callable[[torch.Tensor, int], torch.Tensor],
        store: nn.ParameterList,
    ) -> torch.Tensor | None:
        r"""Module Internal: reduces and retrieves an update.

        Args:
            reduction (Callable[[torch.Tensor, int], torch.Tensor]):
            store (nn.ParameterList): store the update is retrieved from.

        Returns:
            torch.Tensor: updates stacked and reduced.
        """
        if len(store):
            return reduction(torch.stack([*store], 0), 0)
        else:
            return None

    def weight_update(
        self, pos_update: torch.Tensor | None, neg_update: torch.Tensor | None
    ) -> None:
        r"""Adds weight update terms.

        The weight updates are applied in the following manner.

        .. code-block:: python

            weight = weight + pos_update - neg_update

        Args:
            pos_update (torch.Tensor | None): positive weight update component.
            neg_update (torch.Tensor | None): negative weight update component.
        """
        if pos_update is not None:
            self._add_update(pos_update, self._pos_weight_updates)
        if neg_update is not None:
            self._add_update(neg_update, self._neg_weight_updates)

    def bias_update(
        self, pos_update: torch.Tensor | None, neg_update: torch.Tensor | None
    ) -> None:
        r"""Adds bias update terms.

        The bias updates are applied in the following manner.

        .. code-block:: python

            bias = bias + pos_update - neg_update

        Args:
            pos_update (torch.Tensor | None): positive bias update component.
            neg_update (torch.Tensor | None): negative bias update component.
        """
        if pos_update is not None:
            self._add_update(pos_update, self._pos_bias_updates)
        if neg_update is not None:
            self._add_update(neg_update, self._neg_bias_updates)

    def delay_update(
        self, pos_update: torch.Tensor | None, neg_update: torch.Tensor | None
    ) -> None:
        r"""Adds delay update terms.

        The delay updates are applied in the following manner.

        .. code-block:: python

            delay = delay + pos_update - neg_update

        Args:
            pos_update (torch.Tensor | None): positive delay update component.
            neg_update (torch.Tensor | None): negative delay update component.
        """
        if pos_update is not None:
            self._add_update(pos_update, self._pos_delay_updates)
        if neg_update is not None:
            self._add_update(neg_update, self._neg_delay_updates)

    def forward(self) -> None:
        r"""Applies stored updates and resets for the next set.

        Note:
            This does not check if a connection has a trainable bias or trainable delay
            before performing the update. If the updates are not manually set nor added
            by a trainer, this behaves normally. If they are, the included connections
            will silently not update parameters they don't have, but this isn't
            guaranteed behavior for 3rd party ones.
        """
        # weight updates
        w_pos, w_neg = self.pos_weight_update, self.neg_weight_update
        if w_pos is not None or w_neg is not None:
            w_pos, w_neg = 0 if w_pos is None else w_pos, 0 if w_neg is None else w_neg
            self.connection.weight = self.connection.weight + w_pos - w_neg

        # bias updates
        b_pos, b_neg = self.pos_bias_update, self.neg_bias_update
        if b_pos is not None or b_neg is not None:
            b_pos, b_neg = 0 if b_pos is None else b_pos, 0 if b_neg is None else b_neg
            self.connection.bias = self.connection.bias + b_pos - b_neg

        # delay updates
        d_pos, d_neg = self.pos_delay_update, self.neg_delay_update
        if d_pos is not None or d_neg is not None:
            d_pos, d_neg = 0 if d_pos is None else d_pos, 0 if d_neg is None else d_neg
            self.connection.delay = self.connection.delay + d_pos - d_neg

        # wipes internal state
        self._pos_weight_updates = nn.ParameterList()
        self._pos_w_up = None
        self._neg_weight_updates = nn.ParameterList()
        self._neg_w_up = None

        self._pos_bias_updates = nn.ParameterList()
        self._pos_b_up = None
        self._neg_bias_updates = nn.ParameterList()
        self._neg_b_up = None

        self._pos_delay_updates = nn.ParameterList()
        self._pos_d_up = None
        self._neg_delay_updates = nn.ParameterList()
        self._neg_d_up = None
