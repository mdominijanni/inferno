from __future__ import annotations
from .monitors import Monitor, MonitorConstructor
from .. import Module
from .._internal import MapAccessor, getitem, rgetitem, unique
from collections.abc import Iterable, Iterator, Mapping
from itertools import chain
import torch.nn as nn
from typing import Any
import weakref


class Observable:
    r"""Object to which monitors should be registered relative to a basis module.

    Args:
        basis (Module): module which monitors should be registered relative to.
        realign (str): name of the method bound to ``basis`` which realigns attributes
            relative to it.
        realign_args (tuple[Any, ...] | None): positional arguments prepended to output
            of ``local_remap``.
        realign_kwargs (dict[str, Any] | None): keyword arguments added to output of
            ``local_remap``.

    Important:
        Monitors are not stored in an ``Observable`` but are only weakly referenced,
        primarily for the purpose of monitoring monitors.
    """

    def __init__(
        self,
        basis: Module,
        realign: str,
        realign_args: tuple[Any, ...] | None,
        realign_kwargs: dict[str, Any] | None,
    ):
        self.__monitors = weakref.WeakValueDictionary()
        self.__basis = weakref.ref(basis)
        self.__basis_realign = weakref.WeakMethod(getattr(basis, realign))
        self.__realign_args = realign_args if realign_args else ()
        self.__realign_kwargs = realign_kwargs if realign_kwargs else {}

    @property
    def monitors(self) -> MapAccessor:
        r"""Access to monitors added to this observable.

        Monitors available via this property need not have been added to a pool, but
        must have been added or retrieved through :py:meth:`add_monitor`.

        Monitors are never explicitly removed, but if all references to them are
        destroyed, then they will no longer be accessible.

        Returns:
            MapAccessor: attribute-like accessor for monitors.
        """
        return MapAccessor(self.__monitors)

    def local_remap(self, attr: str) -> tuple[tuple[Any, ...], dict[str, Any]]:
        r"""Locally remaps an attribute for pooled monitors.

        This method should alias any local attributes being referenced
        as required. The callback ``realign`` given on initialization will
        accept the output of this as positional and keyword arguments.

        Args:
            attr (str): dot-separated attribute relative to self, to realign.

        Returns:
            tuple[tuple[Any, ...], dict[str, Any]]: tuple of positional arguments and
            keyword arguments for ``realign`` method specified on initialization.

        Raises:
            NotImplementedError: this must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__}(Observable) must implement the method 'local_remap'"
        )

    def realign_attribute(self, attr: str) -> str:
        r"""Realigns an attribute for pooled monitors.

        The class which acts as a factory for ``Observable`` should, provide a means to
        set the nested attribute path relative to ``Observable``, relative to itself
        instead.

        Args:
            attr (str): dot-separated attribute relative to self, to realign.

        Returns:
            str: dot-chained nested attribute name relative to the parent of self.
        """
        args, kwargs = self.local_remap(attr)

        if not self.__basis_realign():
            raise RuntimeError("observable's basis is no longer in memory")
        else:
            return self.__basis_realign()(
                *self.__realign_args, *args, **self.__realign_kwargs, **kwargs
            )

    def add_monitor(
        self,
        name: str,
        attr: str,
        constructor: MonitorConstructor,
        pool: Iterable[tuple[Observable, Mapping[str, Monitor]]] | None,
        /,
        **tags: Any,
    ) -> Monitor:
        r"""Creates or alias a monitor on the observable's basis.

        Args:
            name (str): name of the monitor to add.
            attr (str): dot-separated attribute path, relative to this observable, to
                monitor, when an empty-string, the observable is directly targeted.
            constructor (MonitorConstructor): partial constructor for the monitor to add.
            pool (Iterable[tuple[Observable, Mapping[str, Monitor]]] | None): pool to
                search for compatible monitor, always creates a new one if ``None``.
                Defaults to ``None``.
            **tags (Any): tags to determine if the monitor is unique amongst monitors
                with the same name, targeting the same attribute (aligned to the basis).

        Returns:
            Monitor: added or retrieved monitor.

        Important:
            If a monitor name is not a valid identifier, it cannot be accessed with
            dot-notation via :py:attr:`monitors`.
        """
        # get realigned attribute
        attr = self.realign_attribute(attr)

        # create a new monitor if no pool is specified
        if not pool:
            monitor = constructor(attr, self.__basis())
            self.__monitors[name] = monitor
            return monitor

        # monitor tags
        tags = tags | {"_attr": attr}

        # try to find if a compatible monitor exists
        found = None
        for obs, monitors in pool:
            # skip invalid cells or cells from a different layer
            if (not obs.__basis()) or id(obs.__basis() != id(self.__basis())):
                continue

            # skip if the named monitor doesn't exist
            if name not in monitors:
                continue

            # create the alias if tags match
            if hasattr(monitors[name], "_tags") and monitors[name]._tags == tags:
                found = monitors[name]

                # break if identical cell as this is the "best match"
                if id(obs) == id(self):
                    break

        # return alias or create new monitor
        if found:
            self.__monitors[name] = found
            return found
        else:
            monitor = constructor(attr, self.__basis())
            monitor._tags = tags
            self.__monitors[name] = monitor
            return monitor


class MonitorPool(Module):
    r"""Collection of shared monitors."""

    def __init__(self):
        # call superclass constructor
        Module.__init__(self)

        # inner containers
        self.monitors_ = nn.ModuleDict()
        self.observed_ = weakref.WeakValueDictionary()

    @property
    def monitors(self) -> Iterator[Monitor]:
        r"""Added monitors.

        Because monitors for each ``Pool`` are pooled together for each trainer,
        duplicate monitors are not created where possible. The number of monitors here
        may be less than the number of added monitors.

        Yields:
            Monitor: added monitors.
        """
        return unique(chain.from_iterable(m.values() for m in self.monitors_.values()))

    @property
    def named_monitors(self) -> Iterator[tuple[tuple[str, str], Monitor]]:
        r"""Iterable of added monitors and tuples of the cell and monitor name.

        Yields:
            tuple[tuple[str, str], Monitor]: tuple of an added monitor and a tuple
            of the observable name and monitor name corresponding to it.
        """
        return chain.from_iterable(
            (((o, n), m) for n, m in md.items()) for o, md in self.monitors_.items()
        )

    def named_monitors_of(self, observed: str) -> Iterator[tuple[str, Monitor]]:
        r"""Monitors associated with a given observable.

        Args:
            observed (str): name of the observable to get associated monitors of.

        Yields:
            tuple[str, Monitor]: associated monitors and their names.
        """
        return ((n, m) for n, m in getitem(self.monitors_, observed, {}).items())

    @property
    def pool(self) -> Iterator[tuple[Observable, dict[str, Monitor]]]:
        r"""Pool of monitors used by Observable.

        Yields:
            tuple[Observable, dict[str, Monitor]]: tuple of observable, and its monitors.
        """
        return (
            (obs, {mname: mon for mname, mon in self.monitors_[oname].items()})
            for oname, obs in self.observed_.items()
            if oname in self.monitors_
        )

    def add_observed(self, name: str, value: Observable) -> Observable:
        r"""Adds an observable.

        Args:
            name (str): name of the observable to add.
            value (Observable): observable to add to this pool.

        Raises:
            RuntimeError: an observable with the specified name already exists.
            RuntimeError: an observable with the specified name existed and was
                registered, but removed from memory without deleting it from the pool.

        Returns:
            Observable: added observable.
        """
        if name in self.observed_:
            raise RuntimeError(f"'name' ('{name}') is already a registered observable")
        elif name in self.monitors_:
            raise RuntimeError(
                f"name ('{name}') was a registered observable and never deleted, "
                "call 'del_observed' first"
            )
        else:
            self.observed_[name] = value

        return value

    def get_observed(self, name: str) -> Observable:
        r"""Gets an added observable.

        Args:
            name (str): name of the observable to get.

        Returns:
            Observable: specified observable, if it exists.
        """
        if name in self.observed_:
            return self.observed_[name]
        else:
            raise KeyError(f"'name' ('{name}') is not a registered observable")

    def del_observed(self, name: str) -> None:
        r"""Deletes an added observable.

        Args:
            name (str): name of the observable to delete.

        Important:
            This does not strictly delete the observable, it is still owned by its
            basis. Its monitors are however deleted.
        """
        if name in self.monitors_:
            for monitor in self.monitors_[name].values():
                monitor.deregister()
            del self.monitors_[name]

        if name in self.observed_:
            del self.observed_[name]

    def add_monitor(
        self,
        observed: str,
        name: str,
        attr: str,
        constructor: MonitorConstructor,
        unique: bool = False,
        /,
        **tags: Any,
    ) -> Monitor:
        r"""Adds a monitor to an observable.

        Args:
            observed (str): name of the observable to which the monitor will be added.
            name (str): name of the monitor to add (unique to the observable).
            attr (str): dot-separated attribute to monitor, relative to the observable.
            constructor (MonitorConstructor): partial constructor for the monitor.
            unique (bool, optional): if the monitor should never be aliased from the
                pool. Defaults to ``False``.
            **tags (Any): tags to determine if the monitor is unique amongst monitors
                with the same name, targeting the same attribute (aligned to the basis).

        Raises:
            AttributeError: specified observable does not exist.

        Returns:
            Monitor: added or retrieved monitor.
        """
        # check if the observable exists
        if observed not in self.observed_:
            raise AttributeError(
                f"'observed' ('{observed}') is not the name of an added observable"
            )

        # if the monitor exists and is not unique, return it, delete if unique
        monitor = rgetitem(self.monitors_, (observed, name), None)
        if monitor:
            if unique:
                del self.monitors_[observed][name]
            else:
                return monitor

        # create monitor via the observable
        monitor = self.get_observed(observed).add_monitor(
            name, attr, constructor, None if unique else self.pool, **tags
        )

        # deregister if not currently training
        if not self.training:
            monitor.deregister()

        # add monitor to the pool
        if observed not in self.monitors_:
            self.monitors_[observed] = nn.ModuleDict()
        self.monitors_[observed][name] = monitor

        return monitor

    def get_monitor(self, observed: str, monitor: str) -> Monitor | None:
        r"""Gets an added monitor.

        Args:
            observed (str): name of the observable to which the monitor was added.
            monitor (str): name of the monitor.

        Returns:
            Monitor | None: specified monitor, if it exists.
        """
        return rgetitem(self.monitors_, (observed, monitor), None)

    def del_monitor(self, observed: str, monitor: str) -> None:
        r"""Deletes an added monitor.

        Args:
            observed (str): name of the observable to which the monitor was added.
            monitor (str): name of the monitor.

        Raises:
            AttributeError: specified observable does not exist, or does not have a
                monitor with the specified name added to it.
        """
        # check that the observable has monitors
        if observed not in self.monitors_ or observed not in self.observed_:
            raise AttributeError(
                f"'observed' ('{observed}') is either not the name of an added "
                "observable or is an observable with no added monitors"
            )

        # check that the monitor to delete exists
        if monitor not in self.monitors_[observed]:
            raise AttributeError(
                f"'monitor' ('{monitor}') is not the name of a monitor added on "
                f"observable with name '{observed}'"
            )

        # delete the monitor
        self.monitors_[observed][monitor].deregister()
        del self.monitors_[observed][monitor]

        # delete group if empty
        if not len(self.monitors_[observed]):
            del self.monitors_[observed]
