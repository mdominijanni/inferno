from . import Connection, Neuron, Synapse
from . import Normalization, Clamping  # noqa:F401; ignore, used for docs
from abc import ABC, abstractmethod
from collections.abc import Iterable
import functools
from inferno import Module
from inferno._internal import argtest, rgetattr, Proxy
from inferno.observe import ManagedMonitor, MonitorConstructor
import torch
import torch.nn as nn
from typing import Any, Callable
import warnings


class Updater(Module):
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

    @property
    def weight(self) -> torch.Tensor:
        return self.connection.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.connection.bias

    @property
    def delay(self) -> torch.Tensor | None:
        return self.connection.delay

    @property
    def trainablebias(self) -> bool:
        return self.bias is not None

    @property
    def trainabledelay(self) -> bool:
        return self.delay is not None

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


class Trainable(Module):

    def __init__(self, connection: Updater | Connection, neuron: Neuron):
        r"""A trainable connection-neuron pair.

        Generally, objects of this class should never be constructed by the
        end-user directly.

        This is a construct used to associate a single connection and neuron object
        for the purposes of training. The contained connection may produce output for
        multiple neurons and the neuron may take input from multiple connections.

        When implementing a new updater, the properties here should be used when
        accessing or alering the model parameters.

        If the connection is passed without being wrapped in an updater, it will be
        wrapped in an updater with the default constructor arguments.

        Args:
            connection (Updater | Connection): connection which produces output for the
                neuron, optionally wrapped in an updater.
            neuron (Neuron): neuron which takes output from the connection.
        """
        # call superclass constructor
        Module.__init__(self)

        # component elements
        if not isinstance(connection, Updater):
            self.updater_ = Updater(connection)
        else:
            self.updater_ = connection
        self.neuron_ = neuron

        # key for automatic management
        self._layerkey = None

    @property
    def _key(self) -> str | None:
        r"""Layer managed key.

        Args:
            value (str): write-once key set by :py:class:`Layer`.

        Returns:
            str | None: stored key, if any.
        """
        return self._layerkey

    @_key.setter
    def _key(self, value: str) -> None:
        if self._layerkey is not None:
            raise RuntimeError("layer key cannot be reassigned")
        else:
            self._layerkey = value

    @property
    def updater(self) -> Updater:
        """Updater submodule.

        Returns:
            Updater: composed updater.
        """
        return self.updater_

    @property
    def connection(self) -> Connection:
        r"""Connection submodule.

        Returns:
            Connection: composed connection.
        """
        return self.updater.connection

    @property
    def neuron(self) -> Neuron:
        r"""Neuron submodule.

        Returns:
            Neuron: composed neuron.
        """
        return self.neuron_

    @property
    def synapse(self) -> Synapse:
        r"""Synapse submodule.

        Returns:
            Synapse: composed synapse.
        """
        return self.connection.synapse

    @property
    def precurrent(self) -> torch.Tensor:
        r"""Currents from the synapse at the time last used by the connection.

        Alias for ``connection.syncurrent``.

        Returns:
            torch.Tensor: delay-offset synaptic currents.
        """
        return self.connection.syncurrent

    @property
    def prespike(self) -> torch.Tensor:
        r"""Spikes to the synapse at the time last used by the connection.

        Alias for ``connection.synspike``.

        Returns:
            torch.Tensor: delay-offset synaptic spikes.
        """
        return self.connection.synspike

    @property
    def postvoltage(self) -> torch.Tensor:
        r"""Membrane voltages in millivolts.

        Returns:
            torch.Tensor: membrane voltages.
        """
        return self.neuron.voltage

    @property
    def postspike(self) -> torch.Tensor:
        r"""Action potentials last generated.

        Returns:
            torch.Tensor: membrane voltages.
        """
        return self.neuron.spike

    def forward(self) -> None:
        """Forward call.

        Raises:
            RuntimeError: Trainable cannot have its forward method called.
        """
        raise RuntimeError(
            f"'forward' method of {type(self).__name__}(Trainable) cannot be called"
        )


class Layer(Module, ABC):

    def __init__(self):
        # call superclass constructor
        Module.__init__(self)

        # inner modules
        self.updaters_ = nn.ModuleDict()
        self.neurons_ = nn.ModuleDict()
        self.trainables_ = nn.ModuleDict()

    def add_input(self, name: str, module: Updater | Connection) -> Updater:
        r"""Adds a module that receives input from outside the layer.

        This registers either a :py:class:`Connection` or :py:class:`Updater` as a
        module that receives input from outside of the layer. If the module given is
        not an ``Updater``, this will wrap it in one before registering. This will be
        visible to PyTorch as a submodule.

        This can be accessed later as an ``Updater`` via :py:attr:`updaters`,

        .. code-block:: python

            layer.updaters.name

        a ``Connection`` via :py:attr:`connections`,

        .. code-block:: python

            layer.connections.name

        or a ``Synapse`` via :py:attr:`synapses`.

        .. code-block:: python

            layer.synapses.name

        Any :py:class:`Trainable` objects are also constructed from this input to all
        existing outputs. For each output with name ``output_name``, it can be accessed
        via :py:attr:`trainables` as follows.

        .. code-block:: python

            layer.trainables.name.output_name

        Args:
            name (str): attribute name of the module receiving input from
                outside the layer.
            module (Updater | Connection): module which receives the input and
                generates intermediate output.

        Raises:
            RuntimeError: the name must be unique amongst added inputs.

        Returns:
            Updater: added input module.
        """
        # test that the name is a valid identifier
        _ = argtest.identifier("name", name)

        # check that the name is not taken
        if name in self.updaters_:
            raise RuntimeError(f"'name' ('{name}') already assigned to an input")

        # wraps connection if it is not an updater and assigns
        if not isinstance(module, Updater):
            self.updaters_[name] = Updater(module)
        else:
            self.updaters_[name] = module

        # automatically add trainables
        if name in self.trainables_:
            raise RuntimeError(f"'name' ('{name}') already a first-order trainable key")
        else:
            self.trainables_[name] = nn.ModuleDict()
            for oname, neuron in self.neurons_.items():
                trainable = Trainable(self.updaters_[name], neuron)
                trainable._key = f"{name}.{oname}"
                self.trainables_[name][oname] = trainable

        # return assigned value
        return self.updaters_[name]

    def add_output(self, name: str, module: Neuron) -> Neuron:
        r"""Adds a module that generates output from input modules.

        This registers a :py:class:`Neuron` as a module that receives intermediate
        input and will generate output external to the layer. This will be visible to
        PyTorch as a submodule.


        This can be accessed later as a ``Neuron`` via :py:attr:`neurons`.

        .. code-block:: python

            layer.neurons.name

        Args:
            name (str): attribute name of the module generating output to
                outside the layer.
            module (Neuron): module which receives intermediate output and generates
                the final output.

        Raises:
            RuntimeError: the name must be unique amongst added outputs.

        Returns:
            Neuron: added output module.
        """
        # test that the name is a valid identifier
        _ = argtest.identifier("name", name)

        # check that the name is not taken
        if name in self.neurons_:
            raise RuntimeError(f"'name' ('{name}') already assigned to an output")

        # assigns value
        self.neurons_[name] = module

        # automatically add trainables
        for iname, updater in self.updaters_.items():
            if name in self.trainables_[iname]:
                raise RuntimeError(
                    f"'name' ('{name}') already a second-order trainable key in '{iname}'"
                )

            else:
                trainable = Trainable(updater, self.neurons_[name])
                trainable._key = f"{iname}.{name}"
                self.trainables_[iname][name] = trainable

        # return assigned value
        return self.neurons_[name]

    @property
    def innames(self) -> Iterable[str]:
        r"""Registered input names.

        Yields:
            str: name of a registered input.
        """
        return (k for k in self.updaters_.keys())

    @property
    def outnames(self) -> Iterable[str]:
        r"""Registered output names.

        Yields:
            str: name of a registered output.
        """
        return (k for k in self.neurons_.keys())

    @property
    def connections(self) -> Proxy:
        r"""Registred connections.

        For a given ``name`` registered with :py:meth:`add_input`, its corresponding
        :py:class:`Connection` can be accessed as.

        .. code-block:: python

            layer.connections.name

        And is equivalent to the following.

        .. code-block:: python

            layer.updaters.name.connection

        It can be modified in-place (including setting other attributes, adding
        monitors, etc), but it can neither be deleted nor reassigned.

        Returns:
            Proxy: safe access to registered connections.
        """
        return Proxy(self.updaters_, "connection")

    @property
    def named_connections(self) -> Iterable[tuple[str, Connection]]:
        r"""Iterable of registered connections and their names.

        Yields:
            tuple[str, Connection]: tuple of a registered connection and its name.
        """
        return ((k, v.connection) for k, v in self.updaters_.items())

    @property
    def neurons(self) -> Proxy:
        r"""Registred neurons.

        For a given ``name`` registered with :py:meth:`add_output`, its corresponding
        :py:class:`Neuron` can be accessed as.

        .. code-block:: python

            layer.neurons.name

        It can be modified in-place (including setting other attributes, adding
        monitors, etc), but it can neither be deleted nor reassigned.

        Returns:
            Proxy: safe access to registered neurons.
        """
        return Proxy(self.neurons_, "")

    @property
    def named_neurons(self) -> Iterable[tuple[str, Neuron]]:
        r"""Iterable of registered neurons and their names.

        Yields:
            tuple[str, Neuron]: tuple of a registered neuron and its name.
        """
        return ((k, v) for k, v in self.neurons_.items())

    @property
    def synapses(self) -> Proxy:
        r"""Registred synapses.

        For a given ``name`` registered with :py:meth:`add_input`, its corresponding
        :py:class:`Synapse` can be accessed as.

        .. code-block:: python

            layer.synapses.name

        And is equivalent to the following.

        .. code-block:: python

            layer.updaters.name.connection.synapse

        It can be modified in-place (including setting other attributes, adding
        monitors, etc), but it can neither be deleted nor reassigned.

        Returns:
            Proxy: safe access to registered synapses.
        """
        return Proxy(self.updaters_, "connection.synapse")

    @property
    def named_synapses(self) -> Iterable[tuple[str, Synapse]]:
        r"""Iterable of registered synapses and their names.

        Yields:
            tuple[str, Synapse]: tuple of a registered synapse and its name.
        """
        return ((k, v.connection.synapse) for k, v in self.updaters_.items())

    @property
    def trainables(self) -> Proxy:
        r"""Registered trainables.

        For a given ``input_name`` and ``output_name``, its corresponding
        :py:class:`Trainable` can be accessed as.

        .. code-block:: python

            layer.trainables.input_name.output_name

        Returns:
            Proxy: _description_
        """
        return Proxy(self.trainables_, "", "")

    @property
    def named_trainables(self) -> Iterable[tuple[tuple[str, str], Trainable]]:
        r"""Iterable of registered trainables and tuples of the input and output name.

        Yields:
            tuple[tuple[str, str], torch.Tensor]: tuple of a registered connection and
            a tuple of the input name and output name corresponding to it.
        """
        return ((k, v.connection) for k, v in self.updaters_.items())

    @property
    def updaters(self) -> Proxy:
        r"""Registred updaters.

        For a given ``name`` registered with :py:meth:`add_input`, its corresponding
        :py:class:`Updater` can be accessed as.

        .. code-block:: python

            layer.updaters.name

        It can be modified in-place (including setting other attributes, adding
        monitors, etc), but it can neither be deleted nor reassigned.

        Returns:
            Proxy: safe access to registered synapses.
        """
        return Proxy(self.updaters_, "")

    @property
    def named_updaters(self) -> Iterable[tuple[str, Updater]]:
        r"""Iterable of registered updaters and their names.

        Yields:
            tuple[str, Updater]: tuple of a registered updater and its name.
        """
        return ((k, v) for k, v in self.updaters_.items())

    @abstractmethod
    def wiring(
        self, inputs: dict[str, torch.Tensor], **kwargs
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError(
            f"{type(self).__name__}(Layer) must implement " "the method `wiring`."
        )

    def forward(
        self,
        inputs: dict[str, tuple[torch.Tensor, ...]],
        inkwargs: dict[str, dict[str, Any]] | None = None,
        outkwargs: dict[str, dict[str, Any]] | None = None,
        capture_intermediate: bool = False,
        **kwargs: Any,
    ) -> (
        dict[str, torch.Tensor]
        | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]
    ):
        r"""Computes a forward pass.

        The keys for ``inputs`` and ``inkwargs`` are the names of registered
        :py:class:`Updater` objects correspond to elements in :py:attr`innames`.
        The keys for ``outkwargs`` are the names of the registered :py:class`Neuron`
        objects and correspond to elements in :py:attr:`outnames`.

        Underlying :py:class:`Connection` and :py:class:`Neuron` objects are called
        using :py:meth:`~torch.nn.Module.__call__`, which in turn call
        :py:meth:`Connection.forward` and :py:meth:`Neuron.forward` respectively.
        The keyword argument dictionaries will be unpacked for each call automatically,
        and the inputs will be unpacked as positional arguments for each call.

        Only input modules which have keys in ``inputs`` will be run and added to
        the positional argument of :py:meth:`wiring`.

        Args:
            inputs (dict[str, tuple[torch.Tensor, ...]]): inputs passed to the
                registered connections' forward calls.
            inkwargs (dict[str, dict[str, Any]] | None, optional): keyword arguments
                passed to registered connections' forward calls. Defaults to None.
            outkwargs (dict[str, dict[str, Any]] | None, optional): keyword arguments
                passed to registered neurons' forward calls. Defaults to None.
            capture_intermediate (bool, optional): if output from the connections should
                also be returned. Defaults to False.
            **kwargs (Any): keyword arguments passed to :py:meth:`wiring`.

        Returns:
            dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
            tensors from neurons and the associated neuron names, if ``capture_intermediate``,
            this is the second element of a tuple, the first being a tuple of tensors from
            connections and the associated connection names.
        """
        # replace none with empty dictionaries
        inkwargs = inkwargs if inkwargs else {}
        outkwargs = outkwargs if outkwargs else {}

        # get connection outputs
        res = {
            k: rgetattr(self.updaters_, f"{k}.connection")(*v, **inkwargs.get(k, {}))
            for k, v in inputs
        }

        if capture_intermediate:
            outputs = self.wiring(res, **kwargs)
            outputs = {
                k: rgetattr(self.neurons_, k)(*v, **outkwargs.get(k, {}))
                for k, v in outputs
            }
            return (res, outputs)
        else:
            res = self.wiring(res, **kwargs)
            res = {
                k: rgetattr(self.neurons_, k)(*v, **outkwargs.get(k, {}))
                for k, v in res
            }
            return res


class HomogeneousLayer(Layer, ABC):

    def __init__(self):

        # call superclass constructor
        Layer.__init__(self)


class Serial(Layer):

    def __init__(self, inputs: Updater | Connection, outputs: Neuron):

        # call superclass constructor
        Layer.__init__(self)

        # add connection and neuron
        Layer.add_input(self)

    def add_input(self, name: str, module: Updater | Connection) -> Updater:
        r"""Adds a module that receives input from outside the layer.

        This registers either a :py:class:`Connection` or :py:class:`Updater` as a
        module that receives input from outside of the layer. If the module given is
        not an ``Updater``, this will wrap it in one before registering. This will be
        visible to PyTorch as a submodule.

        This can be accessed later as an ``Updater`` via :py:attr:`updaters`,

        .. code-block:: python

            layer.updaters.name

        a ``Connection`` via :py:attr:`connections`,

        .. code-block:: python

            layer.connections.name

        or a ``Synapse`` via :py:attr:`synapses`.

        .. code-block:: python

            layer.synapses.name

        Any :py:class:`Trainable` objects are also constructed from this input to all
        existing outputs. For each output with name ``output_name``, it can be accessed
        via :py:attr:`trainables` as follows.

        .. code-block:: python

            layer.trainables.name.output_name

        Args:
            name (str): attribute name of the module receiving input from
                outside the layer.
            module (Updater | Connection): module which receives the input and
                generates intermediate output.

        Raises:
            RuntimeError: the name must be unique amongst added inputs.

        Returns:
            Updater: added input module.
        """
        # test that the name is a valid identifier
        _ = argtest.identifier("name", name)

        # check that the name is not taken
        if name in self.updaters_:
            raise RuntimeError(f"'name' ('{name}') already assigned to an input")

        # wraps connection if it is not an updater and assigns
        if not isinstance(module, Updater):
            self.updaters_[name] = Updater(module)
        else:
            self.updaters_[name] = module

        # automatically add trainables
        if name in self.trainables_:
            raise RuntimeError(f"'name' ('{name}') already a first-order trainable key")
        else:
            self.trainables_[name] = nn.ModuleDict()
            for oname, neuron in self.neurons_.items():
                trainable = Trainable(self.updaters_[name], neuron)
                trainable._key = f"{name}.{oname}"
                self.trainables_[name][oname] = trainable

        # return assigned value
        return self.updaters_[name]

    def add_input(self):
        raise RuntimeError(
            f"'add_input' of {type(self).__name__}(Serial) cannot be called."
        )

    def add_output(self):
        raise RuntimeError(
            f"'add_output' of {type(self).__name__}(Serial) cannot be called."
        )


class Biclique(Layer):

    def __init__(
        self,
        inputs: tuple[tuple[str, Updater | Connection], ...],
        outputs: tuple[tuple[str, Neuron], ...],
        reduction: None,
    ):
        # superclass constructor
        Layer.__init__(self)

        # add inputs
        for name, module in inputs:
            self.add_input(name, module)

        # add outputs
        for name, module in outputs:
            self.add_output(name, module)

    def wiring(self, inputs: dict[str, torch.Tensor]):
        pass


class SerialLayer(Module):
    r"""Container for sequential Connection and Neuron objects.

    This is used as the base building block of spiking neural networks in Inferno,
    and is used for training models.

    Args:
        connection (Connection): connection between the layer inputs and the neurons.
        neuron (Neuron): neurons which take their input from the connection and their
            output is returned.
        connection_kwargs (dict[str, str] | None, optional): keyword argument
                mapping for connection methods. Defaults to None.
        neuron_kwargs (dict[str, str] | None, optional): keyword argument
                mapping for neuron methods. Defaults to None.

    Note:
        The keyword argument mappings are a dictionary, where the key is a
        kwarg in a :py:class:`Layer` method, and the corresponding value is
        the name for that kwarg which will be passed to the dependent method in
        in the :py:class:`Connection` or :py:meth:`Neuron`.

        When None, *all kwargs* are passed in. Included classes in Inferno are
        written to avoid conflicts, but that is not always guaranteed.

    Tip:
        The composed :py:class:`Neuron` does not need to be unique to this layer, and
        some architectures explicitly have multiple connections going to the same
        group of neurons. The uniqueness of the composed :py:class:`Connection` is
        not enforced, but unexpected behavior may occur if it is not unique.
    """

    def __init__(
        self,
        connection: Connection,
        neuron: Neuron,
        connection_kwargs: dict[str, str] | None = None,
        neuron_kwargs: dict[str, str] | None = None,
    ):
        Module.__init__(self)
        # warn if connection and neuron are inconsistent
        if connection.dt != neuron.dt:
            warnings.warn(
                f"inconsistent step times, {connection.dt} "
                f"for connection and {neuron.dt} for neuron."
            )

        # error if incompatible
        if connection.bsize != neuron.bsize:
            raise RuntimeError(
                f"incompatible batch sizes, {connection.bsize} "
                f"for connection and {neuron.bsize} for neuron."
            )

        if connection.outshape != neuron.shape:
            raise RuntimeError(
                f"incompatible shapes, {connection.outshape} "
                f"for connection output and {neuron.shape} for neuron."
            )

        # register submodules
        self.register_module("connection_", connection)
        self.register_module("neuron_", neuron)

        # keyword argument mapping functions
        def filterkwargs(kwargs: dict[str, Any], kwamap: dict[str, str | str]):
            return {
                kwamap.get(arg): val for arg, val in kwargs.values() if arg in kwamap
            }

        # kwarg mapping for connection
        if connection_kwargs is None:
            self.kwargmap_c = lambda x: x
        else:
            self.kwargmap_c = functools.partial(filterkwargs, connection_kwargs)

        # kwarg mapping for neuron
        if neuron_kwargs is None:
            self.kwargmap_n = lambda x: x
        else:
            self.kwargmap_n = functools.partial(filterkwargs, neuron_kwargs)

    @property
    def connection(self) -> Connection:
        r"""Connection submodule.

        Args:
            value (Connection): replacement connection.

        Returns:
            Connection: existing connection.
        """
        return self.connection_

    @connection.setter
    def connection(self, value: Neuron):
        self.connection_ = value

    @property
    def neuron(self) -> Neuron:
        r"""Neuron submodule.

        Args:
            value (Neuron): replacement neuron.

        Returns:
            Neuron: existing neuron.
        """
        return self.neuron_

    @neuron.setter
    def neuron(self, value: Connection):
        self.neuron_ = value

    @property
    def synapse(self) -> Synapse:
        r"""Synapse submodule.

        Args:
            value (Synapse): replacement synapse.

        Returns:
            Synapse: existing synapse.
        """
        return self.connection_.synapse

    @synapse.setter
    def synapse(self, value: Synapse):
        self.connection_.synapse = value

    def clear(self, **kwargs):
        r"""Resets connections and neurons to their resting state.

        Keyword arguments are filtered and then passed to :py:meth:`Connection.clear`
        and :py:meth:`Neuron.clear`.
        """
        self.connection.clear(**self.kwargmap_c(kwargs))
        self.neuron.clear(**self.kwargmap_n(kwargs))

    def forward(self, *inputs, **kwargs):
        r"""Runs a simulation step of the connection and then the neurons.

        Keyword arguments are filtered and then passed to :py:meth:`Connection.forward`
        and :py:meth:`Neuron.forward`. It is expected that :py:meth:`Connection.forward`
        outputs a single tensor and :py:meth:`Neuron.forward` and takes a single
        positional argument. The output of the former is used for the input of the
        latter.
        """
        return self.neuron(
            self.connection(*inputs, **self.kwargmap_c(kwargs)),
            **self.kwargmap_n(kwargs),
        )
