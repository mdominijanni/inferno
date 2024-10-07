from .. import IndependentCellTrainer
from ... import Module
from ..._internal import argtest
from ...functional import SpikeTimeHalfKernel
from ...neural import Cell
from ...observe import (
    StateMonitor,
    EventReducer,
)
import torch
from typing import Any, Callable


class KernelSTDP(IndependentCellTrainer):
    r"""General kernel spike-timing dependent plasticity trainer.

    .. math::
        \begin{align*}
            w(t + \Delta t) - w(t) &= K_\text{post}\bigl(t^f_\text{post} - t^f_\text{pre}\bigr) \bigl[t^f_\text{post} \geq t^f_\text{pre}\bigr] \\
            &+ K_\text{pre}\bigl(t^f_\text{post} - t^f_\text{pre}\bigr) \bigl[t^f_\text{post} < t^f_\text{pre}\bigr]
        \end{align*}

    Where:

    Times :math:`t` and :math:`t_n^f` are the current time and the time of the most
    recent spike from neuron :math:`n`, respectively, and :math:`\Delta t` is the
    duration of the simulation step.

    Args:
        kernel_post (~firebrand.functional.SpikeTimeHalfKernel): function for
            determining update strength on postsynaptic spikes, :math:`K_\text{post}`.
        kernel_pre (~firebrand.functional.SpikeTimeHalfKernel): function for
            determining update strength on presynaptic spikes, :math:`K_\text{pre}`.
        kernel_post_kwargs (dict[str, Any]): keyword arguments passed into ``kernel_post``.
        kernel_pre_kwargs (dict[str, Any]): keyword arguments passed into ``kernel_pre``.
        delayed (bool, optional): if the updater should assume that learned delays, if
            present, may change. Defaults to ``False``.
        interp_tolerance (float, optional): maximum difference in time from an observation
            to treat as co-occurring, in :math:`\text{ms}`. Defaults to ``0.0``.
        batch_reduction (Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] | None):
            function to reduce updates over the batch dimension, :py:func:`torch.mean`
            when ``None``. Defaults to ``None``.
        inplace (bool, optional): if :py:class:`~inferno.RecordTensor` write operations
            should be performed in-place. Defaults to ``False``.

    Important:
        It is expected for this to be called after every trainable batch. Variables
        used are not stored (or are invalidated) if multiple batches are given before
        an update.

    Important:
        The :py:class:`~torch.Tensor` values in ``kernel_post_kwargs`` and
        ``kernel_pre_kwargs`` will each be unpacked into a module in the cell's state,
        and registered as buffers.

        If given as a default to the ``KernelSTDP`` constructor, then they will
        be cloned and detached first.

    Note:
        The constructor arguments are hyperparameters and can be overridden on a
        cell-by-cell basis.

    Note:
        ``batch_reduction`` can be one of the functions in PyTorch including but not
        limited to :py:func:`torch.sum`, :py:func:`torch.mean`, and :py:func:`torch.amax`.
        A custom function with similar behavior can also be passed in. Like with the
        included function, it should not keep the original dimensions by default.

    See Also:
        For more details and references, visit
        :ref:`zoo/learning-stdp:Generalized-Kernel Spike-Timing Dependent Plasticity (Kernel STDP)` in the zoo.
    """

    def __init__(
        self,
        kernel_post: SpikeTimeHalfKernel,
        kernel_pre: SpikeTimeHalfKernel,
        kernel_post_kwargs: dict[str, Any],
        kernel_pre_kwargs: dict[str, Any],
        delayed: bool = False,
        interp_tolerance: float = 0.0,
        batch_reduction: (
            Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] | None
        ) = None,
        inplace: bool = False,
        **kwargs,
    ):
        # call superclass constructor
        IndependentCellTrainer.__init__(self, **kwargs)

        # default hyperparameters
        self.kernel_post = kernel_post
        self.kernel_pre = kernel_pre
        self.kernel_post_kwargs = kernel_post_kwargs
        self.kernel_pre_kwargs = kernel_pre_kwargs
        self.delayed = bool(delayed)
        self.tolerance = argtest.gte("interp_tolerance", interp_tolerance, 0, float)
        self.batchreduce = batch_reduction if batch_reduction else torch.mean
        self.inplace = bool(inplace)

    def _build_cell_state(self, **kwargs) -> Module:
        r"""Builds auxiliary state for a cell.

        Keyword arguments will override module-level hyperparameters.

        Returns:
            Module: state module.
        """
        state = Module()

        kernel_post = kwargs.get("kernel_post", self.kernel_post)
        kernel_pre = kwargs.get("kernel_pre", self.kernel_pre)
        kernel_post_kwargs = kwargs.get(
            "kernel_post_kwargs",
            {
                k: v.clone().detach() if isinstance(v, torch.Tensor) else v
                for k, v in self.kernel_post_kwargs.items()
            },
        )
        kernel_pre_kwargs = kwargs.get(
            "kernel_pre_kwargs",
            {
                k: v.clone().detach() if isinstance(v, torch.Tensor) else v
                for k, v in self.kernel_pre_kwargs.items()
            },
        )
        delayed = kwargs.get("delayed", self.delayed)
        interp_tolerance = kwargs.get("interp_tolerance", self.tolerance)
        batch_reduction = kwargs.get("batch_reduction", self.batchreduce)
        inplace = kwargs.get("inplace", self.inplace)

        state.kernel_post = kernel_post
        state.kernel_pre = kernel_pre

        state.kernel_post_kwargs = {}
        state.kernel_post_tensor_kwargs = Module()
        for k, v in kernel_post_kwargs.items():
            if isinstance(v, torch.Tensor):
                state.kernel_post_tensor_kwargs.register_buffer(k, v)
            else:
                state.kernel_post_kwargs[k] = v

        state.kernel_pre_kwargs = {}
        state.kernel_pre_tensor_kwargs = Module()
        for k, v in kernel_pre_kwargs.items():
            if isinstance(v, torch.Tensor):
                state.kernel_pre_tensor_kwargs.register_buffer(k, v)
            else:
                state.kernel_pre_kwargs[k] = v

        state.delayed = bool(delayed)
        state.tolerance = argtest.gte("interp_tolerance", interp_tolerance, 0, float)
        state.batchreduce = (
            batch_reduction if (batch_reduction is not None) else torch.mean
        )
        state.inplace = bool(inplace)

        return state

    def register_cell(
        self,
        name: str,
        cell: Cell,
        /,
        **kwargs: Any,
    ) -> IndependentCellTrainer.Unit:
        r"""Adds a cell with required state.

        Args:
            name (str): name of the cell to add.
            cell (Cell): cell to add.

        Keyword Args:
            kernel_post (~firebrand.functional.SpikeTimeHalfKernel): function for
                determining update strength on postsynaptic spikes.
            kernel_pre (~firebrand.functional.SpikeTimeHalfKernel): function for
                determining update strength on presynaptic spikes.
            kernel_post_kwargs (dict[str, Any]): keyword arguments passed into ``kernel_post``.
            kernel_pre_kwargs (dict[str, Any]): keyword arguments passed into ``kernel_pre``.
            delayed (bool, optional): if the updater should assume that learned delays, if
                present, may change.
            interp_tolerance (float): maximum difference in time from an observation
                to treat as co-occurring.
            batch_reduction (Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] | None):
                function to reduce updates over the batch dimension, :py:func:`torch.mean`
                when ``None``.
            inplace (bool, optional): if :py:class:`~inferno.RecordTensor` write operations
                should be performed in-place.

        Returns:
            IndependentCellTrainer.Unit: specified cell, auxiliary state, and monitors.

        Important:
            Any specified keyword arguments will override the default hyperparameters
            set on initialization. See :py:class:`KernelSTDP` for details.
        """
        # add the cell with additional hyperparameters
        cell, state = self.add_cell(
            name, cell, self._build_cell_state(**kwargs), ["weight"]
        )

        # if delays should be accounted for
        delayed = state.delayed and cell.connection.delayedby is not None

        # common and derived arguments
        monitor_kwargs = {
            "as_prehook": False,
            "train_update": True,
            "eval_update": False,
            "prepend": True,
        }

        # postsynaptic event-time monitor
        self.add_monitor(
            name,
            "spike_post",
            "neuron.spike",
            StateMonitor.partialconstructor(
                reducer=EventReducer(
                    cell.connection.dt,
                    lambda x: x.bool(),
                    initial="nan",
                    duration=0.0,
                    inclusive=True,
                    inplace=state.inplace,
                ),
                **monitor_kwargs,
            ),
            False,
            dt=cell.connection.dt,
            inplace=state.inplace,
        )

        # presynaptic event-time monitor
        # when the delayed condition is true, using synapse.spike records the raw
        # spike times rather than the delay adjusted times of synspike.
        self.add_monitor(
            name,
            "spike_pre",
            "synapse.spike" if delayed else "connection.synspike",
            StateMonitor.partialconstructor(
                reducer=EventReducer(
                    cell.connection.dt,
                    lambda x: x.bool(),
                    initial="nan",
                    duration=cell.connection.delayedby if delayed else 0.0,
                    inclusive=True,
                    inplace=state.inplace,
                ),
                **monitor_kwargs,
            ),
            False,
            dt=cell.connection.dt,
            delayed=delayed,
            inplace=state.inplace,
        )

        return self.get_unit(name)

    def forward(self) -> None:
        r"""Processes update for given layers based on current monitor stored data."""
        # iterate through self
        for cell, state, monitors in self:
            # skip if self or cell is not in training mode or has no updater
            if not cell.training or not self.training or not cell.updater:
                continue

            # relative spike times, reshaped into receptive format
            t_post = cell.connection.postsyn_receptive(monitors["spike_post"].peek())
            t_pre = cell.connection.presyn_receptive(
                monitors["spike_pre"].view(cell.connection.selector, state.tolerance)
                if state.delayed and cell.connection.delayedby
                else monitors["spike_pre"].peek()
            )

            # unadjusted time difference
            t_delta = t_pre - t_post

            # partial updates
            dpost = state.kernel_post(
                t_delta,
                **(
                    state.kernel_post_kwargs
                    | {k: v for k, v in state.kernel_post_tensor_kwargs.named_buffers()}
                ),
            )
            dpre = state.kernel_pre(
                t_delta,
                **(
                    state.kernel_pre_kwargs
                    | {k: v for k, v in state.kernel_pre_tensor_kwargs.named_buffers()}
                ),
            )

            # accumulate partials
            cell.updater.weight = (
                state.batchreduce(dpost.clamp_min(0.0).nansum(dim=-1), 0)
                + state.batchreduce(dpre.clamp_min(0.0).nansum(dim=-1), 0),
                -(
                    state.batchreduce(dpost.clamp_max(0.0).nansum(dim=-1), 0)
                    + state.batchreduce(dpre.clamp_max(0.0).nansum(dim=-1), 0)
                ),
            )


class DelayAdjustedKernelSTDP(IndependentCellTrainer):
    r"""Delay-adjusted general kernel spike-timing dependent plasticity trainer.

    .. math::
        \begin{align*}
            w(t + \Delta t) - w(t) &= K_\text{post}(t_\Delta(t)) [t_\Delta(t) \geq 0] \\
            &+ K_\text{pre}(t_\Delta(t)) [t_\Delta(t) < 0] \\
            t_\Delta(t) &= t^f_\text{post} - t^f_\text{pre} - d(t)
        \end{align*}

    Where:

    Times :math:`t` and :math:`t_n^f` are the current time and the time of the most
    recent spike from neuron :math:`n` respectively, :math:`\Delta t` is the duration of
    the simulation step, and :math:`d(t)` are the learned delays.

    Args:
        kernel_post (~firebrand.functional.SpikeTimeHalfKernel): function for
            determining update strength on postsynaptic spikes, :math:`K_\text{post}`.
        kernel_pre (~firebrand.functional.SpikeTimeHalfKernel): function for
            determining update strength on presynaptic spikes, :math:`K_\text{pre}`.
        kernel_post_kwargs (dict[str, Any]): keyword arguments passed into ``kernel_post``.
        kernel_pre_kwargs (dict[str, Any]): keyword arguments passed into ``kernel_pre``.
        batch_reduction (Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] | None):
            function to reduce updates over the batch dimension, :py:func:`torch.mean`
            when ``None``. Defaults to ``None``.
        inplace (bool, optional): if :py:class:`~inferno.RecordTensor` write operations
            should be performed in-place. Defaults to ``False``.

    Important:
        It is expected for this to be called after every trainable batch. Variables
        used are not stored (or are invalidated) if multiple batches are given before
        an update.

    Important:
        The :py:class:`~torch.Tensor` values in ``kernel_post_kwargs`` and
        ``kernel_pre_kwargs`` will each be unpacked into a module in the cell's state,
        and registered as buffers.

        If given as a default to the ``DelayAdjustedKernelSTDP`` constructor, then they
        will be cloned and detached first.

    Note:
        The constructor arguments are hyperparameters and can be overridden on a
        cell-by-cell basis.

    Note:
        ``batch_reduction`` can be one of the functions in PyTorch including but not
        limited to :py:func:`torch.sum`, :py:func:`torch.mean`, and :py:func:`torch.amax`.
        A custom function with similar behavior can also be passed in. Like with the
        included function, it should not keep the original dimensions by default.

    See Also:
        For more details and references, visit
        :ref:`zoo/learning-stdp:Generalized-Kernel Spike-Timing Dependent Plasticity (Kernel STDP)` and
        :ref:`zoo/learning-stdp:Delay-Adjusted Spike-Timing Dependent Plasticity (Delay-Adjusted STDP)` in the zoo.
    """

    def __init__(
        self,
        kernel_post: SpikeTimeHalfKernel,
        kernel_pre: SpikeTimeHalfKernel,
        kernel_post_kwargs: dict[str, Any],
        kernel_pre_kwargs: dict[str, Any],
        batch_reduction: (
            Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] | None
        ) = None,
        inplace: bool = False,
        **kwargs,
    ):
        # call superclass constructor
        IndependentCellTrainer.__init__(self, **kwargs)

        # default hyperparameters
        self.kernel_post = kernel_post
        self.kernel_pre = kernel_pre
        self.kernel_post_kwargs = kernel_post_kwargs
        self.kernel_pre_kwargs = kernel_pre_kwargs
        self.batchreduce = batch_reduction if batch_reduction else torch.mean
        self.inplace = bool(inplace)

    def _build_cell_state(self, **kwargs) -> Module:
        r"""Builds auxiliary state for a cell.

        Keyword arguments will override module-level hyperparameters.

        Returns:
            Module: state module.
        """
        state = Module()

        kernel_post = kwargs.get("kernel_post", self.kernel_post)
        kernel_pre = kwargs.get("kernel_pre", self.kernel_pre)
        kernel_post_kwargs = kwargs.get(
            "kernel_post_kwargs",
            {
                k: v.clone().detach() if isinstance(v, torch.Tensor) else v
                for k, v in self.kernel_post_kwargs.items()
            },
        )
        kernel_pre_kwargs = kwargs.get(
            "kernel_pre_kwargs",
            {
                k: v.clone().detach() if isinstance(v, torch.Tensor) else v
                for k, v in self.kernel_pre_kwargs.items()
            },
        )
        batch_reduction = kwargs.get("batch_reduction", self.batchreduce)
        inplace = kwargs.get("inplace", self.inplace)

        state.kernel_post = kernel_post
        state.kernel_pre = kernel_pre

        state.kernel_post_kwargs = {}
        state.kernel_post_tensor_kwargs = Module()
        for k, v in kernel_post_kwargs.items():
            if isinstance(v, torch.Tensor):
                state.kernel_post_tensor_kwargs.register_buffer(k, v)
            else:
                state.kernel_post_kwargs[k] = v

        state.kernel_pre_kwargs = {}
        state.kernel_pre_tensor_kwargs = Module()
        for k, v in kernel_pre_kwargs.items():
            if isinstance(v, torch.Tensor):
                state.kernel_pre_tensor_kwargs.register_buffer(k, v)
            else:
                state.kernel_pre_kwargs[k] = v

        state.batchreduce = (
            batch_reduction if (batch_reduction is not None) else torch.mean
        )
        state.inplace = bool(inplace)

        return state

    def register_cell(
        self,
        name: str,
        cell: Cell,
        /,
        **kwargs: Any,
    ) -> IndependentCellTrainer.Unit:
        r"""Adds a cell with required state.

        Args:
            name (str): name of the cell to add.
            cell (Cell): cell to add.

        Keyword Args:
            kernel_post (~firebrand.functional.SpikeTimeHalfKernel): function for
                determining update strength on postsynaptic spikes.
            kernel_pre (~firebrand.functional.SpikeTimeHalfKernel): function for
                determining update strength on presynaptic spikes.
            kernel_post_kwargs (dict[str, Any]): keyword arguments passed into ``kernel_post``.
            kernel_pre_kwargs (dict[str, Any]): keyword arguments passed into ``kernel_pre``.
            batch_reduction (Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] | None):
                function to reduce updates over the batch dimension, :py:func:`torch.mean`
                when ``None``.
            inplace (bool, optional): if :py:class:`~inferno.RecordTensor` write operations
                should be performed in-place.

        Returns:
            IndependentCellTrainer.Unit: specified cell, auxiliary state, and monitors.

        Important:
            Any specified keyword arguments will override the default hyperparameters
            set on initialization. See :py:class:`KernelSTDP` for details.
        """
        # add the cell with additional hyperparameters
        cell, state = self.add_cell(
            name, cell, self._build_cell_state(**kwargs), ["weight"]
        )

        # common and derived arguments
        monitor_kwargs = {
            "as_prehook": False,
            "train_update": True,
            "eval_update": False,
            "prepend": True,
        }

        # postsynaptic event-time monitor
        self.add_monitor(
            name,
            "spike_post",
            "neuron.spike",
            StateMonitor.partialconstructor(
                reducer=EventReducer(
                    cell.connection.dt,
                    lambda x: x.bool(),
                    initial="nan",
                    duration=0.0,
                    inclusive=True,
                    inplace=state.inplace,
                ),
                **monitor_kwargs,
            ),
            False,
            dt=cell.connection.dt,
            inplace=state.inplace,
        )

        # presynaptic event-time monitor
        self.add_monitor(
            name,
            "spike_pre",
            "synapse.spike",
            StateMonitor.partialconstructor(
                reducer=EventReducer(
                    cell.connection.dt,
                    lambda x: x.bool(),
                    initial="nan",
                    duration=0.0,
                    inclusive=True,
                    inplace=state.inplace,
                ),
                **monitor_kwargs,
            ),
            False,
            dt=cell.connection.dt,
            inplace=state.inplace,
        )

        return self.get_unit(name)

    def forward(self) -> None:
        r"""Processes update for given layers based on current monitor stored data."""
        # iterate through self
        for cell, state, monitors in self:
            # skip if self or cell is not in training mode or has no updater
            if not cell.training or not self.training or not cell.updater:
                continue

            # relative spike times, reshaped into receptive format
            t_post = cell.connection.postsyn_receptive(monitors["spike_post"].peek())
            t_pre = cell.connection.presyn_receptive(monitors["spike_pre"].peek())

            # adjusted time difference
            t_delta = t_pre - t_post - cell.connection.delay.unsqueeze(-1)

            # partial updates
            dpost = state.kernel_post(
                t_delta,
                **(
                    state.kernel_post_kwargs
                    | {k: v for k, v in state.kernel_post_tensor_kwargs.named_buffers()}
                ),
            )
            dpre = state.kernel_pre(
                t_delta,
                **(
                    state.kernel_pre_kwargs
                    | {k: v for k, v in state.kernel_pre_tensor_kwargs.named_buffers()}
                ),
            )

            # accumulate partials
            cell.updater.weight = (
                state.batchreduce(dpost.clamp_min(0.0).nansum(dim=-1), 0)
                + state.batchreduce(dpre.clamp_min(0.0).nansum(dim=-1), 0),
                -(
                    state.batchreduce(dpost.clamp_max(0.0).nansum(dim=-1), 0)
                    + state.batchreduce(dpre.clamp_max(0.0).nansum(dim=-1), 0)
                ),
            )


class DelayAdjustedKernelSTDPD(IndependentCellTrainer):
    r"""Delay-adjusted general kernel spike-timing dependent plasticity delay trainer.

    .. math::
        \begin{align*}
            d(t + \Delta t) - d(t) &= K_\text{post}(t_\Delta(t)) [t_\Delta(t) \geq 0] \\
            &+ K_\text{pre}(t_\Delta(t)) [t_\Delta(t) < 0] \\
            t_\Delta(t) &= t^f_\text{post} - t^f_\text{pre} - d(t)
        \end{align*}

    Where:

    Times :math:`t` and :math:`t_n^f` are the current time and the time of the most
    recent spike from neuron :math:`n` respectively, :math:`\Delta t` is the duration of
    the simulation step, and :math:`d(t)` are the learned delays.

    Args:
        kernel_post (~firebrand.functional.SpikeTimeHalfKernel): function for
            determining update strength on postsynaptic spikes, :math:`K_\text{post}`.
        kernel_pre (~firebrand.functional.SpikeTimeHalfKernel): function for
            determining update strength on presynaptic spikes, :math:`K_\text{pre}`.
        kernel_post_kwargs (dict[str, Any]): keyword arguments passed into ``kernel_post``.
        kernel_pre_kwargs (dict[str, Any]): keyword arguments passed into ``kernel_pre``.
        batch_reduction (Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] | None):
            function to reduce updates over the batch dimension, :py:func:`torch.mean`
            when ``None``. Defaults to ``None``.
        inplace (bool, optional): if :py:class:`~inferno.RecordTensor` write operations
            should be performed in-place. Defaults to ``False``.

    Important:
        It is expected for this to be called after every trainable batch. Variables
        used are not stored (or are invalidated) if multiple batches are given before
        an update.

    Important:
        The :py:class:`~torch.Tensor` values in ``kernel_post_kwargs`` and
        ``kernel_pre_kwargs`` will each be unpacked into a module in the cell's state,
        and registered as buffers.

        If given as a default to the ``DelayAdjustedKernelSTDPD`` constructor, then they
        will be cloned and detached first.

    Note:
        The constructor arguments are hyperparameters and can be overridden on a
        cell-by-cell basis.

    Note:
        ``batch_reduction`` can be one of the functions in PyTorch including but not
        limited to :py:func:`torch.sum`, :py:func:`torch.mean`, and :py:func:`torch.amax`.
        A custom function with similar behavior can also be passed in. Like with the
        included function, it should not keep the original dimensions by default.

    See Also:
        For more details and references, visit
        :ref:`zoo/learning-stdp:Generalized-Kernel Spike-Timing Dependent Plasticity (Kernel STDP)` and
        :ref:`zoo/learning-stdp:Delay-Adjusted Spike-Timing Dependent Plasticity of Delays (Delay-Adjusted STDPD)` in the zoo.
    """

    def __init__(
        self,
        kernel_post: SpikeTimeHalfKernel,
        kernel_pre: SpikeTimeHalfKernel,
        kernel_post_kwargs: dict[str, Any],
        kernel_pre_kwargs: dict[str, Any],
        batch_reduction: (
            Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] | None
        ) = None,
        inplace: bool = False,
        **kwargs,
    ):
        # call superclass constructor
        IndependentCellTrainer.__init__(self, **kwargs)

        # default hyperparameters
        self.kernel_post = kernel_post
        self.kernel_pre = kernel_pre
        self.kernel_post_kwargs = kernel_post_kwargs
        self.kernel_pre_kwargs = kernel_pre_kwargs
        self.batchreduce = batch_reduction if batch_reduction else torch.mean
        self.inplace = bool(inplace)

    def _build_cell_state(self, **kwargs) -> Module:
        r"""Builds auxiliary state for a cell.

        Keyword arguments will override module-level hyperparameters.

        Returns:
            Module: state module.
        """
        state = Module()

        kernel_post = kwargs.get("kernel_post", self.kernel_post)
        kernel_pre = kwargs.get("kernel_pre", self.kernel_pre)
        kernel_post_kwargs = kwargs.get(
            "kernel_post_kwargs",
            {
                k: v.clone().detach() if isinstance(v, torch.Tensor) else v
                for k, v in self.kernel_post_kwargs.items()
            },
        )
        kernel_pre_kwargs = kwargs.get(
            "kernel_pre_kwargs",
            {
                k: v.clone().detach() if isinstance(v, torch.Tensor) else v
                for k, v in self.kernel_pre_kwargs.items()
            },
        )
        batch_reduction = kwargs.get("batch_reduction", self.batchreduce)
        inplace = kwargs.get("inplace", self.inplace)

        state.kernel_post = kernel_post
        state.kernel_pre = kernel_pre

        state.kernel_post_kwargs = {}
        state.kernel_post_tensor_kwargs = Module()
        for k, v in kernel_post_kwargs.items():
            if isinstance(v, torch.Tensor):
                state.kernel_post_tensor_kwargs.register_buffer(k, v)
            else:
                state.kernel_post_kwargs[k] = v

        state.kernel_pre_kwargs = {}
        state.kernel_pre_tensor_kwargs = Module()
        for k, v in kernel_pre_kwargs.items():
            if isinstance(v, torch.Tensor):
                state.kernel_pre_tensor_kwargs.register_buffer(k, v)
            else:
                state.kernel_pre_kwargs[k] = v

        state.batchreduce = (
            batch_reduction if (batch_reduction is not None) else torch.mean
        )
        state.inplace = bool(inplace)

        return state

    def register_cell(
        self,
        name: str,
        cell: Cell,
        /,
        **kwargs: Any,
    ) -> IndependentCellTrainer.Unit:
        r"""Adds a cell with required state.

        Args:
            name (str): name of the cell to add.
            cell (Cell): cell to add.

        Keyword Args:
            kernel_post (~firebrand.functional.SpikeTimeHalfKernel): function for
                determining update strength on postsynaptic spikes.
            kernel_pre (~firebrand.functional.SpikeTimeHalfKernel): function for
                determining update strength on presynaptic spikes.
            kernel_post_kwargs (dict[str, Any]): keyword arguments passed into ``kernel_post``.
            kernel_pre_kwargs (dict[str, Any]): keyword arguments passed into ``kernel_pre``.
            batch_reduction (Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] | None):
                function to reduce updates over the batch dimension, :py:func:`torch.mean`
                when ``None``.
            inplace (bool, optional): if :py:class:`~inferno.RecordTensor` write operations
                should be performed in-place.

        Returns:
            IndependentCellTrainer.Unit: specified cell, auxiliary state, and monitors.

        Important:
            Any specified keyword arguments will override the default hyperparameters
            set on initialization. See :py:class:`KernelSTDP` for details.
        """
        # add the cell with additional hyperparameters
        cell, state = self.add_cell(
            name, cell, self._build_cell_state(**kwargs), ["delay"]
        )

        # common and derived arguments
        monitor_kwargs = {
            "as_prehook": False,
            "train_update": True,
            "eval_update": False,
            "prepend": True,
        }

        # postsynaptic event-time monitor
        self.add_monitor(
            name,
            "spike_post",
            "neuron.spike",
            StateMonitor.partialconstructor(
                reducer=EventReducer(
                    cell.connection.dt,
                    lambda x: x.bool(),
                    initial="nan",
                    duration=0.0,
                    inclusive=True,
                    inplace=state.inplace,
                ),
                **monitor_kwargs,
            ),
            False,
            dt=cell.connection.dt,
            inplace=state.inplace,
        )

        # presynaptic event-time monitor
        self.add_monitor(
            name,
            "spike_pre",
            "synapse.spike",
            StateMonitor.partialconstructor(
                reducer=EventReducer(
                    cell.connection.dt,
                    lambda x: x.bool(),
                    initial="nan",
                    duration=0.0,
                    inclusive=True,
                    inplace=state.inplace,
                ),
                **monitor_kwargs,
            ),
            False,
            dt=cell.connection.dt,
            inplace=state.inplace,
        )

        return self.get_unit(name)

    def forward(self) -> None:
        r"""Processes update for given layers based on current monitor stored data."""
        # iterate through self
        for cell, state, monitors in self:
            # skip if self or cell is not in training mode or has no updater
            if not cell.training or not self.training or not cell.updater:
                continue

            # relative spike times, reshaped into receptive format
            t_post = cell.connection.postsyn_receptive(monitors["spike_post"].peek())
            t_pre = cell.connection.presyn_receptive(monitors["spike_pre"].peek())

            # adjusted time difference
            t_delta = t_pre - t_post - cell.connection.delay.unsqueeze(-1)

            # partial updates
            dpost = state.kernel_post(
                t_delta,
                **(
                    state.kernel_post_kwargs
                    | {k: v for k, v in state.kernel_post_tensor_kwargs.named_buffers()}
                ),
            )
            dpre = state.kernel_pre(
                t_delta,
                **(
                    state.kernel_pre_kwargs
                    | {k: v for k, v in state.kernel_pre_tensor_kwargs.named_buffers()}
                ),
            )

            # accumulate partials
            cell.updater.delay = (
                state.batchreduce(dpost.clamp_min(0.0).nansum(dim=-1), 0)
                + state.batchreduce(dpre.clamp_min(0.0).nansum(dim=-1), 0),
                -(
                    state.batchreduce(dpost.clamp_max(0.0).nansum(dim=-1), 0)
                    + state.batchreduce(dpre.clamp_max(0.0).nansum(dim=-1), 0)
                ),
            )
