from __future__ import annotations
from .. import IndependentCellTrainer
from ... import Module
from ..._internal import argtest
from ...neural import Cell
from ...observe import (
    StateMonitor,
    CumulativeTraceReducer,
    NearestTraceReducer,
    PassthroughReducer,
)
import einops as ein
import torch
from typing import Any, Callable, Literal


class STDP(IndependentCellTrainer):
    r"""Pair-based spike-timing dependent plasticity trainer.

    .. math::
        w(t + \Delta t) - w(t) = x_\text{pre}(t) \bigl[t = t^f_\text{post}\bigr] +
        x_\text{post}(t) \bigl[t = t^f_\text{pre}\bigr]

    When ``trace_mode = "cumulative"``:

    .. math::
        \begin{align*}
            x_\text{pre}(t) &= x_\text{pre}(t - \Delta t)
            \exp\left(-\frac{\Delta t}{\tau_\text{pre}}\right) +
            \eta_\text{post} \left[t = t_\text{pre}^f\right] \\
            x_\text{post}(t) &= x_\text{post}(t - \Delta t)
            \exp\left(-\frac{\Delta t}{\tau_\text{post}}\right) +
            \eta_\text{pre} \left[t = t_\text{post}^f\right]
        \end{align*}

    When ``trace_mode = "nearest"``:

    .. math::
        \begin{align*}
            x_\text{pre}(t) &=
            \begin{cases}
                \eta_\text{post} & t = t_\text{pre}^f \\
                x_\text{pre}(t - \Delta t)
                \exp\left(-\frac{\Delta t}{\tau_\text{pre}}\right)
                & t \neq t_\text{pre}^f
            \end{cases} \\
            x_\text{post}(t) &=
            \begin{cases}
                \eta_\text{pre} & t = t_\text{post}^f \\
                x_\text{post}(t - \Delta t)
                \exp\left(-\frac{\Delta t}{\tau_\text{post}}\right)
                & t \neq t_\text{post}^f
            \end{cases}
        \end{align*}

    Where:

    Times :math:`t` and :math:`t_n^f` are the current time and the time of the most
    recent spike from neuron :math:`n`, respectively, and :math:`\Delta t` is the
    duration of the simulation step.

    The signs of the learning rates :math:`\eta_\text{post}` and :math:`\eta_\text{pre}`
    control which terms are potentiative and which terms are depressive. The terms can
    be scaled for weight dependence on updating.

    +-------------------+--------------------------------------+-------------------------------------+-------------------------------------------+-------------------------------------------+
    | Mode              | :math:`\text{sgn}(\eta_\text{post})` | :math:`\text{sgn}(\eta_\text{pre})` | LTP Term(s)                               | LTD Term(s)                               |
    +===================+======================================+=====================================+===========================================+===========================================+
    | Hebbian           | :math:`+`                            | :math:`-`                           | :math:`\eta_\text{post}`                  | :math:`\eta_\text{pre}`                   |
    +-------------------+--------------------------------------+-------------------------------------+-------------------------------------------+-------------------------------------------+
    | Anti-Hebbian      | :math:`-`                            | :math:`+`                           | :math:`\eta_\text{pre}`                   | :math:`\eta_\text{post}`                  |
    +-------------------+--------------------------------------+-------------------------------------+-------------------------------------------+-------------------------------------------+
    | Potentiative Only | :math:`+`                            | :math:`+`                           | :math:`\eta_\text{post}, \eta_\text{pre}` | None                                      |
    +-------------------+--------------------------------------+-------------------------------------+-------------------------------------------+-------------------------------------------+
    | Depressive Only   | :math:`-`                            | :math:`-`                           | None                                      | :math:`\eta_\text{post}, \eta_\text{pre}` |
    +-------------------+--------------------------------------+-------------------------------------+-------------------------------------------+-------------------------------------------+

    Args:
        lr_post (float): learning rate for updates on postsynaptic spikes,
            :math:`\eta_\text{post}`.
        lr_pre (float): learning rate for updates on presynaptic spikes,
            :math:`\eta_\text{pre}`.
        tc_post (float): time constant of exponential decay of postsynaptic trace,
            :math:`\tau_\text{post}`, in :math:`ms`.
        tc_pre (float): time constant of exponential decay of presynaptic trace,
            :math:`\tau_\text{pre}`, in :math:`ms`.
        delayed (bool, optional): if the updater should assume that learned delays, if
            present, may change. Defaults to ``False``.
        interp_tolerance (float, optional): maximum difference in time from an observation
            to treat as co-occurring, in :math:`\text{ms}`. Defaults to ``0.0``.
        trace_mode (Literal["cumulative", "nearest"], optional): method to use for
            calculating spike traces. Defaults to ``"cumulative"``.
        batch_reduction (Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] | None):
            function to reduce updates over the batch dimension, :py:func:`torch.mean`
            when ``None``. Defaults to ``None``.

    Important:
        When ``delayed`` is ``True``, the history for the presynaptic activity (spike
        traces and spike activity) is preserved in its un-delayed form and is then
        accessed using the connection's :py:attr:`~inferno.neural.Connection.selector`.

        When ``delayed`` is ``False``, only the most recent delay-adjusted presynaptic
        activity is preserved.

    Important:
        It is expected for this to be called after every trainable batch. Variables
        used are not stored (or are invalidated) if multiple batches are given before
        an update.

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
        :ref:`zoo/learning-stdp:Spike-Timing Dependent Plasticity (STDP)` in the zoo.
    """

    def __init__(
        self,
        lr_post: float,
        lr_pre: float,
        tc_post: float,
        tc_pre: float,
        delayed: bool = False,
        interp_tolerance: float = 0.0,
        trace_mode: Literal["cumulative", "nearest"] = "cumulative",
        batch_reduction: (
            Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] | None
        ) = None,
        **kwargs,
    ):
        # call superclass constructor
        IndependentCellTrainer.__init__(self, **kwargs)

        # default hyperparameters
        self.lr_post = float(lr_post)
        self.lr_pre = float(lr_pre)
        self.tc_post = argtest.gt("tc_post", tc_post, 0, float)
        self.tc_pre = argtest.gt("tc_pre", tc_pre, 0, float)
        self.delayed = bool(delayed)
        self.tolerance = argtest.gte("interp_tolerance", interp_tolerance, 0, float)
        self.trace = argtest.oneof(
            "trace_mode", trace_mode, "cumulative", "nearest", op=(lambda x: x.lower())
        )
        self.batchreduce = batch_reduction if batch_reduction else torch.mean

    def _build_cell_state(self, **kwargs) -> Module:
        r"""Builds auxiliary state for a cell.

        Keyword arguments will override module-level hyperparameters.

        Returns:
            Module: state module.
        """
        state = Module()

        lr_post = kwargs.get("lr_post", self.lr_post)
        lr_pre = kwargs.get("lr_pre", self.lr_pre)
        tc_post = kwargs.get("tc_post", self.tc_post)
        tc_pre = kwargs.get("tc_pre", self.tc_pre)
        delayed = kwargs.get("delayed", self.delayed)
        interp_tolerance = kwargs.get("interp_tolerance", self.tolerance)
        trace_mode = kwargs.get("trace_mode", self.trace)
        batch_reduction = kwargs.get("batch_reduction", self.batchreduce)

        state.lr_post = float(lr_post)
        state.lr_pre = float(lr_pre)
        state.tc_post = argtest.gt("tc_post", tc_post, 0, float)
        state.tc_pre = argtest.gt("tc_pre", tc_pre, 0, float)
        state.delayed = bool(delayed)
        state.tolerance = argtest.gte("interp_tolerance", interp_tolerance, 0, float)
        state.tracemode = argtest.oneof(
            "trace_mode", trace_mode, "cumulative", "nearest", op=(lambda x: x.lower())
        )
        match state.tracemode:
            case "cumulative":
                state.tracecls = CumulativeTraceReducer
            case "nearest":
                state.tracecls = NearestTraceReducer
            case "_":
                raise RuntimeError(
                    f"an invalid trace mode of '{state.tracemode}' has been set, "
                    "expected one of: 'cumulative', 'nearest'"
                )
        state.batchreduce = (
            batch_reduction if (batch_reduction is not None) else torch.mean
        )

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
            lr_post (float): learning rate for updates on postsynaptic spikes.
            lr_pre (float): learning rate for updates on presynaptic spikes.
            tc_post (float): time constant of exponential decay of postsynaptic trace.
            tc_pre (float): time constant of exponential decay of presynaptic trace.
            delayed (bool): if the updater should assume that learned delays,
                if present, may change.
            interp_tolerance (float): maximum difference in time from an observation
                to treat as co-occurring.
            trace_mode (Literal["cumulative", "nearest"]): method to use for
                calculating spike traces.
            batch_reduction (Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor]):
                function to reduce updates over the batch dimension.

        Returns:
            IndependentCellTrainer.Unit: specified cell, auxiliary state, and monitors.

        Important:
            Any specified keyword arguments will override the default hyperparameters
            set on initialization. See :py:class:`STDP` for details.
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

        # postsynaptic trace monitor (weighs hebbian LTD)
        self.add_monitor(
            name,
            "trace_post",
            "neuron.spike",
            StateMonitor.partialconstructor(
                reducer=state.tracecls(
                    cell.connection.dt,
                    state.tc_post,
                    amplitude=abs(state.lr_pre),
                    target=True,
                    duration=0.0,
                    inclusive=True,
                ),
                **monitor_kwargs,
            ),
            False,
            dt=cell.connection.dt,
            amp=abs(state.lr_pre),
            tc=state.tc_post,
            trace=state.tracemode,
        )

        # postsynaptic spike monitor (triggers hebbian LTP)
        self.add_monitor(
            name,
            "spike_post",
            "neuron.spike",
            StateMonitor.partialconstructor(
                reducer=PassthroughReducer(
                    cell.connection.dt,
                    duration=0.0,
                    inclusive=True,
                ),
                **monitor_kwargs,
            ),
            False,
            dt=cell.connection.dt,
        )

        # presynaptic trace monitor (weighs hebbian LTP)
        # when the delayed condition is true, using synapse.spike records the raw
        # spike times rather than the delay adjusted times of synspike.
        self.add_monitor(
            name,
            "trace_pre",
            "synapse.spike" if delayed else "connection.synspike",
            StateMonitor.partialconstructor(
                reducer=state.tracecls(
                    cell.connection.dt,
                    state.tc_pre,
                    amplitude=abs(state.lr_post),
                    target=True,
                    duration=cell.connection.delayedby if delayed else 0.0,
                    inclusive=True,
                ),
                **monitor_kwargs,
            ),
            False,
            dt=cell.connection.dt,
            amp=abs(state.lr_post),
            tc=state.tc_pre,
            trace=state.tracemode,
            delayed=delayed,
        )

        # presynaptic spike monitor (triggers hebbian LTD)
        # when the delayed condition is true, using synapse.spike records the raw
        # spike times rather than the delay adjusted times of synspike.
        self.add_monitor(
            name,
            "spike_pre",
            "synapse.spike" if delayed else "connection.synspike",
            StateMonitor.partialconstructor(
                reducer=PassthroughReducer(
                    cell.connection.dt,
                    duration=cell.connection.delayedby if delayed else 0.0,
                    inclusive=True,
                ),
                **monitor_kwargs,
            ),
            False,
            dt=cell.connection.dt,
            delayed=delayed,
        )

        return self.get_unit(name)

    def forward(self) -> None:
        r"""Processes update for given layers based on current monitor stored data."""
        # iterate through self
        for cell, state, monitors in self:
            # skip if self or cell is not in training mode or has no updater
            if not cell.training or not self.training or not cell.updater:
                continue

            # spike traces, reshaped into receptive format
            x_post = cell.connection.postsyn_receptive(monitors["trace_post"].peek())
            x_pre = cell.connection.presyn_receptive(
                monitors["trace_pre"].view(cell.connection.selector, state.tolerance)
                if state.delayed and cell.connection.delayedby
                else monitors["trace_pre"].peek()
            )

            # spike presence, reshaped into receptive format
            i_post = cell.connection.postsyn_receptive(monitors["spike_post"].peek())
            i_pre = cell.connection.presyn_receptive(
                monitors["spike_pre"].view(cell.connection.selector, state.tolerance)
                if state.delayed and cell.connection.delayedby
                else monitors["spike_pre"].peek()
            )

            # partial updates
            dpost = state.batchreduce(
                ein.einsum(i_post, x_pre, "b ... r, b ... r -> b ..."), 0
            )
            dpre = state.batchreduce(
                ein.einsum(i_pre, x_post, "b ... r, b ... r -> b ..."), 0
            )

            # accumulate partials with mode condition
            match (state.lr_post >= 0, state.lr_pre >= 0):
                case (False, False):  # depressive
                    cell.updater.weight = (None, dpost + dpre)
                case (False, True):  # anti-hebbian
                    cell.updater.weight = (dpre, dpost)
                case (True, False):  # hebbian
                    cell.updater.weight = (dpost, dpre)
                case (True, True):  # potentiative
                    cell.updater.weight = (dpost + dpre, None)


class StableSTDP(IndependentCellTrainer):
    r"""Pair-based spike-timing dependent plasticity trainer.

    Rather than recording trace values with amplitudes specified by the learning rates,
    this uses an amplitude of 1. With some testing the difference appears to be minor.
    With limited testing, the maximum difference between this and the "unstable"
    implementation is around 2e-6 times the average weight. Not included with the
    default exports.

    .. math::
        w(t + \Delta t) - w(t) = \eta_\text{post} x_\text{pre}(t) \bigl[t = t^f_\text{post}\bigr]
        + eta_\text{pre} x_\text{post}(t) \bigl[t = t^f_\text{pre}\bigr]

    When ``trace_mode = "cumulative"``:

    .. math::
        \begin{align*}
            x_\text{pre}(t) &= x_\text{pre}(t - \Delta t)
            \exp\left(-\frac{\Delta t}{\tau_\text{pre}}\right)
            + \left[t = t_\text{pre}^f\right] \\
            x_\text{post}(t) &= x_\text{post}(t - \Delta t)
            \exp\left(-\frac{\Delta t}{\tau_\text{post}}\right)
            + \left[t = t_\text{post}^f\right]
        \end{align*}

    When ``trace_mode = "nearest"``:

    .. math::
        \begin{align*}
            x_\text{pre}(t) &=
            \begin{cases}
                1 & t = t_\text{pre}^f \\
                x_\text{pre}(t - \Delta t)
                \exp\left(-\frac{\Delta t}{\tau_\text{pre}}\right)
                & t \neq t_\text{pre}^f
            \end{cases} \\
            x_\text{post}(t) &=
            \begin{cases}
                1 & t = t_\text{post}^f \\
                x_\text{post}(t - \Delta t)
                \exp\left(-\frac{\Delta t}{\tau_\text{post}}\right)
                & t \neq t_\text{post}^f
            \end{cases}
        \end{align*}

    Where:

    Times :math:`t` and :math:`t_n^f` are the current time and the time of the most
    recent spike from neuron :math:`n`, respectively, and :math:`\Delta t` is the
    duration of the simulation step.

    The signs of the learning rates :math:`\eta_\text{post}` and :math:`\eta_\text{pre}`
    control which terms are potentiative and which terms are depressive. The terms can
    be scaled for weight dependence on updating.

    +-------------------+--------------------------------------+-------------------------------------+-------------------------------------------+-------------------------------------------+
    | Mode              | :math:`\text{sgn}(\eta_\text{post})` | :math:`\text{sgn}(\eta_\text{pre})` | LTP Term(s)                               | LTD Term(s)                               |
    +===================+======================================+=====================================+===========================================+===========================================+
    | Hebbian           | :math:`+`                            | :math:`-`                           | :math:`\eta_\text{post}`                  | :math:`\eta_\text{pre}`                   |
    +-------------------+--------------------------------------+-------------------------------------+-------------------------------------------+-------------------------------------------+
    | Anti-Hebbian      | :math:`-`                            | :math:`+`                           | :math:`\eta_\text{pre}`                   | :math:`\eta_\text{post}`                  |
    +-------------------+--------------------------------------+-------------------------------------+-------------------------------------------+-------------------------------------------+
    | Potentiative Only | :math:`+`                            | :math:`+`                           | :math:`\eta_\text{post}, \eta_\text{pre}` | None                                      |
    +-------------------+--------------------------------------+-------------------------------------+-------------------------------------------+-------------------------------------------+
    | Depressive Only   | :math:`-`                            | :math:`-`                           | None                                      | :math:`\eta_\text{post}, \eta_\text{pre}` |
    +-------------------+--------------------------------------+-------------------------------------+-------------------------------------------+-------------------------------------------+

    Args:
        lr_post (float): learning rate for updates on postsynaptic spikes,
            :math:`\eta_\text{post}`.
        lr_pre (float): learning rate for updates on presynaptic spikes,
            :math:`\eta_\text{pre}`.
        tc_post (float): time constant of exponential decay of postsynaptic trace,
            :math:`\tau_\text{post}`, in :math:`ms`.
        tc_pre (float): time constant of exponential decay of presynaptic trace,
            :math:`\tau_\text{pre}`, in :math:`ms`.
        delayed (bool, optional): if the updater should assume that learned delays, if
            present, may change. Defaults to ``False``.
        interp_tolerance (float, optional): maximum difference in time from an observation
            to treat as co-occurring, in :math:`\text{ms}`. Defaults to ``0.0``.
        trace_mode (Literal["cumulative", "nearest"], optional): method to use for
            calculating spike traces. Defaults to ``"cumulative"``.
        batch_reduction (Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] | None):
            function to reduce updates over the batch dimension, :py:func:`torch.mean`
            when ``None``. Defaults to ``None``.

    Important:
        When ``delayed`` is ``True``, the history for the presynaptic activity (spike
        traces and spike activity) is preserved in its un-delayed form and is then
        accessed using the connection's :py:attr:`~inferno.neural.Connection.selector`.

        When ``delayed`` is ``False``, only the most recent delay-adjusted presynaptic
        activity is preserved.

    Important:
        It is expected for this to be called after every trainable batch. Variables
        used are not stored (or are invalidated) if multiple batches are given before
        an update.

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
        :ref:`zoo/learning-stdp:Spike-Timing Dependent Plasticity (STDP)` in the zoo.
    """

    def __init__(
        self,
        lr_post: float,
        lr_pre: float,
        tc_post: float,
        tc_pre: float,
        delayed: bool = False,
        interp_tolerance: float = 0.0,
        trace_mode: Literal["cumulative", "nearest"] = "cumulative",
        batch_reduction: (
            Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] | None
        ) = None,
        **kwargs,
    ):
        # call superclass constructor
        IndependentCellTrainer.__init__(self, **kwargs)

        # default hyperparameters
        self.lr_post = float(lr_post)
        self.lr_pre = float(lr_pre)
        self.tc_post = argtest.gt("tc_post", tc_post, 0, float)
        self.tc_pre = argtest.gt("tc_pre", tc_pre, 0, float)
        self.delayed = bool(delayed)
        self.tolerance = argtest.gte("interp_tolerance", interp_tolerance, 0, float)
        self.trace = argtest.oneof(
            "trace_mode", trace_mode, "cumulative", "nearest", op=(lambda x: x.lower())
        )
        self.batchreduce = batch_reduction if batch_reduction else torch.mean

    def _build_cell_state(self, **kwargs) -> Module:
        r"""Builds auxiliary state for a cell.

        Keyword arguments will override module-level hyperparameters.

        Returns:
            Module: state module.
        """
        state = Module()

        lr_post = kwargs.get("lr_post", self.lr_post)
        lr_pre = kwargs.get("lr_pre", self.lr_pre)
        tc_post = kwargs.get("tc_post", self.tc_post)
        tc_pre = kwargs.get("tc_pre", self.tc_pre)
        delayed = kwargs.get("delayed", self.delayed)
        interp_tolerance = kwargs.get("interp_tolerance", self.tolerance)
        trace_mode = kwargs.get("trace_mode", self.trace)
        batch_reduction = kwargs.get("batch_reduction", self.batchreduce)

        state.lr_post = float(lr_post)
        state.lr_pre = float(lr_pre)
        state.tc_post = argtest.gt("tc_post", tc_post, 0, float)
        state.tc_pre = argtest.gt("tc_pre", tc_pre, 0, float)
        state.delayed = bool(delayed)
        state.tolerance = argtest.gte("interp_tolerance", interp_tolerance, 0, float)
        state.tracemode = argtest.oneof(
            "trace_mode", trace_mode, "cumulative", "nearest", op=(lambda x: x.lower())
        )
        match state.tracemode:
            case "cumulative":
                state.tracecls = CumulativeTraceReducer
            case "nearest":
                state.tracecls = NearestTraceReducer
            case "_":
                raise RuntimeError(
                    f"an invalid trace mode of '{state.tracemode}' has been set, "
                    "expected one of: 'cumulative', 'nearest'"
                )
        state.batchreduce = (
            batch_reduction if (batch_reduction is not None) else torch.mean
        )

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
            lr_post (float): learning rate for updates on postsynaptic spikes.
            lr_pre (float): learning rate for updates on presynaptic spikes.
            tc_post (float): time constant of exponential decay of postsynaptic trace.
            tc_pre (float): time constant of exponential decay of presynaptic trace.
            delayed (bool): if the updater should assume that learned delays,
                if present, may change.
            interp_tolerance (float): maximum difference in time from an observation
                to treat as co-occurring.
            trace_mode (Literal["cumulative", "nearest"]): method to use for
                calculating spike traces.
            batch_reduction (Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor]):
                function to reduce updates over the batch dimension.

        Returns:
            IndependentCellTrainer.Unit: specified cell, auxiliary state, and monitors.

        Important:
            Any specified keyword arguments will override the default hyperparameters
            set on initialization. See :py:class:`STDP` for details.
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

        # postsynaptic trace monitor (weighs hebbian LTD)
        self.add_monitor(
            name,
            "trace_post",
            "neuron.spike",
            StateMonitor.partialconstructor(
                reducer=state.tracecls(
                    cell.connection.dt,
                    state.tc_post,
                    amplitude=1.0,
                    target=True,
                    duration=0.0,
                    inclusive=True,
                ),
                **monitor_kwargs,
            ),
            False,
            dt=cell.connection.dt,
            tc=state.tc_post,
            trace=state.tracemode,
        )

        # postsynaptic spike monitor (triggers hebbian LTP)
        self.add_monitor(
            name,
            "spike_post",
            "neuron.spike",
            StateMonitor.partialconstructor(
                reducer=PassthroughReducer(
                    cell.connection.dt,
                    duration=0.0,
                    inclusive=True,
                ),
                **monitor_kwargs,
            ),
            False,
            dt=cell.connection.dt,
        )

        # presynaptic trace monitor (weighs hebbian LTP)
        # when the delayed condition is true, using synapse.spike records the raw
        # spike times rather than the delay adjusted times of synspike.
        self.add_monitor(
            name,
            "trace_pre",
            "synapse.spike" if delayed else "connection.synspike",
            StateMonitor.partialconstructor(
                reducer=state.tracecls(
                    cell.connection.dt,
                    state.tc_pre,
                    amplitude=1.0,
                    target=True,
                    duration=cell.connection.delayedby if delayed else 0.0,
                    inclusive=True,
                ),
                **monitor_kwargs,
            ),
            False,
            dt=cell.connection.dt,
            tc=state.tc_pre,
            trace=state.tracemode,
            delayed=delayed,
        )

        # presynaptic spike monitor (triggers hebbian LTD)
        # when the delayed condition is true, using synapse.spike records the raw
        # spike times rather than the delay adjusted times of synspike.
        self.add_monitor(
            name,
            "spike_pre",
            "synapse.spike" if delayed else "connection.synspike",
            StateMonitor.partialconstructor(
                reducer=PassthroughReducer(
                    cell.connection.dt,
                    duration=cell.connection.delayedby if delayed else 0.0,
                    inclusive=True,
                ),
                **monitor_kwargs,
            ),
            False,
            dt=cell.connection.dt,
            delayed=delayed,
        )

        return self.get_unit(name)

    def forward(self) -> None:
        r"""Processes update for given layers based on current monitor stored data."""
        # iterate through self
        for cell, state, monitors in self:
            # skip if self or cell is not in training mode or has no updater
            if not cell.training or not self.training or not cell.updater:
                continue

            # spike traces, reshaped into receptive format
            x_post = cell.connection.postsyn_receptive(
                monitors["trace_post"].peek() * abs(state.lr_pre)
            )
            x_pre = cell.connection.presyn_receptive(
                (
                    monitors["trace_pre"].view(
                        cell.connection.selector, state.tolerance
                    )
                    if state.delayed and cell.connection.delayedby
                    else monitors["trace_pre"].peek()
                )
                * abs(state.lr_post)
            )

            # spike presence, reshaped into receptive format
            i_post = cell.connection.postsyn_receptive(monitors["spike_post"].peek())
            i_pre = cell.connection.presyn_receptive(
                monitors["spike_pre"].view(cell.connection.selector, state.tolerance)
                if state.delayed and cell.connection.delayedby
                else monitors["spike_pre"].peek()
            )

            # partial updates
            dpost = state.batchreduce(
                ein.einsum(i_post, x_pre, "b ... r, b ... r -> b ..."), 0
            )
            dpre = state.batchreduce(
                ein.einsum(i_pre, x_post, "b ... r, b ... r -> b ..."), 0
            )

            # accumulate partials with mode condition
            match (state.lr_post >= 0, state.lr_pre >= 0):
                case (False, False):  # depressive
                    cell.updater.weight = (None, dpost + dpre)
                case (False, True):  # anti-hebbian
                    cell.updater.weight = (dpre, dpost)
                case (True, False):  # hebbian
                    cell.updater.weight = (dpost, dpre)
                case (True, True):  # potentiative
                    cell.updater.weight = (dpost + dpre, None)


class TripletSTDP(IndependentCellTrainer):
    r"""Triplet-based spike-timing dependent plasticity trainer.

    .. math::
        \begin{align*}
            w(t + \Delta t) - w(t) &= x_a(t)\left(1 + y_b(t - \Delta t) \right) \bigl[ t = t^f_\text{post} \bigr] \\
            &+ y_a(t)\left(1 + x_b(t - \Delta t) \right) \bigl[ t = t^f_\text{pre} \bigr]
        \end{align*}

    When ``trace_mode = "cumulative"``:

    .. math::
        \begin{align*}
            x_a(t) &= x_a(t - \Delta t) \exp \left(-\frac{\Delta t}{\tau_+}\right)
            + \alpha_\text{post} \bigl[t = t^f_\text{pre}\bigr] \\
            x_b(t) &= x_b(t - \Delta t) \exp \left(-\frac{\Delta t}{\tau_x}\right)
            + \frac{\beta_\text{pre}}{\alpha_\text{pre}} \bigl[t = t^f_\text{pre}\bigr] \\
            y_a(t) &= y_a(t - \Delta t) \exp \left(-\frac{\Delta t}{\tau_-}\right)
            + \alpha_\text{pre} \bigl[t = t^f_\text{post}\bigr] \\
            y_b(t) &= y_b(t - \Delta t) \exp \left(-\frac{\Delta t}{\tau_y}\right)
            + \frac{\beta_\text{post}}{\alpha_\text{post}} \bigl[t = t^f_\text{post}\bigr]
        \end{align*}

    When ``trace_mode = "nearest"``:

    .. math::
        \begin{align*}
            x_\text{a}(t) &=
            \begin{cases}
                \alpha_\text{post} & t = t_\text{pre}^f \\
                x_\text{a}(t - \Delta t)
                \exp\left(-\frac{\Delta t}{\tau_+}\right)
                & t \neq t_\text{pre}^f
            \end{cases} \\
            x_\text{b}(t) &=
            \begin{cases}
                \beta_\text{pre} / \alpha_\text{pre} & t = t_\text{pre}^f \\
                x_\text{b}(t - \Delta t)
                \exp\left(-\frac{\Delta t}{\tau_x}\right)
                & t \neq t_\text{pre}^f
            \end{cases} \\
            y_\text{a}(t) &=
            \begin{cases}
                \alpha_\text{pre} & t = t_\text{post}^f \\
                y_\text{a}(t - \Delta t)
                \exp\left(-\frac{\Delta t}{\tau_-}\right)
                & t \neq t_\text{post}^f
            \end{cases} \\
            y_\text{b}(t) &=
            \begin{cases}
                \beta_\text{post} / \alpha_\text{post} & t = t_\text{post}^f \\
                y_\text{b}(t - \Delta t)
                \exp\left(-\frac{\Delta t}{\tau_y}\right)
                & t \neq t_\text{post}^f
            \end{cases}
        \end{align*}

    Where:

    Times :math:`t` and :math:`t_n^f` are the current time and the time of the most
    recent spike from neuron :math:`n`, respectively, and :math:`\Delta t` is the
    duration of the simulation step. The following constraints are enforced.

    .. math::
        \begin{align*}
            0 &< \tau_+ < \tau_x \\
            0 &< \tau_- < \tau_y \\
            0 &\neq \alpha_\text{post} \\
            0 &\neq \alpha_\text{pre}
        \end{align*}

    The signs of the learning rates :math:`\alpha_\text{post}`and :math:`\alpha_\text{pre}`
    control which terms are potentiative and which terms are depressive. The terms can
    be scaled for weight dependence on updating.

    +-------------------+----------------------------------------+---------------------------------------+------------------------------------------------------------------------------------+------------------------------------------------------------------------------------+
    | Mode              | :math:`\text{sgn}(\alpha_\text{post})` | :math:`\text{sgn}(\alpha_\text{pre})` | LTP Term(s)                                                                        | LTD Term(s)                                                                        |
    +===================+========================================+=======================================+====================================================================================+====================================================================================+
    | Hebbian           | :math:`+`                              | :math:`-`                             | :math:`\alpha_\text{post}, \beta_\text{post}`                                      | :math:`\alpha_\text{pre}, \beta_\text{pre}`                                        |
    +-------------------+----------------------------------------+---------------------------------------+------------------------------------------------------------------------------------+------------------------------------------------------------------------------------+
    | Anti-Hebbian      | :math:`-`                              | :math:`+`                             | :math:`\alpha_\text{pre}, \beta_\text{pre}`                                        | :math:`\alpha_\text{post}, \beta_\text{post}`                                      |
    +-------------------+----------------------------------------+---------------------------------------+------------------------------------------------------------------------------------+------------------------------------------------------------------------------------+
    | Potentiative Only | :math:`+`                              | :math:`+`                             | :math:`\alpha_\text{post}, \alpha_\text{pre}, \beta_\text{post}, \beta_\text{pre}` | None                                                                               |
    +-------------------+----------------------------------------+---------------------------------------+------------------------------------------------------------------------------------+------------------------------------------------------------------------------------+
    | Depressive Only   | :math:`-`                              | :math:`-`                             | None                                                                               | :math:`\alpha_\text{post}, \alpha_\text{pre}, \beta_\text{post}, \beta_\text{pre}` |
    +-------------------+----------------------------------------+---------------------------------------+------------------------------------------------------------------------------------+------------------------------------------------------------------------------------+

    For clarity, if the traces were unscaled, the update would be written as follows.

    .. math::
        \begin{align*}
            w(t + \Delta t) - w(t)
            &= x_a(t)\left(\alpha_\text{post} + y_b(t - \Delta t) \beta_\text{post} \right) \bigl[ t = t^f_\text{post} \bigr] \\
            &+ y_a(t)\left(\alpha_\text{pre} + x_b(t - \Delta t) \beta_\text{pre} \right) \bigl[ t = t^f_\text{pre} \bigr]
        \end{align*}

    Args:
        lr_post_pair (float): learning rate for spike pair updates on postsynaptic
            spikes, :math:`\alpha_\text{post}`.
        lr_post_triplet (float): learning rate for spike triplet updates on postsynaptic
            spikes, :math:`\beta_\text{post}`.
        lr_pre_pair (float): learning rate for spike pair updates on presynaptic
            spikes, :math:`\alpha_\text{pre}`.
        lr_pre_triplet (float): learning rate for spike triplet updates on presynaptic
            spikes, :math:`\beta_\text{pre}`.
        tc_post_fast (float): time constant of exponential decay for postsynaptic trace
            of pairs (fast), :math:`\tau_-`, in :math:`ms`.
        tc_post_slow (float): time constant of exponential decay for postsynaptic trace
            of triplets (slow), :math:`\tau_y`, in :math:`ms`.
        tc_pre_fast (float): time constant of exponential decay for presynaptic trace
            of pairs (fast), :math:`\tau_+`, in :math:`ms`.
        tc_pre_slow (float): time constant of exponential decay for presynaptic trace
            of triplets (slow), :math:`\tau_x`, in :math:`ms`.
        delayed (bool, optional): if the updater should assume that learned delays, if
            present, may change. Defaults to ``False``.
        interp_tolerance (float, optional): maximum difference in time from an observation
            to treat as co-occurring, in :math:`\text{ms}`. Defaults to ``0.0``.
        trace_mode (Literal["cumulative", "nearest"], optional): method to use for
            calculating spike traces. Defaults to ``"cumulative"``.
        batch_reduction (Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] | None):
            function to reduce updates over the batch dimension, :py:func:`torch.mean`
            when ``None``. Defaults to ``None``.
        inplace (bool, optional): if :py:class:`~inferno.RecordTensor` write operations
            should be performed in-place. Defaults to ``False``.

    Important:
        When ``delayed`` is ``True``, the history for the presynaptic activity (spike
        traces and spike activity) is preserved in its un-delayed form and is then
        accessed using the connection's :py:attr:`~inferno.neural.Connection.selector`.

        When ``delayed`` is ``False``, only the most recent delay-adjusted presynaptic
        activity is preserved.

    Important:
        It is expected for this to be called after every trainable batch. Variables
        used are not stored (or are invalidated) if multiple batches are given before
        an update.

    Note:
        The constructor arguments are hyperparameters and can be overridden on a
        cell-by-cell basis.

    Note:
        The absolute values of ``lr_post_triplet`` and ``lr_pre_triplet`` are taken
        to enforce they are positive values.

    Note:
        ``batch_reduction`` can be one of the functions in PyTorch including but not
        limited to :py:func:`torch.sum`, :py:func:`torch.mean`, and :py:func:`torch.amax`.
        A custom function with similar behavior can also be passed in. Like with the
        included function, it should not keep the original dimensions by default.

    See Also:
        For more details and references, visit
        :ref:`zoo/learning-stdp:Triplet Spike-Timing Dependent Plasticity (Triplet STDP)` in the zoo.
    """

    def __init__(
        self,
        lr_post_pair: float,
        lr_post_triplet: float,
        lr_pre_pair: float,
        lr_pre_triplet: float,
        tc_post_fast: float,
        tc_post_slow: float,
        tc_pre_fast: float,
        tc_pre_slow: float,
        delayed: bool = False,
        interp_tolerance: float = 0.0,
        trace_mode: Literal["cumulative", "nearest"] = "cumulative",
        batch_reduction: (
            Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] | None
        ) = None,
        inplace: bool = False,
        **kwargs,
    ):
        # call superclass constructor
        IndependentCellTrainer.__init__(self, **kwargs)

        # default hyperparameters
        self.lr_post_pair = argtest.neq("lr_post_pair", lr_post_pair, 0, float)
        self.lr_post_triplet = abs(float(lr_post_triplet))
        self.lr_pre_pair = argtest.neq("lr_pre_pair", lr_pre_pair, 0, float)
        self.lr_pre_triplet = abs(float(lr_pre_triplet))
        self.tc_post_fast = argtest.gt("tc_post_fast", tc_post_fast, 0, float)
        self.tc_post_slow = argtest.gt(
            "tc_post_slow", tc_post_slow, tc_post_fast, float, "tc_post_fast"
        )
        self.tc_pre_fast = argtest.gt("tc_pre_fast", tc_pre_fast, 0, float)
        self.tc_pre_slow = argtest.gt(
            "tc_pre_slow", tc_pre_slow, tc_pre_fast, float, "tc_pre_fast"
        )
        self.delayed = bool(delayed)
        self.tolerance = argtest.gte("interp_tolerance", interp_tolerance, 0, float)
        self.trace = argtest.oneof(
            "trace_mode", trace_mode, "cumulative", "nearest", op=(lambda x: x.lower())
        )
        self.batchreduce = batch_reduction if batch_reduction else torch.mean
        self.inplace = bool(inplace)

    def _build_cell_state(self, **kwargs) -> Module:
        r"""Builds auxiliary state for a cell.

        Keyword arguments will override module-level hyperparameters.

        Returns:
            Module: state module.
        """
        state = Module()

        lr_post_pair = kwargs.get("lr_post_pair", self.lr_post_pair)
        lr_post_triplet = kwargs.get("lr_post_triplet", self.lr_post_triplet)
        lr_pre_pair = kwargs.get("lr_pre_pair", self.lr_pre_pair)
        lr_pre_triplet = kwargs.get("lr_pre_triplet", self.lr_pre_triplet)
        tc_post_fast = kwargs.get("tc_post_fast", self.tc_post_fast)
        tc_post_slow = kwargs.get("tc_post_slow", self.tc_post_slow)
        tc_pre_fast = kwargs.get("tc_pre_fast", self.tc_pre_fast)
        tc_pre_slow = kwargs.get("tc_pre_slow", self.tc_pre_slow)
        delayed = kwargs.get("delayed", self.delayed)
        interp_tolerance = kwargs.get("interp_tolerance", self.tolerance)
        trace_mode = kwargs.get("trace_mode", self.trace)
        batch_reduction = kwargs.get("batch_reduction", self.batchreduce)
        inplace = kwargs.get("inplace", self.inplace)

        state.lr_post_pair = argtest.neq("lr_post_pair", lr_post_pair, 0, float)
        state.lr_post_triplet = abs(float(lr_post_triplet))
        state.lr_pre_pair = argtest.neq("lr_pre_pair", lr_pre_pair, 0, float)
        state.lr_pre_triplet = abs(float(lr_pre_triplet))
        state.tc_post_fast = argtest.gt("tc_post_fast", tc_post_fast, 0, float)
        state.tc_post_slow = argtest.gt(
            "tc_post_slow", tc_post_slow, tc_post_fast, float, "tc_post_fast"
        )
        state.tc_pre_fast = argtest.gt("tc_pre_fast", tc_pre_fast, 0, float)
        state.tc_pre_slow = argtest.gt(
            "tc_pre_slow", tc_pre_slow, tc_pre_fast, float, "tc_pre_fast"
        )
        state.delayed = bool(delayed)
        state.tolerance = argtest.gte("interp_tolerance", interp_tolerance, 0, float)
        state.tracemode = argtest.oneof(
            "trace_mode", trace_mode, "cumulative", "nearest", op=(lambda x: x.lower())
        )
        match state.tracemode:
            case "cumulative":
                state.tracecls = CumulativeTraceReducer
            case "nearest":
                state.tracecls = NearestTraceReducer
            case "_":
                raise RuntimeError(
                    f"an invalid trace mode of '{state.tracemode}' has been set, "
                    "expected one of: 'cumulative', 'nearest'"
                )
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
            lr_post_pair (float): learning rate for spike pair updates on
                postsynaptic spikes.
            lr_post_triplet (float): learning rate for spike triplet updates on
                postsynaptic spikes.
            lr_pre_pair (float): learning rate for spike pair updates on
                presynaptic spikes.
            lr_pre_triplet (float): learning rate for spike triplet updates on
                presynaptic spikes.
            tc_post_fast (float): time constant of exponential decay for postsynaptic
                trace of pairs (fast).
            tc_post_slow (float): time constant of exponential decay for postsynaptic
                trace of triplets (slow).
            tc_pre_fast (float): time constant of exponential decay for presynaptic
                trace of pairs (fast).
            tc_pre_slow (float): time constant of exponential decay for presynaptic
                trace of triplets (slow).
            delayed (bool): if the updater should assume that learned delays,
                if present, may change.
            interp_tolerance (float): maximum difference in time from an observation
                to treat as co-occurring.
            trace_mode (Literal["cumulative", "nearest"]): method to use for
                calculating spike traces.
            batch_reduction (Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor]):
                function to reduce updates over the batch dimension.
            inplace (bool, optional): if :py:class:`~inferno.RecordTensor` write operations
                should be performed in-place. Defaults to ``False``.

        Returns:
            IndependentCellTrainer.Unit: specified cell, auxiliary state, and monitors.

        Important:
            Any specified keyword arguments will override the default hyperparameters
            set on initialization. See :py:class:`TripletSTDP` for details.
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

        # postsynaptic fast trace monitor (weighs hebbian LTD)
        self.add_monitor(
            name,
            "trace_post_fast",
            "neuron.spike",
            StateMonitor.partialconstructor(
                reducer=state.tracecls(
                    cell.connection.dt,
                    state.tc_post_fast,
                    amplitude=abs(state.lr_pre_pair),
                    target=True,
                    duration=cell.connection.dt,
                    inclusive=True,
                    inplace=state.inplace,
                ),
                **monitor_kwargs,
            ),
            False,
            dt=cell.connection.dt,
            amp=abs(state.lr_pre_pair),
            tc=state.tc_post_fast,
            trace=state.tracemode,
            timing="fast",
        )

        # postsynaptic slow trace monitor (weighs hebbian LTP)
        self.add_monitor(
            name,
            "trace_post_slow",
            "neuron.spike",
            StateMonitor.partialconstructor(
                reducer=state.tracecls(
                    cell.connection.dt,
                    state.tc_post_slow,
                    amplitude=abs(state.lr_post_triplet / state.lr_post_pair),
                    target=True,
                    duration=2 * cell.connection.dt,
                    inclusive=True,
                    inplace=state.inplace,
                ),
                **monitor_kwargs,
            ),
            False,
            dt=cell.connection.dt,
            amp=abs(state.lr_post_triplet / state.lr_post_pair),
            tc=state.tc_post_slow,
            trace=state.tracemode,
            timing="slow",
        )

        # postsynaptic spike monitor (triggers hebbian LTP)
        self.add_monitor(
            name,
            "spike_post",
            "neuron.spike",
            StateMonitor.partialconstructor(
                reducer=PassthroughReducer(
                    cell.connection.dt,
                    duration=0.0,
                    inclusive=True,
                    inplace=state.inplace,
                ),
                **monitor_kwargs,
            ),
            False,
            dt=cell.connection.dt,
        )

        # presynaptic fast trace monitor (weighs hebbian LTP)
        # when the delayed condition is true, using synapse.spike records the raw
        # spike times rather than the delay adjusted times of synspike.
        self.add_monitor(
            name,
            "trace_pre_fast",
            "synapse.spike" if delayed else "connection.synspike",
            StateMonitor.partialconstructor(
                reducer=state.tracecls(
                    cell.connection.dt,
                    state.tc_pre_fast,
                    amplitude=abs(state.lr_post_pair),
                    target=True,
                    duration=(
                        cell.connection.delayedby if delayed else cell.connection.dt
                    ),
                    inclusive=True,
                    inplace=state.inplace,
                ),
                **monitor_kwargs,
            ),
            False,
            dt=cell.connection.dt,
            amp=abs(state.lr_post_pair),
            tc=state.tc_pre_fast,
            trace=state.tracemode,
            delayed=delayed,
            timing="fast",
        )

        # presynaptic slow trace monitor (weighs hebbian LTD)
        # when the delayed condition is true, using synapse.spike records the raw
        # spike times rather than the delay adjusted times of synspike.
        self.add_monitor(
            name,
            "trace_pre_slow",
            "synapse.spike" if delayed else "connection.synspike",
            StateMonitor.partialconstructor(
                reducer=state.tracecls(
                    cell.connection.dt,
                    state.tc_pre_slow,
                    amplitude=abs(state.lr_pre_triplet / state.lr_pre_pair),
                    target=True,
                    duration=(
                        cell.connection.delayedby + cell.connection.dt
                        if delayed
                        else 2 * cell.connection.dt
                    ),
                    inclusive=True,
                    inplace=state.inplace,
                ),
                **monitor_kwargs,
            ),
            False,
            dt=cell.connection.dt,
            amp=abs(state.lr_pre_triplet / state.lr_pre_pair),
            tc=state.tc_pre_slow,
            trace=state.tracemode,
            delayed=delayed,
            timing="slow",
        )

        # presynaptic spike monitor (triggers hebbian LTD)
        # when the delayed condition is true, using synapse.spike records the raw
        # spike times rather than the delay adjusted times of synspike.
        self.add_monitor(
            name,
            "spike_pre",
            "synapse.spike" if delayed else "connection.synspike",
            StateMonitor.partialconstructor(
                reducer=PassthroughReducer(
                    cell.connection.dt,
                    duration=(cell.connection.delayedby if delayed else 0.0),
                    inclusive=True,
                    inplace=state.inplace,
                ),
                **monitor_kwargs,
            ),
            False,
            dt=cell.connection.dt,
            delayed=delayed,
        )

        return self.get_unit(name)

    def forward(self) -> None:
        r"""Processes update for given layers based on current monitor stored data."""
        # iterate through self
        for cell, state, monitors in self:
            # skip if self or cell is not in training mode or has no updater
            if not cell.training or not self.training or not cell.updater:
                continue

            # fast spike traces, reshaped into receptive format
            y_a = cell.connection.postsyn_receptive(monitors["trace_post_fast"].peek())
            x_a = cell.connection.presyn_receptive(
                monitors["trace_pre_fast"].view(
                    cell.connection.selector, state.tolerance
                )
                if state.delayed and cell.connection.delayedby
                else monitors["trace_pre_fast"].peek()
            )

            # slow spike traces, reshaped into receptive format
            y_b = monitors["trace_post_slow"].reducer.data_.read(2)
            x_b = (
                monitors["trace_pre_slow"].reducer.data_.select(
                    cell.connection.selector,
                    monitors["trace_post_slow"].interpolate,
                    tolerance=state.tolerance,
                    offset=2,
                )
                if state.delayed and cell.connection.delayedby
                else monitors["trace_pre_slow"].reducer.data_.read(2)
            )

            # spike presence, reshaped into receptive format
            y = monitors["spike_post"].peek()
            x = (
                monitors["spike_pre"].view(cell.connection.selector, state.tolerance)
                if state.delayed and cell.connection.delayedby
                else monitors["spike_pre"].peek()
            )

            # combine like terms
            y = cell.connection.postsyn_receptive((1.0 + y_b) * y)
            x = cell.connection.presyn_receptive((1.0 + x_b) * x)

            # partial updates
            dpost = state.batchreduce(
                ein.einsum(y, x_a, "b ... r, b ... r -> b ..."), 0
            )
            dpre = state.batchreduce(ein.einsum(x, y_a, "b ... r, b ... r -> b ..."), 0)

            # accumulate partials with mode condition
            match (state.lr_post_pair >= 0, state.lr_pre_pair >= 0):
                case (False, False):  # depressive
                    cell.updater.weight = (None, dpost + dpre)
                case (False, True):  # anti-hebbian
                    cell.updater.weight = (dpre, dpost)
                case (True, False):  # hebbian
                    cell.updater.weight = (dpost, dpre)
                case (True, True):  # potentiative
                    cell.updater.weight = (dpost + dpre, None)


class StableTripletSTDP(IndependentCellTrainer):
    r"""Triplet-based spike-timing dependent plasticity trainer.

    Rather than recording trace values with amplitudes specified by the learning rates,
    this uses an amplitude of 1. With some testing the difference appears to be minor.
    With limited testing, the maximum difference between this and the "unstable"
    implementation is around 3e-6 times the average weight. Not included with the
    default exports.

    .. math::
        \begin{align*}
            w(t + \Delta t) - w(t)
            &= x_a(t)\left(\alpha_\text{post} + y_b(t - \Delta t) \beta_\text{post} \right) \bigl[ t = t^f_\text{post} \bigr] \\
            &+ y_a(t)\left(\alpha_\text{pre} + x_b(t - \Delta t) \beta_\text{pre} \right) \bigl[ t = t^f_\text{pre} \bigr]
        \end{align*}

    When ``trace_mode = "cumulative"``:

    .. math::
        \begin{align*}
            x_a(t) &= x_a(t - \Delta t) \exp \left(-\frac{\Delta t}{\tau_+}\right)
            + \bigl[t = t^f_\text{pre}\bigr] \\
            x_b(t) &= x_b(t - \Delta t) \exp \left(-\frac{\Delta t}{\tau_x}\right)
            + \bigl[t = t^f_\text{pre}\bigr] \\
            y_a(t) &= y_a(t - \Delta t) \exp \left(-\frac{\Delta t}{\tau_-}\right)
            + \bigl[t = t^f_\text{post}\bigr] \\
            y_b(t) &= y_b(t - \Delta t) \exp \left(-\frac{\Delta t}{\tau_y}\right)
            + \bigl[t = t^f_\text{post}\bigr]
        \end{align*}

    When ``trace_mode = "nearest"``:

    .. math::
        \begin{align*}
            x_\text{a}(t) &=
            \begin{cases}
                1 & t = t_\text{pre}^f \\
                x_\text{a}(t - \Delta t)
                \exp\left(-\frac{\Delta t}{\tau_+}\right)
                & t \neq t_\text{pre}^f
            \end{cases} \\
            x_\text{b}(t) &=
            \begin{cases}
                1 & t = t_\text{pre}^f \\
                x_\text{b}(t - \Delta t)
                \exp\left(-\frac{\Delta t}{\tau_x}\right)
                & t \neq t_\text{pre}^f
            \end{cases} \\
            y_\text{a}(t) &=
            \begin{cases}
                1 & t = t_\text{post}^f \\
                y_\text{a}(t - \Delta t)
                \exp\left(-\frac{\Delta t}{\tau_-}\right)
                & t \neq t_\text{post}^f
            \end{cases} \\
            y_\text{b}(t) &=
            \begin{cases}
                1 / \alpha_\text{post} & t = t_\text{post}^f \\
                y_\text{b}(t - \Delta t)
                \exp\left(-\frac{\Delta t}{\tau_y}\right)
                & t \neq t_\text{post}^f
            \end{cases}
        \end{align*}

    Where:

    Times :math:`t` and :math:`t_n^f` are the current time and the time of the most
    recent spike from neuron :math:`n`, respectively, and :math:`\Delta t` is the
    duration of the simulation step. The following constraints are enforced.

    .. math::
        \begin{align*}
            0 &< \tau_+ < \tau_x \\
            0 &< \tau_- < \tau_y \\
        \end{align*}

    The signs of the learning rates :math:`\alpha_\text{post}`and :math:`\alpha_\text{pre}`
    control which terms are potentiative and which terms are depressive. The terms can
    be scaled for weight dependence on updating.

    +-------------------+----------------------------------------+---------------------------------------+------------------------------------------------------------------------------------+------------------------------------------------------------------------------------+
    | Mode              | :math:`\text{sgn}(\alpha_\text{post})` | :math:`\text{sgn}(\alpha_\text{pre})` | LTP Term(s)                                                                        | LTD Term(s)                                                                        |
    +===================+========================================+=======================================+====================================================================================+====================================================================================+
    | Hebbian           | :math:`+`                              | :math:`-`                             | :math:`\alpha_\text{post}, \beta_\text{post}`                                      | :math:`\alpha_\text{pre}, \beta_\text{pre}`                                        |
    +-------------------+----------------------------------------+---------------------------------------+------------------------------------------------------------------------------------+------------------------------------------------------------------------------------+
    | Anti-Hebbian      | :math:`-`                              | :math:`+`                             | :math:`\alpha_\text{pre}, \beta_\text{pre}`                                        | :math:`\alpha_\text{post}, \beta_\text{post}`                                      |
    +-------------------+----------------------------------------+---------------------------------------+------------------------------------------------------------------------------------+------------------------------------------------------------------------------------+
    | Potentiative Only | :math:`+`                              | :math:`+`                             | :math:`\alpha_\text{post}, \alpha_\text{pre}, \beta_\text{post}, \beta_\text{pre}` | None                                                                               |
    +-------------------+----------------------------------------+---------------------------------------+------------------------------------------------------------------------------------+------------------------------------------------------------------------------------+
    | Depressive Only   | :math:`-`                              | :math:`-`                             | None                                                                               | :math:`\alpha_\text{post}, \alpha_\text{pre}, \beta_\text{post}, \beta_\text{pre}` |
    +-------------------+----------------------------------------+---------------------------------------+------------------------------------------------------------------------------------+------------------------------------------------------------------------------------+

    Args:
        lr_post_pair (float): learning rate for spike pair updates on postsynaptic
            spikes, :math:`\alpha_\text{post}`.
        lr_post_triplet (float): learning rate for spike triplet updates on postsynaptic
            spikes, :math:`\beta_\text{post}`.
        lr_pre_pair (float): learning rate for spike pair updates on presynaptic
            spikes, :math:`\alpha_\text{pre}`.
        lr_pre_triplet (float): learning rate for spike triplet updates on presynaptic
            spikes, :math:`\beta_\text{pre}`.
        tc_post_fast (float): time constant of exponential decay for postsynaptic trace
            of pairs (fast), :math:`\tau_-`, in :math:`ms`.
        tc_post_slow (float): time constant of exponential decay for postsynaptic trace
            of triplets (slow), :math:`\tau_y`, in :math:`ms`.
        tc_pre_fast (float): time constant of exponential decay for presynaptic trace
            of pairs (fast), :math:`\tau_+`, in :math:`ms`.
        tc_pre_slow (float): time constant of exponential decay for presynaptic trace
            of triplets (slow), :math:`\tau_x`, in :math:`ms`.
        delayed (bool, optional): if the updater should assume that learned delays, if
            present, may change. Defaults to ``False``.
        interp_tolerance (float, optional): maximum difference in time from an observation
            to treat as co-occurring, in :math:`\text{ms}`. Defaults to ``0.0``.
        trace_mode (Literal["cumulative", "nearest"], optional): method to use for
            calculating spike traces. Defaults to ``"cumulative"``.
        batch_reduction (Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] | None):
            function to reduce updates over the batch dimension, :py:func:`torch.mean`
            when ``None``. Defaults to ``None``.
        inplace (bool, optional): if :py:class:`~inferno.RecordTensor` write operations
            should be performed in-place. Defaults to ``False``.

    Important:
        When ``delayed`` is ``True``, the history for the presynaptic activity (spike
        traces and spike activity) is preserved in its un-delayed form and is then
        accessed using the connection's :py:attr:`~inferno.neural.Connection.selector`.

        When ``delayed`` is ``False``, only the most recent delay-adjusted presynaptic
        activity is preserved.

    Important:
        It is expected for this to be called after every trainable batch. Variables
        used are not stored (or are invalidated) if multiple batches are given before
        an update.

    Note:
        The constructor arguments are hyperparameters and can be overridden on a
        cell-by-cell basis.

    Note:
        The absolute values of ``lr_post_triplet`` and ``lr_pre_triplet`` are taken
        to enforce they are positive values.

    Note:
        ``batch_reduction`` can be one of the functions in PyTorch including but not
        limited to :py:func:`torch.sum`, :py:func:`torch.mean`, and :py:func:`torch.amax`.
        A custom function with similar behavior can also be passed in. Like with the
        included function, it should not keep the original dimensions by default.

    See Also:
        For more details and references, visit
        :ref:`zoo/learning-stdp:Triplet Spike-Timing Dependent Plasticity (Triplet STDP)` in the zoo.
    """

    def __init__(
        self,
        lr_post_pair: float,
        lr_post_triplet: float,
        lr_pre_pair: float,
        lr_pre_triplet: float,
        tc_post_fast: float,
        tc_post_slow: float,
        tc_pre_fast: float,
        tc_pre_slow: float,
        delayed: bool = False,
        interp_tolerance: float = 0.0,
        trace_mode: Literal["cumulative", "nearest"] = "cumulative",
        batch_reduction: (
            Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] | None
        ) = None,
        inplace: bool = False,
        **kwargs,
    ):
        # call superclass constructor
        IndependentCellTrainer.__init__(self, **kwargs)

        # default hyperparameters
        self.lr_post_pair = argtest.neq("lr_post_pair", lr_post_pair, 0, float)
        self.lr_post_triplet = abs(float(lr_post_triplet))
        self.lr_pre_pair = argtest.neq("lr_pre_pair", lr_pre_pair, 0, float)
        self.lr_pre_triplet = abs(float(lr_pre_triplet))
        self.tc_post_fast = argtest.gt("tc_post_fast", tc_post_fast, 0, float)
        self.tc_post_slow = argtest.gt(
            "tc_post_slow", tc_post_slow, tc_post_fast, float, "tc_post_fast"
        )
        self.tc_pre_fast = argtest.gt("tc_pre_fast", tc_pre_fast, 0, float)
        self.tc_pre_slow = argtest.gt(
            "tc_pre_slow", tc_pre_slow, tc_pre_fast, float, "tc_pre_fast"
        )
        self.delayed = bool(delayed)
        self.tolerance = argtest.gte("interp_tolerance", interp_tolerance, 0, float)
        self.trace = argtest.oneof(
            "trace_mode", trace_mode, "cumulative", "nearest", op=(lambda x: x.lower())
        )
        self.batchreduce = batch_reduction if batch_reduction else torch.mean
        self.inplace = bool(inplace)

    def _build_cell_state(self, **kwargs) -> Module:
        r"""Builds auxiliary state for a cell.

        Keyword arguments will override module-level hyperparameters.

        Returns:
            Module: state module.
        """
        state = Module()

        lr_post_pair = kwargs.get("lr_post_pair", self.lr_post_pair)
        lr_post_triplet = kwargs.get("lr_post_triplet", self.lr_post_triplet)
        lr_pre_pair = kwargs.get("lr_pre_pair", self.lr_pre_pair)
        lr_pre_triplet = kwargs.get("lr_pre_triplet", self.lr_pre_triplet)
        tc_post_fast = kwargs.get("tc_post_fast", self.tc_post_fast)
        tc_post_slow = kwargs.get("tc_post_slow", self.tc_post_slow)
        tc_pre_fast = kwargs.get("tc_pre_fast", self.tc_pre_fast)
        tc_pre_slow = kwargs.get("tc_pre_slow", self.tc_pre_slow)
        delayed = kwargs.get("delayed", self.delayed)
        interp_tolerance = kwargs.get("interp_tolerance", self.tolerance)
        trace_mode = kwargs.get("trace_mode", self.trace)
        batch_reduction = kwargs.get("batch_reduction", self.batchreduce)
        inplace = kwargs.get("inplace", self.inplace)

        state.lr_post_pair = argtest.neq("lr_post_pair", lr_post_pair, 0, float)
        state.lr_post_triplet = abs(float(lr_post_triplet))
        state.lr_pre_pair = argtest.neq("lr_pre_pair", lr_pre_pair, 0, float)
        state.lr_pre_triplet = abs(float(lr_pre_triplet))
        state.tc_post_fast = argtest.gt("tc_post_fast", tc_post_fast, 0, float)
        state.tc_post_slow = argtest.gt(
            "tc_post_slow", tc_post_slow, tc_post_fast, float, "tc_post_fast"
        )
        state.tc_pre_fast = argtest.gt("tc_pre_fast", tc_pre_fast, 0, float)
        state.tc_pre_slow = argtest.gt(
            "tc_pre_slow", tc_pre_slow, tc_pre_fast, float, "tc_pre_fast"
        )
        state.delayed = bool(delayed)
        state.tolerance = argtest.gte("interp_tolerance", interp_tolerance, 0, float)
        state.tracemode = argtest.oneof(
            "trace_mode", trace_mode, "cumulative", "nearest", op=(lambda x: x.lower())
        )
        match state.tracemode:
            case "cumulative":
                state.tracecls = CumulativeTraceReducer
            case "nearest":
                state.tracecls = NearestTraceReducer
            case "_":
                raise RuntimeError(
                    f"an invalid trace mode of '{state.tracemode}' has been set, "
                    "expected one of: 'cumulative', 'nearest'"
                )
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
            lr_post_pair (float): learning rate for spike pair updates on
                postsynaptic spikes.
            lr_post_triplet (float): learning rate for spike triplet updates on
                postsynaptic spikes.
            lr_pre_pair (float): learning rate for spike pair updates on
                presynaptic spikes.
            lr_pre_triplet (float): learning rate for spike triplet updates on
                presynaptic spikes.
            tc_post_fast (float): time constant of exponential decay for postsynaptic
                trace of pairs (fast).
            tc_post_slow (float): time constant of exponential decay for postsynaptic
                trace of triplets (slow).
            tc_pre_fast (float): time constant of exponential decay for presynaptic
                trace of pairs (fast).
            tc_pre_slow (float): time constant of exponential decay for presynaptic
                trace of triplets (slow).
            delayed (bool): if the updater should assume that learned delays,
                if present, may change.
            interp_tolerance (float): maximum difference in time from an observation
                to treat as co-occurring.
            trace_mode (Literal["cumulative", "nearest"]): method to use for
                calculating spike traces.
            batch_reduction (Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor]):
                function to reduce updates over the batch dimension.
            inplace (bool, optional): if :py:class:`~inferno.RecordTensor` write operations
                should be performed in-place. Defaults to ``False``.

        Returns:
            IndependentCellTrainer.Unit: specified cell, auxiliary state, and monitors.

        Important:
            Any specified keyword arguments will override the default hyperparameters
            set on initialization. See :py:class:`TripletSTDP` for details.
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

        # postsynaptic fast trace monitor (weighs hebbian LTD)
        self.add_monitor(
            name,
            "trace_post_fast",
            "neuron.spike",
            StateMonitor.partialconstructor(
                reducer=state.tracecls(
                    cell.connection.dt,
                    state.tc_post_fast,
                    amplitude=1.0,
                    target=True,
                    duration=cell.connection.dt,
                    inclusive=True,
                    inplace=state.inplace,
                ),
                **monitor_kwargs,
            ),
            False,
            dt=cell.connection.dt,
            tc=state.tc_post_fast,
            trace=state.tracemode,
            timing="fast",
        )

        # postsynaptic slow trace monitor (weighs hebbian LTP)
        self.add_monitor(
            name,
            "trace_post_slow",
            "neuron.spike",
            StateMonitor.partialconstructor(
                reducer=state.tracecls(
                    cell.connection.dt,
                    state.tc_post_slow,
                    amplitude=1.0,
                    target=True,
                    duration=2 * cell.connection.dt,
                    inclusive=True,
                    inplace=state.inplace,
                ),
                **monitor_kwargs,
            ),
            False,
            dt=cell.connection.dt,
            tc=state.tc_post_slow,
            trace=state.tracemode,
            timing="slow",
        )

        # postsynaptic spike monitor (triggers hebbian LTP)
        self.add_monitor(
            name,
            "spike_post",
            "neuron.spike",
            StateMonitor.partialconstructor(
                reducer=PassthroughReducer(
                    cell.connection.dt,
                    duration=0.0,
                    inclusive=True,
                    inplace=state.inplace,
                ),
                **monitor_kwargs,
            ),
            False,
            dt=cell.connection.dt,
        )

        # presynaptic fast trace monitor (weighs hebbian LTP)
        # when the delayed condition is true, using synapse.spike records the raw
        # spike times rather than the delay adjusted times of synspike.
        self.add_monitor(
            name,
            "trace_pre_fast",
            "synapse.spike" if delayed else "connection.synspike",
            StateMonitor.partialconstructor(
                reducer=state.tracecls(
                    cell.connection.dt,
                    state.tc_pre_fast,
                    amplitude=1.0,
                    target=True,
                    duration=(
                        cell.connection.delayedby if delayed else cell.connection.dt
                    ),
                    inclusive=True,
                    inplace=state.inplace,
                ),
                **monitor_kwargs,
            ),
            False,
            dt=cell.connection.dt,
            tc=state.tc_pre_fast,
            trace=state.tracemode,
            delayed=delayed,
            timing="fast",
        )

        # presynaptic slow trace monitor (weighs hebbian LTD)
        # when the delayed condition is true, using synapse.spike records the raw
        # spike times rather than the delay adjusted times of synspike.
        self.add_monitor(
            name,
            "trace_pre_slow",
            "synapse.spike" if delayed else "connection.synspike",
            StateMonitor.partialconstructor(
                reducer=state.tracecls(
                    cell.connection.dt,
                    state.tc_pre_slow,
                    amplitude=1.0,
                    target=True,
                    duration=(
                        cell.connection.delayedby + cell.connection.dt
                        if delayed
                        else 2 * cell.connection.dt
                    ),
                    inclusive=True,
                    inplace=state.inplace,
                ),
                **monitor_kwargs,
            ),
            False,
            dt=cell.connection.dt,
            tc=state.tc_pre_slow,
            trace=state.tracemode,
            delayed=delayed,
            timing="slow",
        )

        # presynaptic spike monitor (triggers hebbian LTD)
        # when the delayed condition is true, using synapse.spike records the raw
        # spike times rather than the delay adjusted times of synspike.
        self.add_monitor(
            name,
            "spike_pre",
            "synapse.spike" if delayed else "connection.synspike",
            StateMonitor.partialconstructor(
                reducer=PassthroughReducer(
                    cell.connection.dt,
                    duration=(cell.connection.delayedby if delayed else 0.0),
                    inclusive=True,
                    inplace=state.inplace,
                ),
                **monitor_kwargs,
            ),
            False,
            dt=cell.connection.dt,
            delayed=delayed,
        )

        return self.get_unit(name)

    def forward(self) -> None:
        r"""Processes update for given layers based on current monitor stored data."""
        # iterate through self
        for cell, state, monitors in self:
            # skip if self or cell is not in training mode or has no updater
            if not cell.training or not self.training or not cell.updater:
                continue

            # fast spike traces, reshaped into receptive format
            y_a = cell.connection.postsyn_receptive(monitors["trace_post_fast"].peek())
            x_a = cell.connection.presyn_receptive(
                monitors["trace_pre_fast"].view(
                    cell.connection.selector, state.tolerance
                )
                if state.delayed and cell.connection.delayedby
                else monitors["trace_pre_fast"].peek()
            )

            # slow spike traces, reshaped into receptive format
            y_b = monitors["trace_post_slow"].reducer.data_.read(2)
            x_b = (
                monitors["trace_pre_slow"].reducer.data_.select(
                    cell.connection.selector,
                    monitors["trace_post_slow"].interpolate,
                    tolerance=state.tolerance,
                    offset=2,
                )
                if state.delayed and cell.connection.delayedby
                else monitors["trace_pre_slow"].reducer.data_.read(2)
            )

            # spike presence, reshaped into receptive format
            y = monitors["spike_post"].peek()
            x = (
                monitors["spike_pre"].view(cell.connection.selector, state.tolerance)
                if state.delayed and cell.connection.delayedby
                else monitors["spike_pre"].peek()
            )

            # combine like terms
            y = cell.connection.postsyn_receptive(
                (abs(state.lr_post_pair) + state.lr_post_triplet * y_b) * y
            )
            x = cell.connection.presyn_receptive(
                (abs(state.lr_pre_pair) + state.lr_pre_triplet * x_b) * x
            )

            # partial updates
            dpost = state.batchreduce(
                ein.einsum(y, x_a, "b ... r, b ... r -> b ..."), 0
            )
            dpre = state.batchreduce(ein.einsum(x, y_a, "b ... r, b ... r -> b ..."), 0)

            # accumulate partials with mode condition
            match (state.lr_post_pair >= 0, state.lr_pre_pair >= 0):
                case (False, False):  # depressive
                    cell.updater.weight = (None, dpost + dpre)
                case (False, True):  # anti-hebbian
                    cell.updater.weight = (dpre, dpost)
                case (True, False):  # hebbian
                    cell.updater.weight = (dpost, dpre)
                case (True, True):  # potentiative
                    cell.updater.weight = (dpost + dpre, None)
