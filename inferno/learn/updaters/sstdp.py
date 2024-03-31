from __future__ import annotations
from .. import IndependentTrainer
from ... import Module, scalar
from ..._internal import argtest
from ...functional import interp_expdecay
from ...neural import Cell
from ...observe import (
    StateMonitor,
    MultiStateMonitor,
    CumulativeTraceReducer,
    PassthroughReducer,
    FoldingReducer,
)
from functools import partial
import torch
from typing import Any, Callable
import weakref


class EligibilityTrace(FoldingReducer):
    r"""Simple eligibility trace reducer.

    .. math::
        z(t + \Delta t) = z(t) \exp \left( -\frac{\Delta t}{\tau_z} \right)
        + \frac{\zeta(t)}{\tau_z}

    Args:
        step_time (float): length of the discrete step time, :math:`\Delta t`.
        time_constant (float): time constant of exponential decay, :math:`\tau_z`.
        duration (float): length of time over which results should be stored.
        trace_reshape (weakref.WeakMethod): method to reshape trace spikes to
            presynaptic or postsynaptic receptive fields.
        spike_reshape (weakref.WeakMethod): method to reshape condition spikes to
            presynaptic or postsynaptic receptive fields.
        field_reduction (Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor]):
            function to reduce eligibility over the receptive field dimension.
    """

    def __init__(
        self,
        step_time: float,
        time_constant: float,
        *,
        duration: float,
    ):
        # call superclass constructor
        FoldingReducer.__init__(self, step_time, duration)

        # register hyperparameters
        self.register_buffer(
            "time_constant",
            torch.tensor(argtest.gt("time_constant", time_constant, 0, float)),
            persistent=False,
        )
        self.register_buffer(
            "decay",
            torch.exp(-self.dt / self.time_constant),
            persistent=False,
        )

    @property
    def dt(self) -> float:
        return FoldingReducer.dt.fget(self)

    @dt.setter
    def dt(self, value: float):
        FoldingReducer.dt.fset(self, value)
        self.decay = scalar(torch.exp(-self.dt / self.time_constant), self.decay)

    def fold(
        self, obs: torch.Tensor, spike: torch.Tensor, state: torch.Tensor | None
    ) -> torch.Tensor:
        if state is None:
            return obs / self.time_constant
        else:
            return (self.decay * state) + (obs / self.time_constant)

    def initialize(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.fill_(0)

    def interpolate(
        self,
        prev_data: torch.Tensor,
        next_data: torch.Tensor,
        sample_at: torch.Tensor,
        step_time: float | torch.Tensor,
    ) -> torch.Tensor:
        return interp_expdecay(
            prev_data, next_data, sample_at, step_time, self.time_constant
        )


class MSTDPET(IndependentTrainer):
    r"""Modulated spike-timing dependent plasticity with eligibility trace updater.

    .. math::
        w(t + \Delta t) - w(t) = \gamma  r(t + \Delta t)
        [z_\text{post}(t + \Delta t) + z_\text{pre}(t + \Delta t)]
        \Delta t

    Where:

    .. math::
        \begin{align*}
            z_\text{post}(t + \Delta t) &= z_\text{post}(t) \exp\left(-\frac{\Delta t}{\tau_z}\right)
            + \frac{x_\text{pre}(t)}{\tau_z}\left[t = t_\text{post}^f\right] \\
            z_\text{pre}(t + \Delta t) &= z_\text{pre}(t) \exp\left(-\frac{\Delta t}{\tau_z}\right)
            + \frac{x_\text{post}(t)}{\tau_z}\left[t = t_\text{pre}^f\right] \\
            x_\text{pre}(t) &= x_\text{pre}(t - \Delta t) \exp \left(-\frac{\Delta t}{\tau_\text{pre}}\right)
            + \eta_\text{pre}\left[t = t_\text{pre}^f\right] \\
            x_\text{post}(t) &= x_\text{post}(t - \Delta t) \exp \left(-\frac{\Delta t}{\tau_\text{post}}\right)
            + \eta_\text{post}\left[t = t_\text{post}^f\right]
        \end{align*}

    Times :math:`t` and :math:`t_n^f` are the current time and the time of the most recent
    spike from neuron :math:`n`, respectively.

    The signs of the learning rates :math:`\eta_\text{post}` and :math:`\eta_\text{pre}`
    controls which terms are potentiative and which terms are depressive. The terms
    (when expanded) can be scaled for weight dependence on updating. :math:`r` is a
    reinforcement term given on each update. Note that this implementation splits the
    eligibility trace into two terms, so weight dependence can scale the magnitude of each.

    +-------------------+--------------------------------------+-------------------------------------+-------------------------------------------+-------------------------------------------+
    | Mode              | :math:`\text{sgn}(\eta_\text{post})` | :math:`\text{sgn}(\eta_\text{pre})` | LTP Term(s)                               | LTD Term(s)                               |
    +===================+======================================+=====================================+===========================================+===========================================+
    | Hebbian           | :math:`+`                            | :math:`-`                           | :math:`\eta_\text{post}`                  | :math:`\eta_\text{pre}`                   |
    +-------------------+--------------------------------------+-------------------------------------+-------------------------------------------+-------------------------------------------+
    | Anti-Hebbian      | :math:`-`                            | :math:`+`                           | :math:`\eta_\text{pre}`                   | :math:`\eta_\text{post}`                  |
    +-------------------+--------------------------------------+-------------------------------------+-------------------------------------------+-------------------------------------------+
    | Depressive Only   | :math:`-`                            | :math:`-`                           | :math:`\eta_\text{post}, \eta_\text{pre}` | None                                      |
    +-------------------+--------------------------------------+-------------------------------------+-------------------------------------------+-------------------------------------------+
    | Potentiative Only | :math:`+`                            | :math:`+`                           | None                                      | :math:`\eta_\text{post}, \eta_\text{pre}` |
    +-------------------+--------------------------------------+-------------------------------------+-------------------------------------------+-------------------------------------------+

    Args:
        step_time (float): length of a simulation time step, :math:`\Delta t`,
            in :math:`\text{ms}`.
        lr_post (float): learning rate for updates on postsynaptic spike updates,
            :math:`\eta_\text{post}`.
        lr_pre (float): learning rate for updates on presynaptic spike updates,
            :math:`\eta_\text{pre}`.
        tc_post (float): time constant for exponential decay of postsynaptic trace,
            :math:`tau_\text{post}`, in :math:`ms`.
        tc_pre (float): time constant for exponential decay of presynaptic trace,
            :math:`tau_\text{pre}`, in :math:`ms`.
        tc_eligibility (float): time constant for exponential decay of eligibility trace,
            :math:`tau_z`, in :math:`ms`.
        scale (float, optional): scaling term for both the postsynaptic and presynaptic
            updates, :math:`\gamma`. Defaults to 1.0.
        delayed (bool, optional): if the updater should assume that learned delays, if
            present, may change. Defaults to False.
        interp_tolerance (float): maximum difference in time from an observation
            to treat as co-occurring, in :math:`\text{ms}`. Defaults to 0.0.
        trace_mode (Literal["cumulative", "nearest"], optional): method to use for
            calculating spike traces. Defaults to "cumulative".
        batch_reduction (Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] | None):
            function to reduce updates over the batch dimension, :py:func:`torch.mean`
            when None. Defaults to None.
        field_reduction (Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] | None):
            function to reduce updates over the receptive field dimension,
            :py:func:`torch.sum` when None. Defaults to None.

    Important:
        When ``delayed`` is ``True``, the history for the required variables is stored
        over the length of time the delay may be, and the selection is performed using
        the learned delays. When ``delayed`` is ``False``, the last state is used even
        if a change in delay occurs. This may be the desired behavior even if delays are
        updated along with weights.

    Important:
        It is expected for this to be called after every trainable batch. Variables
        used are not stored (or are invalidated) if multiple batches are given before
        an update.

    Note:
        The constructor arguments are hyperparameters and can be overridden on a
        cell-by-cell basis.

    Note:
        ``batch_reduction`` can be one of the functions in PyTorch including but not
        limited to :py:func:`torch.sum`, :py:func:`torch.max` and :py:func:`torch.max`.
        A custom function with similar behavior can also be passed in. Like with the
        included function, it should not keep the original dimensions by default.


    See Also:
        For more details and references, visit
        :ref:`zoo/learning-stdp:Modulated Spike-Timing Dependent Plasticity with Eligibility Trace (MSTDPET)`
        in the zoo.
    """

    def __init__(
        self,
        step_time: float,
        lr_post: float,
        lr_pre: float,
        tc_post: float,
        tc_pre: float,
        tc_eligibility: float,
        scale: float = 1.0,
        delayed: bool = False,
        interp_tolerance: float = 0.0,
        batch_reduction: (
            Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] | None
        ) = None,
        field_reduction: (
            Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] | None
        ) = None,
        **kwargs,
    ):
        # call superclass constructor
        IndependentTrainer.__init__(self, **kwargs)

        # default hyperparameters
        self.step_time = argtest.gt("step_time", step_time, 0, float)
        self.lr_post = float(lr_post)
        self.lr_pre = float(lr_pre)
        self.tc_post = argtest.gt("tc_post", tc_post, 0, float)
        self.tc_pre = argtest.gt("tc_pre", tc_pre, 0, float)
        self.tc_eligibility = argtest.gt("tc_eligibility", tc_eligibility, 0, float)
        self.scale = argtest.gt("scale", scale, 0, float)
        self.delayed = bool(delayed)
        self.tolerance = argtest.gte("interp_tolerance", interp_tolerance, 0, float)
        self.batchreduce = batch_reduction if batch_reduction else torch.mean
        self.fieldreduce = field_reduction if field_reduction else torch.sum

    def _build_cell_state(self, **kwargs) -> Module:
        r"""Builds auxiliary state for a cell.

        Keyword arguments will override module-level hyperparameters.

        Returns:
            Module: state module.
        """
        state = Module()

        step_time = kwargs.get("step_time", self.step_time)
        lr_post = kwargs.get("lr_post", self.lr_post)
        lr_pre = kwargs.get("lr_pre", self.lr_pre)
        tc_post = kwargs.get("tc_post", self.tc_post)
        tc_pre = kwargs.get("tc_pre", self.tc_pre)
        tc_eligibility = kwargs.get("tc_eligibility", self.tc_eligibility)
        scale = kwargs.get("scale", self.scale)
        delayed = kwargs.get("delayed", self.delayed)
        interp_tolerance = kwargs.get("interp_tolerance", self.tolerance)
        trace_mode = kwargs.get("trace_mode", self.trace)
        batch_reduction = kwargs.get("batch_reduction", self.batchreduce)
        field_reduction = kwargs.get("field_reduction", self.fieldreduce)

        state.step_time = argtest.gt("step_time", step_time, 0, float)
        state.register_buffer("lr_post", torch.tensor(float(lr_post)), persistent=False)
        state.register_buffer("lr_pre", torch.tensor(float(lr_pre)), persistent=False)
        state.tc_post = argtest.gt("tc_post", tc_post, 0, float)
        state.tc_pre = argtest.gt("tc_pre", tc_pre, 0, float)
        state.tc_eligibility = argtest.gt("tc_eligibility", tc_eligibility, 0, float)
        state.register_buffer(
            "scale",
            torch.tensor(argtest.gt("scale", scale, 0, float)),
            persistent=False,
        )
        state.delayed = bool(delayed)
        state.register_buffer(
            "tolerance",
            torch.tensor(argtest.gte("interp_tolerance", interp_tolerance, 0, float)),
            persistent=False,
        )
        state.trace = argtest.oneof(
            "trace_mode", trace_mode, "cumulative", "nearest", op=(lambda x: x.lower())
        )
        state.batchreduce = batch_reduction if batch_reduction else torch.mean
        state.fieldreduce = field_reduction if field_reduction else torch.sum

        return state

    def register_cell(
        self,
        name: str,
        cell: Cell,
        /,
        **kwargs: Any,
    ) -> MSTDPET.Unit:
        r"""Adds a cell with required state.

        Args:
            name (str): name of the cell to add.
            cell (Cell): cell to add.

        Keyword Args:
            step_time (float): length of a simulation time step.
            lr_post (float): learning rate for updates on postsynaptic spike updates.
            lr_pre (float): learning rate for updates on presynaptic spike updates.
            tc_post (float): time constant for exponential decay of postsynaptic trace.
            tc_pre (float): time constant for exponential decay of presynaptic trace.
            tc_eligibility (float): time constant for exponential decay of eligibility trace.
            scale (float): scaling term for both the postsynaptic and presynaptic updates.
            delayed (bool): if the updater should assume that learned delays,
                if present, may change.
            interp_tolerance (float): maximum difference in time from an observation
                to treat as co-occurring.
            trace_mode (Literal["cumulative", "nearest"]): method to use for
                calculating spike traces.
            batch_reduction (Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor]):
                function to reduce updates over the batch dimension.
            field_reduction (Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] | None):
                function to reduce updates over the receptive field dimension,
                :py:func:`torch.sum` when None. Defaults to None.

        Returns:
            IndependentTrainer.Unit: specified cell, auxiliary state, and monitors.

        Important:
            Any specified keyword arguments will override the default hyperparameters
            set on initialization. See :py:class:`MSTDPET` for details.
        """
        # add the cell with additional hyperparameters
        cell, state = self.add_cell(name, cell, self._build_cell_state(**kwargs))

        # if delays should be accounted for
        delayed = state.delayed and cell.connection.delayedby is not None

        # common arguments
        monitor_kwargs = {
            "as_prehook": False,
            "train_update": True,
            "eval_update": False,
        }

        # postsynaptic trace monitor (weighs hebbian LTD)
        self.add_monitor(
            name,
            "trace_post",
            "neuron.spike",
            StateMonitor.partialconstructor(
                reducer=CumulativeTraceReducer(
                    state.step_time,
                    state.tc_post,
                    amplitude=state.lr_post.abs().item(),
                    target=True,
                    duration=0.0,
                ),
                **monitor_kwargs,
                prepend=True,
            ),
            False,
            dt=state.step_time,
            tc=state.tc_post,
            trace=state.trace,
        )

        # postsynaptic spike monitor (triggers hebbian LTP)
        self.add_monitor(
            name,
            "spike_post",
            "neuron.spike",
            StateMonitor.partialconstructor(
                reducer=PassthroughReducer(state.step_time, duration=0.0),
                **monitor_kwargs,
                prepend=True,
            ),
            False,
            dt=state.step_time,
        )

        # presynaptic trace monitor (weighs hebbian LTP)
        # when the delayed condition is true, using synapse.spike records the raw
        # spike times rather than the delay adjusted times of synspike.
        self.add_monitor(
            name,
            "trace_pre",
            "synapse.spike" if delayed else "connection.synspike",
            StateMonitor.partialconstructor(
                reducer=CumulativeTraceReducer(
                    state.step_time,
                    state.tc_pre,
                    amplitude=state.lr_pre.abs().item(),
                    target=True,
                    duration=0.0,
                ),
                **monitor_kwargs,
                prepend=True,
            ),
            False,
            dt=state.step_time,
            tc=state.tc_pre,
            trace=state.trace,
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
                    state.step_time,
                    duration=0.0,
                ),
                **monitor_kwargs,
                prepend=True,
            ),
            False,
            dt=state.step_time,
            delayed=delayed,
        )

        # partial eligibility calculation
        def eligibility(trace, spike, trs, srs, fr):
            return (fr()(trs()(trace) * srs()(spike), -1),)

        # presynaptic-scaled postsynaptic-triggered eligibility trace (hebbian LTP)
        self.add_monitor(
            name,
            "elig_post",
            "monitors",
            MultiStateMonitor.partialconstructor(
                reducer=EligibilityTrace(
                    state.step_time,
                    state.tc_eligibility,
                    duration=cell.connection.delayedby if delayed else 0.0,
                ),
                subattrs=("trace_pre.peeked", "spike_post.peeked"),
                **monitor_kwargs,
                prepend=False,
                map_=partial(
                    eligibility,
                    trs=weakref.WeakMethod(cell.connection.presyn_receptive),
                    srs=weakref.WeakMethod(cell.connection.postsyn_receptive),
                    fr=weakref.ref(state.fieldreduce),
                ),
            ),
        )

        # postsynaptic-scaled presynaptic-triggered eligibility trace (hebbian LTD)
        self.add_monitor(
            name,
            "elig_pre",
            "monitors",
            MultiStateMonitor.partialconstructor(
                reducer=EligibilityTrace(
                    state.step_time,
                    state.tc_eligibility,
                    duration=cell.connection.delayedby if delayed else 0.0,
                ),
                subattrs=("trace_post.peeked", "spike_pre.peeked"),
                **monitor_kwargs,
                prepend=False,
                map_=partial(
                    eligibility,
                    trs=weakref.WeakMethod(cell.connection.postsyn_receptive),
                    srs=weakref.WeakMethod(cell.connection.presyn_receptive),
                    fr=weakref.ref(state.fieldreduce),
                ),
            ),
        )

        return name

    def clear(self, **kwargs):
        """Clears all of the monitors and additional state for the trainer.

        Note:
            Keyword arguments are passed to :py:meth:`Monitor.clear` call.

        Note:
            If a subclassed trainer has additional state, this should be overridden
            to delete that state as well. This however doesn't delete updater state
            as it may be shared across trainers.
        """
        for monitor in self.monitors:
            monitor.clear(**kwargs)

        for _, state in self.cells:
            state.e_post_trace.clear(**kwargs)
            state.e_pre_trace.clear(**kwargs)

    def forward(self, reward: float | torch.Tensor) -> None:
        r"""Processes update for given layers based on current monitor stored data.

        A reward term (``reward``) is used as an additional scaling term applied to
        the update. When a :py:class:`float`, it is applied to all batch samples.

        The sign of ``reward`` for a given element will affect if the update is considered
        potentiative or depressive for the purposes of weight dependence.

        Args:
            reward (float | torch.Tensor): reward for the trained batch.

        .. admonition:: Shape
            :class: tensorshape

            ``reward``:

            :math:`B`

            Where:
                * :math:`B` is the batch size.
        """
        # iterate through self
        for cell, state, monitors in self:

            # skip if self or cell is not in training mode or has no updater
            if not cell.training or not self.training or not cell.updater:
                continue

            # get eligibility traces
            if state.delayed and cell.connection.delayedby:
                zpost = monitors["elig_post"].view(
                    cell.connection.delay.unsqueeze(0).expand(
                        cell.connection.batchsz, *cell.connection.delay.shape
                    ),
                    state.tolerance,
                )
                zpre = monitors["elig_pre"].view(
                    cell.connection.delay.unsqueeze(0).expand(
                        cell.connection.batchsz, *cell.connection.delay.shape
                    ),
                    state.tolerance,
                )
            else:
                zpost = monitors["elig_post"].peek()
                zpre = monitors["elig_pre"].peek()

            # process update
            if isinstance(reward, torch.Tensor):
                # reward subterms
                reward_abs = reward.abs()
                reward_pos = torch.argwhere(reward_abs >= 0)
                reward_neg = torch.argwhere(reward_abs < 0)

                # partial updates
                dpost = zpost * (reward_abs * state.scale)
                dpre = zpre * (reward_abs * state.scale)

                dpost_reg, dpost_inv = dpost[reward_pos], dpost[reward_neg]
                dpre_reg, dpre_inv = dpre[reward_pos], dpre[reward_neg]

                # join partials
                match (state.lr_post >= 0, state.lr_pre >= 0):
                    case (False, False):  # depressive
                        dpos = torch.cat(dpost_inv, dpre_inv, 0)
                        dneg = torch.cat(dpost_reg, dpre_reg, 0)
                    case (False, True):  # anti-hebbian
                        dpos = torch.cat(dpost_inv, dpre_reg, 0)
                        dneg = torch.cat(dpost_reg, dpre_inv, 0)
                    case (True, False):  # hebbian
                        dpos = torch.cat(dpost_reg, dpre_inv, 0)
                        dneg = torch.cat(dpost_inv, dpre_reg, 0)
                    case (True, True):  # potentiative
                        dpos = torch.cat(dpost_reg, dpre_reg, 0)
                        dneg = torch.cat(dpost_inv, dpre_inv, 0)

                # apply update
                cell.updater.weight = (
                    state.batchreduce(dpos, 0) if dpos.numel() else None,
                    state.batchreduce(dneg, 0) if dneg.numel() else None,
                )

            else:
                # partial updates
                dpost = state.batchreduce(zpost * abs(reward) * state.scale, 0)
                dpre = state.batchreduce(zpre * abs(reward) * state.scale, 0)

                # accumulate partials with mode condition
                match (state.lr_post * reward >= 0, state.lr_pre * reward >= 0):
                    case (False, False):  # depressive
                        cell.updater.weight = (None, dpost + dpre)
                    case (False, True):  # anti-hebbian
                        cell.updater.weight = (dpre, dpost)
                    case (True, False):  # hebbian
                        cell.updater.weight = (dpost, dpre)
                    case (True, True):  # potentiative
                        cell.updater.weight = (dpost + dpre, None)
