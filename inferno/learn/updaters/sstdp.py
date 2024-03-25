from __future__ import annotations
from .. import LayerwiseTrainer
from ... import Module, interpolation
from ..._internal import argtest
from ...neural import Cell
from ...observe import (
    StateMonitor,
    CumulativeTraceReducer,
    PassthroughReducer,
    FoldingReducer,
)
import math
import torch
from typing import Any, Callable


class MSTDPET(LayerwiseTrainer):
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
        **kwargs,
    ):
        # call superclass constructor
        LayerwiseTrainer.__init__(self, **kwargs)

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

    class EligibilityTraceReducer(FoldingReducer):
        r"""Reducer for eligibility trace.

        Args:
            step_time (float): length of the discrete step time.
            time_constant (float): time constant of exponential decay.
            duration (float): length of time over which results should be stored.
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

            # register state
            self.register_buffer(
                "time_constant",
                argtest.gt("time_constant", time_constant, 0, float),
                persistent=False,
            )
            self.register_buffer(
                "decay",
                math.exp(-self.dt / self.time_constant),
                persistent=False,
            )

        @property
        def dt(self) -> float:
            return FoldingReducer.dt.fget(self)

        @dt.setter
        def dt(self, value: float):
            FoldingReducer.dt.fset(self, value)
            self.decay = torch.exp(-self.dt / self.time_constant)

        def fold(
            self, obs: torch.Tensor, cond: torch.Tensor, state: torch.Tensor | None
        ) -> torch.Tensor:
            if state is None:
                return torch.sum((obs / self.time_constant) * cond, 0)
            else:
                return (self.decay * state) + torch.sum(
                    (obs / self.time_constant) * cond, 0
                )

        def initialize(self, inputs: torch.Tensor) -> torch.Tensor:
            return inputs.fill_(0)

        def interpolate(
            self,
            prev_data: torch.Tensor,
            next_data: torch.Tensor,
            sample_at: torch.Tensor,
            step_time: float | torch.Tensor,
        ) -> torch.Tensor:
            return interpolation.expdecay(
                prev_data, next_data, sample_at, step_time, self.time_constant
            )

    class State(Module):
        r"""MSTDPET Auxiliary State

        Args:
            trainer (MSTDPET): MSTDPET trainer.
            cell (Cell): cell.
            **kwargs (Any): default argument overrides.
        """

        def __init__(self, trainer: MSTDPET, cell: Cell, **kwargs: Any):
            # call superclass constructor
            Module.__init__(self)

            # map arguments
            if "step_time" in kwargs:
                self.step_time = argtest.gt("step_time", kwargs["step_time"], 0, float)
            else:
                self.step_time = trainer.step_time

            if "lr_post" in kwargs:
                self.lr_post = float(kwargs["lr_post"])
            else:
                self.lr_post = trainer.lr_post

            if "lr_pre" in kwargs:
                self.lr_pre = float(kwargs["lr_pre"])
            else:
                self.lr_pre = trainer.lr_pre

            if "tc_post" in kwargs:
                self.tc_post = argtest.gt("tc_post", kwargs["tc_post"], 0, float)
            else:
                self.tc_post = trainer.tc_post

            if "tc_pre" in kwargs:
                self.tc_pre = argtest.gt("tc_pre", kwargs["tc_pre"], 0, float)
            else:
                self.tc_pre = trainer.tc_pre

            if "tc_eligibility" in kwargs:
                self.tc_eligibility = argtest.gt(
                    "tc_eligibility", kwargs["tc_eligibility"], 0, float
                )
            else:
                self.tc_eligibility = trainer.tc_eligibility

            if "scale" in kwargs:
                self.scale = argtest.gt("scale", kwargs["scale"], 0, float)
            else:
                self.scale = trainer.scale

            if "delayed" in kwargs:
                self.delayed = bool(kwargs["delayed"])
            else:
                self.delayed = trainer.delayed

            if "interp_tolerance" in kwargs:
                self.tolerance = argtest.gte(
                    "interp_tolerance", kwargs["interp_tolerance"], 0, float
                )
            else:
                self.tolerance = trainer.tolerance

            if "batch_reduction" in kwargs:
                self.batchreduce = (
                    kwargs["batch_reduction"]
                    if kwargs["batch_reduction"]
                    else torch.mean
                )
            else:
                self.batchreduce = trainer.batchreduce

            # eligibility trace reducers
            self.e_post_trace = self.EligibilityTraceReducer(
                self.step_time,
                self.tc_eligibility,
                duration=cell.connection.delayedby if self.delayed else 0.0,
            )

            self.e_pre_trace = self.EligibilityTraceReducer(
                self.step_time,
                self.tc_eligibility,
                duration=cell.connection.delayedby if self.delayed else 0.0,
            )

            # eligibility traces
            self.add_module("e_post_trace", None)
            self.add_module("e_pre_trace", None)

    def add_cell(
        self,
        name: str,
        cell: Cell,
        **kwargs: Any,
    ) -> str:
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

        Returns:
            str: name of the added cell.

        Important:
            Any specified keyword arguments will override the default hyperparameters
            set on initialization. See :py:class:`STDP` for details.
        """
        # add the cell with additional hyperparameters
        cell, state = self._add_cell(name, cell, self.State(self, **kwargs))

        # if delays should be accounted for
        delayed = state.delayed and cell.connection.delayedby is not None

        # common arguments
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
                reducer=CumulativeTraceReducer(
                    state.step_time,
                    state.tc_post,
                    amplitude=abs(state.lr_post),
                    target=True,
                    duration=0.0,
                ),
                **monitor_kwargs,
            ),
            False,
            dt=state.step_time,
            trace=state.trace,
            tc=state.tc_post,
        )

        # postsynaptic spike monitor (triggers hebbian LTP)
        self.add_monitor(
            name,
            "spike_post",
            "neuron.spike",
            StateMonitor.partialconstructor(
                reducer=PassthroughReducer(state.step_time, duration=0.0),
                **monitor_kwargs,
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
                    amplitude=abs(state.lr_pre),
                    target=True,
                    duration=0.0,
                ),
                **monitor_kwargs,
            ),
            False,
            dt=state.step_time,
            trace=state.trace,
            tc=state.tc_pre,
            delayed=delayed,
        )

        # presynaptic spike monitor (triggers hebbian LTD)
        # when the delayed condition is true, using synapse.spike records the raw
        # spike times rather than the delay adjusted times of synspike.
        self.add_monitor(
            name,
            "spike_pre",
            "synapse.spike" if delayed else "connection.synspike",
            StateMonitor.CumulativeTraceReducer(
                reducer=PassthroughReducer(
                    state.step_time,
                    duration=0.0,
                ),
                **monitor_kwargs,
            ),
            False,
            dt=state.step_time,
            delayed=delayed,
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

            # update eligibility traces
            state.e_post_trace(
                cell.connection.presyn_receptive(monitors["trace_pre"].peek()),
                cell.connection.postsyn_receptive(monitors["spike_post"].peek()),
            )

            state.e_pre_trace(
                cell.connection.postsyn_receptive(monitors["trace_post"].peek()),
                cell.connection.presyn_receptive(monitors["spike_pre"].peek()),
            )

            # retrieve eligbility traces
            if state.delayed and cell.connection.delayedby:
                z_post = state.e_post_trace.view(
                    cell.connection.delay.unsqueeze(0).expand(
                        cell.connection.batchsz, *cell.connection.delay.shape
                    ),
                    state.tolerance,
                )
                z_pre = state.e_pre_trace.view(
                    cell.connection.delay.unsqueeze(0).expand(
                        cell.connection.batchsz, *cell.connection.delay.shape
                    ),
                    state.tolerance,
                )
            else:
                z_post = state.e_post_trace.peek()
                z_pre = state.e_pre_trace.peek()

            # process update
            if isinstance(reward, torch.Tensor):
                # reward subterms
                reward_abs = reward.abs()
                reward_pos = torch.argwhere(reward_abs >= 0)
                reward_neg = torch.argwhere(reward_abs < 0)

                # partial updates
                dpost = z_post * (reward_abs * state.scale)
                dpre = z_pre * (reward_abs * state.scale)

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
                dpost = state.batchreduce(z_post * abs(reward), 0) * state.scale
                dpre = state.batchreduce(z_pre * abs(reward), 0) * state.scale

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
