from __future__ import annotations
from .. import LayerwiseTrainer
from ... import Module
from ..._internal import argtest
from ...neural import Cell
from ...observe import StateMonitor, CumulativeTraceReducer, PassthroughReducer
import torch
from typing import Any, Callable, Literal


class MSTDP(LayerwiseTrainer):
    pass


class MSTDPET(LayerwiseTrainer):
    r"""Modulated spike-timing dependent plasticity with eligibility trace updater.

    .. math::
        w(t + \Delta t) - w(t) = \gamma  r(t + \Delta t) [z_+(t + \Delta t) + z_-(t + \Delta t)] \Delta t

    Where:

    .. math::
        \begin{align*}
            z_+(t + \Delta t) &= z_+(t) \exp\left(-\frac{\Delta t}{\tau_z}\right)
            + \frac{x_\text{pre}(t)}{\tau_z}\left[t = t_\text{post}^f\right] \\
            z_-(t + \Delta t) &= z_-(t) \exp\left(-\frac{\Delta t}{\tau_z}\right)
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
        The constructor arguments are hyperparameters for MSTDPET and can be overridden on
        a cell-by-cell basis.

    Note:
        ``batch_reduction`` can be one of the functions in PyTorch including but not
        limited to :py:func:`torch.sum`, :py:func:`torch.max` and :py:func:`torch.max`.
        A custom function with similar behavior can also be passed in. Like with the
        included function, it should not keep the original dimensions by default.

    See Also:
        For more details and references, visit
        :ref:`zoo/learning-stdp:Modulated Spike-Timing Dependent Plasticity (MSTDPET)`
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
        trace_mode: Literal["cumulative", "nearest"] = "cumulative",
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

    class State(Module):
        r"""MSTDPET Auxiliary State

        Args:
            trainer (MSTDPET): MSTDPET trainer.
            **kwargs (Any): default argument overrides.
        """

        def __init__(self, trainer: MSTDPET, **kwargs: Any):
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
                    amplitude=1.0,
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
        delayed = state.delayed and cell.connection.delayedby is not None
        self.add_monitor(
            name,
            "trace_pre",
            "synapse.spike" if delayed else "connection.synspike",
            StateMonitor.partialconstructor(
                reducer=CumulativeTraceReducer(
                    state.step_time,
                    state.tc_pre,
                    amplitude=1.0,
                    target=True,
                    duration=cell.connection.delayedby if delayed else 0.0,
                ),
                **monitor_kwargs,
            ),
            False,
            dt=state.step_time,
            trace=state.trace,
            tc=state.tc_pre,
        )

        # presynaptic spike monitor (triggers hebbian LTD)
        self.add_monitor(
            name,
            "spike_pre",
            "connection.synspike",
            StateMonitor.CumulativeTraceReducer(
                reducer=PassthroughReducer(state.step_time, duration=0.0),
                **monitor_kwargs,
            ),
            False,
            dt=state.step_time,
        )

        return name

    def forward(self) -> None:
        """Processes update for given layers based on current monitor stored data."""
        # iterate through self
        for cell, aux, monitors in self:

            # skip if self or cell is not in training mode or has no updater
            if not cell.training or not self.training or not cell.updater:
                continue

            # spike traces, reshaped into receptive format
            x_post = cell.connection.postsyn_receptive(monitors["trace_post"].peek())
            x_pre = cell.connection.presyn_receptive(
                monitors["trace_pre"].view(cell.connection.selector, aux.tolerance)
                if aux.delayed and cell.connection.delayedby
                else monitors["trace_pre"].peek()
            )

            # spike presence, reshaped into receptive format
            i_post = cell.connection.postsyn_receptive(monitors["spike_post"].peek())
            i_pre = cell.connection.presyn_receptive(monitors["spike_pre"].peek())

            # partial updates
            dpost = aux.batchreduce(torch.sum(i_post * x_pre, -1), 0) * abs(aux.lr_post)
            dpre = aux.batchreduce(torch.sum(i_pre * x_post, -1), 0) * abs(aux.lr_pre)

            # accumulate partials with mode condition
            match (aux.lr_post >= 0, aux.lr_pre >= 0):
                case (False, False):  # depressive
                    cell.updater.weight = (None, dpost + dpre)
                case (False, True):  # anti-hebbian
                    cell.updater.weight = (dpre, dpost)
                case (True, False):  # hebbian
                    cell.updater.weight = (dpost, dpre)
                case (True, True):  # potentiative
                    cell.updater.weight = (dpost + dpre, None)
