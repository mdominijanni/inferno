from __future__ import annotations
from .. import IndependentTrainer
from ... import Module
from ..._internal import argtest
from ...neural import Cell
from ...observe import (
    StateMonitor,
    MultiStateMonitor,
    CumulativeTraceReducer,
    ConditionalCumulativeTraceReducer,
    PassthroughReducer,
)
import torch
from typing import Any, Callable
import weakref


class EligibilityTraceReducer(ConditionalCumulativeTraceReducer):
    r"""Reducer used by MSTDPET for eligibility trace.

    Identical to :py:class:`ConditionalCumulativeTraceReducer`, except it will wrap the
    :py:meth:`fold` function so inputs are reshaped and outputs are reduced along the
    field dimension.
    """

    def __init__(
        self,
        step_time: float,
        time_constant: float,
        amplitude: int | float | complex,
        scale: int | float | complex,
        *,
        obs_reshape: weakref.WeakMethod,
        cond_reshape: weakref.WeakMethod,
        field_reduce: weakref.ReferenceType,
        duration: float = 0.0,
    ):
        # call superclass constructor
        ConditionalCumulativeTraceReducer.__init__(
            self,
            step_time=step_time,
            time_constant=time_constant,
            amplitude=amplitude,
            scale=scale,
            duration=duration,
        )

        # add reshape and reduce references
        self.obs_reshape = obs_reshape
        self.cond_reshape = cond_reshape
        self.field_reduce = field_reduce

    def fold(
        self, obs: torch.Tensor, cond: torch.Tensor, state: torch.Tensor | None
    ) -> torch.Tensor:
        r"""Application of scaled cumulative trace.

        Args:
            obs (torch.Tensor): observation to incorporate into state.
            cond (torch.Tensor): condition if observations match for the trace.
            state (torch.Tensor | None): state from the prior time step,
                None if no prior observations.

        Returns:
            torch.Tensor: state for the current time step.
        """
        return self.field_reduce()(
            ConditionalCumulativeTraceReducer.fold(
                self, self.obs_reshape()(obs), self.cond_reshape()(cond), state
            )
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

    Because this logic is extended to the sign of the reward signal, the size of the
    batch for the potentiative and depressive update components may not be the same as
    the input batch size. Keep this in mind when selecting a ``batch_reduction``. For
    this reason, the default is :py:func:`torch.sum`. Additionally, the scale
    :math:`\gamma` can be passed in along with the reward signal to account for this.

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
        interp_tolerance (float): maximum difference in time from an observation
            to treat as co-occurring, in :math:`\text{ms}`. Defaults to 0.0.
        trace_mode (Literal["cumulative", "nearest"], optional): method to use for
            calculating spike traces. Defaults to "cumulative".
        batch_reduction (Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] | None):
            function to reduce updates over the batch dimension, :py:func:`torch.sum`
            when None. Defaults to None.
        field_reduction (Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] | None):
            function to reduce updates over the receptive field dimension,
            :py:func:`torch.sum` when None. Defaults to None.

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
        self.tolerance = argtest.gte("interp_tolerance", interp_tolerance, 0, float)
        self.batchreduce = batch_reduction if batch_reduction else torch.sum
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
        interp_tolerance = kwargs.get("interp_tolerance", self.tolerance)
        batch_reduction = kwargs.get("batch_reduction", self.batchreduce)
        field_reduction = kwargs.get("field_reduction", self.fieldreduce)

        state.step_time = argtest.gt("step_time", step_time, 0, float)
        state.lr_post = float(lr_post)
        state.lr_post = float(lr_pre)
        state.tc_post = argtest.gt("tc_post", tc_post, 0, float)
        state.tc_pre = argtest.gt("tc_pre", tc_pre, 0, float)
        state.tc_eligibility = argtest.gt("tc_eligibility", tc_eligibility, 0, float)
        state.tolerance = argtest.gte("interp_tolerance", interp_tolerance, 0, float)
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
            interp_tolerance (float): maximum difference in time from an observation
                to treat as co-occurring.
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
                    amplitude=state.lr_pre,
                    target=True,
                    duration=0.0,
                ),
                prepend=True,
                **monitor_kwargs,
            ),
            False,
            dt=state.step_time,
        )

        # postsynaptic spike monitor (triggers hebbian LTP)
        self.add_monitor(
            name,
            "spike_post",
            "neuron.spike",
            StateMonitor.partialconstructor(
                reducer=PassthroughReducer(state.step_time, duration=0.0),
                prepend=True,
                **monitor_kwargs,
            ),
            False,
            dt=state.step_time,
        )

        # presynaptic trace monitor (weighs hebbian LTP)
        self.add_monitor(
            name,
            "trace_pre",
            "connection.synspike",
            StateMonitor.partialconstructor(
                reducer=CumulativeTraceReducer(
                    state.step_time,
                    state.tc_pre,
                    amplitude=state.lr_pre,
                    target=True,
                    duration=0.0,
                ),
                prepend=True,
                **monitor_kwargs,
            ),
            False,
            dt=state.step_time,
        )

        # presynaptic spike monitor (triggers hebbian LTD)
        self.add_monitor(
            name,
            "spike_pre",
            "connection.synspike",
            StateMonitor.partialconstructor(
                reducer=PassthroughReducer(state.step_time, duration=0.0),
                prepend=True,
                **monitor_kwargs,
            ),
            False,
            dt=state.step_time,
        )

        # presynaptic-scaled postsynaptic-triggered eligibility trace (hebbian LTP)
        self.add_monitor(
            name,
            "elig_post",
            "monitors",
            MultiStateMonitor.partialconstructor(
                reducer=EligibilityTraceReducer(
                    state.step_time,
                    state.tc_eligibility,
                    amplitude=0.0,
                    scale=(1 / state.tc_eligibility),
                    obs_reshape=weakref.WeakMethod(cell.connection.presyn_receptive),
                    cond_reshape=weakref.WeakMethod(cell.connection.postsyn_receptive),
                    field_reduce=weakref.ref(state.fieldreduce),
                    duration=0.0,
                ),
                subattrs=("trace_pre.latest", "spike_post.latest"),
                prepend=False,
                **monitor_kwargs,
            ),
            True,
        )

        # postsynaptic-scaled presynaptic-triggered eligibility trace (hebbian LTD)
        self.add_monitor(
            name,
            "elig_pre",
            "monitors",
            MultiStateMonitor.partialconstructor(
                reducer=EligibilityTraceReducer(
                    state.step_time,
                    state.tc_eligibility,
                    amplitude=0.0,
                    scale=(1 / state.tc_eligibility),
                    obs_reshape=weakref.WeakMethod(cell.connection.postsyn_receptive),
                    cond_reshape=weakref.WeakMethod(cell.connection.presyn_receptive),
                    field_reduce=weakref.ref(state.fieldreduce),
                    duration=0.0,
                ),
                subattrs=("trace_post.latest", "spike_pre.latest"),
                prepend=False,
                **monitor_kwargs,
            ),
            True,
        )

        return name

    def forward(self, reward: float | torch.Tensor, scale: float = 1.0) -> None:
        r"""Processes update for given layers based on current monitor stored data.

        A reward term (``reward``) is used as an additional scaling term applied to
        the update. When a :py:class:`float`, it is applied to all batch samples.

        The sign of ``reward`` for a given element will affect if the update is considered
        potentiative or depressive for the purposes of weight dependence.

        Args:
            reward (float | torch.Tensor): reward for the trained batch.
            scale (float, optional): scaling factor used for the updates, this value
                is expected to be nonnegative, and its absolute value will be used.
                Defaults to 1.0.

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

            # eligibility traces (shaped like batched weights)
            z_post = monitors["elig_post"].peek()
            z_pre = monitors["elig_pre"].peek()

            # process update
            if isinstance(reward, torch.Tensor):
                # reward subterms
                reward_abs = reward.abs()
                reward_pos = torch.argwhere(reward >= 0).view(-1)
                reward_neg = torch.argwhere(reward < 0).view(-1)

                # partial updates
                dpost = z_post * (reward_abs * abs(scale))
                dpre = z_pre * (reward_abs * abs(scale))

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
                dpost = state.batchreduce(z_post * abs(reward) * abs(scale), 0)
                dpre = state.batchreduce(z_pre * abs(reward) * abs(scale), 0)

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
