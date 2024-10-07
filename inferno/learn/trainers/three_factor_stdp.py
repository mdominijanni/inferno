from __future__ import annotations
from collections.abc import Sequence
from .. import IndependentCellTrainer
from ... import Module, trace_cumulative_value
from ..._internal import argtest
from ...functional import interp_expdecay
from ...neural import Cell
from ...observe import (
    StateMonitor,
    MultiStateMonitor,
    CumulativeTraceReducer,
    NearestTraceReducer,
    FoldReducer,
    PassthroughReducer,
)
import einops as ein
from itertools import repeat
import math
import torch
from typing import Any, Callable, Literal
import weakref


class EligibilityTraceReducer(FoldReducer):
    r"""Reducer used by MSTDPET for eligibility trace.

    Identical to :py:class:`ConditionalCumulativeTraceReducer`, except it will wrap the
    :py:meth:`fold` function so inputs are reshaped and outputs are reduced along the
    field dimension.
    """

    def __init__(
        self,
        step_time: float,
        time_constant: float,
        *,
        obs_reshape: weakref.WeakMethod,
        cond_reshape: weakref.WeakMethod,
        duration: float = 0.0,
        inclusive: bool = True,
    ):
        # call superclass constructor
        FoldReducer.__init__(self, step_time, duration, inclusive, 0)

        # register state
        self.time_constant = argtest.gt("time_constant", time_constant, 0, float)
        self.decay = math.exp(-self.dt / self.time_constant)
        self.scale = 1 / self.time_constant

        # add reshape and reduce references
        self.obs_reshape = obs_reshape
        self.cond_reshape = cond_reshape

    @property
    def dt(self) -> float:
        r"""Length of the simulation time step, in milliseconds.

        Args:
            value (float): new simulation time step length.

        Returns:
            float: length of the simulation time step.
        """
        return FoldReducer.dt.fget(self)

    @dt.setter
    def dt(self, value: float) -> None:
        FoldReducer.dt.fset(self, value)
        self.decay = math.exp(-self.dt / self.time_constant)

    def fold(
        self, obs: torch.Tensor, cond: torch.Tensor, state: torch.Tensor | None
    ) -> torch.Tensor:
        r"""Application of scaled cumulative trace.

        Args:
            obs (torch.Tensor): observation to incorporate into state.
            cond (torch.Tensor): condition if observations match for the trace.
            state (torch.Tensor | None): state from the prior time step,
                ``None`` if no prior observations.

        Returns:
            torch.Tensor: state for the current time step.
        """
        return trace_cumulative_value(
            ein.einsum(
                self.obs_reshape()(obs),
                self.cond_reshape()(cond),
                "b ... r, b ... r -> b ...",
            ),
            state,
            decay=self.decay,
            scale=self.scale,
        )

    def interpolate(
        self,
        prev_data: torch.Tensor,
        next_data: torch.Tensor,
        sample_at: torch.Tensor,
        step_time: float,
    ) -> torch.Tensor:
        r"""Exponential decay interpolation between observations.

        Args:
            prev_data (torch.Tensor): most recent observation prior to sample time.
            next_data (torch.Tensor): most recent observation subsequent to sample time.
            sample_at (torch.Tensor): relative time at which to sample data.
            step_time (float): length of time between the prior and
                subsequent observations.

        Returns:
            torch.Tensor: interpolated data at sample time.
        """
        return interp_expdecay(
            prev_data, next_data, sample_at, step_time, time_constant=self.time_constant
        )


class MSTDPET(IndependentCellTrainer):
    r"""Modulated spike-timing dependent plasticity with eligibility trace trainer.

    .. math::
        w(t + \Delta t) - w(t) = \gamma  M(t)
        [z_\text{post}(t) + z_\text{pre}(t)]
        \Delta t

    .. math::
        \begin{align*}
            z_\text{post}(t) &= z_\text{post}(t - \Delta t) \exp\left(-\frac{\Delta t}{\tau_z}\right)
            + \frac{x_\text{pre}(t)}{\tau_z}\left[t = t_\text{post}^f\right] \\
            z_\text{pre}(t) &= z_\text{pre}(t - \Delta t) \exp\left(-\frac{\Delta t}{\tau_z}\right)
            + \frac{x_\text{post}(t)}{\tau_z}\left[t = t_\text{pre}^f\right]
        \end{align*}

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
    control which terms are potentiative and depressive updates (these are applied to
    the opposite trace). The terms (when expanded) can be scaled for weight dependence
    on updating. :math:`M` is a reinforcement term given on each update. Note that
    this implementation splits the eligibility trace into two terms, so weight
    dependence can scale the magnitude of each.

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

    Because this logic is extended to the sign of the modulation signal, the size of the
    batch for the potentiative and depressive update components may not be the same as
    the input batch size. Keep this in mind when selecting a ``batch_reduction``. For
    this reason, the default is :py:func:`torch.sum`. Additionally, the scale
    :math:`\gamma` can be passed in along with the modulation signal to account for this.

    Args:
        lr_post (float): learning rate for updates on postsynaptic spikes,
            :math:`\eta_\text{post}`.
        lr_pre (float): learning rate for updates on presynaptic spikes,
            :math:`\eta_\text{pre}`.
        tc_post (float): time constant of exponential decay of postsynaptic trace,
            :math:`tau_\text{post}`, in :math:`ms`.
        tc_pre (float): time constant of exponential decay of presynaptic trace,
            :math:`tau_\text{pre}`, in :math:`ms`.
        tc_eligibility (float): time constant of exponential decay of eligibility trace,
            :math:`tau_z`, in :math:`ms`.
        interp_tolerance (float, optional): maximum difference in time from an observation
            to treat as co-occurring, in :math:`\text{ms}`. Defaults to ``0.0``.
        trace_mode (Literal["cumulative", "nearest"], optional): method to use for
            calculating spike traces. Defaults to ``"cumulative"``.
        batch_reduction (Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] | None):
            function to reduce updates over the batch dimension, :py:func:`torch.sum`
            when ``None``. Defaults to ``None``.

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
        :ref:`zoo/learning-stdp:Modulated Spike-Timing Dependent Plasticity with Eligibility Trace (MSTDPET)`
        in the zoo.
    """

    def __init__(
        self,
        lr_post: float,
        lr_pre: float,
        tc_post: float,
        tc_pre: float,
        tc_eligibility: float,
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
        self.tc_eligibility = argtest.gt("tc_eligibility", tc_eligibility, 0, float)
        self.tolerance = argtest.gte("interp_tolerance", interp_tolerance, 0, float)
        self.trace = argtest.oneof(
            "trace_mode", trace_mode, "cumulative", "nearest", op=(lambda x: x.lower())
        )
        self.batchreduce = batch_reduction if batch_reduction else torch.sum

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
        tc_eligibility = kwargs.get("tc_eligibility", self.tc_eligibility)
        interp_tolerance = kwargs.get("interp_tolerance", self.tolerance)
        trace_mode = kwargs.get("trace_mode", self.trace)
        batch_reduction = kwargs.get("batch_reduction", self.batchreduce)

        state.lr_post = float(lr_post)
        state.lr_pre = float(lr_pre)
        state.tc_post = argtest.gt("tc_post", tc_post, 0, float)
        state.tc_pre = argtest.gt("tc_pre", tc_pre, 0, float)
        state.tc_eligibility = argtest.gt("tc_eligibility", tc_eligibility, 0, float)
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
            batch_reduction if (batch_reduction is not None) else torch.sum
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
            tc_eligibility (float): time constant of exponential decay of eligibility trace.
            scale (float): scaling term for both the postsynaptic and presynaptic updates.
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
            set on initialization. See :py:class:`MSTDPET` for details.
        """
        # add the cell with additional hyperparameters
        cell, state = self.add_cell(
            name, cell, self._build_cell_state(**kwargs), ["weight"]
        )

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
                reducer=state.tracecls(
                    cell.connection.dt,
                    state.tc_post,
                    amplitude=abs(state.lr_pre),
                    target=True,
                    duration=0.0,
                    inclusive=True,
                ),
                prepend=True,
                **monitor_kwargs,
            ),
            False,
            dt=cell.connection.dt,
            amp=abs(state.lr_pre),
            tc=state.tc_post,
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
                prepend=True,
                **monitor_kwargs,
            ),
            False,
            dt=cell.connection.dt,
        )

        # presynaptic trace monitor (weighs hebbian LTP)
        self.add_monitor(
            name,
            "trace_pre",
            "connection.synspike",
            StateMonitor.partialconstructor(
                reducer=state.tracecls(
                    cell.connection.dt,
                    state.tc_pre,
                    amplitude=abs(state.lr_post),
                    target=True,
                    duration=0.0,
                    inclusive=True,
                ),
                prepend=True,
                **monitor_kwargs,
            ),
            False,
            dt=cell.connection.dt,
            amp=abs(state.lr_post),
            tc=state.tc_pre,
        )

        # presynaptic spike monitor (triggers hebbian LTD)
        self.add_monitor(
            name,
            "spike_pre",
            "connection.synspike",
            StateMonitor.partialconstructor(
                reducer=PassthroughReducer(
                    cell.connection.dt,
                    duration=0.0,
                    inclusive=True,
                ),
                prepend=True,
                **monitor_kwargs,
            ),
            False,
            dt=cell.connection.dt,
        )

        # presynaptic-scaled postsynaptic-triggered eligibility trace (hebbian LTP)
        self.add_monitor(
            name,
            "elig_post",
            "monitors",
            MultiStateMonitor.partialconstructor(
                reducer=EligibilityTraceReducer(
                    cell.connection.dt,
                    state.tc_eligibility,
                    obs_reshape=weakref.WeakMethod(cell.connection.presyn_receptive),
                    cond_reshape=weakref.WeakMethod(cell.connection.postsyn_receptive),
                    duration=0.0,
                    inclusive=True,
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
                    cell.connection.dt,
                    state.tc_eligibility,
                    obs_reshape=weakref.WeakMethod(cell.connection.postsyn_receptive),
                    cond_reshape=weakref.WeakMethod(cell.connection.presyn_receptive),
                    duration=0.0,
                    inclusive=True,
                ),
                subattrs=("trace_post.latest", "spike_pre.latest"),
                prepend=False,
                **monitor_kwargs,
            ),
            True,
        )

        return self.get_unit(name)

    def forward(
        self,
        signal: float | torch.Tensor,
        scale: float = 1.0,
        cells: Sequence[str] | None = None,
    ) -> None:
        r"""Processes update for given layers based on current monitor stored data.

        A signal (``signal``) is used as an additional scaling term applied to
        the update. When a :py:class:`float`, it is applied to all batch samples.

        The sign of ``signal`` for a given element will affect if the update is considered
        potentiative or depressive for the purposes of weight dependence.

        Args:
            signal (float | torch.Tensor): signal for the trained batch, :math:`M(t)`.
            scale (float, optional): scaling factor used for the updates, this value
                is expected to be nonnegative, and its absolute value will be used,
                :math:`\gamma`. Defaults to ``1.0``.
            cells (Sequence[str] | None): names of the cells to update, all cells if
                ``None``. Defaults to ``None``.

        .. admonition:: Shape
            :class: tensorshape

            ``signal``:

            :math:`B`

            Where:
                * :math:`B` is the batch size.

        Warning:
            For performance reasons, when ``signal`` is a scalar, it and ``scale``
            are applied after the ``batch_reduction`` function is called. Therefore,
            if ``batch_reduction`` is not homogeneous of degree 1, the result will be
            incorrect. A function :math:`f` is homogeneous degree 1 if it preserves
            scalar multiplication, i.e. :math:`a f(X) = f(aX)`.

        Important:
            By default, the sum of results along the batch axis is taken rather than the
            more conventional choice of the mean. This is because potentiative and
            depressive components are split before the batch reduction is performed. To
            take the mean over all samples in the batch, the ``scale`` term should be
            set to :math:`(\text{batch size})^{-1}`.
        """
        # iterate through self
        for name, (cell, state, monitors) in zip(self.cells_, self):

            # skip if cell is not in a non-none training list
            if cells is not None and name not in cells:
                continue

            # skip if self or cell is not in training mode or has no updater
            if not cell.training or not self.training or not cell.updater:
                continue

            # eligibility traces (shaped like batched weights)
            z_post = monitors["elig_post"].peek()
            z_pre = monitors["elig_pre"].peek()

            # process update
            if isinstance(signal, torch.Tensor):
                # signal subterms
                scaledsignal = (
                    (signal * scale).abs().view(-1, *repeat(1, z_post.ndim - 1))
                )
                signal_pos = torch.argwhere(signal >= 0).view(-1)
                signal_neg = torch.argwhere(signal < 0).view(-1)

                # partial updates
                dpost = z_post * scaledsignal
                dpre = z_pre * scaledsignal

                dpost_reg, dpost_inv = dpost[signal_pos], dpost[signal_neg]
                dpre_reg, dpre_inv = dpre[signal_pos], dpre[signal_neg]

                # join partials
                match (state.lr_post >= 0, state.lr_pre >= 0):
                    case (False, False):  # depressive
                        dpos = torch.cat((dpost_inv, dpre_inv), 0)
                        dneg = torch.cat((dpost_reg, dpre_reg), 0)
                    case (False, True):  # anti-hebbian
                        dpos = torch.cat((dpost_inv, dpre_reg), 0)
                        dneg = torch.cat((dpost_reg, dpre_inv), 0)
                    case (True, False):  # hebbian
                        dpos = torch.cat((dpost_reg, dpre_inv), 0)
                        dneg = torch.cat((dpost_inv, dpre_reg), 0)
                    case (True, True):  # potentiative
                        dpos = torch.cat((dpost_reg, dpre_reg), 0)
                        dneg = torch.cat((dpost_inv, dpre_inv), 0)

                # accumulate update
                cell.updater.weight = (
                    state.batchreduce(dpos, 0) if dpos.numel() else None,
                    state.batchreduce(dneg, 0) if dneg.numel() else None,
                )

            else:
                # partial updates
                dpost = state.batchreduce(z_post, 0) * abs(signal * scale)
                dpre = state.batchreduce(z_pre, 0) * abs(signal * scale)

                # accumulate partials with mode condition
                match (state.lr_post * signal >= 0, state.lr_pre * signal >= 0):
                    case (False, False):  # depressive
                        cell.updater.weight = (None, dpost + dpre)
                    case (False, True):  # anti-hebbian
                        cell.updater.weight = (dpre, dpost)
                    case (True, False):  # hebbian
                        cell.updater.weight = (dpost, dpre)
                    case (True, True):  # potentiative
                        cell.updater.weight = (dpost + dpre, None)


class MSTDP(IndependentCellTrainer):
    r"""Modulated spike-timing dependent plasticity trainer.

    .. math::
        w(t + \Delta t) - w(t) = \gamma M(t) \left(x_\text{pre}(t)
        \bigl[t = t^f_\text{post}\bigr] +
        x_\text{post}(t) \bigl[t = t^f_\text{pre}\bigr] \right)

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
    control which terms are potentiative and depressive updates (these are applied to
    the opposite trace). The terms (when expanded) can be scaled for weight dependence
    on updating. :math:`M` is a reinforcement term given on each update.

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

    Because this logic is extended to the sign of the modulation signal, the size of the
    batch for the potentiative and depressive update components may not be the same as
    the input batch size. Keep this in mind when selecting a ``batch_reduction``. For
    this reason, the default is :py:func:`torch.sum`. Additionally, the scale
    :math:`\gamma` can be passed in along with the modulation signal to account for this.

    Args:
        lr_post (float): learning rate for updates on postsynaptic spikes,
            :math:`\eta_\text{post}`.
        lr_pre (float): learning rate for updates on presynaptic spikes,
            :math:`\eta_\text{pre}`.
        tc_post (float): time constant of exponential decay of postsynaptic trace,
            :math:`tau_\text{post}`, in :math:`ms`.
        tc_pre (float): time constant of exponential decay of presynaptic trace,
            :math:`tau_\text{pre}`, in :math:`ms`.
        delayed (bool, optional): if the updater should assume that learned delays, if
            present, may change. Defaults to ``False``.
        interp_tolerance (float, optional): maximum difference in time from an observation
            to treat as co-occurring, in :math:`\text{ms}`. Defaults to ``0.0``.
        trace_mode (Literal["cumulative", "nearest"], optional): method to use for
            calculating spike traces. Defaults to ``"cumulative"``.
        batch_reduction (Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] | None):
            function to reduce updates over the batch dimension, :py:func:`torch.sum`
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
        :ref:`zoo/learning-stdp:Modulated Spike-Timing Dependent Plasticity (MSTDP)`
        in the zoo.
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
        self.batchreduce = batch_reduction if batch_reduction else torch.sum

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
            batch_reduction if (batch_reduction is not None) else torch.sum
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
            set on initialization. See :py:class:`MSTDP` for details.
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

    def forward(
        self,
        signal: float | torch.Tensor,
        scale: float = 1.0,
        cells: Sequence[str] | None = None,
    ) -> None:
        r"""Processes update for given layers based on current monitor stored data.

        A signal (``signal``) is used as an additional scaling term applied to
        the update. When a :py:class:`float`, it is applied to all batch samples.

        The sign of ``signal`` for a given element will affect if the update is considered
        potentiative or depressive for the purposes of weight dependence.

        Args:
            signal (float | torch.Tensor): signal for the trained batch, :math:`M(t)`.
            scale (float, optional): scaling factor used for the updates, this value
                is expected to be nonnegative, and its absolute value will be used,
                :math:`\gamma`. Defaults to ``1.0``.
            cells (Sequence[str] | None): names of the cells to update, all cells if
                ``None``. Defaults to ``None``.

        .. admonition:: Shape
            :class: tensorshape

            ``signal``:

            :math:`B`

            Where:
                * :math:`B` is the batch size.

        Warning:
            For performance reasons, when ``signal`` is a scalar, it and ``scale``
            are applied after the ``batch_reduction`` function is called. Therefore,
            if ``batch_reduction`` is not homogeneous of degree 1, the result will be
            incorrect. A function :math:`f` is homogeneous degree 1 if it preserves
            scalar multiplication, i.e. :math:`a f(X) = f(aX)`.

        Important:
            By default, the sum of results along the batch axis is taken rather than the
            more conventional choice of the mean. This is because potentiative and
            depressive components are split before the batch reduction is performed. To
            take the mean over all samples in the batch, the ``scale`` term should be
            set to :math:`(\text{batch size})^{-1}`.
        """
        # iterate through self
        for name, (cell, state, monitors) in zip(self.cells_, self):

            # skip if cell is not in a non-none training list
            if cells is not None and name not in cells:
                continue

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

            # unscaled partial updates
            dpost = ein.einsum(i_post, x_pre, "b ... r, b ... r -> b ...")
            dpre = ein.einsum(i_pre, x_post, "b ... r, b ... r -> b ...")

            # process update
            if isinstance(signal, torch.Tensor):
                # signal subterms
                scaledsignal = (
                    (signal * scale).abs().view(-1, *repeat(1, dpost.ndim - 1))
                )
                signal_pos = torch.argwhere(signal >= 0).view(-1)
                signal_neg = torch.argwhere(signal < 0).view(-1)

                # scale partial updates
                dpost = dpost * scaledsignal
                dpre = dpre * scaledsignal

                # select partials by mode
                dpost_reg, dpost_inv = dpost[signal_pos], dpost[signal_neg]
                dpre_reg, dpre_inv = dpre[signal_pos], dpre[signal_neg]

                # join partials
                match (state.lr_post >= 0, state.lr_pre >= 0):
                    case (False, False):  # depressive
                        dpos = torch.cat((dpost_inv, dpre_inv), 0)
                        dneg = torch.cat((dpost_reg, dpre_reg), 0)
                    case (False, True):  # anti-hebbian
                        dpos = torch.cat((dpost_inv, dpre_reg), 0)
                        dneg = torch.cat((dpost_reg, dpre_inv), 0)
                    case (True, False):  # hebbian
                        dpos = torch.cat((dpost_reg, dpre_inv), 0)
                        dneg = torch.cat((dpost_inv, dpre_reg), 0)
                    case (True, True):  # potentiative
                        dpos = torch.cat((dpost_reg, dpre_reg), 0)
                        dneg = torch.cat((dpost_inv, dpre_inv), 0)

                # accumulate update
                cell.updater.weight = (
                    state.batchreduce(dpos, 0) if dpos.numel() else None,
                    state.batchreduce(dneg, 0) if dneg.numel() else None,
                )

            else:
                # scale and reduce partial updates
                dpost = state.batchreduce(dpost, 0) * abs(signal * scale)
                dpre = state.batchreduce(dpre, 0) * abs(signal * scale)

                # accumulate partials with mode condition
                match (state.lr_post * signal >= 0, state.lr_pre * signal >= 0):
                    case (False, False):  # depressive
                        cell.updater.weight = (None, dpost + dpre)
                    case (False, True):  # anti-hebbian
                        cell.updater.weight = (dpre, dpost)
                    case (True, False):  # hebbian
                        cell.updater.weight = (dpost, dpre)
                    case (True, True):  # potentiative
                        cell.updater.weight = (dpost + dpre, None)
