from .. import IndependentCellTrainer
from ... import Module
from ..._internal import argtest
from ...neural import Cell
from ...observe import (
    StateMonitor,
    EventReducer,
)
import torch
from typing import Any, Callable


class DelayAdjustedSTDP(IndependentCellTrainer):
    r"""Delay-adjusted pair-based spike-timing dependent plasticity trainer.

    .. math::
        \begin{align*}
            w(t + \Delta t) - w(t) &=
            \eta_+ \exp\left(-\frac{\lvert t_\Delta(t) \rvert}{\tau_+} \right) [t_\Delta(t) \geq 0] \\
            &+ \eta_- \exp\left(-\frac{\lvert t_\Delta(t) \rvert}{\tau_-} \right) [t_\Delta(t) < 0] \\
            t_\Delta(t) &= t^f_\text{post} - t^f_\text{pre} - d(t)
        \end{align*}

    Where:

    Times :math:`t` and :math:`t_n^f` are the current time and the time of the most
    recent spike from neuron :math:`n` respectively, :math:`\Delta t` is the duration of
    the simulation step, and :math:`d(t)` are the learned delays.

    The signs of the learning rates :math:`\eta_+` and :math:`\eta_-`
    control which terms are potentiative and which terms are depressive. The terms can
    be scaled for weight dependence on updating.

    +-------------------+----------------------------+----------------------------+------------------------+------------------------+
    | Mode              | :math:`\text{sgn}(\eta_+)` | :math:`\text{sgn}(\eta_-)` | LTP Term(s)            | LTD Term(s)            |
    +===================+============================+============================+========================+========================+
    | Hebbian           | :math:`+`                  | :math:`-`                  | :math:`\eta_+`         | :math:`\eta_-`         |
    +-------------------+----------------------------+----------------------------+------------------------+------------------------+
    | Anti-Hebbian      | :math:`-`                  | :math:`+`                  | :math:`\eta_-`         | :math:`\eta_+`         |
    +-------------------+----------------------------+----------------------------+------------------------+------------------------+
    | Potentiative Only | :math:`+`                  | :math:`+`                  | :math:`\eta_+, \eta_-` | None                   |
    +-------------------+----------------------------+----------------------------+------------------------+------------------------+
    | Depressive Only   | :math:`-`                  | :math:`-`                  | None                   | :math:`\eta_+, \eta_-` |
    +-------------------+----------------------------+----------------------------+------------------------+------------------------+

    Args:
        lr_pos (float): learning rate for updates when the last postsynaptic spike
            was more recent, :math:`\eta_+`.
        lr_neg (float): learning rate for updates when the last presynaptic spike
            was more recent, :math:`\eta_-`.
        tc_pos (float): time constant of exponential decay of adjusted trace when,
            the last postsynaptic was more recent, :math:`\tau_+`, in :math:`ms`.
        tc_neg (float): time constant of exponential decay of adjusted trace when,
            the last presynaptic was more recent, :math:`\tau_-`, in :math:`ms`.
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
        :ref:`zoo/learning-stdp:Delay-Adjusted Spike-Timing Dependent Plasticity (Delay-Adjusted STDP)` in the zoo.
    """

    def __init__(
        self,
        lr_pos: float,
        lr_neg: float,
        tc_pos: float,
        tc_neg: float,
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
        self.lr_pos = float(lr_pos)
        self.lr_neg = float(lr_neg)
        self.tc_pos = argtest.gt("tc_pos", tc_pos, 0, float)
        self.tc_neg = argtest.gt("tc_neg", tc_neg, 0, float)
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

        lr_pos = kwargs.get("lr_pos", self.lr_pos)
        lr_neg = kwargs.get("lr_neg", self.lr_neg)
        tc_pos = kwargs.get("tc_pos", self.tc_pos)
        tc_neg = kwargs.get("tc_neg", self.tc_neg)
        interp_tolerance = kwargs.get("interp_tolerance", self.tolerance)
        batch_reduction = kwargs.get("batch_reduction", self.batchreduce)
        inplace = kwargs.get("inplace", self.inplace)

        state.lr_pos = float(lr_pos)
        state.lr_neg = float(lr_neg)
        state.tc_pos = argtest.gt("tc_pos", tc_pos, 0, float)
        state.tc_neg = argtest.gt("tc_neg", tc_neg, 0, float)
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
            lr_pos (float): learning rate for updates when the last postsynaptic spike
                was more recent.
            lr_neg (float): learning rate for updates when the last presynaptic spike
                was more recent.
            tc_pos (float): time constant of exponential decay of adjusted trace when,
                the last postsynaptic was more recent.
            tc_neg (float): time constant of exponential decay of adjusted trace when,
                the last presynaptic was more recent.
            interp_tolerance (float): maximum difference in time from an observation
                to treat as co-occurring.
            batch_reduction (Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor]):
                function to reduce updates over the batch dimension.
            inplace (bool, optional): if :py:class:`~inferno.RecordTensor` write operations
                should be performed in-place. Defaults to ``False``.

        Returns:
            IndependentCellTrainer.Unit: specified cell, auxiliary state, and monitors.

        Important:
            Any specified keyword arguments will override the default hyperparameters
            set on initialization. See :py:class:`DelayAdjustedSTDP` for details.
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
            t_delta_abs = t_delta.abs()

            # partial updates
            dpos = state.batchreduce(
                (
                    torch.exp(t_delta_abs / (-state.tc_pos))
                    * (abs(state.lr_pos) * (t_delta >= 0).to(dtype=t_delta_abs.dtype))
                ).nansum(-1),
                0,
            )
            dneg = state.batchreduce(
                (
                    torch.exp(t_delta_abs / (-state.tc_neg))
                    * (abs(state.lr_neg) * (t_delta < 0).to(dtype=t_delta_abs.dtype))
                ).nansum(-1),
                0,
            )

            # accumulate partials with mode condition
            match (state.lr_pos >= 0, state.lr_neg >= 0):
                case (False, False):  # depressive
                    cell.updater.weight = (None, dpos + dneg)
                case (False, True):  # anti-hebbian
                    cell.updater.weight = (dneg, dpos)
                case (True, False):  # hebbian
                    cell.updater.weight = (dpos, dneg)
                case (True, True):  # potentiative
                    cell.updater.weight = (dpos + dneg, None)


class DelayAdjustedSTDPD(IndependentCellTrainer):
    r"""Delay-adjusted pair-based spike-timing dependent plasticity delay trainer.

    .. math::
        \begin{align*}
            d(t + \Delta t) - d(t) &=
            \eta_- \exp\left(-\frac{\lvert t_\Delta(t) \rvert}{\tau_-} \right) [t_\Delta(t) \geq 0] \\
            &+ \eta_+ \exp\left(-\frac{\lvert t_\Delta(t) \rvert}{\tau_+} \right) [t_\Delta(t) < 0] \\
            t_\Delta(t) &= t^f_\text{post} - t^f_\text{pre} - d(t)
        \end{align*}

    Where:

    Times :math:`t` and :math:`t_n^f` are the current time and the time of the most
    recent spike from neuron :math:`n` respectively, :math:`\Delta t` is the duration of
    the simulation step, and :math:`d(t)` are the learned delays.

    The signs of the learning rates :math:`\eta_-` and :math:`\eta_+`
    control which terms are potentiative and which terms are depressive. The terms
    can be scaled for weight dependence on updating.

    +-------------------+----------------------------+----------------------------+------------------------+------------------------+
    | Mode              | :math:`\text{sgn}(\eta_-)` | :math:`\text{sgn}(\eta_+)` | Potentiative Term(s)   | Depressive Term(s)     |
    +===================+============================+============================+========================+========================+
    | Hebbian           | :math:`-`                  | :math:`+`                  | :math:`\eta_-`         | :math:`\eta_+`         |
    +-------------------+----------------------------+----------------------------+------------------------+------------------------+
    | Anti-Hebbian      | :math:`+`                  | :math:`-`                  | :math:`\eta_+`         | :math:`\eta_-`         |
    +-------------------+----------------------------+----------------------------+------------------------+------------------------+
    | Potentiative Only | :math:`-`                  | :math:`-`                  | :math:`\eta_-, \eta_+` | None                   |
    +-------------------+----------------------------+----------------------------+------------------------+------------------------+
    | Depressive Only   | :math:`+`                  | :math:`+`                  | None                   | :math:`\eta_-, \eta_+` |
    +-------------------+----------------------------+----------------------------+------------------------+------------------------+

    Args:
        lr_neg (float): learning rate for updates when the last postsynaptic spike
            was more recent, :math:`\eta_-`.
        lr_pos (float): learning rate for updates when the last presynaptic spike
            was more recent, :math:`\eta_+`.
        tc_neg (float): time constant of exponential decay of adjusted trace when,
            the last postsynaptic was more recent, :math:`\tau_-`, in :math:`ms`.
        tc_pos (float): time constant of exponential decay of adjusted trace when,
            the last presynaptic was more recent, :math:`\tau_+`, in :math:`ms`.
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
        :ref:`zoo/learning-stdp:Delay-Adjusted Spike-Timing Dependent Plasticity of Delays (Delay-Adjusted STDPD)` in the zoo.
    """

    def __init__(
        self,
        lr_neg: float,
        lr_pos: float,
        tc_neg: float,
        tc_pos: float,
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
        self.lr_neg = float(lr_neg)
        self.lr_pos = float(lr_pos)
        self.tc_neg = argtest.gt("tc_neg", tc_neg, 0, float)
        self.tc_pos = argtest.gt("tc_pos", tc_pos, 0, float)
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

        lr_neg = kwargs.get("lr_neg", self.lr_neg)
        lr_pos = kwargs.get("lr_pos", self.lr_pos)
        tc_neg = kwargs.get("tc_neg", self.tc_neg)
        tc_pos = kwargs.get("tc_pos", self.tc_pos)
        interp_tolerance = kwargs.get("interp_tolerance", self.tolerance)
        batch_reduction = kwargs.get("batch_reduction", self.batchreduce)
        inplace = kwargs.get("inplace", self.inplace)

        state.lr_neg = float(lr_neg)
        state.lr_pos = float(lr_pos)
        state.tc_neg = argtest.gt("tc_neg", tc_neg, 0, float)
        state.tc_pos = argtest.gt("tc_pos", tc_pos, 0, float)
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
            lr_neg (float): learning rate for updates when the last postsynaptic spike
                was more recent.
            lr_pos (float): learning rate for updates when the last presynaptic spike
                was more recent.
            tc_neg (float): time constant of exponential decay of adjusted trace when,
                the last postsynaptic was more recent.
            tc_pos (float): time constant of exponential decay of adjusted trace when,
                the last presynaptic was more recent.
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
            set on initialization. See :py:class:`DelayAdjustedSTDP` for details.
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
            t_delta_abs = t_delta.abs()

            # partial updates
            dneg = state.batchreduce(
                (
                    torch.exp(t_delta_abs / (-state.tc_neg))
                    * (abs(state.lr_neg) * (t_delta >= 0).to(dtype=t_delta_abs.dtype))
                ).nansum(-1),
                0,
            )
            dpos = state.batchreduce(
                (
                    torch.exp(t_delta_abs / (-state.tc_pos))
                    * (abs(state.lr_pos) * (t_delta < 0).to(dtype=t_delta_abs.dtype))
                ).nansum(-1),
                0,
            )

            # accumulate partials with mode condition
            match (state.lr_neg < 0, state.lr_pos < 0):
                case (True, True):  # potentiative
                    cell.updater.delay = (None, dpos + dneg)
                case (True, False):  # hebbian
                    cell.updater.delay = (dpos, dneg)
                case (False, True):  # anti-hebbian
                    cell.updater.delay = (dneg, dpos)
                case (False, False):  # depressive
                    cell.updater.delay = (dpos + dneg, None)
