from .. import LayerwiseUpdater, LayerwiseTrainer
from ... import Module
from inferno._internal import numeric_limit, argtest
from inferno.neural import Layer, Cell
from inferno.observe import (
    StateMonitor,
    CumulativeTraceReducer,
    NearestTraceReducer,
    PassthroughReducer,
)
import torch
from typing import Any, Callable, Literal


class STDP(LayerwiseTrainer):
    r"""Spike-timing dependent plasticity updater.

    .. math::
        \Delta w = A_+ x_\text{pre} \bigl[t = t^f_\text{post}\bigr] +
        A_- x_\text{post} \bigl[t = t^f_\text{pre}\bigr]

    Where:

    .. math::
        x^{(t)} =
        \begin{cases}
            1 + x^{(t-\Delta t)} \exp(-\Delta t / \tau) & t = t^f \text{ and cumulative trace} \\
            1 & t = t^f \text{ and nearest trace} \\
            x^{(t-\Delta t)} \exp(-\Delta t / \tau) & t \neq t^f
        \end{cases}

    :math:`t` and :math:`t^f` are the current time and the time of the most recent
    spike, respectively.

    The terms :math:`A_+` and :math:`A_-` are equal to the learning rates
    :math:`\eta_\text{post}` and :math:`\eta_\text{pre}` respectively, although they
    may be scaled by weight dependence at the updater level. The "mode" changes based on
    the sign of the learning rates, and updates are applied based on any potentiative
    and depressive components.

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
        delayed (bool, optional): if the updater should assume that learned delays, if
            present, may change. Defaults to False.
        interp_tolerance (float): maximum difference in time from an observation
            to treat as co-occurring, in :math:`\text{ms}`. Defaults to 0.0.
        trace_mode (Literal["cumulative", "nearest"], optional): method to use for
            calculating spike traces. Defaults to "cumulative".
        batch_reduction (Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] | None):
            function to reduce updates over the batch dimension, :py:func:`torch.mean`
            when None. Defaults to None.
        name (str | None, optional): name of the trainer, for layer monitor pooling,
            generated uniquely when None. Defaults to None.

    Important:
        The constructor arguments (except for ``name``) are hyperparameters for STDP
        and can be overridden on a cell-by-cell basis.

    Note:
        ``batch_reduction`` can be one of the functions in PyTorch including but not
        limited to :py:func:`torch.sum`, :py:func:`torch.max` and :py:func:`torch.max`.
        A custom function with similar behavior can also be passed in. Like with the
        included function, it should not keep the original dimensions by default.

    See Also:
        For more details and references, visit
        :ref:`zoo/learning-stdp:Spike Timing-Dependent Plasticity (STDP)`,
        :ref:`zoo/learning-stdp:Weight Dependence, Soft Bounding`, and
        :ref:`zoo/learning-stdp:Weight Dependence, Hard Bounding` in the zoo.
    """

    def __init__(
        self,
        step_time: float,
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
        name: str | None = None,
        **kwargs,
    ):
        # call superclass constructor
        LayerwiseTrainer.__init__(self, name, **kwargs)

        # default hyperparameters
        self.step_time = argtest.gt("step_time", step_time, 0, float)
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

    def _hyperparam_config(self, **kwargs: Any) -> dict[str, Any]:
        r"""Module Internal: construct a dictionary of hyperparameters with defaults.

        Args:
            **kwargs (Any) hyperparameter overrides.

        Returns:
            dict[str, Any]: dictionary of hyperparameters
        """
        # hyperparameter store
        d = {}

        # validation functions
        gt0 = lambda n, h: argtest.gt(n, h, 0, float)  # noqa:E731;
        gte0 = lambda n, h: argtest.gte(n, h, 0, float)  # noqa:E731;
        trace_valid = lambda n, h: argtest.oneof(  # noqa:E731;
            n, h, "cumulative", "nearest", op=(lambda x: x.lower())
        )
        identf = lambda n, h: h  # noqa:E731;
        floatf = lambda n, h: float(h)  # noqa:E731;

        # iterate over expected keys
        hparams = {
            "step_time": ("step_time", gt0),
            "lr_post": ("lr_post", floatf),
            "lr_pre": ("lr_pre", floatf),
            "tc_post": ("tc_post", gt0),
            "tc_pre": ("tc_pre", gt0),
            "delayed": ("delayed", floatf),
            "interp_tolerance": ("tolerance", gte0),
            "trace_mode": ("trace", trace_valid),
            "batch_reduction": ("batchreduce", identf),
        }

        for key, (param, func) in hparams.items():
            if key in kwargs:
                d[param] = func(kwargs[key])
            else:
                d[param] = getattr(self, param)

        return d

    def add_cell(
        self,
        name: str,
        cell: Cell,
        **kwargs: Any,
    ) -> tuple[Cell, Module]:
        r"""Adds a cell with required state.

        Args:
            name (str): name of the trainable to add.
            cell (Cell): cell to add.

        Keyword Args:
            step_time (float): length of a simulation time step.
            lr_post (float): learning rate for updates on postsynaptic spike updates.
            lr_pre (float): learning rate for updates on presynaptic spike updates.
            tc_post (float): time constant for exponential decay of postsynaptic trace.
            tc_pre (float): time constant for exponential decay of presynaptic trace.
            delayed (bool): if the updater should assume that learned delays,
                if present, may change.
            interp_tolerance (float): maximum difference in time from an observation
                to treat as co-occurring.
            trace_mode (Literal["cumulative", "nearest"]): method to use for
                calculating spike traces.
            batch_reduction (Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor]):
                function to reduce updates over the batch dimension.

        Returns:
            tuple[Cell, Module]: added cell and required state.

        Important:
            Any specified keyword arguments will override the default hyperparameters
            set on initialization. See :py:class:`STDP` for details.
        """
        # add the cell with additional hyperparameters
        cell, state = LayerwiseTrainer.add_cell(
            self, name, cell, **self._hyperparam_config(**kwargs)
        )

        # TODO: add appropriate monitors, unpool trace
        _ = self.add_monitor(name, "trace_post", "neuron.spike", StateMonitor.partialconstructor())


class STDPv1(LayerwiseUpdater):
    r"""Spike-timing dependent plasticity updater.

    .. math::

        \Delta w = A_+ x_\text{pre} \bigl[t = t^f_\text{post}\bigr] -
        A_- x_\text{post} \bigl[t = t^f_\text{pre}\bigr]

    Where:

    .. math::

        x^{(t)} &=
        \begin{cases}
            1 + x^{(t-\Delta t)} \exp(-\Delta t / \tau) & t = t^f \text{ and cumulative trace} \\
            1 & t = t^f \text{ and nearest trace} \\
            x^{(t-\Delta t)} \exp(-\Delta t / \tau) & t \neq t^f
        \end{cases}

    :math:`t` and :math:`t^f` are the current time and the time of the most recent
    spike, respectively. :math:`A_+` and :math:`A_-` vary with weight dependence, but
    by default are.

    .. math::

        A_+ = \eta_\text{post} \qquad A_- = \eta_\text{pre}

    Args:
        step_time (float): length of a simulation time step, :math:`\Delta t`,
            in :math:`\text{ms}`.
        lr_post (float): learning rate for updates on postsynaptic spike updates (LTP),
            :math:`\eta_\text{post}`.
        lr_pre (float): learning rate for updates on presynaptic spike updates (LTD),
            :math:`\eta_\text{pre}`.
        tc_post (float): time constant for exponential decay of postsynaptic trace,
            :math:`tau_\text{post}`, in :math:`ms`.
        tc_pre (float): time constant for exponential decay of presynaptic trace,
            :math:`tau_\text{pre}`, in :math:`ms`.
        delayed (bool, optional): if the updater should assume that learned delays, if
            present, may change. Defaults to False.
        interp_tolerance (float): maximum difference in time from an observation
            to treat as co-occurring, in :math:`\text{ms}`. Defaults to 0.0.
        trace_mode (Literal["cumulative", "nearest"], optional): method to use for
            calculating spike traces. Defaults to "cumulative".
        wd_lower (~inferno.learn.functional.UpdateBounding | None, optional):
            callable for applying weight dependence on a lower bound, no dependence if
            None. Defaults to None.
        wd_upper (~inferno.learn.functional.UpdateBounding | None, optional):
            callable for applying weight dependence on an bound, no dependence if
            None. Defaults to None.
        batch_reduction (Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] | None):
            function to reduce updates over the batch dimension, :py:func:`torch.mean`
            when None. Defaults to None.

    Note:
        ``batch_reduction`` can be one of the functions in PyTorch including but not
        limited to :py:func:`torch.sum`, :py:func:`torch.max` and :py:func:`torch.max`.
        A custom function with similar behavior can also be passed in. Like with the
        included function, it should not keep the original dimensions by default.

    See Also:
        For more details and references, visit
        :ref:`zoo/learning-stdp:Spike Timing-Dependent Plasticity (STDP)`,
        :ref:`zoo/learning-stdp:Weight Dependence, Soft Bounding`, and
        :ref:`zoo/learning-stdp:Weight Dependence, Hard Bounding` in the zoo.
    """

    def __init__(
        self,
        *layers: Layer,
        step_time: float,
        lr_post: float,
        lr_pre: float,
        tc_post: float,
        tc_pre: float,
        delayed: bool = False,
        interp_tolerance: float = 0.0,
        trace_mode: Literal["cumulative", "nearest"] = "cumulative",
        wd_lower: lf.UpdateBounding | None = None,
        wd_upper: lf.UpdateBounding | None = None,
        batch_reduction: (
            Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] | None
        ) = None,
        **kwargs,
    ):
        # call superclass constructor, registers monitors
        LayerwiseUpdater.__init__(self, *layers)

        # validate string parameters
        trace_mode = trace_mode.lower()
        if trace_mode not in ("cumulative", "nearest"):
            raise ValueError(
                "`trace_mode` must be one of: 'cumulative', 'nearest'; "
                f"received {trace_mode}."
            )

        # updater hyperparameters
        self.step_time, e = numeric_limit("step_time", step_time, 0, "gt", float)
        if e:
            raise e
        self.lr_post = float(lr_post)
        self.lr_pre = float(lr_pre)
        self.tc_post, e = numeric_limit("tc_post", tc_post, 0, "gt", float)
        if e:
            raise e
        self.tc_pre, e = numeric_limit("tc_pre", tc_pre, 0, "gt", float)
        if e:
            raise e
        self.delayed = delayed
        self.tolerance = float(interp_tolerance)
        self.trace = trace_mode
        self.wdlower = wd_lower
        self.wdupper = wd_upper
        self.batchreduce = batch_reduction if batch_reduction else torch.mean

    @property
    def dt(self) -> float:
        r"""Length of the simulation time step, in milliseconds.

        Args:
            value (float): new simulation time step length.

        Returns:
            float: present simulation time step length.
        """
        return self.step_time

    @dt.setter
    def dt(self, value: float) -> None:
        # assign new step time
        self.step_time, e = numeric_limit("step_time", value, 0, "gt", float)
        if e:
            raise e

        # update monitors accordingly
        for monitor, _ in self.monitors:
            monitor.reducer.dt = self.step_time

    @property
    def lrpost(self) -> float:
        r"""Learning rate for updates on postsynaptic spikes (potentiative).

        Args:
            value (float): new postsynaptic learning rate.

        Returns:
            float: present postsynaptic learning rate.
        """
        return self.lr_post

    @lrpost.setter
    def lrpost(self, value: float) -> None:
        self.lr_post = float(value)

    @property
    def lrpre(self) -> float:
        r"""Learning rate for updates on postsynaptic spikes (depressive).

        Args:
            value (float): new postsynaptic learning rate.

        Returns:
            float: present postsynaptic learning rate.
        """
        return self.lr_pre

    @lrpre.setter
    def lrpre(self, value: float) -> None:
        self.lr_pre = float(value)

    def add_monitors(self, trainable: Layer) -> bool:
        r"""Associates base layout of monitors required by the updater with the layer.

        Args:
            trainable (Layer): layer to which monitors should be added.

        Returns:
            bool: if the monitors were successfully added.

        Note:
            This adds four prepended monitors named: "spike_post", "spike_pre",
            "trace_post", and "trace_pre". This method will fail to assign monitors
            if a monitor with these names is already associated with ``trainable``.
        """
        # do not alter state if conflicting monitors are associated
        if any(
            name in ("spike_post", "spike_pre", "trace_post", "trace_pre")
            for _, name in self.get_monitors(trainable)
        ):
            return False

        # trace reducer class
        match self.trace:
            case "cumulative":
                reducer_cls = CumulativeTraceReducer
            case "nearest":
                reducer_cls = NearestTraceReducer
            case "_":
                raise RuntimeError(
                    f"an invalid trace mode of '{self.trace}' has been set, expected "
                    "one of: 'cumulative', 'nearest'."
                )

        # like parameters
        mon_kwargs = {
            "as_prehook": False,
            "train_update": True,
            "eval_update": False,
            "prepend": True,
        }
        amp, tgt = 1.0, True

        # postsynaptic trace monitor (weights LTD)
        self.add_monitor(
            trainable,
            "trace_post",
            StateMonitor(
                reducer=reducer_cls(
                    self.dt,
                    self.tc_post,
                    amplitude=amp,
                    target=tgt,
                    history_len=0.0,
                ),
                attr="neuron.spike",
                **mon_kwargs,
            ),
        )

        # postsynaptic spike monitor (triggers LTP)
        self.add_monitor(
            trainable,
            "spike_post",
            StateMonitor(
                reducer=PassthroughReducer(self.dt, history_len=0.0),
                attr="neuron.spike",
                **mon_kwargs,
            ),
        )

        # presynaptic trace monitor (weights LTP)
        delayed = self.delayed and trainable.connection.delayedby is not None
        self.add_monitor(
            trainable,
            "trace_pre",
            StateMonitor(
                reducer=reducer_cls(
                    self.dt,
                    self.tc_pre,
                    amplitude=amp,
                    target=tgt,
                    history_len=(trainable.connection.delayedby if delayed else 0.0),
                ),
                attr=("synapse.spike" if delayed else "connection.synspike"),
                **mon_kwargs,
            ),
        )

        # presynaptic spike monitor (triggers LTD)
        self.add_monitor(
            trainable,
            "spike_pre",
            StateMonitor(
                reducer=PassthroughReducer(self.dt, history_len=0.0),
                attr="connection.synspike",
                **mon_kwargs,
            ),
        )

        return True

    def forward(self) -> None:
        """Processes update for given layers based on current monitor stored data."""
        # iterate over trainable layers
        for layer in self.trainables:
            # skip if layer not in training mode
            if not layer.training or not self.training:
                continue

            # get reference to composed connection
            conn = layer.connection

            # post and pre synaptic traces
            x_post = self.get_monitor(layer, "trace_post").peek()
            x_pre = (
                self.get_monitor(layer, "trace_pre").view(conn.selector, self.tolerance)
                if self.delayed and conn.delayedby is not None
                else self.get_monitor(layer, "trace_pre").peek()
            )

            # post and pre synaptic spikes
            i_post = self.get_monitor(layer, "spike_post").peek()
            i_pre = self.get_monitor(layer, "spike_pre").peek()

            # reshape postsynaptic
            x_post = layer.connection.postsyn_receptive(x_post)
            i_post = layer.connection.postsyn_receptive(i_post)

            # reshape presynaptic
            x_pre = layer.connection.presyn_receptive(x_pre)
            i_pre = layer.connection.presyn_receptive(i_pre)

            # update amplitudes
            a_pos, a_neg = 1.0, 1.0

            # apply bounding factors
            if self.wdupper:
                a_pos = self.wdupper(conn.weight, a_pos)
            if self.wdlower:
                a_neg = self.wdlower(conn.weight, a_neg)

            # apply learning rates
            a_pos, a_neg = a_pos * self.lrpost, a_neg * self.lrpre

            # mask traces with spikes, reduce dimensionally, apply amplitudes
            dw_pos = self.batchreduce(torch.sum(i_post * x_pre, dim=-1), 0) * a_pos
            dw_neg = self.batchreduce(torch.sum(i_pre * x_post, dim=-1), 0) * a_neg

            # update weights
            conn.weight = conn.weight + dw_pos - dw_neg
