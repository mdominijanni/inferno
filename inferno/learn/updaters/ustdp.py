from .. import LayerwiseUpdater
from .. import functional as lf
from inferno._internal import numeric_limit
from inferno.neural import Layer
from inferno.observe import (
    StateMonitor,
    CumulativeTraceReducer,
    NearestTraceReducer,
    PassthroughReducer,
)
import torch
from typing import Literal


class STDP(LayerwiseUpdater):
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
        delayed (bool, optional): if the updater should assume learned delays, if
            present, may change. Defaults to False.
        interp_tolerance (float): maximum difference in time from an observation
            to treat as co-occurring, in :math:`\text{ms}`. Defaults to 0.0.
        trace_mode (Literal["cumulative", "nearest"], optional): method to use for
            calculating spike traces. Defaults to "cumulative".
        wd_lower (~inferno.learn.functional.WeightDependence | None, optional):
            function for applying weight dependence on a lower bound, no dependence if
            None. Defaults to None.
        wd_upper (~inferno.learn.functional.WeightDependence | None, optional):
            function for applying weight dependence on an bound, no dependence if
            None. Defaults to None.
        wmin (float | None, optional): lower bound for weights, :math:`w_\text{min}`.
            Defaults to None.
        wmax (float | None, optional): upper bound for weights., :math:`w_\text{max}`
            Defaults to None.

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
        wd_lower: lf.WeightDependence | None = None,
        wd_upper: lf.WeightDependence | None = None,
        wmin: float | None = None,
        wmax: float | None = None,
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

        # validate weight dependence
        if wd_lower and wmin is None:
            raise RuntimeError("`wmin` cannot be None if `wd_lower` is specified.")
        if wd_upper and wmax is None:
            raise RuntimeError("`wmax` cannot be None if `wd_upper` is specified.")

        # updater hyperparameters
        self.step_time = numeric_limit("`step_time`", step_time, 0, "gt", float)
        self.lr_post = float(lr_post)
        self.lr_pre = float(lr_pre)
        self.tc_post = numeric_limit("`tc_post`", tc_post, 0, "gt", float)
        self.tc_pre = numeric_limit("`tc_pre`", tc_pre, 0, "gt", float)
        self.delayed = delayed
        self.tolerance = float(interp_tolerance)
        self.trace = trace_mode
        self.wdlower = wd_lower
        self.wdupper = wd_upper
        self.wmin = float(wmin)
        self.wmax = float(wmax)

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
        self.step_time = numeric_limit("`step_time`", value, 0, "gt", float)

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
                self.get_monitor(layer, "trace_pre").view(
                    conn.selector, self.tolerance
                )
                if self.delayed and conn.delayedby is not None
                else self.get_monitor(layer, "trace_pre").peek()
            )

            # post and pre synaptic spikes
            i_post = self.get_monitor(layer, "spike_post").peek()
            i_pre = self.get_monitor(layer, "spike_pre").peek()

            # reshape postsynaptic
            x_post = layer.postsyn_receptive(x_post)
            i_post = layer.postsyn_receptive(i_post)

            # reshape presynaptic
            x_pre = layer.presyn_receptive(x_pre)
            i_pre = layer.presyn_receptive(i_pre)

            # update amplitudes
            a_pos, a_neg = 1.0, 1.0

            # apply bounding factors
            if self.wdupper:
                a_pos = self.wdupper(conn.weight, self.wmax, a_pos)
            if self.wdlower:
                a_neg = self.wdlower(conn.weight, self.wmin, a_neg)

            # apply learning rates
            a_pos, a_neg = a_pos * self.lrpost, a_neg * self.lrpre

            # mask traces with spikes, reduce dimensionally, apply amplitudes
            dw_pos = torch.mean(torch.sum(i_post * x_pre, dim=-1), dim=0) * a_pos
            dw_neg = torch.mean(torch.sum(i_pre * x_post, dim=-1), dim=0) * a_neg

            # update weights
            conn.weight = conn.weight + dw_pos - dw_neg
