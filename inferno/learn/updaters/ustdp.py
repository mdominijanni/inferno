from .. import LayerwiseUpdater, LayerwiseTrainer
from .. import functional as lf
from inferno._internal import numeric_limit, argtest
from inferno.neural import Layer, Trainable
from inferno.observe import (
    StateMonitor,
    CumulativeTraceReducer,
    NearestTraceReducer,
    PassthroughReducer,
)
import torch
from typing import Any, Callable, Literal


class STDP(LayerwiseTrainer):
    def __init__(
        self,
        step_time: float,
        lr_post: float,
        lr_pre: float,
        tc_post: float,
        tc_pre: float,
        hebbian: bool = True,
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

    def _hyperparameter_defaults(
        self,
        step_time: float | None = None,
        lr_post: float | None = None,
        lr_pre: float | None = None,
        tc_post: float | None = None,
        tc_pre: float | None = None,
        hebbian: bool | None = None,
        delayed: bool | None = None,
        interp_tolerance: float | None = None,
        trace_mode: Literal["cumulative", "nearest"] | None = None,
        batch_reduction: (
            Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] | None
        ) = None,
    ) -> dict[str, Any]:
        r"""Module Internal: construct a dictionary of hyperparameters with defaults.

        Args:
            step_time (float | None, optional): length of a simulation time step. Defaults to None.
            lr_post (float | None, optional): learning rate for updates on postsynaptic spike updates (LTP). Defaults to None.
            lr_pre (float | None, optional): _description_. Defaults to None.
            tc_post (float | None, optional): _description_. Defaults to None.
            tc_pre (float | None, optional): _description_. Defaults to None.
            hebbian (bool | None, optional): _description_. Defaults to None.
            delayed (bool | None, optional): _description_. Defaults to None.
            interp_tolerance (float | None, optional): _description_. Defaults to None.
            trace_mode (Literal[&quot;cumulative&quot;, &quot;nearest&quot;] | None, optional): _description_. Defaults to None.
            batch_reduction (Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor]  |  None, optional): _description_. Defaults to None.

        Returns:
            dict[str, Any]: _description_
        """

        # conditional dictionary update
        def nullc(store, value, attr, fn):
            if value is None:
                store[attr] = getattr(self, attr)
            else:
                store[attr] = fn(value)

        # alias argtest for compactness
        gt, gte, oneof = argtest.gt, argtest.gte, argtest.oneof

        # hyperparameter store
        d = {}

        # fill store
        nullc(d, step_time, "step_time", lambda x: gt("step_time", x, 0, float))
        nullc(d, lr_post, "lr_post", lambda x: float(x))
        nullc(d, lr_pre, "lr_pre", lambda x: float(x))
        nullc(d, tc_post, "tc_post", gt("tc_post", tc_post, 0, float))
        nullc(d, tc_pre, "tc_pre", gt("tc_pre", tc_pre, 0, float))
        nullc(d, delayed, "delayed", lambda x: bool(x))
        nullc(
            d,
            interp_tolerance,
            "tolerance",
            gte("interp_tolerance", interp_tolerance, 0, float),
        )

        if trace_mode is None:
            d["trace"] = self.trace
        else:
            d["trace"] = argtest.oneof(
                "trace_mode",
                trace_mode,
                "cumulative",
                "nearest",
                op=(lambda x: x.lower()),
            )
        if batch_reduction is None:
            d["batchreduce"] = self.batchreduce
        else:
            d["batchreduce"] = batch_reduction if batch_reduction else torch.mean

        return d

    def add_trainable(
        self,
        name: str,
        trainable: Trainable,
        *,
        step_time: float | None = None,
        lr_post: float | None = None,
        lr_pre: float | None = None,
        tc_post: float | None = None,
        tc_pre: float | None = None,
        delayed: bool | None = None,
        interp_tolerance: float | None = None,
        trace_mode: Literal["cumulative", "nearest"] | None = None,
        batch_reduction: (
            Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] | None
        ) = None,
        **kwargs,
    ) -> Trainable:
        d = self._hyperparameter_defaults(
            step_time,
            lr_post,
            lr_pre,
            tc_post,
            tc_pre,
            delayed,
            interp_tolerance,
            trace_mode,
            batch_reduction,
        )

        return LayerwiseTrainer.add_trainable(self, name, trainable, **d, **kwargs)


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
