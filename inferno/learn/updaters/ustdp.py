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
    r"""Spike-timing dependent plasticity updated.

    Args:
        step_time (float): _description_
        lr_post (float): learning rate for updates on postsynaptic spikes, potentiative.
        lr_pre (float): learning rate for updates on presynaptic spikes, depressive.
        tc_post (float): time constant for exponential decay of postsynaptic trace,
            in :math:`ms`.
        tc_pre (float): time constant for exponential decay of presynaptic trace,
            in :math:`ms`.
        delayed (bool, optional): if the updater should assume learned delays, if
            present, may change. Defaults to False.
        trace_mode (Literal["cumulative", "nearest"], optional): _description_. Defaults to "cumulative".
        bounding_mode (Literal["soft", "hard"] | None, optional): _description_. Defaults to None.
        wmin (float | None, optional): _description_. Defaults to None.
        wmax (float | None, optional): _description_. Defaults to None.
        wddepexp (float, optional): _description_. Defaults to 1.0.
        wdpotexp (float, optional): _description_. Defaults to 1.0.
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
        trace_mode: Literal["cumulative", "nearest"] = "cumulative",
        bounding_mode: Literal["soft", "hard"] | None = None,
        interpolation_tolerance: float = 1e-7,
        wmin: float | None = None,
        wmax: float | None = None,
        wddepexp: float = 1.0,
        wdpotexp: float = 1.0,
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

        if bounding_mode:
            bounding_mode = bounding_mode.lower()
            if bounding_mode not in ("soft", "hard"):
                raise ValueError(
                    "`bounding_mode`, if not None, must be one of 'soft' or 'hard'; "
                    f"received {bounding_mode}."
                )

        # updater hyperparameters
        self.step_time = numeric_limit('`step_time`', step_time, 0, 'gt', float)
        self.lr_post = float(lr_post)
        self.lr_pre = float(lr_pre)
        self.tc_post = numeric_limit('`tc_post`', tc_post, 0, 'gt', float)
        self.tc_pre = numeric_limit('`tc_pre`', tc_pre, 0, 'gt', float)
        self.delayed = delayed
        self.trace = trace_mode
        self.bounding = bounding_mode
        self.tolerance = float(interpolation_tolerance)

        # case-specific hyperparamters
        if self.bounding:
            if wmin is None and max is None:
                raise TypeError("`wmin` and `wmax` cannot both be None.")
            if wmin is not None and wmax is not None and wmin >= wmax:
                raise ValueError(
                    f"received `wmax` of {wmax} which not greater than "
                    f"`wmin` of {wmin}."
                )
            self.wmin = None if wmin is None else float(wmin)
            self.wmax = None if wmin is None else float(wmax)

        if self.bounding == 'soft':
            self.wd_lowerb_exp = float(wddepexp)
            self.wd_upperb_exp = float(wdpotexp)

    def forward(self) -> None:
        # iterate over trainable layers
        for layer in self.trainables:
            # skip if layer not in training mode
            if not layer.training or not self.training:
                continue

            # post and pre synaptic traces
            a_post = self.get_monitor(layer, "trace_post").peek()
            a_pre = (
                self.get_monitor(layer, "trace_pre").view(
                    layer.connection.selector, self.tolerance
                )
                if self.delayed and layer.connection.delayedby is not None
                else self.get_monitor(layer, "trace_pre").peek()
            )

            # post and pre synaptic spikes
            i_post = self.get_monitor(layer, "spike_post").peek()
            i_pre = self.get_monitor(layer, "spike_pre").peek()

            # reshape postsynaptic
            a_post = layer.postsyn_receptive(a_post)
            i_post = layer.postsyn_receptive(i_post)

            # reshape presynaptic
            a_pre = layer.presyn_receptive(a_pre)
            i_pre = layer.presyn_currents(i_pre)

            # base update
            update_ltp = (
                torch.mean(torch.sum(i_post * a_pre, dim=-1), dim=0) * self.lrpost
            )
            update_ltd = (
                torch.mean(torch.sum(i_pre * a_post, dim=-1), dim=0) * self.lrpre
            )

            # apply bounding if specified
            match self.bounding:
                case "soft":
                    if self.wmin is not None:
                        update_ltd *= lf.soft_bounding_dep(
                            layer.connection.weight, self.wmin, self.wd_lowerb_exp
                        )
                    if self.wmax is not None:
                        update_ltp *= lf.soft_bounding_pot(
                            layer.connection.weight, self.wmax, self.wd_upperb_exp
                        )
                case "hard":
                    if self.wmin is not None:
                        update_ltd *= lf.hard_bounding_dep(
                            layer.connection.weight, self.wmin
                        )
                    if self.wmax is not None:
                        update_ltp *= lf.hard_bounding_pot(
                            layer.connection.weight, self.wmax
                        )

            # update weights
            layer.connection.weight = layer.connection.weight + update_ltp - update_ltd

    def add_monitors(self, trainable: Layer):
        # add trainable using superclass method
        LayerwiseUpdater.add_trainable(self, trainable)

        # don't add anything if monitors are already associated with this trainable
        if len(self.get_monitors(trainable)):
            return

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

        # postsynaptic trace monitor (weights LTD)
        self.add_monitor(
            trainable,
            "trace_post",
            StateMonitor(
                reducer=reducer_cls(
                    self.dt,
                    self.tc_post,
                    amplitude=1.0,
                    target=True,
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
                    amplitude=1.0,
                    target=True,
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

    @property
    def dt(self) -> float:
        return self.step_time

    @dt.setter
    def dt(self, value: float) -> None:
        # assign new step time
        self.step_time = numeric_limit('`step_time`', value, 0, 'gt', float)

        # update monitors accordingly
        for monitor, _ in self.monitors:
            monitor.reducer.dt = self.step_time

    @property
    def lrpost(self) -> float:
        return self.lr_post

    @lrpost.setter
    def lrpost(self, value: float) -> None:
        self.lr_post = float(value)

    @property
    def lrpre(self) -> float:
        return self.lr_pre

    @lrpre.setter
    def lrpre(self, value: float) -> None:
        self.lr_pre = float(value)
