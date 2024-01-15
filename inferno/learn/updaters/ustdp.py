from .. import LayerwiseUpdater
from .. import functional as lf
from inferno.neural import Layer
from inferno.observe import (
    StateMonitor,
    OutputMonitor,
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
        lr_post (float): _description_
        lr_pre (float): _description_
        tc_post (float): _description_
        tc_pre (float): _description_
        delayed (bool, optional): if the updater should assume learned delays, if present, may change.
            Defaults to False.
        trace_mode (Literal["cumulative", "nearest"], optional): _description_. Defaults to "cumulative".
        bounding_mode (Literal["soft", "hard"] | None, optional): _description_. Defaults to None.
        wmin (float | None, optional): _description_. Defaults to None.
        wmax (float | None, optional): _description_. Defaults to None.
        wdepexp (float | None, optional): _description_. Defaults to None.
        wpotexp (float | None, optional): _description_. Defaults to None.

    Raises:
        RuntimeError: _description_
        RuntimeError: _description_
        RuntimeError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
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
        wdepexp: float | None = None,
        wpotexp: float | None = None,
        **kwargs,
    ):
        # call superclass constructor
        LayerwiseUpdater.__init__(self, *layers)

        # test hyperparameter validity
        if step_time <= 0:
            raise RuntimeError(f"step time must be positive, received {step_time}")
        if tc_post <= 0:
            raise RuntimeError(
                f"postsynaptic time constant must be positive, received {tc_post}"
            )
        if tc_pre <= 0:
            raise RuntimeError(
                f"presynaptic time constant must be positive, received {tc_pre}"
            )

        trace_mode = trace_mode.lower()
        if trace_mode not in ("cumulative", "nearest"):
            raise ValueError(
                f"trace mode must be 'cumulative' or 'nearest', received {trace_mode}"
            )

        if bounding_mode:
            bounding_mode = bounding_mode.lower()
            if bounding_mode not in ("soft", "hard"):
                raise ValueError(
                    f"bounding mode, if not None, must be 'soft' or 'hard', received {bounding_mode}"
                )

        # register hyperparameters
        self.delayed = delayed
        self.trace = trace_mode
        self.bounding = bounding_mode
        self.tolerance = float(interpolation_tolerance)
        self.register_extra("step_time", float(step_time))
        self.register_extra("lr_post", float(lr_post))
        self.register_extra("lr_pre", float(lr_pre))
        self.register_extra("tc_post", float(tc_post))
        self.register_extra("tc_pre", float(tc_pre))

        # construct monitors
        for layer in self.trainables.values():
            # trace monitors
            match trace_mode.lower():
                # cumulative trace monitors
                case "cumulative":
                    # presynaptic
                    self.add_monitor(
                        layer,
                        "Apre",
                        StateMonitor(
                            reducer=CumulativeTraceReducer(
                                self.step_time,
                                self.tc_pre,
                                amplitude=1.0,
                                target=True,
                                history_len=(
                                    self.delayed if self.delayed is not None else 0.0
                                ),
                            ),
                            attr="connection.synspike",
                            train_update=True,
                            eval_update=False,
                        ),
                    )

                    # postsynaptic
                    self.add_monitor(
                        layer,
                        "Apost",
                        OutputMonitor(
                            reducer=CumulativeTraceReducer(
                                self.step_time,
                                self.tc_post,
                                amplitude=1.0,
                                target=True,
                                history_len=(
                                    self.delayed if self.delayed is not None else 0.0
                                ),
                            ),
                            train_update=True,
                            eval_update=False,
                        ),
                    )

                # nearest trace monitors
                case "nearest":
                    self.add_monitor(
                        layer,
                        "Apre",
                        StateMonitor(
                            reducer=NearestTraceReducer(
                                self.step_time,
                                self.tc_pre,
                                amplitude=1.0,
                                target=True,
                                history_len=(
                                    self.delayed if self.delayed is not None else 0.0
                                ),
                            ),
                            attr="connection.synspike",
                            train_update=True,
                            eval_update=False,
                        ),
                    )

                    self.add_monitor(
                        layer,
                        "Apost",
                        OutputMonitor(
                            reducer=NearestTraceReducer(
                                self.step_time,
                                self.tc_post,
                                amplitude=1.0,
                                target=True,
                                history_len=(
                                    self.delayed if self.delayed is not None else 0.0
                                ),
                            ),
                            train_update=True,
                            eval_update=False,
                        ),
                    )

                case _:
                    raise ValueError(
                        f"invalid trace mode '{trace_mode}' given, "
                        "must be 'cumulative' or 'nearest'."
                    )

            # spike monitors
            self.add_monitor(
                layer,
                "Ipre",
                StateMonitor(
                    reducer=PassthroughReducer(
                        self.step_time,
                        history_len=(self.delayed if self.delayed is not None else 0.0),
                    ),
                    attr="connection.synspike",
                    train_update=True,
                    eval_update=False,
                ),
            )

            self.add_monitor(
                layer,
                "Ipost",
                OutputMonitor(
                    reducer=PassthroughReducer(
                        self.step_time,
                        history_len=(self.delayed if self.delayed is not None else 0.0),
                    ),
                    train_update=True,
                    eval_update=False,
                ),
            )

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
                    layer.selector, self.tolerance
                )
                if layer.connection.delayed and self.delayed
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
            update_ltp = torch.mean(torch.sum(i_post * a_pre, dim=-1), dim=0) * self.lrpost
            update_ltd = torch.mean(torch.sum(i_pre * a_post, dim=-1), dim=0) * self.lrpre

            # apply bounding if specified
            match self.bounding:
                case "soft":
                    if self.wmin is not None:
                        update_ltd *= lf.soft_bounding_dep(layer.connection.weight, self.wmin, self.wd_lower_exp)
                    if self.wmax is not None:
                        update_ltp *= lf.soft_bounding_pot(layer.connection.weight, self.wmax, self.wd_upper_exp)
                case "hard":
                    if self.wmin is not None:
                        update_ltd *= lf.hard_bounding_dep(layer.connection.weight, self.wmin)
                    if self.wmax is not None:
                        update_ltp *= lf.hard_bounding_pot(layer.connection.weight, self.wmax)

            # update weights
            layer.connection.weight = layer.connection.weight + update_ltp - update_ltd

    def add_trainable(self, trainable: Layer):
        # add trainable using superclass method
        LayerwiseUpdater.add_trainable(self, trainable)

        # don't add anything if monitors are already associated with this trainable
        if len(self.get_monitors(trainable)) != 0:
            return

        # add spike monitors
        self.add_monitor(
            trainable,
            "spike_pre",
            StateMonitor(
                reducer=PassthroughReducer(self.dt, 0.0),
                attr="connection.synspike",
                as_prehook=False,
                train_update=True,
                eval_update=False,
                prepend=True,
            ),
        )

        self.add_monitor(
            trainable,
            "spike_post",
            OutputMonitor(
                reducer=PassthroughReducer(self.dt, 0.0),
                train_update=True,
                eval_update=False,
                prepend=True,
            ),
        )

        # trace reducer class
        match self.trace:
            case "cumulative":
                TraceReducer = CumulativeTraceReducer
            case "nearest":
                TraceReducer = NearestTraceReducer
            case "_":
                raise RuntimeError(
                    f"an invalid trace mode of {self.trace}' has been set."
                )

        # postsynaptic trace monitor
        self.add_monitor(
            trainable,
            "trace_post",
            OutputMonitor(
                reducer=TraceReducer(
                    self.dt, self.tc_post, amplitude=1.0, target=True, history_len=0.0
                ),
                train_update=True,
                eval_update=False,
                prepend=True,
            ),
        )

        # postsynaptic spike monitor
        self.add_monitor(
            trainable,
            "spike_post",
            OutputMonitor(
                reducer=PassthroughReducer(self.dt, history_len=0.0),
                train_update=True,
                eval_update=False,
                prepend=True,
            ),
        )

        # presynaptic trace monitor
        delayed = trainable.connection.delayed and self.delayed
        self.add_monitor(
            trainable,
            "trace_pre",
            StateMonitor(
                reducer=TraceReducer(
                    self.dt,
                    self.tc_pre,
                    amplitude=1.0,
                    target=True,
                    history_len=(trainable.connection.delayed if delayed else 0.0),
                ),
                attr=("connection.synapse.spike" if delayed else "connection.synspike"),
                as_prehook=False,
                train_update=True,
                eval_update=False,
                prepend=True,
            ),
        )

        # presynaptic spike monitor
        self.add_monitor(
            trainable,
            "spike_pre",
            StateMonitor(
                reducer=PassthroughReducer(self.dt, history_len=0.0),
                attr="connection.synspike",
                as_prehook=False,
                train_update=True,
                eval_update=False,
                prepend=True,
            ),
        )

    @property
    def dt(self) -> float:
        return self.step_time

    @dt.setter
    def dt(self, value: float) -> None:
        # test for valid dt
        if value <= 0:
            raise RuntimeError(f"step time must be positive, received {value}")

        # assign new step time
        self.step_time = float(value)

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
