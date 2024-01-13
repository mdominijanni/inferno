from .. import LayerwiseUpdater
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
    def __init__(
        self,
        *layers: Layer,
        step_time: float,
        lr_post: float,
        lr_pre: float,
        tc_post: float,
        tc_pre: float,
        trace_mode: Literal["cumulative", "nearest"] = "cumulative",
        bounding_mode: Literal["none", "soft", "hard"] = "none",
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

        # register hyperparameters
        self.register_extra("step_time", float(step_time))
        self.register_extra("lr_post", float(lr_post))
        self.register_extra("lr_pre", float(lr_pre))
        self.register_extra("tc_post", float(tc_post))
        self.register_extra("tc_pre", float(tc_pre))

        # construct monitors
        for layer in self.trainables.values():
            # trace monitors
            match trace_mode.lower():
                case "cumulative":
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
            if not layer.training:
                continue

            # post and pre synaptic traces
            a_post = self.get_monitor(layer, "Apost")
            a_pre = self.get_monitor(layer, "Apre")

            # post and pre synaptic spikes
            i_post = self.get_monitor(layer, "Ipost")
            i_pre = self.get_monitor(layer, "Ipre")

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
