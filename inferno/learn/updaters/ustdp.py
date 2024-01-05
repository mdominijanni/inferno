from .. import PairwiseUpdater
from inferno.neural import Layer
from inferno.observe import StateMonitor, OutputMonitor, NearestTraceReducer, CumulativeTraceReducer
import torch
from typing import Callable, Literal


class STDP(PairwiseUpdater):
    def __init__(
        self,
        *layers: Layer,
        step_time: float,
        lr_post: float,
        lr_pre: float,
        tc_post: float,
        tc_pre: float,
        trace_mode: Literal["cumulative", "nearest"] = "cumulative",
        bounding_function: Callable | None = None,
        **kwargs
    ):
        # call superclass constructor
        PairwiseUpdater.__init__(self, *layers)

        # register hyperparameters
        self.register_extra('step_time', float(step_time))
        self.register_buffer('lr_post', torch.tensor(float(lr_post)))
        self.register_buffer('lr_pre', torch.tensor(float(lr_pre)))
        self.register_extra('tc_post', float(tc_post))
        self.register_extra('tc_pre', float(tc_pre))

        # construct monitors
        for layer in self.trainables.values():
            # trace monitors
            match trace_mode.lower():
                case 'cumulative':
                    self.add_monitor(layer, 'Apre', StateMonitor())
                    self.add_monitor(layer, 'Apost', OutputMonitor())
                case 'nearest':
                    self.add_monitor(layer, 'Apre', StateMonitor())
                    self.add_monitor(layer, 'Apost', OutputMonitor())
                case _:
                    raise ValueError(f"invalid trace mode '{trace_mode}' given, "
                                     "must be 'cumulative' or 'nearest'.")

            # spike monitors
            self.add_monitor(layer, 'Ipre', StateMonitor())
            self.add_monitor(layer, 'Ipost', OutputMonitor())
