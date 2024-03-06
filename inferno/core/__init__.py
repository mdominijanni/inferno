from . import bounding
from . import interpolation

from .math import (
    exp,
    sqrt,
    normalize,
    rescale,
    exponential_smoothing,
    holt_linear_smoothing,
    isi,
)

from .tensor import (
    zeros,
    ones,
    empty,
    full,
    uniform,
    normal,
    scalar,
    astensors,
)

from .trace import (
    trace_nearest,
    trace_cumulative,
    trace_nearest_scaled,
    trace_cumulative_scaled,
)

__all__ = [
    # submodules
    "bounding",
    "interpolation",
    # math
    "exp",
    "sqrt",
    "normalize",
    "rescale",
    "exponential_smoothing",
    "holt_linear_smoothing",
    "isi",
    # tensor creation
    "zeros",
    "ones",
    "empty",
    "full",
    "uniform",
    "normal",
    "scalar",
    "astensors",
    # spike trace
    "trace_nearest",
    "trace_cumulative",
    "trace_nearest_scaled",
    "trace_cumulative_scaled",
]
