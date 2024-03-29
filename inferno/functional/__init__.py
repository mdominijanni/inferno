from .protocols import (
    HalfBounding,
    FullBounding,
    Interpolation,
    Extrapolation,
)

from .bounding import (
    # power half bounding
    bound_upper_power,
    bound_lower_power,
    # multiplicative half bounding
    bound_upper_multiplicative,
    bound_lower_multiplicative,
    # sharp half bounding
    bound_upper_sharp,
    bound_lower_sharp,
    # full bounding
    bound_power,
    bound_multiplicative,
    bound_sharp,
)

from .interpolation import (
    # positional
    interp_previous,
    interp_next,
    interp_nearest,
    # mathematical
    interp_linear,
    interp_expdecay,
    interp_expratedecay,
)

from .extrapolation import (
    # positional
    extrap_previous,
    extrap_next,
    extrap_neighbors,
    extrap_nearest,
    # mathematical
    extrap_linear_forward,
    extrap_linear_backward,
    extrap_expdecay,
    extrap_expratedecay,
)

__all__ = [
    # types
    "HalfBounding",
    "FullBounding",
    "Interpolation",
    "Extrapolation",
    # bounding
    "bound_upper_power",
    "bound_lower_power",
    "bound_upper_multiplicative",
    "bound_lower_multiplicative",
    "bound_upper_sharp",
    "bound_lower_sharp",
    "bound_power",
    "bound_multiplicative",
    "bound_sharp",
    # interpolation
    "interp_previous",
    "interp_next",
    "interp_nearest",
    "interp_linear",
    "interp_expdecay",
    "interp_expratedecay",
    # extrapolation
    "extrap_previous",
    "extrap_next",
    "extrap_neighbors",
    "extrap_nearest",
    "extrap_linear_forward",
    "extrap_linear_backward",
    "extrap_expdecay",
    "extrap_expratedecay",
]
