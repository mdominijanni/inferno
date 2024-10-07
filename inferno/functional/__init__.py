from .protocols import (
    HalfBounding,
    FullBounding,
    Interpolation,
    Extrapolation,
    DimensionReduction,
    SpikeTimeHalfKernel,
)

from .bounding import (
    # power half bounding
    bound_upper_power,
    bound_lower_power,
    bound_upper_scaled_power,
    bound_lower_scaled_power,
    # multiplicative half bounding
    bound_upper_multiplicative,
    bound_lower_multiplicative,
    bound_upper_scaled_multiplicative,
    bound_lower_scaled_multiplicative,
    # sharp half bounding
    bound_upper_sharp,
    bound_lower_sharp,
    # full bounding
    bound_power,
    bound_scaled_power,
    bound_multiplicative,
    bound_scaled_multiplicative,
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

from .dimreductiion import (
    sum,
    nansum,
    divsum,
    nandivsum,
    min,
    absmin,
    max,
    absmax,
    mean,
    nanmean,
    quantile,
    nanquantile,
    median,
    nanmedian,
    geomean,
    nangeomean,
)

from .stdkernels import (
    exp_stdp_post_kernel,
    exp_stdp_pre_kernel,
)

__all__ = [
    # types
    "HalfBounding",
    "FullBounding",
    "Interpolation",
    "Extrapolation",
    "DimensionReduction",
    "SpikeTimeHalfKernel",
    # bounding
    "bound_upper_power",
    "bound_lower_power",
    "bound_upper_scaled_power",
    "bound_lower_scaled_power",
    "bound_upper_multiplicative",
    "bound_lower_multiplicative",
    "bound_upper_scaled_multiplicative",
    "bound_lower_scaled_multiplicative",
    "bound_upper_sharp",
    "bound_lower_sharp",
    "bound_power",
    "bound_scaled_power",
    "bound_multiplicative",
    "bound_scaled_multiplicative",
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
    # dimension reduction
    "sum",
    "nansum",
    "divsum",
    "nandivsum",
    "min",
    "absmin",
    "max",
    "absmax",
    "mean",
    "nanmean",
    "quantile",
    "nanquantile",
    "median",
    "nanmedian",
    "geomean",
    "nangeomean",
    # spike time difference kernels
    "exp_stdp_post_kernel",
    "exp_stdp_pre_kernel",
]
