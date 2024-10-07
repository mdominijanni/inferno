# inferno.functional

```{eval-rst}
.. automodule:: inferno.functional
```

## Protocols

```{eval-rst}
.. autosummary::
    :nosignatures:
    :toctree: generated

    HalfBounding
    FullBounding
    Interpolation
    Extrapolation
    DimensionReduction
    SpikeTimeHalfKernel
```

## Bounding

```{eval-rst}
.. autosummary::
    :nosignatures:
    :toctree: generated

    bound_power
    bound_upper_power
    bound_lower_power
    bound_scaled_power
    bound_upper_scaled_power
    bound_lower_scaled_power
    bound_multiplicative
    bound_upper_multiplicative
    bound_lower_multiplicative
    bound_scaled_multiplicative
    bound_upper_scaled_multiplicative
    bound_lower_scaled_multiplicative
    bound_sharp
    bound_upper_sharp
    bound_lower_sharp
```

## Interpolation

```{eval-rst}
.. autosummary::
    :nosignatures:
    :toctree: generated

    interp_previous
    interp_next
    interp_nearest
    interp_linear
    interp_expdecay
    interp_expratedecay
```

## Extrapolation

```{eval-rst}
.. autosummary::
    :nosignatures:
    :toctree: generated

    extrap_previous
    extrap_next
    extrap_neighbors
    extrap_nearest
    extrap_linear_forward
    extrap_linear_backward
    extrap_expdecay
    extrap_expratedecay
```

## Dimension Reductions

```{eval-rst}
.. autosummary::
    :nosignatures:
    :toctree: generated

    sum
    nansum
    divsum
    nandivsum
    min
    max
    absmin
    absmax
    mean
    nanmean
    median
    nanmedian
    geomean
    nangeomean
    quantile
    nanquantile
```

## Spike Time Kernels
```{eval-rst}
.. autosummary::
    :nosignatures:
    :toctree: generated

    exp_stdp_post_kernel
    exp_stdp_pre_kernel
```