# inferno

```{eval-rst}
.. currentmodule:: inferno
```

## Infrastructure
```{eval-rst}
.. autosummary::
    :nosignatures:
    :toctree: generated

    Module
    DimensionalModule
    RecordModule
    Hook
    ContextualHook
    StateHook
```

## Tensor Creation
```{eval-rst}
.. autosummary::
    :nosignatures:
    :toctree: generated

    zeros
    ones
    empty
    full
    uniform
    normal
    scalar
    astensors
```

## Math Operations
```{eval-rst}
.. autosummary::
    :nosignatures:
    :toctree: generated

    exp
    sqrt
    normalize
    rescale
    exponential_smoothing
    holt_linear_smoothing
    isi
```

## Spike Trace
```{eval-rst}
.. autosummary::
    :nosignatures:
    :toctree: generated

    trace_nearest
    trace_cumulative
    trace_nearest_scaled
    trace_cumulative_scaled
```

## Interpolation
```{eval-rst}
.. autosummary::
    :nosignatures:
    :toctree: generated

    interpolation.Interpolation
    interpolation.previous
    interpolation.nearest
    interpolation.linear
    interpolation.expdecay
    interpolation.expratedecay
```

## Parameter Bounding
```{eval-rst}
.. autosummary::
    :nosignatures:
    :toctree: generated

    bounding.FullBounding
    bounding.HalfBounding
    bounding.power
    bounding.upper_power
    bounding.lower_power
    bounding.multiplicative
    bounding.upper_multiplicative
    bounding.lower_multiplicative
    bounding.sharp
    bounding.upper_sharp
    bounding.lower_sharp
```

## Types
```{eval-rst}
.. autosummary::
    :nosignatures:
    :toctree: generated

    types.OneToOne
    types.ManyToOne
    types.OneToMany
    types.ManyToMany
    types.OneToOneMethod
    types.ManyToOneMethod
    types.OneToManyMethod
    types.ManyToManyMethod
```