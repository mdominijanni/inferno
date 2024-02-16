# inferno.neural

```{eval-rst}
.. automodule:: inferno.neural
```

## Components
```{eval-rst}
.. autosummary::
    :nosignatures:
    :toctree: generated

    Neuron
    Synapse
    Connection
    Layer
```

## Neurons
```{eval-rst}
.. autosummary::
    :nosignatures:
    :toctree: generated

    LIF
    ALIF
    GLIF1
    GLIF2
    QIF
    Izhikevich
    EIF
    AdEx
```

## Synapses
```{eval-rst}
.. autosummary::
    :nosignatures:
    :toctree: generated

    DeltaCurrent
    DeltaPlusCurrent
```

## Connections
```{eval-rst}
.. autosummary::
    :nosignatures:
    :toctree: generated

    LinearDense
    LinearDirect
    LinearLateral
    Conv2D
```

## Encoders
```{eval-rst}
.. autosummary::
    :nosignatures:
    :toctree: generated

    HomogeneousPoissonEncoder
    PoissonIntervalEncoder
```

## Hooks
```{eval-rst}
.. autosummary::
    :nosignatures:
    :toctree: generated

    Clamping
    Normalization
    RemoteClamping
    RemoteNormalization
```

## Types
```{eval-rst}
.. autosummary::
    :nosignatures:
    :toctree: generated

    SynapseConstructor
```

## Mixins
```{eval-rst}
.. autosummary::
    :nosignatures:
    :toctree: generated

    mixins.BatchMixin
    mixins.ShapeMixin
    neurons.mixins.AdaptationMixin
    neurons.mixins.CurrentMixin
    neurons.mixins.RefractoryMixin
    neurons.mixins.SpikeRefractoryMixin
    neurons.mixins.VoltageMixin
    synapses.mixins.CurrentMixin
    synapses.mixins.SpikeMixin
    synapses.mixins.CurrentDerivedSpikeMixin
    synapses.mixins.SpikeDerivedCurrentMixin
    synapses.mixins.SpikeCurrentMixin
    synapses.mixins.DelayedSpikeCurrentAccessorMixin
    connections.mixins.WeightMixin
    connections.mixins.WeightBiasMixin
    connections.mixins.WeightBiasDelayMixin
```