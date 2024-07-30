# inferno.neural

```{eval-rst}
.. automodule:: inferno.neural
```

## Modelling
```{eval-rst}
.. autosummary::
    :nosignatures:
    :toctree: generated

    Layer
    Cell
    Updater
    Updatable
    Accumulator
```

## Layers
```{eval-rst}
.. autosummary::
    :nosignatures:
    :toctree: generated

    Biclique
    Serial
    RecurrentSerial
```

## Components
```{eval-rst}
.. autosummary::
    :nosignatures:
    :toctree: generated

    Neuron
    Synapse
    Connection
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
    HomogeneousPoissonApproxEncoder
    PoissonIntervalEncoder
```

## Hooks
```{eval-rst}
.. autosummary::
    :nosignatures:
    :toctree: generated

    Clamping
    Normalization
```

## Types
```{eval-rst}
.. autosummary::
    :nosignatures:
    :toctree: generated

    SynapseConstructor
```

## Internal Components
```{eval-rst}
.. autosummary::
    :nosignatures:
    :toctree: generated

    InfernoNeuron
    InfernoSynapse
```

## Internal Mixins
```{eval-rst}
.. autosummary::
    :nosignatures:
    :toctree: generated

    mixins.BatchMixin
    mixins.ShapeMixin
    mixins.BatchShapeMixin
    mixins.DelayedMixin
    neurons.mixins.AdaptiveCurrentMixin
    neurons.mixins.AdaptiveThresholdMixin
    neurons.mixins.CurrentMixin
    neurons.mixins.RefractoryMixin
    neurons.mixins.SpikeRefractoryMixin
    neurons.mixins.VoltageMixin
    synapses.mixins.CurrentMixin
    synapses.mixins.SpikeMixin
    synapses.mixins.CurrentDerivedSpikeMixin
    synapses.mixins.SpikeDerivedCurrentMixin
    synapses.mixins.SpikeCurrentMixin
    connections.mixins.WeightMixin
    connections.mixins.WeightBiasMixin
    connections.mixins.WeightBiasDelayMixin
```