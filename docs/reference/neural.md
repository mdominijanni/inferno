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
```

## Synapses
```{eval-rst}
.. autosummary::
    :nosignatures:
    :toctree: generated

    PassthroughSynapse
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

## Hooks
```{eval-rst}
.. autosummary::
    :nosignatures:
    :toctree: generated

    Normalization
    Clamping
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
    synapses.mixins.SpikeCurrentMixin
    synapses.mixins.DelayedSpikeCurrentMixin
    connections.mixins.WeightMixin
    connections.mixins.WeightBiasMixin
    connections.mixins.WeightBiasDelayMixin
```