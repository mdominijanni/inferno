# API Reference

```{toctree}
:hidden:

inferno
functional
neural
neural-functional
learn
observe
stats
```

## Package Overview
### [inferno](<reference/inferno:inferno>)
The common infrastructure used throughout various submodules and various functions for calculations and tensor-creation which may be used either within Inferno or may be helpful for end-users.

### [inferno.functional](<reference/functional:inferno.functional>)
The protocols and various implementations for parameter bounding, interpolation, extrapolation, and dimensionality reduction.

### [inferno.neural](<reference/neural:inferno.neural>)
The basic components for spiking neural networks (neurons, synapses, connections), the infrastructure used for connecting them into a network and for supporting generalized parameter updates (layers, updaters), and encoding non-spiking data into spike trains.

### [inferno.neural.functional](<reference/neural-functional:inferno.neural.functional>)
The functional implementation of various components used by different models as a way to generalize and share functionality, also useful when implementing new classes that represent neural components.

### [inferno.learn](<reference/learn:inferno.learn>)
The components needed for training spiking neural networks, as well as components which may be used for specific inference tasks (e.g. classification).

### [inferno.observe](<reference/observe:inferno.observe>)
The infrastructure and components for monitoring the internal states of components.

### [inferno.stats](<reference/stats:inferno.stats>)
A work-in-progress module containing PyTorch-based implementations of various probability distributions.