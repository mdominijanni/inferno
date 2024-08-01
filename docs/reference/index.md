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
extra
```

## Package Overview

<strong style="font-size: 1.5em;">[inferno](<reference/inferno:inferno>)</strong>

The common infrastructure used throughout various submodules and various functions for calculations and tensor-creation which may be used either within Inferno or may be helpful for end-users.

<strong style="font-size: 1.5em;">[inferno.functional](<reference/functional:inferno.functional>)</strong>

The protocols and various implementations for parameter bounding, interpolation, extrapolation, and dimensionality reduction.

<strong style="font-size: 1.5em;">[inferno.neural](<reference/neural:inferno.neural>)</strong>

The basic components for spiking neural networks, the infrastructure used for connecting them into a network and for supporting generalized parameter updates, and encoding non-spiking data into spike trains.

<strong style="font-size: 1.5em;">[inferno.neural.functional](<reference/neural-functional:inferno.neural.functional>)</strong>

The functional implementation of various components used by different models as a way to generalize and share functionality, also useful when implementing new classes that represent neural components.

<strong style="font-size: 1.5em;">[inferno.learn](<reference/learn:inferno.learn>)</strong>

The components needed for training spiking neural networks, as well as components which may be used for specific inference tasks (e.g. classification).

<strong style="font-size: 1.5em;">[inferno.observe](<reference/observe:inferno.observe>)</strong>

The infrastructure and components for monitoring the internal states of components.

<strong style="font-size: 1.5em;">[inferno.stats](<reference/stats:inferno.stats>)</strong>

A work-in-progress module containing PyTorch-based implementations of various probability distributions.

<strong style="font-size: 1.5em;">[inferno.extra](<reference/extra:inferno.extra>)</strong>

A work-in-progress module containing assorted components which may be useful when attempting to generate visualizations or diagnose issues, or whenever a placeholder is needed.
