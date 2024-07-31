# About Inferno

## Architecture
### Neural Components
```{image} ../images/diagrams/inferno-nseq-light.svg
:alt: Inputs and outputs for each type of neural component.
:class: only-light
:width: 30em
:align: center
```

```{image} ../images/diagrams/inferno-nseq-dark.svg
:alt: Inputs and outputs for each type of neural component.
:class: only-dark
:width: 30em
:align: center
```
Sequence of a forward pass and the data type of the tensors for the inputs and outputs for each neural component. Here, "inj" are optional injected values (which are only supported on some types of synapses) and "*mapping*" is the internal, trainable mapping for a given connection.

#### Neurons

Neurons are responsible for converting values on the wire into sequences of discrete events, called _spikes_ or _action potentials_. When a neuron generates an action potential, it is said to have _spiked_ or _fired_. Each group of neurons is represented by Inferno with the {py:class}`~inferno.neural.Neuron` and {py:class}`~inferno.neural.InfernoNeuron` classes, where the former is the most general interface and the latter includes some implementation to work with provided mixins. Each `Neuron` implements a common mode of dynamics (the neuron model being simulated) with common hyperparameters. Neurons serve a similar role to the non-linear activation functions of artificial neural networks, but are generally far more complex. They need to maintain an internal state, chiefly in the form of _membrane potential_ or _membrane voltage_, which is defined as the difference in electric potential between the interior of a neuron and the extracellular medium that surrounds it.

#### Synapses

Synapses are responsible for converting the discrete outputs from neurons into continuous inputs for neurons. Each group of synapses is represented by Inferno with the {py:class}`~inferno.neural.Synapse` and {py:class}`~inferno.neural.InfernoSynapse` classes, where the former is the most general interface and the latter includes some implementation to work with provided mixins. Each `Synapse` implements a common mode of kinetics (the synapse model being simulated) with common hyperparameters. There is no direct equivalent in artificial neural networks. Inferno also uses `Synapse` classes as the way of working with trainable, heterogeneous delays by storing the state of multiple previous steps and interpolating between them.

#### Connections

Connections are responsible for taking multiple inputs from one or more groups of neurons and combining them for input to another group of neurons. Each connection is represented by Inferno with the {py:class}`~inferno.neural.Connection` class. The responsibility of a connection can be divided in two. The first is converting from discrete spikes into continuous values, which Inferno does by injecting a `Synapse` as a dependency into a `Connection`. The second is applying a trainable mapping of weights (and optionally biases and delays) to the inputs from the composed `Synapse`. This second responsibility is the same as that of a layer in artificial neural networks.

### Modelling Components

Inferno's modelling components do not strictly represent any specific part of the biological neural networks that spiking neural networks seek to model, but instead are used to aid the modelling itself.

#### Layers

Layers manage the wiring between one or more connections that receive input from outside the layer, and one or more groups of neurons that take input from those connections. Each layer is represented by Inferno with the {py:class}`~inferno.neural.Layer` class. Inferno uses these to manage the triggers for monitors ({py:class}`~inferno.observe.Monitor`) that record the state varibles used for training connections.

#### Cells

Cells are a bundling of a connection and a group of neurons which takes its input from the connection output. Each cell is represented by Inferno with the {py:class}`~inferno.neural.Cell` class. Each `Cell` is tied to the `Layer` which created it and is used for training connections. Specifically, it is used to register a connection with training methods that presynaptic and postsynaptic spikes as the basis for parameter updates.

#### Updaters

Updaters are used to update the trainable parameters of a component. Each updater is represented by Inferno with the {py:class}`~inferno.neural.Updater` class, and can be used on any subclass of {py:class}`~inferno.neural.Updatable`.

#### Accumulators

Accumulators are used to store and apply the updates for a specific trainable parameter. Each accumulator is represented by Inferno with the {py:class}`~inferno.neural.Accumulator` class. They are created and managed by the `Updater` for a given object. Each `Accumulator` can not only apply multiple updates, but it can control how multiple updates are reduced together, and how potentiative and depressive updates are applied to keep the updatable parameter within a desired range.

### Component Composition
```{image} ../images/diagrams/inferno-ccarch-light.svg
:alt: Composition of Inferno components.
:class: only-light
:width: 30em
:align: center
```

```{image} ../images/diagrams/inferno-ccarch-dark.svg
:alt: Composition of Inferno components.
:class: only-dark
:width: 30em
:align: center
```

Composition of the different neural and modelling components as defined by the Inferno library.