# Pragmatic Considerations

## Functional or Object-Oriented?
Inferno provides two main ways to interact with it—through its object-oriented and its functional interfaces. The latter can be distinguished by having module names which end in `.functional`.

In general, the object-oriented interface is great at getting off the ground quickly as a lot of the boilerplate code, such as ensuring tensor shape compatibility and automatic reduction of the batch axis is done for you. The functional interface meanwhile provides more control over the flow of logic and allows for greater customization.

The object-oriented interface provides base classes which can be extended, and implementation of custom subclasses using logic from the functional interface can balance these considerations. Dedicated classes provided by this interface in certain cases may be faster, as less type-checking is needed, and parameters which remain the same over multiple calls can be pre-computed on initialization.


## Minibatching
Unlike the artificial neurons typically found in ANNs, the neurons in SNNs are fundamentally stateful. In biological neurons, the electric potential difference between the interior and exterior of a cell is the driving force behind the generation of action potentials. This extends to the simplified models used in SNNs.

When using minibatches with SNNs, it's important to treat these state variables as separated for each of the samples presented during a minibatch. Fixed hyperparameters meanwhile do not need to be duplicated and instead will be broadcast to each sample in the batch.

In some models, parameters may be included to adapt the behavior of a neuron based on the inputs. A common example of this is the use of an adaptive threshold, where the threshold is defined as the minimum membrane potential at which a spike is generated. These parameters should be shared across all samples in a minibatch. Each sample the contributes to a portion of the update. This also applies to the adaptive parameters found in ANNs, such as weights and biases, and are handled the same way in SNNs.

This can be done either by reducing either the inputs or the outputs along the batch dimension. The former is more efficient but makes assumptions about how the adaptation is performed. When using Inferno's object-oriented interface, this is performed automatically. But in certain cases when using the functional interface it might be necessary to do this by hand.

Detailed information on minibatch processing with SNNs can be found at [arXiv:1909.02549](https://arxiv.org/abs/1909.02549).


## Discrete Time Simulations
Because Inferno performs simulations over discrete units of time, there are relevant considerations for how the computations match with the theoretical continuous-time descriptions.

### Refractory Periods
Refractory periods are specified using some length of time, given in milliseconds. On each simulation step, the amount of time that has elapsed is subtracted from the remaining time in the neurons will be in their refractory period (inclusive minimum bound of zero). When equal to zero, a neuron is considered to be out of its refractory period. Therefore, if the refractory period is not evenly divisible by the length of the simualted step time, the practical length of the refractory period is "rounded up" to the next integer multiple of the step time.


## Model Saving and Restoring
Because batch size affects not only the data passed through a model, but the model itself, special consideratin must be taken when saving and loading models. The batch size of a model saved must match the batch size of the model loaded. This same principle extends to modules which record state over time (e.g. reducers and synapses). While a step time and duration are specified, this changes the underlying structure of the data and therefore should match on load and restore. Note that the included property setters can be used to modify this state after loading.