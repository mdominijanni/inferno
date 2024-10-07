# Roadmap

## Conductance Synapse Support
At present, Inferno only support current-based synapses (CUBA), not conductance-based
synapses (COBA). The architecture assumes that synapses only act on the inputs to
a connection. There is no convenient way to incorporate the postsynaptic membrane
potentials into this. A moderate overhaul would be required.

### Planned Implementation
- Rename `Synapse` property `current` to `state` and `Connection` property `syncurrent` to `synoutput`. Refactor accordingly.
- Change "shortcut names" used for monitoring to reflect this.
- Create `CurrentSynapse` and `ConductanceSynapse` classes that inherit from `Synapse`. These will automatically provide properties `current` and `conductance` respectively based off of `output`.
- Create `Connection` properties `syncurrent` and `synconductance`, return `None` if the `Synapse` is not a `CurrentSynapse` or `ConductanceSynapse` respectively.
- Create a `SynapticBridge` protocol. The first argument of which is the `Neuron` to which `Connection` outputs will be given, and afterwards arbitrary positional and keyword arguments. Positional arguments will be the outputs from connected `Connection` objects. Alternatively, create this as a `Module` subclass.
- Change `Layer` objects to use `SynapticBridge` functions (or objects, depending on which is decided upon) rather than an arbitrary reduction function.

Note: More thought on the relation between `SynapticBridge` and `Layer` will need to be given.