# Roadmap

## Conductance Synapse Support
At present, Inferno only support current-based synapses (CUBA), not conductance-based
synapses (COBA). The architecture assumes that synapses only act on the inputs to
a connection. There is no convenient way to incorporate the postsynaptic membrane
potentials into this. A moderate overhaul would be required.
