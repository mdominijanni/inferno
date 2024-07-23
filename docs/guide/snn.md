# About SNNs

*Work in Progress*

## Dynamics of a Neuron
### Refractory Periods
Biological neurons have two kinds of refractory periods, an absolute refractory period (ARP) and a relative refractory period (RRP).

The ARP is a period during which action potentials cannot be generated, and any presynaptic current does not affect the membrane potential (this is the period in which the neuron depolarizes and then repolarizes). The RRP is a period during which the neuron is hyperpolarized, meaning that while it is more difficult for the neuron to fire in this period, it is not impossible.

The ARP is modeled using a refractory period during which a neuron cannot fire and its state is fixed at some model-defined reset value (this state includes membrane voltage and can also include neuronal adaptations). The RRP is modeled using by the *reset voltage* of the neuron, which determines how strongly the neuron is hyperpolarized following the action potential. In some models, this reset voltage is a constant value whereas in others it is parameterized additional model state.