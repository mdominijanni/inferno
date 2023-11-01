# Neurons and Neuronal Systems

## Dynamics of a Neuron

### Refractory Periods
Biological neurons have two kinds of refractory periods— an absolute refractory period (ARP) and a relative refractory period (RRP).

The ARP is a period during which action potentials cannot be generated, and any presynaptic current does not affect the membrane potential (this is the period in which the neuron depolarizes and then repolarizes). The RRP is a period during which the neuron is hyperpolarized, meaning that while it is more difficult for the neuron to fire in this period, it is not impossible.

The ARP is modeled using a refractory period (in integer multiples of the simulation step time) in which a neuron cannot fire and the voltage of neurons in this period is fixed. This can also apply to neuronal adaptations. Meanwhile, RRP is modeled through the reset voltage, which determines how strongly the neuron is hyperpolarized following the action potential.