# Synapses, Current Models

## Delta
### Formulation
$$
I(t) = Q \delta(t - t_f) + I_x
$$

*Where the Dirac delta function $\delta(x)$ in discrete time simulations is:*

$$
\delta(x) =
\begin{cases}
    1 / \Delta t & x = 0 \\
    0 & x \neq 0
\end{cases}
$$

*Where:*
- $I$, total input current applied to the neuron $(\text{nA})$
- $Q$, electrical charge carried by an action potential $(\text{pC})$.
- $t$, current time of the simulation $(\text{ms})$
- $t_f$, time of the last presynaptic spike $(\text{ms})$
- $\Delta t$, length of time over which each simulation step occurs $(\text{ms})$
- $I_x$, any external passed to through the synapse $(\text{nA})$.

Note that the Dirac delta function for a value $x$ has units inverse of $x$. So in this
case, $\delta(t - t_f)$ has units $\text{ms}^{-1}$.

### Description
This is a very simplified model for a synapse. In simulations its role is to normalize
the current delivered for a given spike such that simulation time step does not dramatically
change the integration of current from action potentials into the neuron.

### References
1. [ISBN:9780262548083](https://mitpress.ublish.com/ebook/modeling-neural-circuits-made-simple-with-python-preview/12788/Cover)