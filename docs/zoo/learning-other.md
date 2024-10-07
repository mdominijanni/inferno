# Learning, Other Methods

## Linear Homeostatic Plasticity
### Formulation
$$
v(t + \Delta t) - v(t) = \lambda \frac{r^* - r}{r^*}
$$

*Where:*
- $v$, an updatable parameter
- $r^*$, target spike rate
- $r$, observed spike rate
- $\lambda$, learning rate
- $t$, current simulation time
- $\Delta t$, duration of the simulation step

### Description
This method is used for regulating the trainable parameters of a network trained with another plasticity method based on a target spiking rate. As originally provided, weight updates use a positive value for $\lambda$ and delay updates use a negative value for $\lambda$.

### References
1. [DOI:10.1162/neco_a_01674](https://arxiv.org/abs/2011.09380)
