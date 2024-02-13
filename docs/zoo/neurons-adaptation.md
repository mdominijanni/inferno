# Neuronal Adaptation

## Overview
The dynamics imposed by most neuron models (without adaptation) cannot respond to changes with regard to their input. This implies that, for a given constant input, the neuron's firing frequency will remain steady over time. Adaptation, in this context, gives the neuron model one or more additional degrees of freedom, allowing it to respond to the presented input.

Generally, this will either be applied as a modification to the presynaptic current (either dampening or amplifying the input) or as a modification to the membrane voltage threshold at which the neuron will fire (either raising or lowering this threshold). Dampening the input or raising the threshold will depress the neuron's spiking behavior, and amplifying the input or lowering the threshold will potentiate it.

Additionally, two different parameters of a neuron are typically monitored to inform this adaptation—postsynaptic spikes and membrane potential. The former will potentiate the firing rate as said rate decreases and depress it as it increases. The latter will potentiate the firing rate as the membrane voltage when the voltage is below the equilibrium (rest) voltage and depress the firing rate as it goes above the rest voltage. These effects are in proportion to the magnitude of the difference.

## Adaptive Current, Linear
### Formulation
$$
\begin{align*}
    I(t) &= I_x(t) - \sum_k w_k(t) \\
    \tau_k \frac{dw_k(t)}{dt} &= a_k \left[ V_m(t) - V_\text{rest} \right] - w_k(t) \\
\end{align*}
$$

*With approximation:*

$$w_k(t + \Delta t) \approx \frac{\Delta t}{\tau_k}\left[ a_k \left[ V_m(t) - V_\text{rest} \right] - w_k(t) \right] + w_k(t)$$

*After an action potential is generated:*

$$w_k(t) \leftarrow w_k(t) + b_k$$

*Where:*
- $I$, total input current applied to the neuron $(\text{nA})$
- $I_x$, input current before adaptation $(\text{nA})$
- $V_m$, electric potential difference across the cell membrane $(\text{mV})$
- $V_\text{rest}$, equilibrium of the membrane potential $(\text{mV})$
- $a_k$, subthreshold adaptation, voltage-current coupling $(\mu\text{S})$
- $b_k$, spike-triggered current adaptation $(\text{nA})$
- $\tau_k$, adaptation time constant $(\text{ms})$
- $t$, current time of the simulation $(\text{ms})$
- $\Delta t$, length of time over which each simulation step occurs $(\text{ms})$

### References
1. [DOI:10.1017/CBO9781107447615 (§6.1)](https://neuronaldynamics.epfl.ch/online/Ch6.S1.html)
2. [DOI:10.1152/jn.00686.2005](https://journals.physiology.org/doi/full/10.1152/jn.00686.2005)

## Adaptive Current, Linear Spike-Dependent
***(To Be Implemented)***
### Formulation
$$
\begin{align*}
    I(t) &= I_x(t) - \sum_k w_k(t) \\
    \tau_k \frac{dw_k(t)}{dt} &= - w_k(t)
\end{align*}
$$

*With solution:*

$$w_k(t + \Delta t) = w_k(t)\exp\left(-\frac{\Delta t}{\tau_k}\right)$$

*After an action potential is generated:*

$$w_k(t) \leftarrow w_k(t) + b_k$$

*Where:*
- $I$, total input current applied to the neuron $(\text{nA})$
- $I_x$, input current before adaptation $(\text{nA})$
- $b_k$, spike-triggered current adaptation $(\text{nA})$
- $\tau_k$, adaptation time constant $(\text{ms})$
- $t$, current time of the simulation $(\text{ms})$
- $\Delta t$, length of time over which each simulation step occurs $(\text{ms})$

### References
1. [DOI:10.1038/s41467-017-02717-4](https://www.nature.com/articles/s41467-017-02717-4)
2. [Allen Institute GLIF Whitepaper](http://web.archive.org/web/20230428012128/https://help.brain-map.org/download/attachments/8323525/glifmodels.pdf)

## Adaptive Threshold, Linear Voltage-Dependent
### Formulation
$$
\begin{align*}
    \Theta(t) &= \Theta_\infty + \sum_k \theta_k(t) \\
    \frac{d\theta_k(t)}{dt} &= a_k \left[ V_m(t) - V_\text{rest} \right] - b_k \theta_k(t)
\end{align*}
$$

*With approximation:*

$$
\theta_k(t + \Delta t) \approx \Delta t \left[a_k \left[ V_m(t) - V_\text{rest} \right] - b_k \theta_k(t)\right] + \theta_k(t)
$$

*After an action potential is generated:*

$$\theta_k(t) \leftarrow \max(\theta_k(t), \theta_\text{reset})$$

*Where:*
- $\Theta$, membrane potential at which an action potential is generated $(\text{mV})$
- $\Theta_\infty$, equilibrium of the firing threshold $(\text{mV})$
- $\theta_k$, adaptive component of the firing threshold $(\text{mV})$
- $\theta_\text{reset}$, reset value of the adaptive component of the firing threshold $(\text{mV})$
- $V_m$, electric potential difference across the cell membrane $(\text{mV})$
- $V_\text{rest}$, equilibrium of the membrane potential $(\text{mV})$
- $a_k$, threshold adaptation exponential decay constant $(\text{ms}^{-1})$
- $b_k$, threshold rebound exponential decay constant $(\text{ms}^{-1})$
- $t$, current time of the simulation $(\text{ms})$
- $\Delta t$, length of time over which each simulation step occurs $(\text{ms})$

### References
1. [DOI:10.1162/neco.2008.12-07-680](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2954058/)
2. [DOI:10.1162/NECO_a_00196](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3513351/)

## Adaptive Threshold, Linear Spike-Dependent
### Formulation
$$
\begin{align*}
    \Theta(t) &= \Theta_\infty + \sum_k \theta_k(t) \\
    \tau_k \frac{d\theta_k(t)}{dt} &= -\theta_k(t)
\end{align*}
$$

*With solution:*

$$\theta_k(t + \Delta t) = \theta_k(t) \exp\left(-\frac{\Delta t}{\tau_k}\right)$$

*After an action potential is generated:*

$$\theta_k(t) \leftarrow \theta_k(t) + d_k$$

*Where:*
- $\Theta$, membrane potential at which an action potential is generated $(\text{mV})$
- $\Theta_\infty$, equilibrium of the firing threshold $(\text{mV})$
- $\theta_k$, adaptive component of the firing threshold $(\text{mV})$
- $\theta_\text{reset}$, reset value of the adaptive component of the firing threshold $(\text{mV})$
- $d_k$, spike-triggered voltage threshold adaptation $(\text{mV})$
- $\tau_k$, adaptation time constant $(\text{ms})$
- $t$, current time of the simulation $(\text{ms})$
- $\Delta t$, length of time over which each simulation step occurs $(\text{ms})$

### References
1. [arXiv:1803.09574](https://arxiv.org/abs/1803.09574)
2. [DOI:10.1038/s41467-021-24427-8](https://www.nature.com/articles/s41467-021-24427-8)
3. [DOI:10.1152/jn.00234.2019](https://journals.physiology.org/doi/full/10.1152/jn.00234.2019)
4. [DOI:10.1038/s41467-017-02717-4](https://www.nature.com/articles/s41467-017-02717-4)