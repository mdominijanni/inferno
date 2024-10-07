# Synapses, Current Models

## Delta
### Formulation
$$
I(t) = Q \delta(t - t_f)
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
- $Q$, electrical charge carried by an action potential $(\text{pC})$
- $t$, current time of the simulation $(\text{ms})$
- $t_f$, time of the last presynaptic spike $(\text{ms})$
- $\Delta t$, length of time over which each simulation step occurs $(\text{ms})$

Note that the Dirac delta function for a value $x$ has units inverse of $x$. So in this
case, $\delta(t - t_f)$ has units $\text{ms}^{-1}$.

### Description
This is a very simplified model for a synapse. In simulations its role is to normalize
the current delivered for a given spike such that simulation time step does not dramatically
change the integration of current from action potentials into the neuron.

### References
1. [ISBN:9780262548083](https://mitpress.ublish.com/ebook/modeling-neural-circuits-made-simple-with-python-preview/12788/Cover)


## Single Exponential
### Formulation
$$
\tau \frac{dI(t)}{dt} = -I(t) + Q \sum_{\mathcal{F}} \delta \left(t - t_f\right)
$$

*With solution:*

$$
I(t + \Delta t) = I(t) \exp\left(-\frac{\Delta t}{\tau}\right) + \frac{Q}{\tau}\left[t = t_f\right]
$$

*Where:*
- $I$, total input current applied to the neuron $(\text{nA})$
- $Q$, electrical charge induced by an action potential $(\text{pC})$
- $\tau$, time constant of current decay $(\text{ms})$
- $t$, current time of the simulation $(\text{ms})$
- $t_f$, time of the last presynaptic spike $(\text{ms})$
- $\Delta t$, length of time over which each simulation step occurs $(\text{ms})$
- $\delta(t)$, [Dirac delta function](<guide/mathematics:Dirac Delta Function>) $(\text{ms}^{-1})$

$[\cdots]$ is the Iverson bracket and equals $1$ if the inner statement is true and $0$ if it is false (unitless).

### Description
This model of synaptic kinetics assumes that the neurotransmitters responsible for inducing the synaptic current instantly bind to receptors and that their influence decays exponentially over time.

### References
1. [DOI:10.3390/brainsci12070863](https://www.mdpi.com/2076-3425/12/7/863)


## Double Exponential
### Formulation
$$
\begin{align*}
    \frac{dI(t)}{dt} &= -\frac{I(t)}{\tau_d} + A(t) \\
    \frac{dA(t)}{dt} &= -\frac{A(t)}{\tau_r} + \frac{Q}{\tau_r \tau_d} \sum_{t_f < t} \delta \left(t - t_f\right)
\end{align*}
$$

*Alternatively:*

$$
\tau_d\tau_r \frac{d^2 I(t)}{dt^2} + (\tau_d + \tau_r) \frac{dI(t)}{dt} = -I(t) + Q \sum_{t_f < t} \delta \left(t - t_f\right)
$$

*With solutions:*

$$
\begin{align*}
    I(t + \Delta t) &= \left[I(t) + \frac{\tau_d\tau_r}{\tau_d - \tau_r} A(t) \right] \exp\left(-\frac{\Delta t}{\tau_d}\right) - \frac{\tau_d\tau_r}{\tau_d - \tau_r} A(t + \Delta t) + \frac{Q}{\tau_d - \tau_r}\left[t = t_f\right] \\
A(t + \Delta t) &= A(t) \exp\left(-\frac{\Delta t}{\tau_r}\right) + \frac{Q}{\tau_d\tau_r}\left[t = t_f\right]
\end{align*}
$$

*Equivalently:*

$$
\begin{align*}
    I(t) &= I_d(t) - I_r(t) \\
    I_d(t + \Delta t) &= I_d(t) \exp \left(-\frac{\Delta t}{\tau_d}\right) + \frac{Q}{\tau_d - \tau_r} \left[t = t_f\right] \\
    I_r(t + \Delta t) &= I_r(t) \exp \left(-\frac{\Delta t}{\tau_r}\right) + \frac{Q}{\tau_d - \tau_r} \left[t = t_f\right]
\end{align*}
$$

*Where:*
- $I$, total input current applied to the neuron $(\text{nA})$
- $A$, rate of current change on an action potential $(\text{nA} \cdot \text{ms}^{-1})$
- $Q$, electrical charge induced by an action potential $(\text{pC})$
- $\tau_d$, slow time constant of current decay $(\text{ms})$
- $\tau_r$, fast time constant of current rise $(\text{ms})$
- $t$, current time of the simulation $(\text{ms})$
- $t_f$, time of the last presynaptic spike $(\text{ms})$
- $\Delta t$, length of time over which each simulation step occurs $(\text{ms})$
- $\delta(t)$, [Dirac delta function](<guide/mathematics:Dirac Delta Function>) $(\text{ms}^{-1})$

Under the condition $\tau_d > \tau_r > 0$.

### Description
This models a synapse where the binding of neurotransmitters is not instantaneous, but instead has both a rise and fall, each controlled by a separate exponential term. For instance, this is typically used when modelling synapses with AMPA and NMDA receptors. AMPA activity is mediated by $\tau_r$ and NMDA is mediated by $\tau_d$, where $\tau_d$ is roughly an order of magnitude larger than $\tau_r$.

The peak current following a spike at time $t_f$ is equal to $t_\text{peak} + t_f$ where $t_\text{peak}$ is computed as follows.

$$t_\text{peak} = \frac{\tau_d\tau_r}{\tau_d - \tau_r} \ln\left(\frac{\tau_d}{\tau_r}\right)$$

This is also used to determine the value for $Q$ for which the peak current will equal 1.
$$K = \frac{\tau_d - \tau_r}{\exp(-\frac{t_\text{peak}}{\tau_d}) - \exp(-\frac{t_\text{peak}}{\tau_r})}$$

### References
1. [DOI:10.3390/brainsci12070863](https://www.mdpi.com/2076-3425/12/7/863)
1. [DOI:10.3389/fncom.2022.806086](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2022.806086/full)