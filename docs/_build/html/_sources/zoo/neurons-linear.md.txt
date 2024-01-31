# Neuron Models, Linear

## Leaky Integrate-and-Fire (LIF)
### Formulation
$$
\begin{align*}
    \tau_m \frac{dV_m(t)}{dt} &= - \left[V_m(t) - V_\text{rest}\right] + R_mI(t) \\
    V_m(t + \Delta t) &= \left[V_m(t) - V_\text{rest}\right] \exp\left(-\frac{t}{\tau_m}\right) + V_\text{rest} + R_mI(t)
\end{align*}
$$

*After an action potential is generated:*

$$V_m(t) \leftarrow V_\text{reset}$$

*Where:*
- $I$, total input current applied to the neuron $(\text{nA})$
- $V_m$, electric potential difference across the cell membrane $(\text{mV})$
- $V_\text{rest}$, equilibrium of the membrane potential $(\text{mV})$
- $V_\text{reset}$, membrane potential difference set after spiking $(\text{mV})$
- $\tau_m$, membrane time constant $(\text{ms})$
- $t$, current time of the simulation $(\text{ms})$
- $\Delta t$, length of time over which each simulation step occurs $(\text{ms})$

### Description
This model defines the basic leaky integrate-and-fire neuron, without the incorporation of adaptive behavior or "biologically defined reset rules". This is equivalent to the [GLIF{sub}`1`](https://www.nature.com/articles/s41467-017-02717-4) (generalized leaky integrate-and-fire) model.

### Alternative Formulations
$$
\begin{align*}
C_m \frac{dV_m(t)}{dt} &= - \frac{1}{R_m}\left[V_m(t) - V_\text{rest}\right] + I(t) \\
\tau_m \frac{dV_m(t)}{dt} &= - \left[V_m(t) - V_\text{rest}\right] + \frac{1}{g_L}I(t) \\
\end{align*}
$$

In the first alternative formulation, the membrane resistance $R_m$, given in $\text{M\Omega}$, is not multiplied into each side, leaving the membrane capacitance $C_m$, given in $\text{nF}$.

In the second alternative formulation, rather than considering the resistance of the membrane, it is instead phrased in terms of the membrane's leak conductance $g_L$, given in $\text{\mu S}$.

These formulations are all equivalent, but expose different underlying properties of the neuron. Given the formulation used in Inferno, the other values can be calculated as follows.

$$
\begin{align*}
    C_m &= \tau_m R_m^{-1} \\
    g_L &= R_m^{-1}
\end{align*}
$$

### References
1. [DOI:10.1017/CBO9781107447615 (Chapter 1.3)](https://neuronaldynamics.epfl.ch/online/Ch1.S3.html)
2. [ISBN:9780262548083](https://github.com/RobertRosenbaum/ModelingNeuralCircuits/blob/main/ModelingNeuralCircuits.pdf)
3. [DOI:10.1038/s41467-017-02717-4](https://www.nature.com/articles/s41467-017-02717-4)

## Adaptive Leaky Integrate-and-Fire (ALIF)
### Formulation
$$
\begin{align*}
    \tau_m \frac{dV_m(t)}{dt} &= - \left[V_m(t) - V_\text{rest}\right] + R_mI(t) \\
    \Theta(t) &= \Theta_\infty + \sum_k \theta_k(t) \\
    \tau_k \frac{d\theta_k(t)}{dt} &= -\theta_k(t) \\
\end{align*}
$$

*With solutions:*

$$
\begin{align*}
    V_m(t + \Delta t) &= \left[V_m(t) - V_\text{rest}\right] \exp\left(-\frac{t}{\tau_m}\right) + V_\text{rest} + R_mI(t) \\
    \theta_k(t + \Delta t) &= \theta_k(t) \exp\left(-\frac{\Delta t}{\tau_k}\right)
\end{align*}
$$

*After an action potential is generated:*

$$
\begin{align*}
    V_m(t) &\leftarrow V_\text{reset} \\
    \theta_k(t) &\leftarrow \theta_k(t) + a_k
\end{align*}
$$

*Where:*
- $I$, total input current applied to the neuron $(\text{nA})$
- $V_m$, electric potential difference across the cell membrane $(\text{mV})$
- $V_\text{rest}$, equilibrium of the membrane potential $(\text{mV})$
- $\tau_m$, membrane time constant $(\text{ms})$
- $\Theta$, membrane potential at which an action potential is generated $(\text{mV})$
- $\Theta_\infty$, equilibrium of the firing threshold $(\text{mV})$
- $\theta_k$, adaptive component of the firing threshold $(\text{mV})$
- $\theta_\text{reset}$, reset value of the adaptive component of the firing threshold $(\text{mV})$
- $a_k$, spike-triggered voltage threshold adaptation $(\text{mV})$
- $\tau_k$, adaptation time constant $(\text{ms})$
- $t$, current time of the simulation $(\text{ms})$
- $\Delta t$, length of time over which each simulation step occurs $(\text{ms})$

### Description
This model uses the underlying dynamics of the leaky integrate-and-fire neuron, but it incorporates a spike-dependent adaptive threshold. This is equivalent to the [GLIF{sub}`2`](https://www.nature.com/articles/s41467-017-02717-4) (generalized leaky integrate-and-fire) model with the exception of membrane reset voltage behavior.

### References
1. [DOI:10.1038/s41467-017-02717-4](https://www.nature.com/articles/s41467-017-02717-4)
2. [DOI:10.1038/s41467-020-17236-y](https://www.nature.com/articles/s41467-020-17236-y)
3. [Allen Institute GLIF Whitepaper](http://web.archive.org/web/20230428012128/https://help.brain-map.org/download/attachments/8323525/glifmodels.pdf)


## Generalized Leaky Integrate-and-Fire 1 (GLIF{sub}`1`)
See: [Leaky Integrate-and-Fire (LIF)](#leaky-integrate-and-fire-lif)

## Generalized Leaky Integrate-and-Fire 2 (GLIF{sub}`2`)
### Formulation
$$
\begin{align*}
    \tau_m \frac{dV_m(t)}{dt} &= - \left[V_m(t) - V_\text{rest}\right] + R_mI(t) \\
    \Theta(t) &= \Theta_\infty + \sum_k \theta_k(t) \\
    \tau_k \frac{d\theta_k(t)}{dt} &= -\theta_k(t) \\
\end{align*}
$$

*With solutions:*

$$
\begin{align*}
    V_m(t + \Delta t) &= \left[V_m(t) - V_\text{rest}\right] \exp\left(-\frac{t}{\tau_m}\right) + V_\text{rest} + R_mI(t) \\
    \theta_k(t + \Delta t) &= \theta_k(t) \exp\left(-\frac{\Delta t}{\tau_k}\right)
\end{align*}
$$

*After an action potential is generated:*

$$
\begin{align*}
    V_m(t) &\leftarrow V_\text{rest} + m_v \left[ V_m(t) - V_\text{rest} \right] - b_v \\
    \theta_k(t) &\leftarrow \theta_k(t) + a_k
\end{align*}
$$

*Where:*
- $I$, total input current applied to the neuron $(\text{nA})$
- $V_m$, electric potential difference across the cell membrane $(\text{mV})$
- $V_\text{rest}$, equilibrium of the membrane potential $(\text{mV})$
- $m_v$, spike-triggered voltage fraction, as slope $(\text{1})$
- $b_v$, spike-triggered voltage addition, as intercept $(\text{mV})$
- $\tau_m$, membrane time constant $(\text{ms})$
- $\Theta$, membrane potential at which an action potential is generated $(\text{mV})$
- $\Theta_\infty$, equilibrium of the firing threshold $(\text{mV})$
- $\theta_k$, adaptive component of the firing threshold $(\text{mV})$
- $\theta_\text{reset}$, reset value of the adaptive component of the firing threshold $(\text{mV})$
- $a_k$, spike-triggered voltage threshold adaptation $(\text{mV})$
- $\tau_k$, adaptation time constant $(\text{ms})$
- $t$, current time of the simulation $(\text{ms})$
- $\Delta t$, length of time over which each simulation step occurs $(\text{ms})$

### Description
Also called "leaky integrate-and-fire with biologically defined reset rules (LIF-R)", this model uses conventional LIF dynamics, with a linear spike-dependent adaptive threshold and a reset voltage which is contingent on the membrane voltage reached when spiking.

### References
1. [DOI:10.1038/s41467-017-02717-4](https://www.nature.com/articles/s41467-017-02717-4)
2. [Allen Institute GLIF Whitepaper](http://web.archive.org/web/20230428012128/https://help.brain-map.org/download/attachments/8323525/glifmodels.pdf)

## Generalized Leaky Integrate-and-Fire 3 (GLIF{sub}`3`)
***(To Be Implemented)***

## Generalized Leaky Integrate-and-Fire 4 (GLIF{sub}`4`)
***(To Be Implemented)***

## Generalized Leaky Integrate-and-Fire 5 (GLIF{sub}`5`)
***(To Be Implemented)***
