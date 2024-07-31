# Neurons, Linear Models

## Leaky Integrate-and-Fire (LIF)
### Formulation
$$
\tau_m \frac{dV_m(t)}{dt} = - \left[V_m(t) - V_\text{rest}\right] + R_mI(t)
$$

*With solution:*

$$
V_m(t + \Delta t) = \left[V_m(t) - V_\text{rest} - R_mI(t)\right] \exp\left(-\frac{\Delta t}{\tau_m}\right) + V_\text{rest} + R_mI(t)
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
This model defines the basic leaky integrate-and-fire neuron, without the incorporation of adaptive behavior or "biologically defined reset rules". This is equivalent to the [GLIF{sub}`1`](https://www.nature.com/articles/s41467-017-02717-4) (generalized leaky integrate-and-fire) model. Without presynaptic current, the membrane voltage decays back towards the rest voltage with rate in inverse proportion to the membrane time constant.

```{image} ../images/plots/lif-slope-field-light.png
:alt: Linear Integrate-and-Fire Slope Field of Membrane Voltage
:class: only-light
:width: 30em
:align: center
```

```{image} ../images/plots/lif-slope-field-dark.png
:alt: Linear Integrate-and-Fire Slope Field of Membrane Voltage
:class: only-dark
:width: 30em
:align: center
```

Slope field of the membrane voltage without any input current showing the decay towards
the rest voltage $(V_R = -60 \text{ mV})$ over time. Plotted with value $\tau_m=1 \text{ ms}$
over a time of $2 \text{ ms}$.

### Alternative Formulations
$$
\begin{align*}
C_m \frac{dV_m(t)}{dt} &= - \frac{1}{R_m}\left[V_m(t) - V_\text{rest}\right] + I(t) \\
\tau_m \frac{dV_m(t)}{dt} &= - \left[V_m(t) - V_\text{rest}\right] + \frac{1}{g_L}I(t) \\
\end{align*}
$$

In the first alternative formulation, the membrane resistance $R_m$, given in $\text{M}\Omega$, is not multiplied into each side, leaving the membrane capacitance $C_m$, given in $\text{nF}$.

In the second alternative formulation, rather than considering the resistance of the membrane, it is instead phrased in terms of the membrane's leak conductance $g_L$, given in $\mu\text{S}$.

These formulations are all equivalent, but expose different underlying properties of the neuron. Given the formulation used in Inferno, the other values can be calculated as follows.

$$
\begin{align*}
    C_m &= \tau_m R_m^{-1} \\
    g_L &= R_m^{-1}
\end{align*}
$$

### References
1. [DOI:10.1017/CBO9781107447615 (§1.3)](https://neuronaldynamics.epfl.ch/online/Ch1.S3.html)
1. [ISBN:9780262548083](https://mitpress.ublish.com/ebook/modeling-neural-circuits-made-simple-with-python-preview/12788/Cover)
1. [DOI:10.1038/s41467-017-02717-4](https://www.nature.com/articles/s41467-017-02717-4)

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
    V_m(t + \Delta t) &= \left[V_m(t) - V_\text{rest} - R_mI(t)\right] \exp\left(-\frac{\Delta t}{\tau_m}\right) + V_\text{rest} + R_mI(t) \\
    \theta_k(t + \Delta t) &= \theta_k(t) \exp\left(-\frac{\Delta t}{\tau_k}\right)
\end{align*}
$$

*After an action potential is generated:*

$$
\begin{align*}
    V_m(t) &\leftarrow V_\text{reset} \\
    \theta_k(t) &\leftarrow \theta_k(t) + d_k
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
- $d_k$, spike-triggered voltage threshold adaptation $(\text{mV})$
- $\tau_k$, adaptation time constant $(\text{ms})$
- $t$, current time of the simulation $(\text{ms})$
- $\Delta t$, length of time over which each simulation step occurs $(\text{ms})$

### Description
This model uses the underlying dynamics of the leaky integrate-and-fire neuron, but it incorporates a spike-dependent adaptive threshold. This is equivalent to the [GLIF{sub}`2`](https://www.nature.com/articles/s41467-017-02717-4) (generalized leaky integrate-and-fire) model with the exception of membrane reset voltage behavior.

### References
1. [DOI:10.1038/s41467-017-02717-4](https://www.nature.com/articles/s41467-017-02717-4)
1. [DOI:10.1038/s41467-020-17236-y](https://www.nature.com/articles/s41467-020-17236-y)
1. [Allen Institute GLIF Whitepaper](https://community.brain-map.org/uploads/short-url/8Q1u3ecpUDRHIuCXF05cAd6PEeE.pdf)


## Generalized Leaky Integrate-and-Fire 1 (GLIF{sub}`1`)
See: [Leaky Integrate-and-Fire (LIF)](#leaky-integrate-and-fire-lif)

## Generalized Leaky Integrate-and-Fire 2 (GLIF{sub}`2`)
### Formulation
$$
\begin{align*}
    \tau_m \frac{dV_m(t)}{dt} &= - \left[V_m(t) - V_\text{rest}\right] + R_mI(t) \\
    \Theta(t) &= \Theta_\infty + \sum_k \theta_k(t) \\
    \frac{d\theta_k(t)}{dt} &= -\lambda \theta_k(t) \\
\end{align*}
$$

*With solutions:*

$$
\begin{align*}
    V_m(t + \Delta t) &= \left[V_m(t) - V_\text{rest} - R_mI(t)\right] \exp\left(-\frac{\Delta t}{\tau_m}\right) + V_\text{rest} + R_mI(t) \\
    \theta_k(t + \Delta t) &= \theta_k(t) \exp\left(-\lambda_k \Delta t\right)
\end{align*}
$$

*After an action potential is generated:*

$$
\begin{align*}
    V_m(t) &\leftarrow V_\text{rest} + m_v \left[ V_m(t) - V_\text{rest} \right] - b_v \\
    \theta_k(t) &\leftarrow \theta_k(t) + d_k
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
- $d_k$, spike-triggered voltage threshold adaptation $(\text{mV})$
- $\lambda_k$, adaptation rate constant $(\text{ms}^{-1})$
- $t$, current time of the simulation $(\text{ms})$
- $\Delta t$, length of time over which each simulation step occurs $(\text{ms})$

### Description
Also called "leaky integrate-and-fire with biologically defined reset rules (LIF-R)", this model uses conventional LIF dynamics, with a linear spike-dependent adaptive threshold and a reset voltage which is contingent on the membrane voltage reached when spiking. Note that the adaptation decay is defined in terms of a rate constant $\lambda_k$ rather than a time constant $\tau_k$.

### References
1. [DOI:10.1038/s41467-017-02717-4](https://www.nature.com/articles/s41467-017-02717-4)
1. [Allen Institute GLIF Whitepaper](https://community.brain-map.org/uploads/short-url/8Q1u3ecpUDRHIuCXF05cAd6PEeE.pdf)

## Generalized Leaky Integrate-and-Fire 3 (GLIF{sub}`3`)
```{admonition} Work In Progress
This is not yet implemented and the documentation is incomplete.
```

## Generalized Leaky Integrate-and-Fire 4 (GLIF{sub}`4`)
```{admonition} Work In Progress
This is not yet implemented and the documentation is incomplete.
```

## Generalized Leaky Integrate-and-Fire 5 (GLIF{sub}`5`)
```{admonition} Work In Progress
This is not yet implemented and the documentation is incomplete.
```