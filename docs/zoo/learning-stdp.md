# STDP-Like Learning Methods
## Hebbian Learning
Typically STDP is parameterized such that it performs Hebbian learning (often summarized as "cells that fire together wire together"). In Hebbian learning methods, if there exists a "postsynaptic" neuron which receives input from a "presynaptic" neuron, then if the presynaptic neuron fires and afterwards the postsynaptic neuron fires, its considered to be "causal". Likewise, if the postsynaptic neuron fires then the presynaptic neuron fires, its considered to be "anti-causal". Hebbian learning increases the connection weight on neuron firings which are causal and decreases it on anti-causal firings.

Most STDP methods are phrased in the context of Hebbian learning, although they do not need to be. For instance, STDP can reverse the direction of the weight updates for causal and anti-causal firings (this is called anti-Hebbian learning). For the below methods, they will be written as they are most commonly described, although Inferno typically supports generalization.

## Spike Timing-Dependent Plasticity (STDP)
### Formulation
$$
\frac{dw}{dt} = A_+ x_\text{pre}(t) \sum_{\mathcal{F}_\text{post}} \delta(t - t^f_\text{post}) - A_- x_\text{post}(t) \sum_{\mathcal{F}_\text{pre}} \delta(t - t^f_\text{pre})
$$

*With solution:*

$$
w(t + \Delta t) - w(t) = A_+ x_\text{pre}(t) \left[ t = t^f_\text{post} \right]
- A_- x_\text{post}(t) \left[ t = t^f_\text{pre} \right]
$$

*Where:*
- $w$, connection weight
- $A_+$, update magnitude for Hebbian long-term potentiation (LTP)
- $A_-$, update magnitude for Hebbian long-term depression (LTD)
- $x_\text{post}$, [spike trace](<zoo/spike-trace:Overview>) of postsynaptic (output) spikes, parameterized by time constant $\tau_\text{post}$
- $x_\text{pre}$, [spike trace](<zoo/spike-trace:Overview>) of presynaptic (input) spikes, parameterized by time constant $\tau_\text{pre}$
- $t$ is the current time step
- $t^f_\text{post}$, time of (the most recent) postsynaptic spike
- $t^f_\text{pre}$, time of (the most recent) presynaptic spike
- $\mathcal{F}_\text{post}$, set of prior postsynaptic spikes
- $\mathcal{F}_\text{pre}$, set of prior presynaptic spikes
- $\delta$, [Dirac delta function](<guide/mathematics:Dirac Delta Function>)

$[\cdots]$ is the Iverson bracket and equals $1$ if the inner statement is true and $0$ if it is false.

## Delayed Spike Timing-Dependent Plasticity (Delayed STDP)
### Formulation
$$
\Delta w_t =
\begin{cases}
    A_+ \exp \left(-\frac{\Delta t_{pp}}{\tau_+}\right) &\Delta t_{pp} > 0 \\
    0 &\Delta t_{pp} = 0 \\
    -A_- \exp \left(\frac{\Delta t_{pp}}{\tau_-}\right) &\Delta t_{pp} < 0
\end{cases}
$$

$$
\Delta t_{pp} = t^f_\text{post} - t^f_\text{pre} - d_{pp}
$$

*Where:*
- $\Delta w_t$, change in weight
- $A_+$, update magnitude for long-term potentiation (LTP)
- $A_-$, update magnitude for long-term depression (LTD)
- $\tau_+$, time constant for potentiation
- $\tau_-$, time constant for depression
- $\Delta t_{pp}$, adjusted time between neighboring post and presynaptic spikes
- $t^f_\text{post}$, time of the most recent postsynaptic spike
- $t^f_\text{pre}$, time of the most recent presynaptic spike
- $d_{pp}$, length of the delay between input the neuron

## Modulated Spike-Timing Dependent Plasticity (MSTDP)
### Formulation
$$
\begin{align*}
    \frac{dw}{dt} &= \gamma \, r(t) \, \xi(t) \\
    \xi(t) &= P^+ \Phi_\text{post}(t) + P^- \Phi_\text{pre}(t) \\
    \tau_+ \frac{dP^+}{dt} &= -P^+ + A_+ \Phi_\text{pre}(t) \\
    \tau_- \frac{dP^-}{dt} &= -P^- + A_- \Phi_\text{post}(t) \\
    \Phi_n(t) &= \sum_{\mathcal{F}_n} \delta(t - t_n^f)
\end{align*}
$$

*With solutions:*

$$
\begin{align*}
    w(t + \Delta t) - w(t) &= \gamma \, r(t) \, \zeta(t) \\
    \zeta(t) &= P^+ \left[t = t_\text{post}^f\right] + P^- \left[t = t_\text{pre}^f\right] \\
    P^-(t) &= P^+(t - \Delta t) \exp \left(-\frac{\Delta t}{\tau_+}\right) + A_+\left[t = t_\text{pre}^f\right] \\
    P^-(t) &= P^-(t - \Delta t) \exp \left(-\frac{\Delta t}{\tau_-}\right) + A_- \left[t = t_\text{post}^f\right]
\end{align*}
$$

$[\cdots]$ is the Iverson bracket and equals $1$ if the inner statement is true and $0$ if it is false.

*Where:*
- $w$, connection weight
- $\gamma$, common learning rate
- $r$, reward term
- $A_+$, update magnitude for Hebbian long-term potentiation (LTP)
- $A_-$, update magnitude for Hebbian long-term depression (LTD)
- $P^-$, [spike trace](<zoo/spike-trace:Overview>) of postsynaptic (output) spikes, parameterized by time constant $\tau_\text{post}$
- $P^+$, [spike trace](<zoo/spike-trace:Overview>) of presynaptic (input) spikes, parameterized by time constant $\tau_\text{pre}$
- $t$ is the current time step
- $t^f_\text{post}$, time of (the most recent) postsynaptic spike
- $t^f_\text{pre}$, time of (the most recent) presynaptic spike
- $\mathcal{F}_\text{post}$, set of prior postsynaptic spikes
- $\mathcal{F}_\text{pre}$, set of prior presynaptic spikes
- $\delta$, [Dirac delta function](<guide/mathematics:Dirac Delta Function>)


### Description
This is equivalent to [STDP](#spike-timing-dependent-plasticity-stdp) except scaled by a time-dependent reward term $r$. Note that $P^+$ is the presynaptic spike trace and $P^-$ is the postsynaptic spike trace (calculated as [cumulative trace](<zoo/spike-trace:Cumulative Trace>)).

### References
1. [10.1162/neco.2007.19.6.1468](https://florian.io/papers/2007_Florian_Modulated_STDP.pdf)

## Modulated Spike-Timing Dependent Plasticity with Eligibility Trace (MSTDPET)
$$
\begin{align*}
    \frac{dw}{dt} &= \gamma \, r(t) \, z(t) \\
    \tau_z \frac{dz}{dt} &= -z(t) + \xi(t) \\
    \xi(t) &= P^+ \Phi_\text{post}(t) + P^- \Phi_\text{pre}(t) \\
    \tau_+ \frac{dP^+}{dt} &= -P^+ + A_+ \Phi_\text{pre}(t) \\
    \tau_- \frac{dP^-}{dt} &= -P^- + A_- \Phi_\text{post}(t) \\
    \Phi_n(t) &= \sum_{\mathcal{F}_n} \delta(t - t_n^f)
\end{align*}
$$

*With approximations and solutions:*

$$
\begin{align*}
    w(t + \Delta t) - w(t) &= \gamma \Delta t \, r(t + \Delta t) \, z(t + \Delta t) \\
    z(t + \Delta t) &= z(t) \exp\left(-\frac{\Delta t}{\tau_z}\right) + \frac{\zeta(t)}{\tau_z} \\
    \zeta(t) &= P^+ \left[t = t_\text{post}^f\right] + P^- \left[t = t_\text{pre}^f\right] \\
    P^+(t) &= P^+(t - \Delta t) \exp \left(-\frac{\Delta t}{\tau_+}\right) + A_+\left[t = t_\text{pre}^f\right] \\
    P^-(t) &= P^-(t - \Delta t) \exp \left(-\frac{\Delta t}{\tau_-}\right) + A_- \left[t = t_\text{post}^f\right]
\end{align*}
$$

*Where:*
- $w$, connection weight
- $z$, eligibility trace
- $\gamma$, common learning rate
- $r$, reward term
- $A_+$, update magnitude for Hebbian long-term potentiation (LTP)
- $A_-$, update magnitude for Hebbian long-term depression (LTD)
- $P^-$, [spike trace](<zoo/spike-trace:Overview>) of postsynaptic (output) spikes, parameterized by time constant $\tau_\text{post}$
- $P^+$, [spike trace](<zoo/spike-trace:Overview>) of presynaptic (input) spikes, parameterized by time constant $\tau_\text{pre}$
- $t$ is the current time step
- $t^f_\text{post}$, time of (the most recent) postsynaptic spike
- $t^f_\text{pre}$, time of (the most recent) presynaptic spike
- $\mathcal{F}_\text{post}$, set of prior postsynaptic spikes
- $\mathcal{F}_\text{pre}$, set of prior presynaptic spikes
- $\delta$, [Dirac delta function](<guide/mathematics:Dirac Delta Function>)

### References
1. [10.1162/neco.2007.19.6.1468](https://florian.io/papers/2007_Florian_Modulated_STDP.pdf)

## Triplet Spike-Timing Dependent Plasticity (STDP)