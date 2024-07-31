# Learning, STDP Methods
Typically STDP is parameterized such that it performs Hebbian learning (often summarized as "cells that fire together wire together"). In Hebbian learning methods there is a *postsynaptic* neuron which receives input from a *presynaptic* neuron. Then, if the presynaptic neuron fires and after the postsynaptic neuron fires, the connection between them is considered to be *causal*. Likewise, if the postsynaptic neuron fires then the presynaptic neuron fires, it's considered to be *anti-causal*. Hebbian learning increases the connection weight on neuron firings which are causal and decreases it on anti-causal firings. Most STDP methods are phrased in the context of Hebbian learning, although some explicitly are not, while others are flexible even if written in terms of Hebbian learning.

```{note}
Although STDP methods are phrased as originally described, Inferno tries to introduce flexibility where possible. The settings for controlling the behavior of each method will be described in the relevant {py:class}`~inferno.learn.CellTrainer` class.
```

## Spike-Timing Dependent Plasticity (STDP)
### Formulation
$$
\begin{align*}
    \frac{dw}{dt} &= x_\text{pre}(t) \sum_{\mathcal{F}_\text{post}} \delta(t - t^f_\text{post}) + x_\text{post}(t) \sum_{\mathcal{F}_\text{pre}} \delta(t - t^f_\text{pre}) \\
    \tau_\text{pre} \frac{dx_\text{pre}}{dt} &= -x_\text{pre} + A_+ \sum_f \delta (t - t_\text{pre}^f) \\
    \tau_\text{post} \frac{dx_\text{post}}{dt} &= -x_\text{post} + A_- \sum_f \delta (t - t_\text{post}^f)
\end{align*}
$$

*With solutions:*

$$
\begin{align*}
    w(t + \Delta t) - w(t) &= x_\text{pre}(t) \left[ t = t^f_\text{post} \right] + x_\text{post}(t) \left[ t = t^f_\text{pre} \right] \\
    x_\text{pre}(t) &= x_\text{pre}(t - \Delta t) \exp \left(-\frac{\Delta t}{\tau_\text{pre}}\right) + A_+ \left[t = t^f_\text{pre}\right] \\
    x_\text{post}(t) &= x_\text{post}(t - \Delta t) \exp \left(-\frac{\Delta t}{\tau_\text{post}}\right) + A_- \left[t = t^f_\text{post}\right]
\end{align*}
$$

*Where:*
- $w$, connection weight
- $A_+$, learning rate for Hebbian long-term potentiation (LTP)
- $A_-$, learning rate for Hebbian long-term depression (LTD)
- $x_\text{post}$, [spike trace](<guide/concepts:Trace>) of postsynaptic (output) spikes, parameterized by time constant $\tau_\text{post}$
- $x_\text{pre}$, [spike trace](<guide/concepts:Trace>) of presynaptic (input) spikes, parameterized by time constant $\tau_\text{pre}$
- $t$ is the current time step
- $t^f_\text{post}$, time of (the most recent) postsynaptic spike
- $t^f_\text{pre}$, time of (the most recent) presynaptic spike
- $\mathcal{F}_\text{post}$, set of prior postsynaptic spikes
- $\mathcal{F}_\text{pre}$, set of prior presynaptic spikes
- $\delta$, [Dirac delta function](<guide/mathematics:Dirac Delta Function>)

$[\cdots]$ is the Iverson bracket and equals $1$ if the inner statement is true and $0$ if it is false.

*Note:*

Training is Hebbian when $A_+$ is positive and $A_-$ is negative. When the sign of $A_+$ or $A_-$ is positive, the weights are potentiated, and when negative, the weights are depressed. The former is applied on a postsynaptic spike and the latter on a presynaptic spike.

### Description
With STDP, each time a presynaptic spike is received or a postsynaptic spike is generated, the connection weights are modified. The update on a postsynaptic spike is proportional to the presynaptic trace, an exponentially decaying value parameterized by a time constant $\tau_\text{pre}$, and the learning rate for the update, often denoted $A_+$, $A_\text{post}$, $\eta_+$, or $\eta_\text{post}$. The update on a presynaptic spike is proportional to the postsynaptic trace, an exponentially decaying value parameterized by a time constant $\tau_\text{post}$, and the learning rate for the update, often denoted $A_-$, $A_\text{pre}$, $\eta_-$, or $\eta_\text{pre}$.

```{image} ../images/plots/stdp-hyperparam-light.png
:alt: Effect of Hyperparameters on Spike-Timing Dependent Plasticity Updates
:class: only-light
:width: 30em
:align: center
```

```{image} ../images/plots/stdp-hyperparam-dark.png
:alt: Effect of Hyperparameters on Spike-Timing Dependent Plasticity Updates
:class: only-dark
:width: 30em
:align: center
```

Plot of the weight update curve using STDP. Values along the $x$-axis indicate how long
after the most recent presynaptic spike did a postsynaptic spike occur, $t^f_\text{post} - t^f_\text{pre}$. Values along the $y$-axis indicate the direction and magnitude of the corresponding weight update. Plotted with $A_\text{post} = 1.0$, $A_\text{pre} = -0.5$, $\tau_\text{pre} = 20 \text{ms}$, and $\tau_\text{post} = 30 \text{ms}$.

```{image} ../images/plots/stdp-mode-light.png
:alt: Different Modes of Spike-Timing Dependent Plasticity
:class: only-light
:width: 30em
:align: center
```

```{image} ../images/plots/stdp-mode-dark.png
:alt: Different Modes of Spike-Timing Dependent Plasticity
:class: only-dark
:width: 30em
:align: center
```

Plot of weight update curves using STDP with the different combinations of signs for learning rates. Note that LTP refers to long-term potentiation and LTD refers to long-term depression. Classical STDP follows the Hebbian curve. Dopaminergic STDP (DA-STDP) follows the "LTP-Only" curve.

### References
1. [DOI:10.1017/CBO9781107447615 (§19.2](https://neuronaldynamics.epfl.ch/online/Ch19.S2.html)

## Delay-Adjusted Spike-Timing Dependent Plasticity (Delay-Adjusted STDP)
```{admonition} Work In Progress
This is not yet implemented and the documentation is incomplete. The information presented may be incorrect.
```
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
    P^+(t) &= P^+(t - \Delta t) \exp \left(-\frac{\Delta t}{\tau_+}\right) + A_+\left[t = t_\text{pre}^f\right] \\
    P^-(t) &= P^-(t - \Delta t) \exp \left(-\frac{\Delta t}{\tau_-}\right) + A_- \left[t = t_\text{post}^f\right]
\end{align*}
$$

*Where:*
- $w$, connection weight
- $\gamma$, common learning rate
- $r$, reward term
- $A_+$, learning rate for Hebbian long-term potentiation (LTP)
- $A_-$, learning rate for Hebbian long-term depression (LTD)
- $P^-$, [spike trace](<guide/concepts:Trace>) of postsynaptic (output) spikes, parameterized by time constant $\tau_\text{post}$
- $P^+$, [spike trace](<guide/concepts:Trace>) of presynaptic (input) spikes, parameterized by time constant $\tau_\text{pre}$
- $t$ is the current time step
- $t^f_\text{post}$, time of (the most recent) postsynaptic spike
- $t^f_\text{pre}$, time of (the most recent) presynaptic spike
- $\mathcal{F}_\text{post}$, set of prior postsynaptic spikes
- $\mathcal{F}_\text{pre}$, set of prior presynaptic spikes
- $\delta$, [Dirac delta function](<guide/mathematics:Dirac Delta Function>)

$[\cdots]$ is the Iverson bracket and equals $1$ if the inner statement is true and $0$ if it is false.

*Note:*

Where $\gamma$ is positive, training is Hebbian when $A_+$ is positive and $A_-$ is negative. When the sign of $A_+$ or $A_-$ is positive, the weights are potentiated, and when negative, the weights are depressed. The former is applied on a postsynaptic spike and the latter on a presynaptic spike.

### Description
This is equivalent to [STDP](#spike-timing-dependent-plasticity-stdp) except scaled by a time-dependent reward term $r$. Note that $P^+$ is the presynaptic spike trace and $P^-$ is the postsynaptic spike trace (calculated as [cumulative trace](<guide/concepts:Cumulative Trace>)).

### References
1. [10.1162/neco.2007.19.6.1468](https://florian.io/papers/2007_Florian_Modulated_STDP.pdf)

## Modulated Spike-Timing Dependent Plasticity with Eligibility Trace (MSTDPET)
### Formulation
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
- $A_+$, learning rate for Hebbian long-term potentiation (LTP)
- $A_-$, learning rate for Hebbian long-term depression (LTD)
- $P^-$, [spike trace](<guide/concepts:Trace>) of postsynaptic (output) spikes, parameterized by time constant $\tau_\text{post}$
- $P^+$, [spike trace](<guide/concepts:Trace>) of presynaptic (input) spikes, parameterized by time constant $\tau_\text{pre}$
- $t$ is the current time step
- $t^f_\text{post}$, time of (the most recent) postsynaptic spike
- $t^f_\text{pre}$, time of (the most recent) presynaptic spike
- $\mathcal{F}_\text{post}$, set of prior postsynaptic spikes
- $\mathcal{F}_\text{pre}$, set of prior presynaptic spikes
- $\delta$, [Dirac delta function](<guide/mathematics:Dirac Delta Function>)

$[\cdots]$ is the Iverson bracket and equals $1$ if the inner statement is true and $0$ if it is false.

*Note:*

Where $\gamma$ is positive, training is Hebbian when $A_+$ is positive and $A_-$ is negative. When the sign of $A_+$ or $A_-$ is positive, the weights are potentiated, and when negative, the weights are depressed. The former is applied on a postsynaptic spike and the latter on a presynaptic spike.

### Description
This is equivalent to [MSTDP](#modulated-spike-timing-dependent-plasticity-mstdp) except the trace of what would have been the update term, the eligibility, is used instead. This has an exponential smoothing effect on the value of the weights. See the [Florian STDP](<examples/florian-stdp:Florian STDP>) example for a visual comparison.

### References
1. [10.1162/neco.2007.19.6.1468](https://florian.io/papers/2007_Florian_Modulated_STDP.pdf)
