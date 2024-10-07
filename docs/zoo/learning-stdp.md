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
    \frac{dx_\text{pre}}{dt} &= -\frac{x_\text{pre}}{\tau_\text{pre}} + A_+ \sum_{\mathcal{F}_\text{pre}} \delta (t - t_\text{pre}^f) \\
    \frac{dx_\text{post}}{dt} &= -\frac{x_\text{post}}{\tau_\text{post}} + A_- \sum_{\mathcal{F}_\text{post}} \delta (t - t_\text{post}^f)
\end{align*}
$$

*With solutions:*

$$
\begin{align*}
    w(t + \Delta t) - w(t) &= x_\text{pre}(t) \bigl[ t = t^f_\text{post} \bigr] + x_\text{post}(t) \bigl[ t = t^f_\text{pre} \bigr] \\
    x_\text{pre}(t) &= x_\text{pre}(t - \Delta t) \exp \left(-\frac{\Delta t}{\tau_\text{pre}}\right) + A_+ \bigl[t = t^f_\text{pre}\bigr] \\
    x_\text{post}(t) &= x_\text{post}(t - \Delta t) \exp \left(-\frac{\Delta t}{\tau_\text{post}}\right) + A_- \bigl[t = t^f_\text{post}\bigr]
\end{align*}
$$

*Where:*
- $w$, connection weight
- $A_+$, learning rate for postsynaptic events, Hebbian long-term potentiation (LTP) when positive
- $A_-$, learning rate for presynaptic events, Hebbian long-term depression (LTD) when negative
- $x_\text{post}$, [trace](<guide/concepts:Trace>) of postsynaptic spikes
- $x_\text{pre}$, trace of presynaptic spikes
- $\tau_\text{post}$, time constant of [exponential decay](<guide/mathematics:Exponential Decay and Time Constants>) for the postsynaptic trace
- $\tau_\text{pre}$, time constant of exponential decay for the presynaptic trace
- $t$, current simulation time
- $\Delta t$, duration of the simulation step
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

## Triplet Spike-Timing Dependent Plasticity (Triplet STDP)
### Formulation
$$
\begin{align*}
    \frac{dw}{dt} &= r_1(t)\left({A_2}^+ + o_2(t - \epsilon){A_3}^+\right) \sum_{\mathcal{F}_\text{post}} \delta(t - t^f_\text{post}) \\
    &+ o_1(t)\left({A_2}^- + r_2(t - \epsilon){A_3}^-\right) \sum_{\mathcal{F}_\text{pre}} \delta(t - t^f_\text{pre}) \\
    \frac{dr_1}{dt} &= -\frac{r_1(t)}{\tau_+} + \sum_{\mathcal{F}_\text{pre}} \delta (t - t_\text{pre}^f) \\
    \frac{dr_2}{dt} &= -\frac{r_2(t)}{\tau_x} + \sum_{\mathcal{F}_\text{pre}} \delta (t - t_\text{pre}^f) \\
    \frac{do_1}{dt} &= -\frac{o_1(t)}{\tau_-} + \sum_{\mathcal{F}_\text{post}} \delta (t - t_\text{post}^f) \\
    \frac{do_2}{dt} &= -\frac{o_2(t)}{\tau_y} + \sum_{\mathcal{F}_\text{post}} \delta (t - t_\text{post}^f)
\end{align*}
$$

*With solutions:*

$$
\begin{align*}
    w(t + \Delta t) - w(t) &= r_1(t)\left({A_2}^+ + o_2(t - \Delta t){A_3}^+\right) \bigl[ t = t^f_\text{post} \bigr] \\
    &+ o_1(t)\left({A_2}^- + r_2(t - \Delta t){A_3}^-\right) \bigl[ t = t^f_\text{pre} \bigr] \\
    r_1(t) &= r_1(t - \Delta t) \exp \left(-\frac{\Delta t}{\tau_+}\right) + \bigl[t = t^f_\text{pre}\bigr] \\
    r_2(t) &= r_2(t - \Delta t) \exp \left(-\frac{\Delta t}{\tau_x}\right) + \bigl[t = t^f_\text{pre}\bigr] \\
    o_1(t) &= o_1(t - \Delta t) \exp \left(-\frac{\Delta t}{\tau_-}\right) + \bigl[t = t^f_\text{post}\bigr] \\
    o_2(t) &= o_2(t - \Delta t) \exp \left(-\frac{\Delta t}{\tau_y}\right) + \bigl[t = t^f_\text{post}\bigr]
\end{align*}
$$

*Where:*
- $w$, connection weight
- $r_1$, fast [trace](<guide/concepts:Trace>) of presynaptic spikes
- $r_2$, slow trace of presynaptic spikes
- $o_1$, fast trace of postsynaptic spikes
- $o_2$, slow trace of postsynaptic spikes
- $\tau_+$, time constant of [exponential decay](<guide/mathematics:Exponential Decay and Time Constants>) for fast presynaptic trace
- $\tau_x$, time constant of exponential decay for slow presynaptic trace
- $\tau_-$, fast time constant of exponential decay for postsynaptic trace
- $\tau_y$, time constant of exponential decay for slow postsynaptic trace
- ${A_2}^+$, learning rate for postsynaptic pair events, Hebbian long-term potentiation (LTP) when positive
- ${A_3}^+$, learning rate for postsynaptic triplet events, unsigned
- ${A_2}^-$, learning rate for presynaptic pair events, Hebbian long-term depression (LTD) when negative
- ${A_3}^-$, learning rate for presynaptic triplet events, unsigned
- $\epsilon$, small positive constant
- $t$, current simulation time
- $\Delta t$, duration of the simulation step
- $t^f_\text{post}$, time of (the most recent) postsynaptic spike
- $t^f_\text{pre}$, time of (the most recent) presynaptic spike
- $\mathcal{F}_\text{post}$, set of prior postsynaptic spikes
- $\mathcal{F}_\text{pre}$, set of prior presynaptic spikes
- $\delta$, [Dirac delta function](<guide/mathematics:Dirac Delta Function>)

$[\cdots]$ is the Iverson bracket and equals $1$ if the inner statement is true and $0$ if it is false.

*And the following conditions hold:*

- $0 < \tau_+ < \tau_x$
- $0 < \tau_- < \tau_y$
- $\text{sgn}({A_3}^+) = 1$
- $\text{sgn}({A_3}^-) = 1$

*Note:*

As a deviation from the original, in this formulation, it is expected that when ${A_2}^-$ is negatively signed and ${A_2}^+$ is positively signed, the updates will be Hebbian. Additionally, the original specifies $\epsilon$ as a "small positive constant" in both continuous and discrete formulations. The term $t - \epsilon$, is used to indicate the value at a time close to but before $t$. In the discrete version, $\Delta t$ is substituted in for $\epsilon$, as it is the smallest meaningful difference in time.

### Description
Unlike classical STDP which only triggers an update for spike pairs, triplet STDP extends this to spike triplets. ${A_2}^+$ scales updates when a postsynaptic spike follows a presynaptic spike $(f_\text{pre} \rightarrow f_\text{post})$. ${A_3}^+$ scales updates when a postsynaptic spike follows a presynaptic spike which itself follows another postsynaptic spike $(f_\text{post} \rightarrow f_\text{pre} \rightarrow f_\text{post})$. ${A_2}^-$ scales updates when a presynaptic spike follows a postsynaptic spike $(f_\text{post} \rightarrow f_\text{pre})$. ${A_3}^-$ scales updates when a presynaptic spike follows a postsynaptic spike which itself follows another presynaptic spike $(f_\text{pre} \rightarrow f_\text{post} \rightarrow f_\text{pre})$.

### References
1. [DOI:10.1523/JNEUROSCI.1425-06.2006](https://www.jneurosci.org/content/26/38/9673)
1. [DOI:10.1007/s00422-008-0233-1](https://link.springer.com/article/10.1007/s00422-008-0233-1)

## Modulated Spike-Timing Dependent Plasticity (MSTDP)
### Formulation
$$
\begin{align*}
    \frac{dw}{dt} &= \gamma \, M(t) \, \xi(t) \\
    \xi(t) &= P^+ \Phi_\text{post}(t) + P^- \Phi_\text{pre}(t) \\
    \frac{dP^+}{dt} &= -\frac{P^+}{\tau_+} + A_+ \Phi_\text{pre}(t) \\
    \frac{dP^-}{dt} &= -\frac{P^-}{\tau_-} + A_- \Phi_\text{post}(t) \\
    \Phi_n(t) &= \sum_{\mathcal{F}_n} \delta(t - t_n^f)
\end{align*}
$$

*With solutions:*

$$
\begin{align*}
    w(t + \Delta t) - w(t) &= \gamma \, M(t) \, \zeta(t) \\
    \zeta(t) &= P^+ \bigl[t = t_\text{post}^f\bigr] + P^- \bigl[t = t_\text{pre}^f\bigr] \\
    P^+(t) &= P^+(t - \Delta t) \exp \left(-\frac{\Delta t}{\tau_+}\right) + A_+\bigl[t = t_\text{pre}^f\bigr] \\
    P^-(t) &= P^-(t - \Delta t) \exp \left(-\frac{\Delta t}{\tau_-}\right) + A_- \bigl[t = t_\text{post}^f\bigr]
\end{align*}
$$

*Where:*
- $w$, connection weight
- $\gamma$, scaling factor
- $M$, modulation term
- $A_+$, learning rate for postsynaptic events, Hebbian long-term potentiation (LTP) when positive
- $A_-$, learning rate for presynaptic events, Hebbian long-term depression (LTD) when negative
- $P^-$, [trace](<guide/concepts:Trace>) of postsynaptic spikes
- $P^+$, trace of presynaptic spikes
- $\tau_-$, time constant of [exponential decay](<guide/mathematics:Exponential Decay and Time Constants>) for the postsynaptic trace
- $\tau_+$, time constant of exponential decay for the presynaptic trace
- $t$, current simulation time
- $\Delta t$, duration of the simulation step
- $t^f_\text{post}$, time of (the most recent) postsynaptic spike
- $t^f_\text{pre}$, time of (the most recent) presynaptic spike
- $\mathcal{F}_\text{post}$, set of prior postsynaptic spikes
- $\mathcal{F}_\text{pre}$, set of prior presynaptic spikes
- $\delta$, [Dirac delta function](<guide/mathematics:Dirac Delta Function>)

$[\cdots]$ is the Iverson bracket and equals $1$ if the inner statement is true and $0$ if it is false.

### Description
This is equivalent to [STDP](#spike-timing-dependent-plasticity-stdp) except scaled by a time-dependent modulation term $M$. The spike traces $P^-$ and $P^+$ are, in the original formulation, calculated as [cumulative trace](<guide/concepts:Cumulative Trace>).

### References
1. [10.1162/neco.2007.19.6.1468](https://florian.io/papers/2007_Florian_Modulated_STDP.pdf)
1. [10.3389/fncir.2015.00085](https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2015.00085/full)

## Modulated Spike-Timing Dependent Plasticity with Eligibility Trace (MSTDPET)
### Formulation
$$
\begin{align*}
    \frac{dw}{dt} &= \gamma \, M(t) \, z(t) \\
    \tau_z \frac{dz}{dt} &= -z(t) + \xi(t) \\
    \xi(t) &= P^+ \Phi_\text{post}(t) + P^- \Phi_\text{pre}(t) \\
    \frac{dP^+}{dt} &= -\frac{P^+}{\tau_+} + A_+ \Phi_\text{pre}(t) \\
    \frac{dP^-}{dt} &= -\frac{P^-}{\tau_-} + A_- \Phi_\text{post}(t) \\
    \Phi_n(t) &= \sum_{\mathcal{F}_n} \delta(t - t_n^f)
\end{align*}
$$

*With solutions:*

$$
\begin{align*}
    w(t + \Delta t) - w(t) &= \gamma \, \Delta t \, M(t) \, z(t) \\
    z(t) &= z(t - \Delta t) \exp\left(-\frac{\Delta t}{\tau_z}\right) + \frac{\zeta(t)}{\tau_z} \\
    \zeta(t) &= P^+ \bigl[t = t_\text{post}^f\bigr] + P^- \bigl[t = t_\text{pre}^f\bigr] \\
    P^+(t) &= P^+(t - \Delta t) \exp \left(-\frac{\Delta t}{\tau_+}\right) + A_+\bigl[t = t_\text{pre}^f\bigr] \\
    P^-(t) &= P^-(t - \Delta t) \exp \left(-\frac{\Delta t}{\tau_-}\right) + A_- \bigl[t = t_\text{post}^f\bigr]
\end{align*}
$$

*Where:*
- $w$, connection weight
- $z$, eligibility [trace](<guide/concepts:Trace>)
- $\tau_z$ time constant of [exponential decay](<guide/mathematics:Exponential Decay and Time Constants>) for eligibility trace
- $\gamma$, scaling factor
- $M$, modulation term
- $A_+$, learning rate for postsynaptic events, Hebbian long-term potentiation (LTP) when positive
- $A_-$, learning rate for presynaptic events, Hebbian long-term depression (LTD) when negative
- $P^-$, trace of postsynaptic spikes
- $P^+$, trace of presynaptic spikes
- $\tau_-$, time constant of exponential decay for postsynaptic trace
- $\tau_+$, time constant of exponential decay for presynaptic trace
- $t$, current simulation time
- $\Delta t$, duration of the simulation step
- $t^f_\text{post}$, time of (the most recent) postsynaptic spike
- $t^f_\text{pre}$, time of (the most recent) presynaptic spike
- $\mathcal{F}_\text{post}$, set of prior postsynaptic spikes
- $\mathcal{F}_\text{pre}$, set of prior presynaptic spikes
- $\delta$, [Dirac delta function](<guide/mathematics:Dirac Delta Function>)

$[\cdots]$ is the Iverson bracket and equals $1$ if the inner statement is true and $0$ if it is false.

### Description
This is equivalent to [MSTDP](#modulated-spike-timing-dependent-plasticity-mstdp) except the trace of what would have been the update term, the eligibility, is used instead. This has an exponential smoothing effect on the value of the \text{pre}. See the [Florian STDP](<examples/florian-stdp:Florian STDP>) example for a visual comparison.

The form of the modulation can vary. For example, in the case of reward-modulated spike-timing dependent plasticity (R-STDP), the modulation term is defined as $M(t) = R(t) - b$ where $R(t)$ is the reward signal at time $t$ and $b$ is some baseline (often the running average of $R$).

Some sources have the formulation of the eligibility trace $z$ differ slightly. The above formulation is sourced from [[1]](https://florian.io/papers/2007_Florian_Modulated_STDP.pdf) whereas the below is given in [[2]](https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2015.00085/full).

$$\tau_z \frac{dz}{dt} = -z(t) + \xi(t)$$

With the following solution.

$$z(t) = z(t - \Delta t) \exp\left(-\frac{\Delta t}{\tau_z}\right) + \zeta(t)$$

This comes from some notational changes made to the online learning in partially observable Markov decision process (OLPOMDP) algorithm, where the original uses a scaling factor $\gamma^0 = = \gamma \Delta t / \tau_z$ rather than $\gamma$.

### References
1. [10.1162/neco.2007.19.6.1468](https://florian.io/papers/2007_Florian_Modulated_STDP.pdf)
1. [10.3389/fncir.2015.00085](https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2015.00085/full)

## Generalized-Kernel Spike-Timing Dependent Plasticity (Kernel STDP)
### Formulation
$$
w(t + \Delta t) - w(t) =
\begin{cases}
    K_\text{post}\bigl(t^f_\text{post} - t^f_\text{pre}\bigr) &t^f_\text{post} \geq t^f_\text{pre} \\
    K_\text{pre}\bigl(t^f_\text{post} - t^f_\text{pre}\bigr) &t^f_\text{post} < t^f_\text{pre}
\end{cases}
$$

*Where:*
- $w$, connection weight
- $K_\text{post}$, kernel to use for cases where the last postsynaptic spike was at least as recent
- $K_\text{pre}$, kernel to use for cases where the last presynaptic spike was more recent
- $t$, current simulation time
- $\Delta t$, duration of the simulation step
- $t^f_\text{post}$, time of (the most recent) postsynaptic spike
- $t^f_\text{pre}$, time of (the most recent) presynaptic spike

### Description
This method generalizes from most variants of STDP by allowing functions other than exponential decay to represent the scale of the update based on the relative time between the most recent postsynaptic and presynaptic spikes.

### References
1. [DOI:10.1371/journal.pone.0101109](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0101109)

## Delay-Adjusted Spike-Timing Dependent Plasticity (Delay-Adjusted STDP)
### Formulation
$$
\begin{align*}
    w(t + \Delta t) - w(t) &=
    \begin{cases}
        A_+ \exp\left(-\frac{\lvert t_\Delta(t) \rvert}{\tau_+} \right) &t_\Delta(t) \geq 0 \\
        A_- \exp\left(-\frac{\lvert t_\Delta(t) \rvert}{\tau_-} \right) &t_\Delta(t) < 0
    \end{cases} \\
    t_\Delta(t) &= t^f_\text{post} - t^f_\text{pre} - d(t)
\end{align*}
$$

*Where:*
- $w$, connection weight
- $d$, connection delay
- $A_+$, learning rate for postsynaptic events, Hebbian long-term potentiation (LTP) when positive
- $A_-$, learning rate for presynaptic events, Hebbian long-term depression (LTD) when negative
- $\tau_+$, time constant of [exponential decay](<guide/mathematics:Exponential Decay and Time Constants>) for the postsynaptic adjusted trace
- $\tau_-$, time constant of exponential decay for the presynaptic adjusted trace
- $t$, current simulation time
- $\Delta t$, duration of the simulation step
- $t^f_\text{post}$, time of (the most recent) postsynaptic spike
- $t^f_\text{pre}$, time of (the most recent) presynaptic spike
- $t_\Delta$, adjusted postsynaptic-presynaptic spike time difference

*Note:*

As a deviation from the original, in this formulation, it is expected that when $A_-$ is negatively signed and $A_+$ is positively signed, the updates will be Hebbian.

### Description
This method applies an adjustment term to the difference in time between the most recent presynaptic and postsynaptic spike based on the learned delay. This is not directly equivalent to nearest neighbor [trace](<guide/concepts:Trace>) even when $d(t) = 0$ as the updates are not applied in an event-based manner. The adjusted traces additionally are not directly traces of postsynaptic and presynaptic spikes.

### References
1. [DOI:10.1162/neco_a_01674](https://arxiv.org/abs/2011.09380)

## Delay-Adjusted Spike-Timing Dependent Plasticity of Delays (Delay-Adjusted STDPD)
### Formulation
$$
\begin{align*}
    d(t + \Delta t) - d(t) &=
    \begin{cases}
        B_- \exp\left(-\frac{\lvert t_\Delta(t) \rvert}{\tau_-} \right) &t_\Delta(t) \geq 0 \\
        B_+ \exp\left(-\frac{\lvert t_\Delta(t) \rvert}{\tau_+} \right) &t_\Delta(t) < 0
    \end{cases} \\
    t_\Delta(t) &= t^f_\text{post} - t^f_\text{pre} - d(t)
\end{align*}
$$

*Where:*
- $d$, connection delay
- $B_-$, learning rate for postsynaptic events, Hebbian when negative
- $B_+$, learning rate for presynaptic events, Hebbian when positive
- $\tau_-$, time constant of [exponential decay](<guide/mathematics:Exponential Decay and Time Constants>) for the postsynaptic adjusted trace
- $\tau_+$, time constant of exponential decay for the presynaptic adjusted trace
- $t$, current simulation time
- $\Delta t$, duration of the simulation step
- $t^f_\text{post}$, time of (the most recent) postsynaptic spike
- $t^f_\text{pre}$, time of (the most recent) presynaptic spike
- $t_\Delta$, adjusted postsynaptic-presynaptic spike time difference

*Note:*

As a deviation from the original, in this formulation, it is expected that when $B_-$ is negatively signed and $B_+$ is positively signed, the updates will be the delay-equivalent of Hebbian.

### Description
This method applies an adjustment term to the difference in time between the most recent presynaptic and postsynaptic spike based on the learned delay. This is not directly equivalent to nearest neighbor [trace](<guide/concepts:Trace>) even when $d(t) = 0$ as the updates are not applied in an event-based manner. The adjusted traces additionally are not directly traces of postsynaptic and presynaptic spikes.

The original paper suggests not updating any delay where $d(t) < c$ for a constant $c$ where $c > \min(B_+, B_-)$ if at least one of $B_+$ or $B_-$ is less than zero. This can be practically achieved either using weight clamping or [parameter dependence](<guide/concepts:Parameter Dependence>).

This is the counterpart of [Delay-Adjusted STDP](<zoo/learning-stdp:Delay-Adjusted Spike-Timing Dependent Plasticity (Delay-Adjusted STDP)>) for delay learning.

### References
1. [DOI:10.1162/neco_a_01674](https://arxiv.org/abs/2011.09380)