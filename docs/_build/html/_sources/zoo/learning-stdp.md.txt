# STDP-Like Learning Methods

## Overview
STDP (spike timing-dependent plasticity) refers to a category of methods that implement Hebbian...

## Spike Timing-Dependent Plasticity (STDP)
### Formulation
$$
\frac{dw}{dt} = A_+ x_\text{pre} \cdot \delta(t - t^f_\text{post}) - A_- x_\text{post} \cdot \delta(t - t^f_\text{pre})
$$

With solution:

$$
\Delta w_t =
\begin{cases}
    A_+ x_\text{pre} &t = t^f_\text{post} \\
    -A_- x_\text{post} &t = t^f_\text{pre}
\end{cases}
$$

Where:
- $\Delta w_t$, change in weight
- $A_+$, update magnitude for long-term potentiation (LTP)
- $A_-$, update magnitude for long-term depression (LTD)
- $x_\text{post}$, [spike trace](<guide/mathematics:trace>) of postsynaptic (output) spikes, parameterized by time constant $\tau_\text{post}$
- $x_\text{pre}$, [spike trace](<guide/mathematics:trace>) of presynaptic (input) spikes, parameterized by time constant $\tau_\text{pre}$
- $t$ is the current time step
- $t^f_\text{post}$, time of the most recent postsynaptic spike
- $t^f_\text{pre}$, time of the most recent presynaptic spike
- $\delta$, [Dirac delta function](<guide/mathematics:Dirac Delta Function>)

## Delayed Spike Timing-Dependent Plasticity (Delayed STDP)
### Formulation
$$
\Delta w_t =
\begin{cases}
    A_+ \exp \left(-\frac{\Delta t_{pp}}{\tau_+}\right) &\Delta t_{pp} > 0 \\
    -A_- \exp \left(\frac{\Delta t_{pp}}{\tau_-}\right) &\Delta t_{pp} < 0 \\
    0 &\Delta t_{pp} = 0
\end{cases}
$$

$$
\Delta t_{pp} = t^f_\text{post} - t^f_\text{pre} - d_{pp}
$$

Where:
- $\Delta w_t$, change in weight
- $A_+$, update magnitude for long-term potentiation (LTP)
- $A_-$, update magnitude for long-term depression (LTD)
- $\tau_+$, time constant for potentiation
- $\tau_-$, time constant for depression
- $\Delta t_{pp}$, adjusted time between neighboring post and presynaptic spikes
- $t^f_\text{post}$, time of the most recent postsynaptic spike
- $t^f_\text{pre}$, time of the most recent presynaptic spike
- $d_{pp}$, length of the delay between input the neuron

## Anti-Hebbian Spike Timing-Dependent Plasticity (aSTDP)
### Formulation
$$
\frac{dw}{dt} = A_+ x_\text{post} \cdot \delta(t - t^f_\text{pre}) - A_- x_\text{pre} \cdot \delta(t - t^f_\text{post})
$$

With solution:

$$
\Delta w_t =
\begin{cases}
    A_+ x_\text{post} &t = t^f_\text{pre} \\
    -A_- x_\text{pre} &t = t^f_\text{post}
\end{cases}
$$

Where:
- $\Delta w_t$, change in weight
- $A_+$, update magnitude for long-term potentiation (LTP)
- $A_-$, update magnitude for long-term depression (LTD)
- $x_\text{post}$, [spike trace](<guide/mathematics:trace>) of postsynaptic (output) spikes, parameterized by time constant $\tau_\text{post}$
- $x_\text{pre}$, [spike trace](<guide/mathematics:trace>) of presynaptic (input) spikes, parameterized by time constant $\tau_\text{pre}$
- $t$ is the current time step
- $t^f_\text{post}$, time of the most recent postsynaptic spike
- $t^f_\text{pre}$, time of the most recent presynaptic spike
- $\delta$, [Dirac delta function](<guide/mathematics:Dirac Delta Function>)

### Description
This flips the sign of the update magnitudes of [STDP](#spike-timing-dependent-plasticity-stdp) to be anti-Hebbian. That is, when a postsynpatic spike follows a presynaptic one, long term depression takes place. And when a presynaptic spike follows a postsynaptic one, long term potentiation takes place.

## Modulated Spike-Timing Dependent Plasticity (MSTDP)

## Modulated Spike-Timing Dependent Plasticity with Eligibility Trace (MSTDPET)

## Soft Weight Bounding (Weight Dependence)
### Formulation
$$
\begin{align*}
    A_+(w) &= (w_\text{max} - w)^{\mu_+}\eta_+ \\
    A_-(w) &= (w - w_\text{min})^{\mu_-}\eta_-
\end{align*}
$$

Where:
- $A_+$, update magnitude for long-term potentiation (LTP)
- $A_-$, update magnitude for long-term depression (LTD)
- $w$, connection weight being updated
- $w_\text{max}$, upper bound for connection weight
- $w_\text{min}$, lower bound for connection weight
- $\eta_+$, learning rate for LTP
- $\eta_-$, learning rate for LTD
- $\mu_+$, order for upper weight bound
- $\mu_-$, order for lower weight bound

### Description
This method penalizes weights that are out of specified bounds by applying a penalty proportional to the amount by which the current weight is over/under the bound. The order parameters $\mu_+$ and $\mu_-$ control the rate of this penalty. When $\mu_+$ and $\mu_-$ are set to $1$, this is referred to as "multiplicative weight dependence", and when not set to one is often referred to as "power law weight dependence".

## Hard Weight Bounding
### Formulation
$$
\begin{align*}
    A_+(w) &= \Theta(w_\text{max} - w)\eta_+ \\
    A_-(w) &= \Theta(w - w_\text{min})\eta_-
\end{align*}
$$

Where:
- $A_+$, update magnitude for long-term potentiation (LTP)
- $A_-$, update magnitude for long-term depression (LTD)
- $w$, connection weight being updated
- $w_\text{max}$, upper bound for connection weight
- $w_\text{min}$, lower bound for connection weight
- $\eta_+$, learning rate for LTP
- $\eta_-$, learning rate for LTD
- $\Theta$, [Heaviside step function](<guide/mathematics:Heaviside Step Function>)