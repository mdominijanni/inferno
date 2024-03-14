# Spike Trace

## Overview
When considering the operations which are performed using spikes (the action potentials of neurons), often these should impact not only the immediate time step, but should have an impact going forward. This is the notion of a "trace" and is typically (but not always) represented as exponential decay over time. These functions appear in synaptic models and training methods.

## Cumulative Trace
### Formulation
$$
x(t) = x(t - \Delta t) \exp \left(-\frac{\Delta t}{\tau_x}\right) + A \left[t = t^f\right]
$$

*From:*

$$
\tau_x \frac{dx}{dt} = -x(t) + A \sum_\mathcal{F} \delta(t - t^f)
$$

*Where:*
- $x$, spike trace
- $A$, amplitude of the trace
- $\tau_x$, time constant of exponential decay $(\text{ms})$
- $t^f$, time of (the most recent) prior spike $(\text{ms})$
- $\mathcal{F}$, set of prior spikes
- $t$, current runtime of the simulation $(\text{ms})$
- $\Delta t$, length of time over which each simulation step occurs $(\text{ms})$

$[\cdots]$ is the Iverson bracket and equals $1$ if the inner statement is true and $0$ if it is false.

## Nearest Trace
### Formulation
$$
x(t) =
\begin{cases}
    x(t - \Delta t) \exp \left(-\frac{\Delta t}{\tau_x}\right) & t \neq t^f \\
    A & t=t^f \\
\end{cases}
$$

*Where:*
- $x$, spike trace
- $A$, amplitude of the trace
- $\tau_x$, time constant of exponential decay $(\text{ms})$
- $t^f$, time of the most recent spike $(\text{ms})$
- $t$, current runtime of the simulation $(\text{ms})$
- $\Delta t$, length of time over which each simulation step occurs $(\text{ms})$