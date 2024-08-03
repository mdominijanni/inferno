# Concepts in SNNs

## Trace
In the context of spiking neural networks, the notion of "trace" frequently arises. When performing computations which care about the history of spiking activity, they may be formally described as summing over all the previous times in which a spike occurred. However, this is not practical for the purposes of simulation, as an all-to-all comparison grows in complexity quadratically with respect to the length of the simulation time.

In the interest of biological plausibility, and to the benefit of computational feasibility, rather than considering every prior spike, it is instead modeled as each prior spike leaving behind a *trace*. A trace is typically represented by the differential equation

$$
\frac{dx}{dt} = -\frac{x(t)}{\tau_x} + A \sum_f \delta (t - t^f),
$$

where $x$ is the trace, $A$ is the amplitude of the trace, $\tau_x$ is the time constant of [exponential decay](<guide/mathematics:Exponential Decay and Time Constants>), $t$ is the simulation time, $t^f$ are the times at which spikes occurred (over the set of spikes $f$), and $\delta$ is the Dirac delta function. The construction $\delta (t - t^f)$, when in the solution of the differential equation, evaluates to $1$ if the current time $t$ is a time at which an action potential was generated and $0$ otherwise. While $f$ typically corresponds to a set of spikes, it can refer to any binary event.

### Cumulative Trace
$$
x(t) = x(t - \Delta t) \exp \left(-\frac{\Delta t}{\tau_x}\right) + A \left[t = t^f\right]
$$

*From:*

$$
\frac{dx}{dt} = -\frac{x(t)}{\tau_x} + A \sum_f \delta(t - t^f)
$$

*Where:*
- $x$, trace
- $A$, amplitude of the trace
- $\tau_x$, time constant of exponential decay $(\text{ms})$
- $t^f$, time of (the most recent) prior event $(\text{ms})$
- $t$, current runtime of the simulation $(\text{ms})$
- $\Delta t$, length of time over which each simulation step occurs $(\text{ms})$

$[\cdots]$ is the Iverson bracket and equals $1$ if the inner statement is true and $0$ if it is false.

### Nearest Trace
$$
x(t) =
\begin{cases}
    A & t=t^f \\
    x(t - \Delta t) \exp \left(-\frac{\Delta t}{\tau_x}\right) & t \neq t^f \\
\end{cases}
$$

*From:*

$$
\frac{dx}{dt} = -\frac{x(t)}{\tau_x} + \left(A - x(t) + \frac{x(t)}{\tau_x}\right) \sum_f \delta(t - t^f)
$$

*Where:*
- $x$, trace
- $A$, amplitude of the trace
- $\tau_x$, time constant of exponential decay $(\text{ms})$
- $t^f$, time of (the most recent) prior event $(\text{ms})$
- $t$, current runtime of the simulation $(\text{ms})$
- $\Delta t$, length of time over which each simulation step occurs $(\text{ms})$

### Comparison and Interpretation
The "cumulative trace" is used to model these all-to-all interactions whereas "nearest trace" is used to model nearest-neighbor interactions.

### References
1. [DOI:10.1007/s00422-008-0233-1](https://link.springer.com/article/10.1007/s00422-008-0233-1)

## Parameter Dependence
Traditionally applied to connection weights, especially with learning methods which contain a potentiative and depressive component, parameter dependence is a technique which limits the range of values for a parameter.

### Soft Dependence
$$
\begin{align*}
    A_+(v) &= (v_\text{max} - v)^{\mu_+}\eta_+ \\
    A_-(v) &= (v - v_\text{min})^{\mu_-}\eta_-
\end{align*}
$$

*Where:*
- $A_+$, adjusted magnitude for long-term potentiation (LTP)
- $A_-$, adjusted magnitude for long-term depression (LTD)
- $v$, connection parameter being updated
- $v_\text{max}$, upper bound for connection parameter
- $v_\text{min}$, lower bound for connection parameter
- $\eta_+$, original magnitude (learning rate) for LTP
- $\eta_-$, original magnitude (learning rate) for LTD
- $\mu_+$, order for upper parameter bound
- $\mu_-$, order for lower parameter bound

This penalizes parameter that are out of specified bounds by applying a penalty proportional to the amount by which the current weight is over/under the bound. The order parameters $\mu_+$ and $\mu_-$ control the rate of this penalty. When $\mu_+$ and $\mu_-$ are set to $1$, this is referred to as "multiplicative dependence", and when not set to $1$ is often referred to as "power law dependence".

### Hard Dependence
$$
\begin{align*}
    A_+(v) &= \Theta(v_\text{max} - v)\eta_+ \\
    A_-(v) &= \Theta(v - v_\text{min})\eta_-
\end{align*}
$$

*Where:*
- $A_+$, update magnitude for long-term potentiation (LTP)
- $A_-$, update magnitude for long-term depression (LTD)
- $v$, connection parameter being updated
- $v_\text{max}$, upper bound for connection parameter
- $v_\text{min}$, lower bound for connection parameter
- $\eta_+$, learning rate for LTP
- $\eta_-$, learning rate for LTD

$\Theta(\cdots)$ is the [Heaviside step function](<guide/mathematics:Heaviside Step Function>) and equals $1$ if the input is nonnegative and $0$ otherwise.

This is functionally similar to clamping the parameter (i.e. setting any values to the minimum/maximum allowed if they go under/over the limits). It filters out any update which would move the parameter further beyond its limit.