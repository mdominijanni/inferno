# Mathematical Notes

## Exponential Decay and Time Constants
Exponential decay is defined as a phenomenon where the rate at which a quantity decreases is proportional to its current value. It can be described by the differential equation

$$
\tau\frac{dN}{dt} = -N,
$$

with a solution of

$$
N(t) = N_0 e^{-\frac{t}{\tau}},
$$

where $N_0 = N(0)$.

This phenomenon is oft-used in spiking neural networks. When used in simulations involving discrete timesteps, such as those performed with Inferno, they will often appear in the form of

$$
N(t + \Delta t) = N(t)\exp\left(-\frac{\Delta t}{\tau}\right)
$$

when the underlying differential equation was solved analytically and

$$
N(t + \Delta t) = \frac{\Delta t}{\tau} \left[-N(t)\right] + N(t)
$$

when a solution was approximated via Euler's method. In both of these equations, $\Delta t$ represents the length of the simulated timestep. When selecting model hyperparameters, an important way of considering time constants is in terms of how long "lived" the corresponding quantity is under exponential decay. In general, this relation can be described as:

$$
Ne^{-k} = N \left[\exp\left(-\frac{\Delta t}{\tau}\right)\right]^{k \tau / \Delta t}.
$$

That is, after $\tau / \Delta t$ repeated applications (timesteps), the quantity $N$ will be reduced to $1/e$ of its starting value. To consider this outside of the notion of $1/e$ life, we can select $\tau$ as a multiple of $1 / \ln(b)$ for the $1/b$ life we want to think in. Let $\tau = \tau_b / \ln(b)$. Then the equation

$$
Nb^{-k} = N \left[\exp\left(-\frac{\Delta t}{\tau}\right)\right]^{k \tau_b / \Delta t}
$$

will express $N$ in terms of $1/b$ life instead. Sometimes exponential decay will instead be represented with a
"rate" parameter $\lambda$, such that $\lambda = \tau^{-1}$.

## Trace
In the context of spiking neural networks, the notion of "trace" frequently arises. When performing computations which care about the history of spiking activity, they may be formally described as summing over all the previous times in which a spike occurred. However, this is not practical for the purposes of simulation, as an all-to-all comparison grows in complexity quadratically with respect to the length of the simulation time.

In the interest of biological plausibility, and to the benefit of computational feasibility, rather than considering every prior spike, it is instead modeled as each prior spike leaving behind a *trace*. A trace is typically represented by the differential equation

$$
\tau_x \frac{dx}{dt} = -x + A \sum_f \delta (t - t^f),
$$

where $x$ is the trace, $A$ is the amplitude of the trace, $\tau_x$ is the time constant of exponential decay, $t$ is the simulation time, $t^f$ are the times at which spikes occurred (over the set of spikes $f$), and $\delta$ is the Dirac delta function. This construction $\delta (t - t^f)$, when in the solution of the differential equation, evaluates to $1$ if the current time $t$ is a time at which an action potential was generated and $0$ otherwise.

This equation can be interpreted as follows: whenever a spike occurs, add some value $A$ to the trace $x$, and let this trace decay exponentially with some time constant $\tau_x$. Instead of only considering spikes, $j$ is the set of true events, occuring at times $t^j$.

### Cumulative Trace
$$
x(t) = x(t - \Delta t) \exp \left(-\frac{\Delta t}{\tau_x}\right) + A \left[t = t^j\right]
$$

*From:*

$$
\tau_x \frac{dx}{dt} = -x(t) + A \sum_j \delta(t - t^j)
$$

*Where:*
- $x$, spike trace
- $A$, amplitude of the trace
- $\tau_x$, time constant of exponential decay $(\text{ms})$
- $t^j$, time of (the most recent) prior event $(\text{ms})$
- $t$, current runtime of the simulation $(\text{ms})$
- $\Delta t$, length of time over which each simulation step occurs $(\text{ms})$

$[\cdots]$ is the Iverson bracket and equals $1$ if the inner statement is true and $0$ if it is false.

### Nearest Trace
$$
x(t) =
\begin{cases}
    A & t=t^j \\
    x(t - \Delta t) \exp \left(-\frac{\Delta t}{\tau_x}\right) & t \neq t^j \\
\end{cases}
$$

*Where:*
- $x$, spike trace
- $A$, amplitude of the trace
- $\tau_x$, time constant of exponential decay $(\text{ms})$
- $t^j$, time of (the most recent) prior event $(\text{ms})$
- $t$, current runtime of the simulation $(\text{ms})$
- $\Delta t$, length of time over which each simulation step occurs $(\text{ms})$

### Scaling Trace
Generalizing trace further, if a new observation $h$ meets a criterion function $J$, then an amplitude proportional to the observation is added to trace. So for $t^j = \{t \mid  J(h^{(t)})\}$ an observation-dependent amplitude $f_A(t) = h^{(t)}s + A$ is used.

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

This method filters out any update which would move the parameter further beyond its limit.

## Dirac Delta and Heaviside Step Functions
### Dirac Delta Function
The Dirac delta function $\delta(x)$ is a generalized function in which the following properties hold.

$$
\begin{align*}
    \delta(x) &= 0 &x \neq 0 \\
    \delta(x) &\neq 0 &x = 0
\end{align*}\\
\int_{-\infty}^{\infty} \delta(x)dx = 1
$$

### Heaviside Step Function
The Heaviside step function $\Theta(x)$ is a step function which is $0$ for negative arguments and $1$ for positive arguments.

$$
\Theta(x) =
\begin{cases}
    1 &x \geq 0 \\
    0 & x < 0
\end{cases}
$$

### Relationship
The Dirac delta function is the derivative of the Heaviside step function.

$$\delta(x) = \frac{d}{dx} \Theta(x)$$

And the Heaviside step function is the integral of the Dirac delta function.

$$\Theta(x) = \int_{-\infty}^x \delta(s) ds$$