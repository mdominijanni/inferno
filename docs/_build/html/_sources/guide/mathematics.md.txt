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

In the interest of biological plausibility, and to the benefit of computational feasibility, rather than considering every prior spike, it is instead modeled as each prior spike leaving behind some sort of *trace*. In its simplest form, a trace is represented by the differential equation

$$
\tau \frac{dx}{dt} = -x + a \sum_f \delta (t - t^f),
$$

where $x$ is the trace, $t$ is the simulation time, $t^f$ are the times at which spikes occurred, $a$ is some amplitude, and $\delta$ is the Dirac delta function. This construction $\delta (t - t^f)$, when in the solution of the differential equation, evaluates to $1$ if the current time $t$ is a time at which an action potential was generated and $0$ otherwise.

This equation can be interpreted as follows: whenever a spike occurs, add some value $a$ to the trace $x$, and let this trace decay exponentially with some time constant $\tau$.

Inferno implements four variants of trace, which will be discussed below. A common feature to all of these is that it generalizes the notion of trace beyond just spikes, but instead considers some value $f$ being traced over time.

### Cumulative Trace
This is the same as previously described. Cumulative comes from the fact that this interpretation factors in the contributions of all previous spikes. Below is the update equation, as it is represented in Inferno.

$$
x_{t + \Delta t} =
\begin{cases}
    a + x_t \exp (\Delta t / \tau) &\lvert f_{t + \Delta t} - f^* \rvert \leq \epsilon \\
    x_t \exp (\Delta t / \tau) &\text{otherwise}
\end{cases}
$$

Here the trace is represented as $x$, the simulation step time is $\Delta t$, the time constant by $\tau$, the and $a$ is the trace amplitude. The variable being traced is represented as $f$, with target $f^*$ and permissible error $\epsilon$.

For tracing spikes, the target can be set as $f^* = 1$ and error as $\epsilon = 0$. In implementation, when the $\epsilon$ term is unspecified, it is computed as an equality test.

### Nearest Trace
Whereas the cumulative trace considers all prior inputs, changing from addition to replacement modifies the trace such that it only considers the most recent event (spike or otherwise).

$$
x_{t + \Delta t} =
\begin{cases}
    a &\lvert f_{t + \Delta t} - f^* \rvert \leq \epsilon \\
    x_t \exp (\Delta t / \tau) &\text{otherwise}
\end{cases}
$$

The parameters are otherwise the same, and which is chosen will depend on the desired properties where it will be used.

### Scaling Traces
The described formulations of cumulative and nearest trace above treat any input that matches, either exactly or with some tolerance, the same. A generalization is that an input is considered to "match" whenever it meets some criterion $K$, and component added to the trace is scaled by the input and some scaling factor $S$.

#### Cumulative Scaled Trace
$$
x_{t + \Delta t} =
\begin{cases}
    a + Sf + x_t \exp (\Delta t / \tau) &K(f_{t + \Delta t}) \\
    x_t \exp (\Delta t / \tau) &\text{otherwise}
\end{cases}
$$

#### Nearest Scaled Trace
$$
x_{t + \Delta t} =
\begin{cases}
    a + Sf &K(f_{t + \Delta t}) \\
    x_t \exp (\Delta t / \tau) &\text{otherwise}
\end{cases}
$$

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