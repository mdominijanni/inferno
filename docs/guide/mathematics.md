# Mathematical Refresher

## Exponential Decay and Time Constants
Because spiking neural networks represent stateful systems that change over time, the notion of *exponential decay* frequently comes up. Exponential decay is defined as a phenomenon where the rate at which a quantity decreases is proportional to its current value. It can be described by the differential equation

$$
\tau\frac{dN}{dt} = -N,
$$

with a solution of

$$
N(t) = N_0 e^{-t / \tau},
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

when a solution was approximated via Euler's method. In both of these equations, $\Delta t$ represents the length of the simulated time step. When selecting model hyperparameters, an important way of considering time constants is in terms of how long "lived" the corresponding quantity is under exponential decay. In general, this relation can be described as:

$$
Ne^{-k} = N \left[\exp\left(-\frac{\Delta t}{\tau}\right)\right]^{k \tau / \Delta t}.
$$

That is, after $\tau / \Delta t$ repeated applications (simulation steps), the quantity $N$ will be reduced to $1/e$ of its starting value. To consider this outside of the notion of $1/e$ life, we can select $\tau$ as a multiple of $1 / \ln(b)$ for the $1/b$ life we want to think in. Let $\tau = \tau_b / \ln(b)$. Then the equation

$$
Nb^{-k} = N \left[\exp\left(-\frac{\Delta t}{\tau}\right)\right]^{k \tau_b / \Delta t}
$$

will express $N$ in terms of $1/b$ life instead. Sometimes exponential decay will instead be represented with a
"rate" parameter $\lambda$, such that $\lambda = \tau^{-1}$.

## Dirac Delta and Heaviside Step Functions
Although spiking neural networks are modelled as differential equations, spike events themselves are represented discretely. Because of this, the Dirac delta function $\delta$ and Heaviside step function $\Theta$ frequently arise in this context. These two functions are related as follows.

$$\delta(x) = \frac{d}{dx} \Theta(x) \qquad \Theta(x) = \int_{-\infty}^x \delta(s) ds$$

### Dirac Delta Function
The Dirac delta function $\delta$ is a generalized function in which the following properties hold.

$$
\begin{align*}
    \delta(x) &= 0 &x \neq 0 \\
    \delta(x) &\neq 0 &x = 0
\end{align*}\\
\int_{-\infty}^{\infty} \delta(x)dx = 1
$$

This is frequently shows up when a spike event should trigger some change in the model state, especially in the form $\delta(t - t^f)$ when in a in a differential equation over time $t$, where $t_f$ is the time of the most recent spike.

### Heaviside Step Function
The Heaviside step function $\Theta$ is a function which evaluates to $0$ for negative arguments and $1$ for non-negative arguments.

$$
\Theta(x) =
\begin{cases}
    1 &x \geq 0 \\
    0 & x < 0
\end{cases}
$$