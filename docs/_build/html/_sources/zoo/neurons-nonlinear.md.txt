# Neuron Models, Nonlinear

## Quadratic Integrate-and-Fire (QIF)
### Formulation
$$
\tau_m \frac{dV_m(t)}{dt} = a \left(V_m(t) - V_\text{rest}\right)\left(V_m(t) - V_\text{crit}\right) + R_mI(t)
$$

*With approximation:*

$$
V_m(t + \Delta t) \approx \frac{\Delta t}{\tau_m} \left[ a \left(V_m(t) - V_\text{rest}\right)\left(V_m(t) - V_\text{crit}\right) + R_mI(t) \right] + V_m(t)
$$

*After an action potential is generated:*

$$V_m(t) \leftarrow V_\text{reset}$$

*Where:*
- $I$, total input current applied to the neuron $(\text{nA})$
- $V_m$, electric potential difference across the cell membrane $(\text{mV})$
- $V_\text{rest}$, equilibrium of the membrane potential $(\text{mV})$
- $V_\text{crit}$, critical value of the membrane potential $(\text{mV})$
- $a$, tendency for the membrane potential to go towards $V_\text{rest}$ and away from $V_\text{crit}$ (unitless).
- $V_\text{reset}$, membrane potential difference set after spiking $(\text{mV})$
- $\tau_m$, membrane time constant $(\text{ms})$
- $t$, current time of the simulation $(\text{ms})$
- $\Delta t$, length of time over which each simulation step occurs $(\text{ms})$

Under the conditions $a > 0$ and $V_\text{crit} > V_\text{rest}$.

### Description
This model approximates exponential integrate-and-fire using a quadratic dynamics. It has
two fixed points: the stable $V_\text{rest}$ and unstable $V_\text{crit}$. The rate at which
the membrane voltage is attracted towards $V_\text{rest}$ and repelled away from $V_\text{crit}$
is controlled by $a$.

```{image} ../images/plots/qif_slope_field.png
:alt: Quadratic Integrate-and-Fire Slope Field of Memrane Voltage
:class: bg-primary
:scale: 30 %
:align: center
```
Slope field of the membrane voltage without any input current showing the relation between it
and the critical voltage $(V_C = -50 \text{ mV})$ and rest voltage $(V_R = -60 \text{ mV})$
parameters. Plotted with values $\tau_m=1 \text{ ms}$ and $a=1$ over a time of $1 \text{ ms}$.

### References
1. [DOI:10.1017/CBO9781107447615 (Chapter 5.3)](https://neuronaldynamics.epfl.ch/online/Ch5.S3.html)

## Izhikevich
*(To Be Completed)*

### References
1. [DOI:10.1017/CBO9781107447615 (Chapter 6.1)](https://neuronaldynamics.epfl.ch/online/Ch6.S1.html)
2. [DOI:10.3390/brainsci12070863](https://www.mdpi.com/2076-3425/12/7/863/pdf)

## Exponential Integrate-and-Fire (EIF)
### Formulation
$$
\tau_m \frac{dV_m(t)}{dt} =
- \left[V_m(t) - V_\text{rest}\right] +
\Delta_T \exp \left(\frac{V_m(t) - V_T}{\Delta_T}\right)
+ R_mI(t)
$$

*With approximation:*

$$
V_m(t + \Delta t) \approx \frac{\Delta t}{\tau_m} \left[
- \left[V_m(t) - V_\text{rest}\right] +
\Delta_T \exp \left(\frac{V_m(t) - V_T}{\Delta_T}\right) + R_mI(t)
\right]+ V_m(t)
$$

*After an action potential is generated:*

$$V_m(t) \leftarrow V_\text{reset}$$

*Where:*
- $I$, total input current applied to the neuron $(\text{nA})$
- $V_m$, electric potential difference across the cell membrane $(\text{mV})$
- $V_\text{rest}$, equilibrium of the membrane potential $(\text{mV})$
- $V_T$, membrane potential approaching the depolarization threshold $(\text{mV})$
- $V_\text{reset}$, membrane potential difference set after spiking $(\text{mV})$
- $\Delta_T$, steepness of depolarization and closeness to $V_T$ $(\text{mV})$
- $\tau_m$, membrane time constant $(\text{ms})$
- $t$, current time of the simulation $(\text{ms})$
- $\Delta t$, length of time over which each simulation step occurs $(\text{ms})$

Under the conditions $\Delta_T > 0$ and $V_T > V_\text{rest}$.

### Description
This model uses exponential dynamics to model the rapid increase in membrane voltage before
(and at the start of) the generation of an action potential. This upswing occurs at a voltage
above $V_T$, specifically, as $\Delta_T \rightarrow 0$, the voltage at which this upswing occurs
approaches $V_T$. $\Delta_T$ also controls the sharpness of upswing.

```{important}
Two of the parameters used in EIF and its derivatives can be easy to confuse with unrelated
parameters.

- $V_T$ is called the "threshold voltage" (although is occasionally called the rheobase
threshold and denoted $\vartheta_{rh}$), but is different than the "voltage threshold",
i.e. the membrane potential at which an action potential is generated. The latter is
often denoted as $\Theta$, $\Theta_\infty$, $\theta$, or $V_\text{thresh}$ and
in Inferno is usually represented by the parameter `thresh_v` or `thresh_eq_v`.
- $\Delta_T$ is called the "slope factor" or "sharpness" and is unrelated to the
"step time" used in discrete-time simuations of neuronal dynamics. The latter is
typically denoted as $\Delta t$ or $\delta t$ and in Inferno is usually represented by
the parameter `step_time` and attribute `dt`.
```

Below are two slope fields showing the relation between the threshold voltage $V_T$ and
rest voltage $V_R$ in relation to the membrane voltage. Examples with two $\Delta_T$ settings
are used to illustrate its effect.

```{image} ../images/plots/eif_slope_field_d1.png
:alt: Exponential Integrate-and-Fire Slope Field of Memrane Voltage ($\Delta_T = 1$)
:class: bg-primary
:scale: 30 %
:align: center
```
Membrane voltage with no input current. Plotted with values $V_R = -60\text{ mV}$,
$V_T = -50$, $\Delta_T = 1$, and $\tau_m=1 \text{ ms}$ over a time of $1 \text{ ms}$.

```{image} ../images/plots/eif_slope_field_d2.png
:alt: Exponential Integrate-and-Fire Slope Field of Memrane Voltage ($\Delta_T = 2$)
:class: bg-primary
:scale: 30 %
:align: center
```
Membrane voltage with no input current. Plotted with values $V_R = -60\text{ mV}$,
$V_T = -50$, $\Delta_T = 2$, and $\tau_m=1 \text{ ms}$ over a time of $1 \text{ ms}$.

### References
1. [DOI:10.1017/CBO9781107447615 (Chapter 6.1)](https://neuronaldynamics.epfl.ch/online/Ch6.S1.html)
2. [ISBN:9780262548083 (Section 1.2)](https://github.com/RobertRosenbaum/ModelingNeuralCircuits/blob/main/ModelingNeuralCircuits.pdf)

## Adaptive Exponential Integrate-and-Fire (AdEx)
### Formulation
$$
\begin{align*}
    \tau_m \frac{dV_m(t)}{dt} &= - \left[V_m(t) - V_\text{rest}\right] +
    \Delta_T \exp \left(\frac{V_m(t) - V_T}{\Delta_T}\right) + R_mI(t) \\
    I(t) &= I_x(t) - \sum_k w_k(t) \\
    \tau_k \frac{dw_k(t)}{dt} &= a_k \left[ V_m(t) - V_\text{rest} \right] - w_k(t) \\
\end{align*}
$$

*With approximations:*

$$
\begin{align*}
    V_m(t + \Delta t) &\approx \frac{\Delta t}{\tau_m} \left[- \left[V_m(t) - V_\text{rest}\right] +
    \Delta_T \exp \left(\frac{V_m(t) - V_T}{\Delta_T}\right) + R_mI(t) \right]+ V_m(t) \\
    w_k(t + \Delta t) &\approx \frac{\Delta t}{\tau_k}\left[ a_k \left[ V_m(t) - V_\text{rest} \right] - w_k(t) \right] + w_k(t) \\
\end{align*}
$$

*After an action potential is generated:*

$$
\begin{align*}
    V_m(t) &\leftarrow V_\text{reset} \\
    w_k(t) &\leftarrow w_k(t) + b_k
\end{align*}
$$

*Where:*
- $I$, total input current applied to the neuron $(\text{nA})$
- $I_x$, input current before adaptation $(\text{nA})$
- $V_m$, electric potential difference across the cell membrane $(\text{mV})$
- $V_\text{rest}$, equilibrium of the membrane potential $(\text{mV})$
- $V_T$, membrane potential approaching the depolarization threshold $(\text{mV})$
- $V_\text{reset}$, membrane potential difference set after spiking $(\text{mV})$
- $\Delta_T$, steepness of depolarization and closeness to $V_T$ $(\text{mV})$
- $\tau_m$, membrane time constant $(\text{ms})$
- $a_k$, subthreshold adaptation, voltage-current coupling $(\mu\text{S})$
- $b_k$, spike-triggered current adaptation $(\text{nA})$
- $t$, current time of the simulation $(\text{ms})$
- $\Delta t$, length of time over which each simulation step occurs $(\text{ms})$

Under the conditions $\Delta_T > 0$ and $V_T > V_\text{rest}$.

### Description
This model uses the same underlying dynamics of the exponential integrate-and-fire neuron, but
incorporates a linear adaptive current depeendent upon output spikes and the membrane voltage.

### References
1. [DOI:10.1017/CBO9781107447615 (Chapter 6.1)](https://neuronaldynamics.epfl.ch/online/Ch6.S1.html)
2. [ISBN:9780262548083 (Section 1.2)](https://github.com/RobertRosenbaum/ModelingNeuralCircuits/blob/main/ModelingNeuralCircuits.pdf)
3. [DOI:10.4249/scholarpedia.8427](http://www.scholarpedia.org/article/Adaptive_exponential_integrate-and-fire_model)
4. [DOI:10.1523/JNEUROSCI.23-37-11628.2003](https://www.jneurosci.org/content/23/37/11628.long)