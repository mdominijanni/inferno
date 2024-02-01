# Neuron Models, Non-Linear

## Quadratic Integrate-and-Fire (QIF)
### Formulation
$$
\tau_m \frac{dV_m(t)}{dt} = a \left(V_m(t) - V_\text{rest}\right)\left(V_m(t) - V_\text{crit}\right) + R_mI(t)
$$

*With approximation:*

$$
V_m(t + \Delta t) \approx \frac{\Delta t}{\tau_m} \left[ a \left(V_m(t) - V_\text{rest}\right)\left(V_m(t) - V_\text{crit}\right) \right] + R_mI(t) + V_m(t)
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
Slope field of the membrane voltage showing the relation between it and the
critical voltage $(V_C = V_\text{crit})$ and rest voltage $(V_R = V_\text{rest})$ parameters.
Plotted with values $\tau_m=1$ and $a=1$.

### References
1. [DOI:10.1017/CBO9781107447615 (Chapter 5.3)](https://neuronaldynamics.epfl.ch/online/Ch5.S3.html)
