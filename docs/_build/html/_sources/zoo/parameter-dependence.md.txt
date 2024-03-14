# Parameter Dependence

## Overview
Traditionally applied to connection weights, especially with learning methods which contain a potentiative and depressive component, parameter dependence is a technique which limits the range of values for a parameter.

## Parameter Dependence, Soft Bounding
### Formulation
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

### Description
This method penalizes parameter that are out of specified bounds by applying a penalty proportional to the amount by which the current weight is over/under the bound. The order parameters $\mu_+$ and $\mu_-$ control the rate of this penalty. When $\mu_+$ and $\mu_-$ are set to $1$, this is referred to as "multiplicative dependence", and when not set to $1$ is often referred to as "power law dependence".

## Parameter Dependence, Hard Bounding
### Formulation
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

### Description
This method filters out any update which would move the parameter further beyond its limit.