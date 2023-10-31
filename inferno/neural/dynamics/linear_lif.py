import torch
import torch.nn as nn

from inferno.neural.dynamics.abstract import AbstractDynamics, ShapeMixin


class PIFDynamics(ShapeMixin, AbstractDynamics):
    """Population of neurons modeled by perfect integrate-and-fire dynamics.

    Perfect integrate-and-fire (PIF) neurons are simulated by a difference equation, performed over discrete
    time steps. This difference equation modulates the membrane voltages by incorporating inputs.
    The following equation is used to model this dynamic.

    .. math::
        V_m(t + \\Delta t) = V_m(t) + R_mI(t)

    Where :math:`t` is the current time, :math:`\\Delta t` is the length of the time step, :math:`V_m` is the membrance voltage,
    :math:`R_m` is the membrane resistance, and :math:`I` is the input current.

    After adding in the new spiking inputs to the membrane voltage, spiking output is then generated if the membrane voltage
    has met or exceed the threshold voltage, :math:`V_m(t) \\geq V_\\text{thresh}`, and only if the neuron is not in an absolute refractory period.
    After a neuron fires, the membrane voltages is set to the reset voltage :math:`V_\\text{reset}` and the absolute refractory period begins
    for some number of timesteps.

    Attributes:
        v_membranes (torch.nn.Parameter): Current membrane voltages for the neurons in the population, shaped like :py:attr:`batched_shape`, in :math:`mV`.
        ts_refrac_membranes (torch.nn.Parameter): Remaining number of simulation time steps for the absolute refractory period of neurons in the population to end, shaped like :py:attr:`batched_shape`.

    Args:
        shape (tuple[int, ...] | int): shape of the population of neurons specified as the lengths along tensor dimensions.
        step_time (float): length of the time steps as simulated by the difference equation, in :math:`ms`.
        v_rest (float, optional): resting membrane voltage of the neurons, in :math:`mV`. Defaults to `-70.0`.
        v_reset (float, optional): membrane voltage a neuron is reset to following action potential, in :math:`mV`. Defaults to `-65.0`.
        v_threshold (float, optional): membrane voltage at which a neuron generates an action potential, in :math:`mV`. Defaults to `-50.0`.
        ts_abs_refrac (int, optional): number of time steps for which a neuron should be in its absolute refractory period. Defaults to `2`.
        r_membrane (float, optional): electrical resistance of a neuron's membrane, in :math:`k\\Omega`. Defaults to `1.0`.
        batch_size (int, optional): number of separate inputs to be passed along the batch (:math:`0^\\text{th}`) axis. Defaults to `1`.

    Raises:
        ValueError: `batch_size` must be a positive integer.
        ValueError: `step_time` must be a non-negative value.
    """
    def __init__(
        self,
        shape: tuple[int, ...] | int,
        step_time: float,
        v_rest: float = -70.0,
        v_reset: float = -65.0,
        v_threshold: float = -50.0,
        ts_abs_refrac: int = 2,
        r_membrane: float = 1.0,
        batch_size: int = 1
    ):
        # call superclass constructors
        AbstractDynamics.__init__(self)
        ShapeMixin.__init__(self, shape, batch_size, ('v_membranes', 'ts_refrac_membranes'))

        # verify a valid step time
        if step_time <= 0:
            raise ValueError(f"step time must be greater than zero, received {step_time}")

        # register hyperparameters as buffers
        self.register_buffer('step_time', torch.tensor(step_time, requires_grad=False))
        self.register_buffer('v_rest', torch.tensor(v_rest, requires_grad=False))
        self.register_buffer('v_reset', torch.tensor(v_reset, requires_grad=False))
        self.register_buffer('v_threshold', torch.tensor(v_threshold, requires_grad=False))
        self.register_buffer('ts_abs_refrac', torch.tensor(ts_abs_refrac, requires_grad=False))
        self.register_buffer('r_membrane', torch.tensor(r_membrane, requires_grad=False))

        # register states as parameters
        self.register_parameter('v_membranes', nn.Parameter(torch.ones(self.batched_shape, requires_grad=False) * self.v_rest, False))
        self.register_parameter('ts_refrac_membranes', nn.Parameter(torch.zeros(self.batched_shape, requires_grad=False), False))

    def clear(self, **kwargs) -> None:
        """Reinitializes the neurons' states.

        This sets the neurons back to their initial condition: their membrane voltages all set to :py:attr:`v_rest` and with their
        absolute refractory period counters zeroed-out. This should be used wherever there is a need to cleanly delineate
        inputs, such as between batches.
        """
        self.v_membranes.fill_(self.v_rest)
        self.ts_refrac_membranes.fill_(0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Processes input currents into the neuron bodies and generates spiking output.

        Args:
            inputs (torch.Tensor): tensor of input currents shaped like the population of neurons.

        Returns:
            torch.Tensor: resulting tensor shaped like the population of neurons indicating which spiked (1) and which did not (0).
       """
        # reset voltages for last depolarized neurons
        self.v_membranes.masked_fill_(self.ts_refrac_membranes == self.ts_abs_refrac, self.v_reset)

        # incorporate input currents
        self.v_membranes.add_((self.ts_refrac_membranes == 0) * (inputs * self.r_membrane))

        # determine output spikes
        spikes = self.v_membranes >= self.v_threshold
        spikes.masked_fill_(self.ts_refrac_membranes != 0, 0)

        # tick down refractory periods
        self.ts_refrac_membranes.sub_(1)
        self.ts_refrac_membranes.clamp_(min=0)

        # set refractory periods of depolarized neurons
        self.ts_refrac_membranes.masked_fill_(spikes, self.ts_abs_refrac)

        return spikes


class LIFDynamics(ShapeMixin, AbstractDynamics):
    """Population of neurons modeled by leaky integrate-and-fire dynamics.

    Leaky integrate-and-fire (LIF) neurons are simulated by a difference equation, performed over discrete
    time steps. This difference equation modulates the membrane voltages by incorporating inputs, and it decays the
    current membrane voltage back to its rest voltage over time. The following equation is used to model this dynamic.

    .. math::
        V_m(t + \\Delta t) = \\left(V_m(t) - V_\\text{rest}\\right)\\exp\\left(\\frac{-\\Delta t}{\\tau_m}\\right) + V_\\text{rest} + R_mI(t)

    Where :math:`t` is the current time, :math:`\\Delta t` is the length of the time step, :math:`V_m` is the membrance voltage, :math:`V_\\text{rest}` is the resting membrane voltage,
    :math:`\\tau_m` is the membrane time constant, :math:`R_m` is the membrane resistance, and :math:`I` is the input current.

    After decaying the membrane voltage and adding in the new spiking inputs, spiking output is then generated if the membrane voltage
    has met or exceed the threshold voltage, :math:`V_m(t) \\geq V_\\text{thresh}`, and only if the neuron is not in an absolute refractory period.
    After a neuron fires, the membrane voltages is set to the reset voltage :math:`V_\\text{reset}` and the absolute refractory period begins
    for some number of timesteps.

    Attributes:
        v_membranes (torch.nn.parameter.Parameter): Current membrane voltages for the neurons in the population, shaped like :py:attr:`batched_shape`, in :math:`mV`.
        ts_refrac_membranes (torch.nn.parameter.Parameter): Remaining number of simulation time steps for the absolute refractory period of neurons in the population to end, shaped like :py:attr:`batched_shape`.

    Args:
        shape (tuple[int, ...] | int): shape of the population of neurons specified as the lengths along tensor dimensions.
        step_time (float): length of the time steps as simulated by the difference equation, in :math:`ms`.
        v_rest (float, optional): resting membrane voltage of the neurons, in :math:`mV`. Defaults to `-70.0`.
        v_reset (float, optional): membrane voltage a neuron is reset to following action potential, in :math:`mV`. Defaults to `-65.0`.
        v_threshold (float, optional): membrane voltage at which a neuron generates an action potential, in :math:`mV`. Defaults to `-50.0`.
        ts_abs_refrac (int, optional): number of time steps for which a neuron should be in its absolute refractory period. Defaults to `2`.
        tc_membrane (float | None, optional): time constant of a neuron's membrane, defined as :math:`\\tau_m = R_mC_m`, in :math:`ms`. Defaults to `20.0`.
        r_membrane (float | None, optional): electrical resistance of a neuron's membrane, in :math:`k\\Omega`. Defaults to `1.0`.
        c_membrane (float | None, optional): capacitance of a neuron's membrane, in :math:`nF`. Defaults to `None`.
        batch_size (int, optional): number of separate inputs to be passed along the batch (:math:`0^\\text{th}`) axis. Defaults to `1`.

    Raises:
        ValueError: `batch_size` must be a positive integer.
        ValueError: `step_time` must be a non-negative value.
        TypeError: at least two of `tc_membrane`, `r_membrane`, and `c_membrane` must not be `None`.
        ValueError: `tc_membrane`, `r_membrane`, and `c_membrane` have mismatched values, ``tc_membrane == r_membrane * c_membrane`` must be true.
    """
    def __init__(
        self,
        shape: tuple[int, ...] | int,
        step_time: float,
        v_rest: float = -70.0,
        v_reset: float = -65.0,
        v_threshold: float = -50.0,
        ts_abs_refrac: int = 2,
        tc_membrane: float | None = 20.0,
        r_membrane: float | None = 1.0,
        c_membrane: float | None = None,
        batch_size: int = 1
    ):
        # call superclass constructors
        AbstractDynamics.__init__(self)
        ShapeMixin.__init__(self, shape, batch_size, ('v_membranes', 'ts_refrac_membranes'))

        # verify a valid step time
        if step_time <= 0:
            raise ValueError(f"step time must be greater than zero, received {step_time}")

        # register hyperparameters as buffers
        self.register_buffer('step_time', torch.tensor(step_time, requires_grad=False))
        self.register_buffer('v_rest', torch.tensor(v_rest, requires_grad=False))
        self.register_buffer('v_reset', torch.tensor(v_reset, requires_grad=False))
        self.register_buffer('v_threshold', torch.tensor(v_threshold, requires_grad=False))
        self.register_buffer('ts_abs_refrac', torch.tensor(ts_abs_refrac, requires_grad=False))
        self.register_buffer('tc_membrane', None)
        self.register_buffer('r_membrane', None)
        self.register_buffer('c_membrane', None)
        if [tc_membrane, r_membrane, c_membrane].count(None) > 1:
            raise TypeError("at least two of 'tc_membrane', 'r_membrane', and 'c_membrane' must be specified")
        elif [tc_membrane, r_membrane, c_membrane].count(None) == 1:
            if tc_membrane is None:
                self.r_membrane = torch.tensor(r_membrane, requires_grad=False)
                self.c_membrane = torch.tensor(c_membrane, requires_grad=False)
                self.tc_membrane = self.r_membrane * self.c_membrane
            if r_membrane is None:
                self.tc_membrane = torch.tensor(tc_membrane, requires_grad=False)
                self.c_membrane = torch.tensor(c_membrane, requires_grad=False)
                self.r_membrane = self.tc_membrane / self.c_membrane
            if c_membrane is None:
                self.tc_membrane = torch.tensor(tc_membrane, requires_grad=False)
                self.r_membrane = torch.tensor(r_membrane, requires_grad=False)
                self.c_membrane = self.tc_membrane / self.r_membrane
        else:
            if tc_membrane == r_membrane * c_membrane:
                self.tc_membrane = torch.tensor(tc_membrane, requires_grad=False)
                self.r_membrane = torch.tensor(r_membrane, requires_grad=False)
                self.c_membrane = torch.tensor(c_membrane, requires_grad=False)
            else:
                raise ValueError("'tc_membrane', 'r_membrane', and 'c_membrane' have mismatched values, \
                    tc_membrane must equal r_membrane * c_membrane")
        self.register_buffer('decay_rate', torch.exp(-self.step_time / self.tc_membrane))

        # register states as parameters
        self.register_parameter('v_membranes', nn.Parameter(torch.ones(self.batched_shape, requires_grad=False) * self.v_rest, False))
        self.register_parameter('ts_refrac_membranes', nn.Parameter(torch.zeros(self.batched_shape, requires_grad=False), False))

    def clear(self, **kwargs) -> None:
        """Reinitializes the neurons' states.

        This sets the neurons back to their initial condition: their membrane voltages all set to :py:attr:`v_rest` and with their
        absolute refractory period counters zeroed-out. This should be used wherever there is a need to cleanly delineate
        inputs, such as between batches.
        """
        self.v_membranes.fill_(self.v_rest)
        self.ts_refrac_membranes.fill_(0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Processes input currents into the neuron bodies and generates spiking output.

        Args:
            inputs (torch.Tensor): tensor of input currents shaped like the population of neurons.

        Returns:
            torch.Tensor: resulting tensor shaped like the population of neurons indicating which spiked (1) and which did not (0).
       """
        # decay voltages
        self.v_membranes.sub_(self.v_rest)
        self.v_membranes.mul_(self.decay_rate)
        self.v_membranes.add_(self.v_rest)

        # incorporate input currents
        self.v_membranes.add_((self.ts_refrac_membranes == 0) * (inputs * self.r_membrane))

        # determine output spikes
        spikes = self.v_membranes >= self.v_threshold
        spikes.masked_fill_(self.ts_refrac_membranes != 0, 0)

        # tick down refractory periods
        self.ts_refrac_membranes.sub_(1)
        self.ts_refrac_membranes.clamp_(min=0)

        # set refractory periods of depolarized neurons
        self.ts_refrac_membranes.masked_fill_(spikes, self.ts_abs_refrac)

        # reset voltages for last depolarized neurons
        self.v_membranes.masked_fill_(self.ts_refrac_membranes == self.ts_abs_refrac, self.v_reset)

        return spikes


class AdaptiveLIFDynamics(ShapeMixin, AbstractDynamics):
    """Population of neurons modeled by leaky integrate-and-fire dynamics with an adaptive threshold.

    Adaptive leaky integrate-and-fire (Adaptive LIF) neurons are simulated by a difference equation, performed over discrete
    time steps. This difference equation modulates the membrane voltages by incorporating inputs, and it decays the
    current membrane voltage back to its rest voltage over time. The following equation is used to model this dynamic.

    .. math::
        V_m(t + \\Delta t) = \\left(V_m(t) - V_\\text{rest}\\right)\\exp\\left(\\frac{-\\Delta t}{\\tau_m}\\right) + V_\\text{rest} + R_mI(t)

    Where :math:`t` is the current time, :math:`\\Delta t` is the length of the time step, :math:`V_m` is the membrance voltage, :math:`V_\\text{rest}` is the resting membrane voltage,
    :math:`\\tau_m` is the membrane time constant, :math:`R_m` is the membrane resistance, and :math:`I` is the input current.

    After decaying the membrane voltage and adding in the new spiking inputs, spiking output is then generated if the membrane voltage
    has met or exceed the threshold voltage plus the adaptive thresholding component, :math:`V_m(t) \\geq V_\\text{thresh} + \\theta(t)`, and only if the neuron is not in an absolute
    refractory period. This adaptive thresholding component :math:`\\theta(t)` is decayed to zero by a time constant :math:`\\tau_\\theta` and is increased by a constant :math:`\\theta_+`
    when the neuron fires.

    .. math::
        \\theta(t + \\Delta t) = \\theta(t)\\exp\\left(\\frac{-\\Delta t}{\\tau_\\theta}\\right) + \\theta_+S_\\text{out}(t)

    Where :math:`S_\\text{out}(t)` represents the spiking output of the neuron at time :math:`t`. After a neuron fires, the membrane voltages is set to the reset voltage
    :math:`V_\\text{reset}` and the absolute refractory period begins for some number of timesteps.

    Attributes:
        v_membranes (torch.nn.parameter.Parameter): Current membrane voltages for the neurons in the population, shaped like :py:attr:`batched_shape`, in :math:`mV`.
        ts_refrac_membranes (torch.nn.parameter.Parameter): Remaining number of simulation time steps for the absolute refractory period of neurons in the population to end, shaped like :py:attr:`batched_shape`.
        theta (torch.nn.parameter.Parameter): Adaptive threshold components for each neuron in the population, shaped like :py:attr:`batched_shape`, in :math:`mV`.

    Args:
        shape (tuple[int, ...] | int): shape of the population of neurons specified as the lengths along tensor dimensions.
        step_time (float): length of the time steps as simulated by the difference equation, in :math:`ms`.
        v_rest (float, optional): resting membrane voltage of the neurons, in :math:`mV`. Defaults to `-70.0`.
        v_reset (float, optional): membrane voltage a neuron is reset to following action potential, in :math:`mV`. Defaults to `-65.0`.
        v_threshold (float, optional): base membrane voltage at which a neuron generates an action potential, in :math:`mV`. Defaults to `-50.0`.
        tc_theta (float, optional): time constant controlling the decay of the firing threshold back to `v_threshold`, inversely proportional to decay rate. Defaults to `30.0`.
        theta_plus (float, optional): increase to the firing threshold after each action potential, in :math:`mV`. Defaults to `0.05`.
        ts_abs_refrac (int, optional): number of time steps for which a neuron should be in its absolute refractory period. Defaults to `2`.
        tc_membrane (float | None, optional): time constant of a neuron's membrane, defined as :math:`\\tau_m = R_mC_m`, in :math:`ms`. Defaults to `20.0`.
        r_membrane (float | None, optional): electrical resistance of a neuron's membrane, in :math:`k\\Omega`. Defaults to `1.0`.
        c_membrane (float | None, optional): capacitance of a neuron's membrane, in :math:`nF`. Defaults to `None`.
        adapt_theta (bool | None, optional): controls if the adaptive threshold should be modified this timestep, when `None` adaptation only occurs when in training mode. Defaults to `None`.
        batch_size (int, optional): number of separate inputs to be passed along the batch (:math:`0^\\text{th}`) axis. Defaults to `1`.

    Raises:
        ValueError: `batch_size` must be a positive integer.
        ValueError: `step_time` must be a non-negative value.
        TypeError: at least two of `tc_membrane`, `r_membrane`, and `c_membrane` must not be `None`.
        ValueError: `tc_membrane`, `r_membrane`, and `c_membrane` have mismatched values, ``tc_membrane == r_membrane * c_membrane`` must be true.
    """
    def __init__(
        self,
        shape: tuple[int, ...] | int,
        step_time: float,
        v_rest: float = -70.0,
        v_reset: float = -65.0,
        v_threshold: float = -50.0,
        tc_theta: float = 30.0,
        theta_plus: float = 0.05,
        ts_abs_refrac: int = 2,
        tc_membrane: float | None = 20.0,
        r_membrane: float | None = 1.0,
        c_membrane: float | None = None,
        adapt_theta: bool | None = None,
        batch_size: int = 1,
    ):
        # call superclass constructors
        AbstractDynamics.__init__(self)
        ShapeMixin.__init__(self, shape, batch_size, ('v_membranes', 'ts_refrac_membranes', 'theta'))

        # verify a valid step time
        if step_time <= 0:
            raise ValueError(f"step time must be greater than zero, received {step_time}")

        # set mode variable
        self.adapt_theta = adapt_theta

        # register hyperparameters as buffers
        self.register_buffer('step_time', torch.tensor(step_time, requires_grad=False))
        self.register_buffer('v_rest', torch.tensor(v_rest, requires_grad=False))
        self.register_buffer('v_reset', torch.tensor(v_reset, requires_grad=False))
        self.register_buffer('v_threshold', torch.tensor(v_threshold, requires_grad=False))
        self.register_buffer('ts_abs_refrac', torch.tensor(ts_abs_refrac, requires_grad=False))
        self.register_buffer('tc_membrane', None)
        self.register_buffer('r_membrane', None)
        self.register_buffer('c_membrane', None)
        if [tc_membrane, r_membrane, c_membrane].count(None) > 1:
            raise TypeError("at least two of 'tc_membrane', 'r_membrane', and 'c_membrane' must be specified")
        elif [tc_membrane, r_membrane, c_membrane].count(None) == 1:
            if tc_membrane is None:
                self.r_membrane = torch.tensor(r_membrane, requires_grad=False)
                self.c_membrane = torch.tensor(c_membrane, requires_grad=False)
                self.tc_membrane = self.r_membrane * self.c_membrane
            if r_membrane is None:
                self.tc_membrane = torch.tensor(tc_membrane, requires_grad=False)
                self.c_membrane = torch.tensor(c_membrane, requires_grad=False)
                self.r_membrane = self.tc_membrane / self.c_membrane
            if c_membrane is None:
                self.tc_membrane = torch.tensor(tc_membrane, requires_grad=False)
                self.r_membrane = torch.tensor(r_membrane, requires_grad=False)
                self.c_membrane = self.tc_membrane / self.r_membrane
        else:
            if tc_membrane == r_membrane * c_membrane:
                self.tc_membrane = torch.tensor(tc_membrane, requires_grad=False)
                self.r_membrane = torch.tensor(r_membrane, requires_grad=False)
                self.c_membrane = torch.tensor(c_membrane, requires_grad=False)
            else:
                raise ValueError("'tc_membrane', 'r_membrane', and 'c_membrane' have mismatched values, \
                    tc_membrane must equal r_membrane * c_membrane")
        self.register_buffer('tc_theta', torch.tensor(tc_theta, requires_grad=False))
        self.register_buffer('theta_plus', torch.tensor(theta_plus, requires_grad=False))
        self.register_buffer('decay_rate', torch.exp(-self.step_time / self.tc_membrane))
        self.register_buffer('theta_decay_rate', torch.exp(-self.step_time / self.tc_theta))

        # register states as parameters
        self.register_parameter('v_membranes', nn.Parameter(torch.ones(self.batched_shape, requires_grad=False) * self.v_rest, False))
        self.register_parameter('ts_refrac_membranes', nn.Parameter(torch.zeros(self.batched_shape, requires_grad=False), False))
        self.register_parameter('theta', nn.Parameter(torch.zeros(self.batched_shape, requires_grad=False), False))

    def clear(self, preserve_theta: bool = False, **kwargs) -> None:
        """Reinitializes the neurons' states.

        This sets the neurons back to their initial condition: their membrane voltages all set to :py:attr:`v_rest` and with their
        absolute refractory period counters zeroed-out. This should be used wherever there is a need to cleanly delineate
        inputs, such as between batches. Whether or not the adaptive threshold component should be reset is optional.

        Args:
            preserve_theta (bool, optional): controls if the adaptive threshold component :py:attr:`theta` should be reset to zero. Defaults to `False`.
        """
        self.v_membranes.fill_(self.v_rest)
        self.ts_refrac_membranes.fill_(0)
        if not preserve_theta:
            self.theta.fill_(0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Processes input currents into the neuron bodies and generates spiking output.

        Args:
            inputs (torch.Tensor): tensor of input currents shaped like the population of neurons.

        Returns:
            torch.Tensor: resulting tensor shaped like the population of neurons indicating which spiked (1) and which did not (0).
        """
        # set theta adaptation
        if self.adapt_theta is None:
            adapt_theta = self.training
        else:
            adapt_theta = self.adapt_theta

        # decay voltages
        self.v_membranes.sub_(self.v_rest)
        self.v_membranes.mul_(self.decay_rate)
        self.v_membranes.add_(self.v_rest)

        # decay theta
        if adapt_theta:
            self.theta.mul_(self.theta_decay_rate)

        # incorporate input currents
        self.v_membranes.add_((self.ts_refrac_membranes == 0) * (inputs * self.r_membrane))

        # determine output spikes
        spikes = self.v_membranes >= (self.v_threshold + self.theta)
        spikes.masked_fill_(self.ts_refrac_membranes != 0, 0)

        # boost theta
        if adapt_theta:
            self.theta.add_(self.theta_plus * spikes)

        # tick down refractory periods
        self.ts_refrac_membranes.sub_(1)
        self.ts_refrac_membranes.clamp_(min=0)

        # set refractory periods of depolarized neurons
        self.ts_refrac_membranes.masked_fill_(spikes, self.ts_abs_refrac)

        # reset voltages for last depolarized neurons
        self.v_membranes.masked_fill_(self.ts_refrac_membranes == self.ts_abs_refrac, self.v_reset)

        return spikes
