from typing import Callable

import torch
import torch.nn as nn

from inferno._internal import agnostic_mean
from inferno.learning import AbstractLayerUpdater, AbstractLayerwiseUpdater
from inferno.neural import AbstractLayer
from inferno.monitoring import InputMonitor, OutputMonitor
from inferno.monitoring import SinglePassthroughReducer, AdditiveTraceReducer, TraceReducer, ScalingTraceReducer


class MSTDPUpdater(AbstractLayerwiseUpdater):
    """Updates a layer's weights using the Modulated Spike-Timing Dependent Plasticity algorithm.

    More details on this algorithm can be found at https://doi.org/10.1162/neco.2007.19.6.1468.

        Args:
            trainables (tuple[AbstractLayer, ...] | AbstractLayer): layers to which MSTDP will be applied.
            step_time (float): length of the time steps used by the model, in :math:`ms`.
            lr_post (float): learning rate for the postsynaptic update component.
            lr_pre (float): learning rate for the presynaptic update component.
            tc_post (float): time constant for postsynaptic additive spike trace.
            tc_pre (float): time constant for presynaptic additive spike trace.
            amp_post (float): amplitude for postsynaptic additive spike trace, generally negative.
            amp_pre (float): amplitude for presynaptic additive spike trace, generally positive.
            weight_bounding_mode (str | None, optional): Mode of operation for weight limiting. Defaults to None.
            batch_reduction (Callable[[torch.Tensor], torch.Tensor], optional): Function to reduce batch dimension of intermediate computations. Defaults to torch.mean.
            spatial_reduction (Callable[[torch.Tensor], torch.Tensor], optional): Function to reduce spatial dimension of intermediate computations. Defaults to torch.mean.
        """
    def __init__(
        self,
        trainables: tuple[AbstractLayer, ...] | AbstractLayer,
        step_time: float,
        lr_post: float,
        lr_pre: float,
        tc_post: float,
        tc_pre: float,
        amp_post: float,
        amp_pre: float,
        weight_min: float | None = None,
        weight_max: float | None = None,
        weight_bounding_mode: str | None = None,
        batch_reduction: Callable[[torch.Tensor], torch.Tensor] = torch.mean,
        spatial_reduction: Callable[[torch.Tensor], torch.Tensor] = torch.mean
    ):
        # call superclass constructor
        AbstractLayerwiseUpdater.__init__(self, trainables)

        # register training hyperparameters
        self.register_buffer('lr_post', torch.tensor(lr_post))
        self.register_buffer('lr_pre', torch.tensor(lr_pre))
        self.register_buffer('tc_post', torch.tensor(tc_post))
        self.register_buffer('tc_pre', torch.tensor(tc_pre))
        self.register_buffer('amp_post', torch.tensor(amp_post))
        self.register_buffer('amp_pre', torch.tensor(amp_pre))

        # register environmental buffer
        self.register_buffer('step_time', torch.tensor(step_time))

        # register weight limits
        if (weight_min is not None) and (weight_max is not None):
            if not (weight_min < weight_max):
                raise ValueError(f"minimum weight {weight_min} must be less than maximum weight {weight_max}")
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.weight_bounding_mode = weight_bounding_mode

        # register axis reduction functions
        self.batch_reduction = batch_reduction
        self.spatial_reduction = spatial_reduction

        # construct necessary monitors
        for sm in self.submodules:
            self.monitors[hex(id(sm))]['p_in'] = InputMonitor(reducer=AdditiveTraceReducer(decay=torch.exp(-self.step_time / self.tc_pre), amplitude=self.amp_pre), train_update=True, eval_update=False, module=sm)
            self.monitors[hex(id(sm))]['p_out'] = OutputMonitor(reducer=AdditiveTraceReducer(decay=torch.exp(-self.step_time / self.tc_post), amplitude=self.amp_post), train_update=True, eval_update=False, module=sm)
            self.monitors[hex(id(sm))]['s_in'] = InputMonitor(reducer=SinglePassthroughReducer(), train_update=True, eval_update=False, module=sm)
            self.monitors[hex(id(sm))]['s_out'] = OutputMonitor(reducer=SinglePassthroughReducer(), train_update=True, eval_update=False, module=sm)

    def forward(self, reward: float | torch.Tensor) -> None:
        """Performs an update step based on previously processed inputs and a reward signal.

        With batch training, the reward should be presented as a vector with a single value for each batch.

        The reward can be presented in a shape compatible with the relevant intermediate (i.e. for a dense connection of 4 to 9, it would need to be compatible with (b, 9, 4))

        Args:
            reward (float | torch.Tensor): Reward signal for the current time step.
        """
        for sm in self.submodules:
            for conn in sm.connections:
                # build presynaptic and postsynaptic spike traces and spikes
                P_plus = conn.reshape_inputs(self.monitors[hex(id(sm))]['p_in'].peak())
                P_minus = conn.reshape_outputs(self.monitors[hex(id(sm))]['p_out'].peak())
                I_post = conn.reshape_outputs(self.monitors[hex(id(sm))]['s_out'].peak()).to(dtype=P_minus.dtype)
                I_pre = conn.reshape_inputs(self.monitors[hex(id(sm))]['s_in'].peak()).to(dtype=P_plus.dtype)

                # compute update, allows for vector input if batch training
                if isinstance(reward, torch.Tensor):
                    post_term = torch.bmm(P_minus, I_pre)
                    post_term.mul_(reward[(...,) + (None,) * (post_term.ndim - reward.ndim)])
                    post_term = self.lr_post * self.batch_reduction(post_term, dim=0, keepdim=False)
                    pre_term = torch.bmm(I_post, P_plus)
                    pre_term.mul_(reward[(...,) + (None,) * (pre_term.ndim - reward.ndim)])
                    pre_term = self.lr_pre * self.batch_reduction(pre_term, dim=0, keepdim=False)
                else:
                    post_term = self.lr_post * reward * self.batch_reduction(torch.bmm(P_minus, I_pre), dim=0, keepdim=False)
                    pre_term = self.lr_pre * reward * self.batch_reduction(torch.bmm(I_post, P_plus), dim=0, keepdim=False)
                match self.weight_bounding_mode:
                    case 'soft':
                        if self.weight_min is not None:
                            sub_mul = self.weight - self.weight_min
                        else:
                            sub_mul = 1.0
                        if self.weight_max is not None:
                            add_mul = self.weight_max - self.weight
                        else:
                            add_mul = 1.0
                        conn.update_weight(conn.reshape_weight_update(sub_mul * post_term, self.spatial_reduction) + conn.reshape_weight_update(add_mul * pre_term, self.spatial_reduction))
                    case 'hard':
                        if self.weight_min is not None:
                            sub_mul = torch.heaviside(self.weight - self.weight_min, torch.tensor([0.0], dtype=post_term.dtype, device=post_term.device, requires_grad=False))
                        else:
                            sub_mul = 1.0
                        if self.weight_max is not None:
                            add_mul = torch.heaviside(self.weight_max - self.weight, torch.tensor([0.0], dtype=pre_term.dtype, device=pre_term.device, requires_grad=False))
                        else:
                            add_mul = 1.0
                        conn.update_weight(conn.reshape_weight_update(sub_mul * post_term, self.spatial_reduction) + conn.reshape_weight_update(add_mul * pre_term, self.spatial_reduction))
                    case 'clamp':
                        conn.update_weight(conn.reshape_weight_update(pre_term) + conn.reshape_weight_update(post_term), self.weight_min, self.weight_max)
                    case _:
                        conn.update_weight(conn.reshape_weight_update(pre_term) + conn.reshape_weight_update(post_term))

                # clean temporary variables
                del P_plus
                del P_minus
                del I_post
                del I_pre
                del post_term
                del pre_term


class MSTDPETUpdater(AbstractLayerwiseUpdater):
    """Updates a layer's weights using the Modulated Spike-Timing Dependent Plasticity algorithm.

    More details on this algorithm can be found at https://doi.org/10.1162/neco.2007.19.6.1468.

        Args:
            trainables (tuple[AbstractLayer, ...] | AbstractLayer): layers to which MSTDP will be applied.
            step_time (float): length of the time steps used by the model, in :math:`ms`.
            lr_post (float): learning rate for the postsynaptic update component.
            lr_pre (float): learning rate for the presynaptic update component.
            tc_post (float): time constant for postsynaptic additive spike trace.
            tc_pre (float): time constant for presynaptic additive spike trace.
            tc_eligibility (float): time constant for additive trace of eligibility.
            amp_post (float): amplitude for postsynaptic additive spike trace, generally negative.
            amp_pre (float): amplitude for presynaptic additive spike trace, generally positive.
            weight_bounding_mode (str | None, optional): Mode of operation for weight limiting. Defaults to None.
            batch_reduction (Callable[[torch.Tensor], torch.Tensor], optional): Function to reduce batch dimension of intermediate computations. Defaults to torch.mean.
            spatial_reduction (Callable[[torch.Tensor], torch.Tensor], optional): Function to reduce spatial dimension of intermediate computations. Defaults to torch.mean.
        """
    def __init__(
        self,
        trainables: tuple[AbstractLayer, ...] | AbstractLayer,
        step_time: float,
        lr_post: float,
        lr_pre: float,
        tc_post: float,
        tc_pre: float,
        tc_eligibility: float,
        amp_post: float,
        amp_pre: float,
        weight_min: float | None = None,
        weight_max: float | None = None,
        weight_bounding_mode: str | None = None,
        batch_reduction: Callable[[torch.Tensor], torch.Tensor] = torch.mean,
        spatial_reduction: Callable[[torch.Tensor], torch.Tensor] = torch.mean
    ):
        # call superclass constructor
        AbstractLayerwiseUpdater.__init__(self, trainables)

        # register training hyperparameters
        self.register_buffer('lr_post', torch.tensor(lr_post))
        self.register_buffer('lr_pre', torch.tensor(lr_pre))
        self.register_buffer('tc_post', torch.tensor(tc_post))
        self.register_buffer('tc_pre', torch.tensor(tc_pre))
        self.register_buffer('tc_eligibility', torch.tensor(tc_eligibility))
        self.register_buffer('amp_post', torch.tensor(amp_post))
        self.register_buffer('amp_pre', torch.tensor(amp_pre))

        # register environmental buffer
        self.register_buffer('step_time', torch.tensor(step_time))

        # register weight limits
        if (weight_min is not None) and (weight_max is not None):
            if not (weight_min < weight_max):
                raise ValueError(f"minimum weight {weight_min} must be less than maximum weight {weight_max}")
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.weight_bounding_mode = weight_bounding_mode

        # register axis reduction functions
        self.batch_reduction = batch_reduction
        self.spatial_reduction = spatial_reduction

        # construct necessary monitors
        for sm in self.submodules:
            self.monitors[hex(id(sm))]['p_in'] = InputMonitor(reducer=AdditiveTraceReducer(decay=torch.exp(-self.step_time / self.tc_pre), amplitude=self.amp_pre), train_update=True, eval_update=False, module=sm)
            self.monitors[hex(id(sm))]['p_out'] = OutputMonitor(reducer=AdditiveTraceReducer(decay=torch.exp(-self.step_time / self.tc_post), amplitude=self.amp_post), train_update=True, eval_update=False, module=sm)
            self.monitors[hex(id(sm))]['s_in'] = InputMonitor(reducer=SinglePassthroughReducer(), train_update=True, eval_update=False, module=sm)
            self.monitors[hex(id(sm))]['s_out'] = OutputMonitor(reducer=SinglePassthroughReducer(), train_update=True, eval_update=False, module=sm)

        # constuct eligibility trace
        self.post_eligibility_trace = nn.ModuleDict()
        self.pre_eligibility_trace = nn.ModuleDict()
        for sm in self.submodules:
            self.post_eligibility_trace[hex(id(sm))] = ScalingTraceReducer(amplitude=torch.exp(1 / self.tc_eligibility), step_time=self.step_time, time_constant=self.tc_eligibility)
            self.pre_eligibility_trace[hex(id(sm))] = ScalingTraceReducer(amplitude=torch.exp(1 / self.tc_eligibility), step_time=self.step_time, time_constant=self.tc_eligibility)

    def clear(self):
        AbstractLayerwiseUpdater.clear()
        for sm in self.submodules:
            self.post_eligibility_trace[hex(id(sm))].clear()
            self.pre_eligibility_trace[hex(id(sm))].clear()

    def forward(self, reward: float | torch.Tensor) -> None:
        """Performs an update step based on previously processed inputs and a reward signal.

        With batch training, the reward should be presented as a vector with a single value for each batch.

        The reward can be presented in a shape compatible with the relevant intermediate (i.e. for a dense connection of 4 to 9, it would need to be compatible with (b, 9, 4))

        Args:
            reward (float | torch.Tensor): Reward signal for the current time step.
        """
        for sm in self.submodules:
            for conn in sm.connections:
                # build presynaptic and postsynaptic spike traces and spikes
                P_plus = conn.reshape_inputs(self.monitors[hex(id(sm))]['p_in'].peak())
                P_minus = conn.reshape_outputs(self.monitors[hex(id(sm))]['p_out'].peak())
                I_post = conn.reshape_outputs(self.monitors[hex(id(sm))]['s_out'].peak()).to(dtype=P_minus.dtype)
                I_pre = conn.reshape_inputs(self.monitors[hex(id(sm))]['s_in'].peak()).to(dtype=P_plus.dtype)

                # compute update, allows for vector input if batch training
                post_term = torch.bmm(P_minus, I_pre)
                pre_term = torch.bmm(I_post, P_plus)
                self.post_eligibility_trace[hex(id(sm))](post_term)
                self.pre_eligibility_trace[hex(id(sm))](pre_term)
                post_term = self.post_eligibility_trace[hex(id(sm))].peak()
                pre_term = self.pre_eligibility_trace[hex(id(sm))].peak()
                if isinstance(reward, torch.Tensor):
                    post_term.mul_(reward[(...,) + (None,) * (post_term.ndim - reward.ndim)])
                    pre_term.mul_(reward[(...,) + (None,) * (pre_term.ndim - reward.ndim)])
                    post_term = self.lr_post * self.batch_reduction(post_term, dim=0, keepdim=False)
                    pre_term = self.lr_pre * self.batch_reduction(pre_term, dim=0, keepdim=False)
                else:
                    post_term = self.lr_post * reward * self.batch_reduction(post_term, dim=0, keepdim=False)
                    pre_term = self.lr_pre * reward * self.batch_reduction(pre_term, dim=0, keepdim=False)
                match self.weight_bounding_mode:
                    case 'soft':
                        if self.weight_min is not None:
                            sub_mul = self.weight - self.weight_min
                        else:
                            sub_mul = 1.0
                        if self.weight_max is not None:
                            add_mul = self.weight_max - self.weight
                        else:
                            add_mul = 1.0
                        conn.update_weight(conn.reshape_weight_update(sub_mul * post_term, self.spatial_reduction) + conn.reshape_weight_update(add_mul * pre_term, self.spatial_reduction))
                    case 'hard':
                        if self.weight_min is not None:
                            sub_mul = torch.heaviside(self.weight - self.weight_min, torch.tensor([0.0], dtype=post_term.dtype, device=post_term.device, requires_grad=False))
                        else:
                            sub_mul = 1.0
                        if self.weight_max is not None:
                            add_mul = torch.heaviside(self.weight_max - self.weight, torch.tensor([0.0], dtype=pre_term.dtype, device=pre_term.device, requires_grad=False))
                        else:
                            add_mul = 1.0
                        conn.update_weight(conn.reshape_weight_update(sub_mul * post_term, self.spatial_reduction) + conn.reshape_weight_update(add_mul * pre_term, self.spatial_reduction))
                    case 'clamp':
                        conn.update_weight(conn.reshape_weight_update(pre_term) + conn.reshape_weight_update(post_term), self.weight_min, self.weight_max)
                    case _:
                        conn.update_weight(conn.reshape_weight_update(pre_term) + conn.reshape_weight_update(post_term))

                # clean temporary variables
                del P_plus
                del P_minus
                del I_post
                del I_pre
                del post_term
                del pre_term


class EDLDelayUpdater(AbstractLayerUpdater):
    """Updates a layer's delays using the Extended Delay Learning algorithm.
    Note this only impelements the delay learning component, not the weight learning component.

    More details on the algorithm can be found at https://doi.org/10.1007/978-3-319-26535-3_22.

    Args:
        trainable (AbstractLayer): layer to which EDL delay learning will be applied.
        step_time (float): length of the time steps used by the model, in :math:`ms`.
        update_max (float): magnitude of the maximum/minimum permissible update, in time steps.
        amp_trace (float): amplitude of the trace calculation.
        tc_trace (float): time constant of the trace calculation.
        exc_inh_mask (torch.Tensor): mask specifying if neuron in a given position is excitatory (mask=1) or inhibitory (mask=0).
        weight_bounding_mode (str | None, optional): Mode of operation for weight limiting. Defaults to None.
        batch_reduction (Callable[[torch.Tensor], torch.Tensor], optional): _description_. Defaults to agnostic_mean.
        spatial_reduction (Callable[[torch.Tensor], torch.Tensor], optional): _description_. Defaults to agnostic_mean.
    """
    def __init__(
        self,
        trainable: AbstractLayer,
        step_time: float,
        update_max: float,
        amp_trace: float,
        tc_trace: float,
        exc_inh_mask: torch.Tensor,
        batch_reduction: Callable[[torch.Tensor], torch.Tensor] = agnostic_mean,
        spatial_reduction: Callable[[torch.Tensor], torch.Tensor] = agnostic_mean
    ):
        # call superclass constructor
        AbstractLayerUpdater.__init__(self, trainable)

        # register buffers
        self.register_buffer('step_time', torch.tensor(step_time))
        self.register_buffer('update_max', torch.tensor(update_max))
        self.register_buffer('amp_trace', torch.tensor(amp_trace))
        self.register_buffer('tc_trace', torch.tensor(tc_trace))
        self.register_buffer('mask', exc_inh_mask.float())

        # register axis reduction functions
        self.batch_reduction = batch_reduction
        self.spatial_reduction = spatial_reduction

        # construct necessary monitors
        self.monitors['input_traces'] = InputMonitor(reducer=TraceReducer(amplitude=self.amp_trace, step_time=self.step_time, time_constant=self.tc_trace), train_update=True, eval_update=False, module=self.submodule)
        self.monitors['output_spikes'] = OutputMonitor(reducer=SinglePassthroughReducer(), train_update=True, eval_update=False, module=self.submodule)

        # construct excitatory and inhibitory trace mask tensors
        self.register_buffer('exc_mask', torch.where(self.mask == 1, 0.0, float('-inf')))
        self.register_buffer('inh_mask', torch.where(self.mask == 0, 0.0, float('-inf')))
        if self.mask.ndim == 2:
            self.exc_mask.unsqueeze_(0)
            self.inh_mask.unsqueeze_(0)
        if self.mask.ndim == 2:
            self.mask.squeeze_(0)

    def forward(self, target: torch.Tensor) -> None:
        """Performs an update step based on previously processed inputs and a target tensor.

        Args:
            target (torch.Tensor): Target spikes for the current time step.
        """
        for conn in self.submodule.connections:
            # get spike traces as an batches x outputs x inputs_per_neuron batched matrix (0 if no spike, -inf if no equivalent mode)
            traces = conn.inputs_as_receptive_areas(self.monitors['input_traces'].peak())
            nearest_exc_traces = torch.amax(traces + self.exc_mask, dim=-1, keepdim=True)
            nearest_inh_traces = torch.amax(traces + self.inh_mask, dim=-1, keepdim=True)

            # get the length of time since the last spikes in milliseconds based on the trace (-inf if no spike, nan if no equivalent mode)
            nearest_exc_spikes = (-self.tc_trace * torch.log(nearest_exc_traces / self.amp_trace))
            nearest_inh_spikes = (-self.tc_trace * torch.log(nearest_inh_traces / self.amp_trace))

            # construct the condition masks
            output_spikes = self.monitors['output_spikes'].peak()
            output_spikes = output_spikes.view(output_spikes.shape[0], -1)
            cond_pot_mask = torch.logical_and(torch.logical_not(output_spikes), target.view(target.shape[0], -1)).float().unsqueeze_(-1)
            cond_dep_mask = torch.logical_and(output_spikes, torch.logical_not(target.view(target.shape[0], -1))).float().unsqueeze_(-1)

            # compute updates in milliseconds, using batches x (output_shape) for target and output_spikes
            exc_update = torch.nan_to_num((nearest_exc_spikes * traces) / nearest_exc_traces, nan=0.0)
            inh_update = torch.nan_to_num((nearest_inh_spikes * traces) / nearest_inh_traces, nan=0.0)

            # compute unified update and convert milliseconds to timesteps (note, pot and dep masks are mutually exclusive)
            update = self.batch_reduction((exc_update * cond_pot_mask) - (exc_update * cond_dep_mask) - (inh_update * cond_pot_mask) + (inh_update * cond_dep_mask), dim=0, keepdim=False)
            update.div_(self.step_time)

            # apply offset according to existing delays
            delays = conn.delays_as_receptive_area()
            update.sub_(delays * self.mask)
            update.add_(delays * torch.abs(self.mask - 1))

            # clamp update to limits
            update.clamp_(-self.update_max, self.update_max)

            # perform update step
            conn.update_delay_ra(update)
