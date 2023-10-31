from typing import Callable

import torch

from inferno.learning import AbstractLayerwiseUpdater
from inferno.neural import AbstractLayer
from inferno.monitoring import InputMonitor, OutputMonitor, TraceReducer, AdditiveTraceReducer


class OnlineSTDPUpdater(AbstractLayerwiseUpdater):
    """Updates layers using an online variant of spiking time-dependent plasticity.

    More details can be found at http://www.scholarpedia.org/article/Spike-timing_dependent_plasticity#Online_implementation_of_STDP_models

    Args:
        trainables (tuple[AbstractLayer, ...] | AbstractLayer): layers to which STDP will be applied.
        step_time (float): length of the time steps used by the model, in :math:`ms`.
        lr_post (float): learning rate for the postsynaptic update component, multiplicative term for added component.
        lr_pre (float): learning rate for the presynaptic update component, multiplicative term for subtracted component.
        tc_post (float): time constant for the postsynaptic update component.
        tc_pre (float): time constant for the presynaptic update component.
        amp_post (float): amplitude for postsynaptic spike trace. Defaults to `1.0`.
        amp_pre (float): amplitude for presynaptic spike trace. Defaults to `1.0`.
        weight_min (float | None, optional): if not `None`, sets a lower bound to the learned weights. Defaults to None.
        weight_max (float | None, optional): if not `None`, sets an upper bound to the learned weights. Defaults to None.
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
        amp_post: float = -1.0,
        amp_pre: float = 1.0,
        weight_min: float | None = None,
        weight_max: float | None = None,
        weight_bounding_mode: str | None = None,
        batch_reduction: Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] = torch.mean,
        spatial_reduction: Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] = torch.mean
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

        self.register_buffer('pApost', torch.empty(0))
        self.register_buffer('pApre', torch.empty(0))

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
            self.monitors[hex(id(sm))]['in'] = InputMonitor(reducer=AdditiveTraceReducer(decay=torch.exp(-self.step_time / self.tc_pre), amplitude=self.amp_pre), train_update=True, eval_update=False, module=sm)
            self.monitors[hex(id(sm))]['out'] = OutputMonitor(reducer=AdditiveTraceReducer(decay=torch.exp(-self.step_time / self.tc_post), amplitude=self.amp_post), train_update=True, eval_update=False, module=sm)

    def forward(self) -> None:
        """Performs an update step based on previously processed inputs.
        """
        for sm in self.submodules:
            for conn in sm.connections:
                # build presynaptic and postsynaptic spike traces and spikes
                A_post = conn.reshape_outputs(self.monitors[hex(id(sm))]['out'].peak())
                A_pre = conn.reshape_inputs(self.monitors[hex(id(sm))]['in'].peak())

                if self.pApost.numel() == 0:
                    self.pApost = A_post.clone().detach()
                if self.pApre.numel() == 0:
                    self.pApre = A_pre.clone().detach()

                I_post = (A_post > self.pApost).to(dtype=A_post.dtype)
                I_pre = (A_pre > self.pApre).to(dtype=A_pre.dtype)

                self.pApost = A_post.clone().detach()
                self.pApre = A_pre.clone().detach()

                # compute update
                dep_term = self.lr_pre * self.batch_reduction(torch.bmm(A_post, I_pre), dim=0, keepdim=False)
                pot_term = self.lr_post * self.batch_reduction(torch.bmm(I_post, A_pre), dim=0, keepdim=False)

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
                        conn.update_weight(conn.reshape_weight_update(sub_mul * dep_term, self.spatial_reduction) - conn.reshape_weight_update(add_mul * pot_term, self.spatial_reduction))
                    case 'hard':
                        if self.weight_min is not None:
                            sub_mul = torch.heaviside(self.weight - self.weight_min, torch.tensor([0.0], dtype=dep_term.dtype, device=dep_term.device, requires_grad=False))
                        else:
                            sub_mul = 1.0
                        if self.weight_max is not None:
                            add_mul = torch.heaviside(self.weight_max - self.weight, torch.tensor([0.0], dtype=pot_term.dtype, device=pot_term.device, requires_grad=False))
                        else:
                            add_mul = 1.0
                        conn.update_weight(conn.reshape_weight_update(sub_mul * dep_term, self.spatial_reduction) - conn.reshape_weight_update(add_mul * pot_term, self.spatial_reduction))
                    case 'clamp':
                        conn.update_weight(conn.reshape_weight_update(pot_term, self.spatial_reduction) - conn.reshape_weight_update(dep_term, self.spatial_reduction), self.weight_min, self.weight_max)
                    case _:
                        conn.update_weight(conn.reshape_weight_update(pot_term, self.spatial_reduction) - conn.reshape_weight_update(dep_term, self.spatial_reduction))

                # clean temporary variables
                del A_post
                del A_pre
                del I_post
                del I_pre
                del pot_term
                del dep_term


class OnlineSTDPUpdaterV2(AbstractLayerwiseUpdater):
    """Updates layers using an online variant of spiking time-dependent plasticity.

    More details can be found at http://www.scholarpedia.org/article/Spike-timing_dependent_plasticity#Online_implementation_of_STDP_models

    Args:
        trainables (tuple[AbstractLayer, ...] | AbstractLayer): layers to which STDP will be applied.
        step_time (float): length of the time steps used by the model, in :math:`ms`.
        lr_post (float): learning rate for the postsynaptic update component, multiplicative term for added component.
        lr_pre (float): learning rate for the presynaptic update component, multiplicative term for subtracted component.
        tc_post (float): time constant for the postsynaptic update component.
        tc_pre (float): time constant for the presynaptic update component.
        amp_post (float): amplitude for postsynaptic spike trace. Defaults to `1.0`.
        amp_pre (float): amplitude for presynaptic spike trace. Defaults to `1.0`.
        weight_min (float | None, optional): if not `None`, sets a lower bound to the learned weights. Defaults to None.
        weight_max (float | None, optional): if not `None`, sets an upper bound to the learned weights. Defaults to None.
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
        amp_post: float = -1.0,
        amp_pre: float = 1.0,
        weight_min: float | None = None,
        weight_max: float | None = None,
        weight_bounding_mode: str | None = None,
        batch_reduction: Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] = torch.mean,
        spatial_reduction: Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] = torch.mean
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
            self.monitors[hex(id(sm))]['in'] = InputMonitor(reducer=TraceReducer(decay=torch.exp(-self.step_time / self.tc_pre), amplitude=self.amp_pre), train_update=True, eval_update=False, module=sm)
            self.monitors[hex(id(sm))]['out'] = OutputMonitor(reducer=TraceReducer(decay=torch.exp(-self.step_time / self.tc_post), amplitude=self.amp_post), train_update=True, eval_update=False, module=sm)

    def forward(self) -> None:
        """Performs an update step based on previously processed inputs.
        """
        for sm in self.submodules:
            for conn in sm.connections:
                # build presynaptic and postsynaptic spike traces and spikes
                A_post = conn.reshape_outputs(self.monitors[hex(id(sm))]['out'].peak())
                A_pre = conn.reshape_inputs(self.monitors[hex(id(sm))]['in'].peak())
                I_post = (A_post == self.amp_post).to(dtype=A_post.dtype)
                I_pre = (A_pre == self.amp_pre).to(dtype=A_pre.dtype)
                # compute update
                pot_term = self.lr_post * self.batch_reduction(torch.bmm(I_post, A_pre), dim=0, keepdim=False)
                dep_term = self.lr_pre * self.batch_reduction(torch.bmm(A_post, I_pre), dim=0, keepdim=False)

                match self.weight_bounding_mode:
                    case 'clamp':
                        conn.update_weight(conn.reshape_weight_update(pot_term, self.spatial_reduction) - conn.reshape_weight_update(dep_term, self.spatial_reduction), self.weight_min, self.weight_max)
                    case _:
                        conn.update_weight(conn.reshape_weight_update(pot_term, self.spatial_reduction) - conn.reshape_weight_update(dep_term, self.spatial_reduction))

                # clean temporary variables
                del A_post
                del A_pre
                del I_post
                del I_pre
                del pot_term
                del dep_term


class OnlineSTDPAdditiveUpdaterV2(AbstractLayerwiseUpdater):
    """Updates layers using an online variant of spiking time-dependent plasticity.

    More details can be found at http://www.scholarpedia.org/article/Spike-timing_dependent_plasticity#Online_implementation_of_STDP_models

    Args:
        trainables (tuple[AbstractLayer, ...] | AbstractLayer): layers to which STDP will be applied.
        step_time (float): length of the time steps used by the model, in :math:`ms`.
        lr_post (float): learning rate for the postsynaptic update component, multiplicative term for added component.
        lr_pre (float): learning rate for the presynaptic update component, multiplicative term for subtracted component.
        tc_post (float): time constant for the postsynaptic update component.
        tc_pre (float): time constant for the presynaptic update component.
        amp_post (float): amplitude for postsynaptic spike trace. Defaults to `1.0`.
        amp_pre (float): amplitude for presynaptic spike trace. Defaults to `1.0`.
        weight_min (float | None, optional): if not `None`, sets a lower bound to the learned weights. Defaults to None.
        weight_max (float | None, optional): if not `None`, sets an upper bound to the learned weights. Defaults to None.
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
        amp_post: float = -1.0,
        amp_pre: float = 1.0,
        weight_min: float | None = None,
        weight_max: float | None = None,
        weight_bounding_mode: str | None = None,
        batch_reduction: Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] = torch.mean,
        spatial_reduction: Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] = torch.mean
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
            self.monitors[hex(id(sm))]['in'] = InputMonitor(reducer=AdditiveTraceReducer(decay=torch.exp(-self.step_time / self.tc_pre), amplitude=self.amp_pre), train_update=True, eval_update=False, module=sm)
            self.monitors[hex(id(sm))]['out'] = OutputMonitor(reducer=AdditiveTraceReducer(decay=torch.exp(-self.step_time / self.tc_post), amplitude=self.amp_post), train_update=True, eval_update=False, module=sm)

    def forward(self) -> None:
        """Performs an update step based on previously processed inputs.
        """
        for sm in self.submodules:
            for conn in sm.connections:
                # build presynaptic and postsynaptic spike traces and spikes
                A_post = conn.reshape_outputs(self.monitors[hex(id(sm))]['out'].peak())
                A_pre = conn.reshape_inputs(self.monitors[hex(id(sm))]['in'].peak())
                I_post = (A_post == self.amp_post).to(dtype=A_post.dtype)
                I_pre = (A_pre == self.amp_pre).to(dtype=A_pre.dtype)
                # compute update
                pot_term = self.lr_post * self.batch_reduction(torch.bmm(I_post, A_pre), dim=0, keepdim=False)
                dep_term = self.lr_pre * self.batch_reduction(torch.bmm(A_post, I_pre), dim=0, keepdim=False)

                match self.weight_bounding_mode:
                    case 'clamp':
                        conn.update_weight(conn.reshape_weight_update(pot_term, self.spatial_reduction) - conn.reshape_weight_update(dep_term, self.spatial_reduction), self.weight_min, self.weight_max)
                    case _:
                        conn.update_weight(conn.reshape_weight_update(pot_term, self.spatial_reduction) - conn.reshape_weight_update(dep_term, self.spatial_reduction))

                # clean temporary variables
                del A_post
                del A_pre
                del I_post
                del I_pre
                del pot_term
                del dep_term


class OnlineSTDPAdditiveUpdaterV4(AbstractLayerwiseUpdater):
    """Updates layers using an online variant of spiking time-dependent plasticity.

    More details can be found at http://www.scholarpedia.org/article/Spike-timing_dependent_plasticity#Online_implementation_of_STDP_models

    Args:
        trainables (tuple[AbstractLayer, ...] | AbstractLayer): layers to which STDP will be applied.
        step_time (float): length of the time steps used by the model, in :math:`ms`.
        lr_post (float): learning rate for the postsynaptic update component, multiplicative term for added component.
        lr_pre (float): learning rate for the presynaptic update component, multiplicative term for subtracted component.
        tc_post (float): time constant for the postsynaptic update component.
        tc_pre (float): time constant for the presynaptic update component.
        amp_post (float): amplitude for postsynaptic spike trace. Defaults to `1.0`.
        amp_pre (float): amplitude for presynaptic spike trace. Defaults to `1.0`.
        weight_min (float | None, optional): if not `None`, sets a lower bound to the learned weights. Defaults to None.
        weight_max (float | None, optional): if not `None`, sets an upper bound to the learned weights. Defaults to None.
        weight_bounding_mode (str | None, optional): Mode of operation for weight limiting. Defaults to None.
        batch_reduction (Callable[[torch.Tensor], torch.Tensor], optional): Function to reduce batch dimension of intermediate computations. Defaults to torch.mean.
        spatial_reduction (Callable[[torch.Tensor], torch.Tensor], optional): Function to reduce spatial dimension of intermediate computations. Defaults to torch.mean.
    """
    def __init__(
        self,
        trainables: tuple[AbstractLayer, ...] | AbstractLayer,
        step_time: float,
        lr_post: float,   # LTP
        lr_pre: float,    # LTD
        tc_post: float,
        tc_pre: float,
        amp_post: float = 1.0,
        amp_pre: float = 1.0,
        weight_min: float | None = None,
        weight_max: float | None = None,
        weight_bounding_mode: str | None = None,
        batch_reduction: Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] = torch.mean,
        spatial_reduction: Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] = torch.mean
    ):
        # call superclass constructor
        AbstractLayerwiseUpdater.__init__(self, trainables)

        # register training hyperparameters
        self.register_buffer('lr_post', torch.tensor(lr_post))
        self.register_buffer('lr_pre', torch.tensor(lr_pre))
        self.register_buffer('tc_post', torch.tensor(tc_post))
        self.register_buffer('tc_pre', torch.tensor(tc_pre))
        self.register_buffer('amp_post', torch.tensor(1.0))
        self.register_buffer('amp_pre', torch.tensor(1.0))

        self.register_buffer('pApost', torch.empty(0))
        self.register_buffer('pApre', torch.empty(0))

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
            self.monitors[hex(id(sm))]['in'] = InputMonitor(reducer=AdditiveTraceReducer(decay=torch.exp(-self.step_time / self.tc_pre), amplitude=self.amp_pre), train_update=True, eval_update=False, module=sm)
            self.monitors[hex(id(sm))]['out'] = OutputMonitor(reducer=AdditiveTraceReducer(decay=torch.exp(-self.step_time / self.tc_post), amplitude=self.amp_post), train_update=True, eval_update=False, module=sm)

    def clear(self):
        self.pApost = torch.empty(0)
        self.pApre = torch.empty(0)
        AbstractLayerwiseUpdater.clear(self)

    def forward(self) -> None:
        """Performs an update step based on previously processed inputs.
        """
        for sm in self.submodules:
            for conn in sm.connections:
                # build presynaptic and postsynaptic spike traces and spikes
                A_post = conn.reshape_outputs(self.monitors[hex(id(sm))]['out'].peak())
                A_pre = conn.reshape_inputs(self.monitors[hex(id(sm))]['in'].peak())

                if self.pApost.numel() == 0:
                    self.pApost = torch.zeros_like(A_post)
                if self.pApre.numel() == 0:
                    self.pApre = torch.zeros_like(A_pre)

                I_post = (A_post > self.pApost).to(dtype=A_post.dtype)
                I_pre = (A_pre > self.pApre).to(dtype=A_pre.dtype)

                self.pApost = A_post.clone().detach()
                self.pApre = A_pre.clone().detach()

                # compute update
                pot_term = self.lr_post * self.batch_reduction(torch.bmm(I_post, A_pre), dim=0, keepdim=False)
                dep_term = self.lr_pre * self.batch_reduction(torch.bmm(A_post, I_pre), dim=0, keepdim=False)

                match self.weight_bounding_mode:
                    case 'clamp':
                        conn.update_weight(conn.reshape_weight_update(pot_term, self.spatial_reduction) - conn.reshape_weight_update(dep_term, self.spatial_reduction), self.weight_min, self.weight_max)
                    case _:
                        conn.update_weight(conn.reshape_weight_update(pot_term, self.spatial_reduction) - conn.reshape_weight_update(dep_term, self.spatial_reduction))

                # clean temporary variables
                del A_post
                del A_pre
                del I_post
                del I_pre
                del pot_term
                del dep_term
