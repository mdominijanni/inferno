from __future__ import annotations
from typing import Callable
import torch
from .base import FoldingReducer
import inferno
from inferno._internal import newtensor
from inferno.typing import OneToOneMethod, ManyToOneMethod, OneToOne


class NearestTraceReducer(FoldingReducer):
    r"""Stores the trace over time, considering the latest match.

    .. math::
        x_{t + \Delta t} =
        \begin{cases}
            a &\lvert f_{t + \Delta t} - f^* \rvert \leq \epsilon \\
            x_t \exp (\Delta t / \tau) &\text{otherwise}
        \end{cases}

    For the trace (state) :math:`x` and observation :math:`f`.

    Args:
        step_time (float | torch.Tensor): length of the discrete step time, :math:`\Delta t`.
        time_constant (float | torch.Tensor): time constant of exponential decay, :math:`\tau`.
        amplitude (int | float | complex | torch.Tensor): value to set trace to for matching elements, :math:`a`.
        target (int | float | bool | complex | torch.Tensor): target value to set trace to, :math:`f^*`.
        tolerance (int | float | torch.Tensor | None, optional): allowable absolute difference to
            still count as a match, :math:`\epsilon`. Defaults to None.
        mapmeth (OneToOneMethod[NearestTraceReducer, torch.Tensor] | ManyToOneMethod[NearestTraceReducer, torch.Tensor] | None, optional): transformation
            to apply to inputs, where the first argument is the reducer itself, no transformation if None. Defaults to None.
        filtermeth (Callable[[NearestTraceReducer, torch.Tensor], bool] | None, optional): conditional test if transformed
            input should be integrated, where the first argument is the reducer itself, always will if None. Defaults to None.

    Note:
        By default, only a single input tensor can be provided to :py:meth:`forward` for this class. A custom
        ``mapmeth`` must be passed in for multiple input support.
    """

    def __init__(
        self,
        step_time: float | torch.Tensor,
        time_constant: float | torch.Tensor,
        amplitude: int | float | complex | torch.Tensor,
        target: int | float | bool | complex | torch.Tensor,
        tolerance: int | float | torch.Tensor | None = None,
        *,
        mapmeth: OneToOneMethod[NearestTraceReducer, torch.Tensor]
        | ManyToOneMethod[NearestTraceReducer, torch.Tensor]
        | None = None,
        filtermeth: Callable[[NearestTraceReducer, torch.Tensor], bool] | None = None,
    ):
        # call superclass constructor, overriding instance methods if specified
        FoldingReducer.__init__(self, mapmeth=mapmeth, filtermeth=filtermeth)

        # register state
        decay = inferno.exp(-step_time / time_constant)
        self.register_buffer("_decay", newtensor(decay))
        self.register_buffer("_amplitude", newtensor(amplitude))
        self.register_buffer("_target", newtensor(target))

        # register tolerance, construct foldfn closure, and assign
        if tolerance is not None:
            self.register_buffer("_tolerance", newtensor(tolerance))

            def foldfn(observation, trace, decay, amplitude, target, tolerance):
                mask = torch.abs(observation - target) <= tolerance
                return torch.where(mask, amplitude, decay * trace)

            self._inner_foldfn = foldfn

        else:
            self.register_buffer("_tolerance", None)

            def foldfn(observation, trace, decay, amplitude, target, _):
                mask = observation == target
                return torch.where(mask, amplitude, decay * trace)

            self._inner_foldfn = foldfn

    def fold_fn(self, obs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self._inner_foldfn(
            obs, state, self._decay, self._amplitude, self._target, self._tolerance
        )

    def map_fn(self, *inputs: torch.Tensor) -> torch.Tensor:
        return inputs[0]

    def filter_fn(self, inputs: torch.Tensor) -> bool:
        return True

    def init_fn(self, inputs: torch.Tensor) -> torch.Tensor:
        return inferno.zeros(self._data, shape=inputs.shape)


class CumulativeTraceReducer(FoldingReducer):
    r"""Stores the trace over time, considering all prior matches.

    .. math::
        x_{t + \Delta t} =
        \begin{cases}
            a + x_t \exp (\Delta t / \tau) &\lvert f_{t + \Delta t} - f^* \rvert \leq \epsilon \\
            x_t \exp (\Delta t / \tau) &\text{otherwise}
        \end{cases}

    For the trace (state) :math:`x` and observation :math:`f`.

    Args:
        step_time (float | torch.Tensor): length of the discrete step time, :math:`\Delta t`.
        time_constant (float | torch.Tensor): time constant of exponential decay, :math:`\tau`.
        amplitude (int | float | complex | torch.Tensor): value to set trace to for matching elements, :math:`a`.
        target (int | float | bool | complex | torch.Tensor): target value to set trace to, :math:`f^*`.
        tolerance (int | float | torch.Tensor | None, optional): allowable absolute difference to
            still count as a match, :math:`\epsilon`. Defaults to None.
        mapmeth (OneToOneMethod[CumulativeTraceReducer, torch.Tensor] | ManyToOneMethod[CumulativeTraceReducer, torch.Tensor] | None, optional): transformation
            to apply to inputs, where the first argument is the reducer itself, no transformation if None. Defaults to None.
        filtermeth (Callable[[CumulativeTraceReducer, torch.Tensor], bool] | None, optional): conditional test if transformed
            input should be integrated, where the first argument is the reducer itself, always will if None. Defaults to None.

    Note:
        By default, only a single input tensor can be provided to :py:meth:`forward` for this class. A custom
        ``mapmeth`` must be passed in for multiple input support.
    """

    def __init__(
        self,
        step_time: float | torch.Tensor,
        time_constant: float | torch.Tensor,
        amplitude: int | float | complex | torch.Tensor,
        target: int | float | bool | complex | torch.Tensor,
        tolerance: int | float | torch.Tensor | None = None,
        *,
        mapmeth: OneToOneMethod[CumulativeTraceReducer, torch.Tensor]
        | ManyToOneMethod[CumulativeTraceReducer, torch.Tensor]
        | None = None,
        filtermeth: Callable[[CumulativeTraceReducer, torch.Tensor], bool]
        | None = None,
    ):
        # call superclass constructor, overriding instance methods if specified
        FoldingReducer.__init__(self, mapmeth=mapmeth, filtermeth=filtermeth)

        # register state
        decay = inferno.exp(-step_time / time_constant)
        self.register_buffer("_decay", newtensor(decay))
        self.register_buffer("_amplitude", newtensor(amplitude))
        self.register_buffer("_target", newtensor(target))

        # register tolerance, construct foldfn closure, and assign
        if tolerance is not None:
            self.register_buffer("_tolerance", newtensor(tolerance))

            def foldfn(observation, trace, decay, amplitude, target, tolerance):
                mask = torch.abs(observation - target) <= tolerance
                return (decay * trace) + (amplitude * mask)

            self._inner_foldfn = foldfn

        else:
            self.register_buffer("_tolerance", None)

            def foldfn(observation, trace, decay, amplitude, target, _):
                mask = observation == target
                return (decay * trace) + (amplitude * mask)

            self._inner_foldfn = foldfn

    def fold_fn(self, obs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self._inner_foldfn(
            obs, state, self._decay, self._amplitude, self._target, self._tolerance
        )

    def map_fn(self, *inputs: torch.Tensor) -> torch.Tensor:
        return inputs[0]

    def filter_fn(self, inputs: torch.Tensor) -> bool:
        return True

    def init_fn(self, inputs: torch.Tensor) -> torch.Tensor:
        return inferno.zeros(self._data, shape=inputs.shape)


class ScaledNearestTraceReducer(FoldingReducer):
    r"""Stores the trace over time, scaled by the input, considering the latest match.

    .. math::
        x_{t + \Delta t} =
        \begin{cases}
            a + Sf &K(f_{t + \Delta t}) \\
            x_t \exp (\Delta t / \tau) &\text{otherwise}
        \end{cases}

    For the trace (state) :math:`x` and observation :math:`f`.

    Args:
        step_time (float | torch.Tensor): length of the discrete step time, :math:`\Delta t`.
        time_constant (float | torch.Tensor): time constant of exponential decay, :math:`\tau`.
        amplitude (int | float | complex | torch.Tensor): value to set trace to for matching elements, :math:`a`.
        scale (int | float | complex | torch.Tensor): multiplicitive scale for contributions to trace, :math:`S`.
        criterion (OneToOne[torch.Tensor]): function to test if the input is considered a match for the purpose of tracing, :math:`K`.
        mapmeth (OneToOneMethod[ScaledNearestTraceReducer, torch.Tensor] | ManyToOneMethod[ScaledNearestTraceReducer, torch.Tensor] | None, optional): transformation
            to apply to inputs, where the first argument is the reducer itself, no transformation if None. Defaults to None.
        filtermeth (Callable[[ScaledNearestTraceReducer, torch.Tensor], bool] | None, optional): conditional test if transformed
            input should be integrated, always will if None. Defaults to None.

    Note:
        The output of ``criterion`` must have a datatype (:py:class:`torch.dtype`) of :py:data:`torch.bool`.

    Note:
        By default, only a single input tensor can be provided to :py:meth:`forward` for this class. A custom
        ``mapmeth`` must be passed in for multiple input support.
    """

    def __init__(
        self,
        step_time: float | torch.Tensor,
        time_constant: float | torch.Tensor,
        amplitude: int | float | complex | torch.Tensor,
        scale: int | float | complex | torch.Tensor,
        criterion: OneToOne[torch.Tensor],
        *,
        mapmeth: OneToOneMethod[ScaledNearestTraceReducer, torch.Tensor]
        | ManyToOneMethod[ScaledNearestTraceReducer, torch.Tensor]
        | None = None,
        filtermeth: Callable[[ScaledNearestTraceReducer, torch.Tensor], bool]
        | None = None,
    ):
        # call superclass constructor, overriding instance methods if specified
        FoldingReducer.__init__(self, mapmeth=mapmeth, filtermeth=filtermeth)

        # register state
        decay = inferno.exp(-step_time / time_constant)
        self.register_buffer("_decay", newtensor(decay))
        self.register_buffer("_amplitude", newtensor(amplitude))
        self.register_buffer("_scale", newtensor(scale))

        # set non-persistent function
        self._criterion = criterion

    def fold_fn(self, obs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        mask = self._criterion(obs)
        return torch.where(
            mask, self._amplitude + self._scale * obs, self._decay * state
        )

    def map_fn(self, *inputs: torch.Tensor) -> torch.Tensor:
        return inputs[0]

    def filter_fn(self, inputs: torch.Tensor) -> bool:
        return True

    def init_fn(self, inputs: torch.Tensor) -> torch.Tensor:
        return inferno.zeros(self._data, shape=inputs.shape)


class ScaledCumulativeTraceReducer(FoldingReducer):
    r"""Stores the trace over time, scaled by the input, considering all prior matches.

    .. math::
        x_{t + \Delta t} =
        \begin{cases}
            a + Sf + x_t \exp (\Delta t / \tau) &K(f_{t + \Delta t}) \\
            x_t \exp (\Delta t / \tau) &\text{otherwise}
        \end{cases}

    For the trace (state) :math:`x` and observation :math:`f`.

    Args:
        step_time (float | torch.Tensor): length of the discrete step time, :math:`\Delta t`.
        time_constant (float | torch.Tensor): time constant of exponential decay, :math:`\tau`.
        amplitude (int | float | complex | torch.Tensor): value to set trace to for matching elements, :math:`a`.
        scale (int | float | complex | torch.Tensor): multiplicitive scale for contributions to trace, :math:`S`.
        criterion (OneToOne[torch.Tensor]): function to test if the input is considered a match for the purpose of tracing, :math:`K`.
        mapmeth (OneToOneMethod[ScaledNearestTraceReducer, torch.Tensor] | ManyToOneMethod[ScaledNearestTraceReducer, torch.Tensor] | None, optional): transformation
            to apply to inputs, where the first argument is the reducer itself, no transformation if None. Defaults to None.
        filtermeth (Callable[[ScaledNearestTraceReducer, torch.Tensor], bool] | None, optional): conditional test if transformed
            input should be integrated, always will if None. Defaults to None.

    Note:
        The output of ``criterion`` must have a datatype (:py:class:`torch.dtype`) of :py:data:`torch.bool`.

    Note:
        By default, only a single input tensor can be provided to :py:meth:`forward` for this class. A custom
        ``mapmeth`` must be passed in for multiple input support.
    """

    def __init__(
        self,
        step_time: float | torch.Tensor,
        time_constant: float | torch.Tensor,
        amplitude: int | float | complex | torch.Tensor,
        scale: int | float | complex | torch.Tensor,
        criterion: OneToOne[torch.Tensor],
        *,
        mapmeth: OneToOneMethod[ScaledNearestTraceReducer, torch.Tensor]
        | ManyToOneMethod[ScaledNearestTraceReducer, torch.Tensor]
        | None = None,
        filtermeth: Callable[[ScaledNearestTraceReducer, torch.Tensor], bool]
        | None = None,
    ):
        # call superclass constructor, overriding instance methods if specified
        FoldingReducer.__init__(self, mapmeth=mapmeth, filtermeth=filtermeth)

        # register state
        decay = inferno.exp(-step_time / time_constant)
        self.register_buffer("_decay", newtensor(decay))
        self.register_buffer("_amplitude", newtensor(amplitude))
        self.register_buffer("_scale", newtensor(scale))

        # set non-persistent function
        self._criterion = criterion

    def fold_fn(self, obs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        mask = self._criterion(obs)
        return self._decay * state + torch.where(
            mask, self._amplitude + self._scale * self._observation, 0
        )

    def map_fn(self, *inputs: torch.Tensor) -> torch.Tensor:
        return inputs[0]

    def filter_fn(self, inputs: torch.Tensor) -> bool:
        return True

    def init_fn(self, inputs: torch.Tensor) -> torch.Tensor:
        return inferno.zeros(self._data, shape=inputs.shape)
