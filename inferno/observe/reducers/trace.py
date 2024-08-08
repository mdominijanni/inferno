from .base import FoldReducer
from ... import (
    exp,
    trace_nearest,
    trace_cumulative,
    trace_nearest_scaled,
    trace_cumulative_scaled,
)
from ..._internal import argtest
from ...functional import interp_expdecay
from ...types import OneToOne
from functools import partial
import math
import torch


class NearestTraceReducer(FoldReducer):
    r"""Stores the trace over time, considering the latest match.

    .. math::
        x(t) =
        \begin{cases}
            A & \lvert h(t) - h^* \rvert \leq \epsilon \\
            x(t - \Delta t) \exp \left(-\frac{\Delta t}{\tau_x}\right)
            & \lvert h(t) - h^* \rvert > \epsilon
        \end{cases}

    For the trace (state) :math:`x` and observation :math:`h`.

    Args:
        step_time (float): length of the discrete step time, :math:`\Delta t`.
        time_constant (float): time constant of exponential decay, :math:`\tau_x`.
        amplitude (int | float | complex): value to set trace to for matching elements,
            :math:`A`.
        target (int | float | bool | complex): target value test for when determining
            if an input is a match, :math:`h^*`.
        tolerance (int | float | None, optional): allowable absolute difference to
            still count as a match, :math:`\epsilon`. Defaults to ``None``.
        duration (float, optional): length of time over which results should be
            stored, in the same units as :math:`\Delta t`. Defaults to ``0.0``.
        inclusive (bool, optional): if the duration should be inclusive.
            Defaults to ``False``.
        inplace (bool, optional): if write operations should be performed
            in-place. Defaults to ``False``.

    Important:
        Because the input tensor to :py:meth:`fold` is treated as an event condition,
        it will have its datatype cast to that of the reducer itself.
    """

    def __init__(
        self,
        step_time: float,
        time_constant: float,
        amplitude: int | float | complex,
        target: int | float | bool | complex,
        tolerance: int | float | None = None,
        *,
        duration: float = 0.0,
        inclusive: bool = False,
        inplace: bool = False,
    ):
        # call superclass constructor
        FoldReducer.__init__(self, step_time, duration, inclusive, inplace, 0)

        # reducer attributes
        self.time_constant = argtest.gt("time_constant", time_constant, 0, float)
        self.decay = math.exp(-self.dt / self.time_constant)
        self.amplitude = argtest.neq("amplitude", amplitude, 0, None)
        self.target = target
        self.tolerance = (
            None if tolerance is None else argtest.gt("tolerance", tolerance, 0, float)
        )

    @property
    def dt(self) -> float:
        r"""Length of the simulation time step, in milliseconds.

        Args:
            value (float): new simulation time step length.

        Returns:
            float: length of the simulation time step.
        """
        return FoldReducer.dt.fget(self)

    @dt.setter
    def dt(self, value: float) -> None:
        FoldReducer.dt.fset(self, value)
        self.decay = exp(-self.dt / self.time_constant)

    def fold(self, obs: torch.Tensor, state: torch.Tensor | None) -> torch.Tensor:
        r"""Application of nearest trace.

        Args:
            obs (torch.Tensor): observation to incorporate into state.
            state (torch.Tensor | None): state from the prior time step,
                ``None`` if no prior observations.

        Returns:
            torch.Tensor: state for the current time step.
        """
        return trace_nearest(
            obs.to(dtype=self.data.dtype),
            state,
            decay=self.decay,
            amplitude=self.amplitude,
            target=self.target,
            tolerance=self.tolerance,
        )

    def interpolate(
        self,
        prev_data: torch.Tensor,
        next_data: torch.Tensor,
        sample_at: torch.Tensor,
        step_time: float,
    ) -> torch.Tensor:
        r"""Exponential decay interpolation between observations.

        Args:
            prev_data (torch.Tensor): most recent observation prior to sample time.
            next_data (torch.Tensor): most recent observation subsequent to sample time.
            sample_at (torch.Tensor): relative time at which to sample data.
            step_time (float): length of time between the prior and
                subsequent observations.

        Returns:
            torch.Tensor: interpolated data at sample time.
        """
        return interp_expdecay(
            prev_data, next_data, sample_at, step_time, time_constant=self.time_constant
        )


class CumulativeTraceReducer(FoldReducer):
    r"""Stores the trace over time, considering all prior matches.

    .. math::
        x(t) = x(t - \Delta t) \exp \left(-\frac{\Delta t}{\tau_x}\right)
        + A \left[\lvert h(t) - h^* \rvert \leq \epsilon\right]

    For the trace (state) :math:`x` and observation :math:`h`.

    Args:
        step_time (float): length of the discrete step time, :math:`\Delta t`.
        time_constant (float): time constant of exponential decay, :math:`\tau_x`.
        amplitude (int | float | complex): value to add to trace for matching elements,
            :math:`A`.
        target (int | float | bool | complex): target value test for when determining
            if an input is a match, :math:`h^*`.
        tolerance (int | float | None, optional): allowable absolute difference to
            still count as a match, :math:`\epsilon`. Defaults to ``None``.
        duration (float, optional): length of time over which results should be
            stored, in the same units as :math:`\Delta t`. Defaults to ` 0.0``.
        inclusive (bool, optional): if the duration should be inclusive.
            Defaults to ``False``.
        inplace (bool, optional): if write operations should be performed
            in-place. Defaults to ``False``.

    Important:
        Because the input tensor to :py:meth:`fold` is treated as an event condition,
        it will have its datatype cast to that of the reducer itself.
    """

    def __init__(
        self,
        step_time: float,
        time_constant: float,
        amplitude: int | float | complex,
        target: int | float | bool | complex,
        tolerance: int | float | None = None,
        *,
        duration: float = 0.0,
        inclusive: bool = False,
        inplace: bool = False,
    ):
        # call superclass constructor
        FoldReducer.__init__(self, step_time, duration, inclusive, inplace, 0)

        # reducer attributes
        self.time_constant = argtest.gt("time_constant", time_constant, 0, float)
        self.decay = math.exp(-self.dt / self.time_constant)
        self.amplitude = argtest.neq("amplitude", amplitude, 0, None)
        self.target = target
        self.tolerance = (
            None if tolerance is None else argtest.gt("tolerance", tolerance, 0, float)
        )

    @property
    def dt(self) -> float:
        r"""Length of the simulation time step, in milliseconds.

        Args:
            value (float): new simulation time step length.

        Returns:
            float: length of the simulation time step.
        """
        return FoldReducer.dt.fget(self)

    @dt.setter
    def dt(self, value: float) -> None:
        FoldReducer.dt.fset(self, value)
        self.decay = exp(-self.dt / self.time_constant)

    def fold(self, obs: torch.Tensor, state: torch.Tensor | None) -> torch.Tensor:
        r"""Application of cumulative trace.

        Args:
            obs (torch.Tensor): observation to incorporate into state.
            state (torch.Tensor | None): state from the prior time step,
                ``None`` if no prior observations.

        Returns:
            torch.Tensor: state for the current time step.
        """
        return trace_cumulative(
            obs.to(dtype=self.data.dtype),
            state,
            decay=self.decay,
            amplitude=self.amplitude,
            target=self.target,
            tolerance=self.tolerance,
        )

    def interpolate(
        self,
        prev_data: torch.Tensor,
        next_data: torch.Tensor,
        sample_at: torch.Tensor,
        step_time: float,
    ) -> torch.Tensor:
        r"""Exponential decay interpolation between observations.

        Args:
            prev_data (torch.Tensor): most recent observation prior to sample time.
            next_data (torch.Tensor): most recent observation subsequent to sample time.
            sample_at (torch.Tensor): relative time at which to sample data.
            step_time (float): length of time between the prior and
                subsequent observations.

        Returns:
            torch.Tensor: interpolated data at sample time.
        """
        return interp_expdecay(
            prev_data, next_data, sample_at, step_time, time_constant=self.time_constant
        )


class ScaledNearestTraceReducer(FoldReducer):
    r"""Stores the trace over time, scaled by the input, considering the latest match.

    .. math::
        x(t) =
        \begin{cases}
            sh + A & J(h) \\
            x(t - \Delta t) \exp \left(-\frac{\Delta t}{\tau_x}\right)
            & \neg J(h)
        \end{cases}

    For the trace (state) :math:`x` and observation :math:`h`.

    Args:
        step_time (float): length of the discrete step time, :math:`\Delta t`.
        time_constant (float): time constant of exponential decay, :math:`\tau_x`.
        amplitude (int | float | complex): value to set trace to for matching elements,
            :math:`A`.
        scale (int | float | complex): multiplicative scale for contributions to trace,
            :math:`s`.
        criterion (OneToOne[torch.Tensor]): function to test if the input is considered
            a match for the purpose of tracing, :math:`J`.
        duration (float, optional): length of time over which results should be
            stored, in the same units as :math:`\Delta t`. Defaults to ``0.0``.
        inclusive (bool, optional): if the duration should be inclusive.
            Defaults to ``False``.
        inplace (bool, optional): if write operations should be performed
            in-place. Defaults to ``False``.

    Note:
        The output of ``criterion`` must have a datatype (:py:class:`torch.dtype`) of
        ``torch.bool``.
    """

    def __init__(
        self,
        step_time: float,
        time_constant: float,
        amplitude: int | float | complex,
        scale: int | float | complex,
        criterion: OneToOne[torch.Tensor],
        *,
        duration: float = 0.0,
        inclusive: bool = False,
        inplace: bool = False,
    ):
        # call superclass constructor
        FoldReducer.__init__(self, step_time, duration, inclusive, inplace, 0)

        # reducer attributes
        self.time_constant = argtest.gt("time_constant", time_constant, 0, float)
        self.decay = math.exp(-self.dt / self.time_constant)
        self.amplitude = float(amplitude)
        self.scale = scale
        self.criterion = criterion

    @property
    def dt(self) -> float:
        r"""Length of the simulation time step, in milliseconds.

        Args:
            value (float): new simulation time step length.

        Returns:
            float: length of the simulation time step.
        """
        return FoldReducer.dt.fget(self)

    @dt.setter
    def dt(self, value: float) -> None:
        FoldReducer.dt.fset(self, value)
        self.decay = exp(-self.dt / self.time_constant)

    def fold(self, obs: torch.Tensor, state: torch.Tensor | None) -> torch.Tensor:
        r"""Application of scaled nearest trace.

        Args:
            obs (torch.Tensor): observation to incorporate into state.
            state (torch.Tensor | None): state from the prior time step,
                ``None`` if no prior observations.

        Returns:
            torch.Tensor: state for the current time step.
        """
        return trace_nearest_scaled(
            obs,
            state,
            decay=self.decay,
            amplitude=self.amplitude,
            scale=self.scale,
            matchfn=self.criterion,
        )

    def interpolate(
        self,
        prev_data: torch.Tensor,
        next_data: torch.Tensor,
        sample_at: torch.Tensor,
        step_time: float,
    ) -> torch.Tensor:
        r"""Exponential decay interpolation between observations.

        Args:
            prev_data (torch.Tensor): most recent observation prior to sample time.
            next_data (torch.Tensor): most recent observation subsequent to sample time.
            sample_at (torch.Tensor): relative time at which to sample data.
            step_time (float): length of time between the prior and
                subsequent observations.

        Returns:
            torch.Tensor: interpolated data at sample time.
        """
        return interp_expdecay(
            prev_data, next_data, sample_at, step_time, time_constant=self.time_constant
        )


class ScaledCumulativeTraceReducer(FoldReducer):
    r"""Stores the trace over time, scaled by the input, considering all prior matches.

    .. math::
        x(t) = x(t - \Delta t) \exp \left(-\frac{\Delta t}{\tau_x}\right)
        + (sh + A) \left[\lvert J(h) \right]

    For the trace (state) :math:`x` and observation :math:`h`.

    Args:
        step_time (float): length of the discrete step time, :math:`\Delta t`.
        time_constant (float): time constant of exponential decay, :math:`\tau_x`.
        amplitude (int | float | complex): value to add to trace for matching elements,
            :math:`A`.
        scale (int | float | complex): multiplicative scale for contributions to trace,
            :math:`s`.
        criterion (OneToOne[torch.Tensor]): function to test if the input is considered
            a match for the purpose of tracing, :math:`J`.
        duration (float, optional): length of time over which results should be
            stored, in the same units as :math:`\Delta t`. Defaults to ``0.0``.
        inclusive (bool, optional): if the duration should be inclusive.
            Defaults to ``False``.
        inplace (bool, optional): if write operations should be performed
            in-place. Defaults to ``False``.

    Note:
        The output of ``criterion`` must have a datatype (:py:class:`torch.dtype`) of
        ``torch.bool``.
    """

    def __init__(
        self,
        step_time: float,
        time_constant: float,
        amplitude: int | float | complex,
        scale: int | float | complex,
        criterion: OneToOne[torch.Tensor],
        *,
        duration: float = 0.0,
        inclusive: bool = False,
        inplace: bool = False,
    ):
        # call superclass constructor
        FoldReducer.__init__(self, step_time, duration, inclusive, inplace, 0)

        # register state
        self.time_constant = argtest.gt("time_constant", time_constant, 0, float)
        self.decay = math.exp(-self.dt / self.time_constant)
        self.amplitude = float(amplitude)
        self.scale = scale
        self.criterion = criterion

    @property
    def dt(self) -> float:
        r"""Length of the simulation time step, in milliseconds.

        Args:
            value (float): new simulation time step length.

        Returns:
            float: length of the simulation time step.
        """
        return FoldReducer.dt.fget(self)

    @dt.setter
    def dt(self, value: float) -> None:
        FoldReducer.dt.fset(self, value)
        self.decay = exp(-self.dt / self.time_constant)

    def fold(self, obs: torch.Tensor, state: torch.Tensor | None) -> torch.Tensor:
        r"""Application of scaled cumulative trace.

        Args:
            obs (torch.Tensor): observation to incorporate into state.
            state (torch.Tensor | None): state from the prior time step,
                ``None`` if no prior observations.

        Returns:
            torch.Tensor: state for the current time step.
        """
        return trace_cumulative_scaled(
            obs,
            state,
            decay=self.decay,
            amplitude=self.amplitude,
            scale=self.scale,
            matchfn=self.criterion,
        )

    def interpolate(
        self,
        prev_data: torch.Tensor,
        next_data: torch.Tensor,
        sample_at: torch.Tensor,
        step_time: float,
    ) -> torch.Tensor:
        r"""Exponential decay interpolation between observations.

        Args:
            prev_data (torch.Tensor): most recent observation prior to sample time.
            next_data (torch.Tensor): most recent observation subsequent to sample time.
            sample_at (torch.Tensor): relative time at which to sample data.
            step_time (float): length of time between the prior and
                subsequent observations.

        Returns:
            torch.Tensor: interpolated data at sample time.
        """
        return interp_expdecay(
            prev_data, next_data, sample_at, step_time, time_constant=self.time_constant
        )


class ConditionalNearestTraceReducer(FoldReducer):
    r"""Stores the trace of over time, scaled by the input, considering the latest condition.

    .. math::
        x(t) =
        \begin{cases}
            sh + A & j^* \\
            x(t - \Delta t) \exp \left(-\frac{\Delta t}{\tau_x}\right)
            & \neg j^*
        \end{cases}

    For the trace (state) :math:`x`, observation :math:`h`, and criterion :math:`j^*`.

    Args:
        step_time (float): length of the discrete step time, :math:`\Delta t`.
        time_constant (float): time constant of exponential decay, :math:`\tau_x`.
        amplitude (int | float | complex): value to set trace to for matching elements,
            :math:`A`.
        scale (int | float | complex): multiplicative scale for contributions to trace,
            :math:`s`.
        duration (float, optional): length of time over which results should be
            stored, in the same units as :math:`\Delta t`. Defaults to ``0.0``.
        inclusive (bool, optional): if the duration should be inclusive. Defaults to ``False``.
        inclusive (bool, optional): if the duration should be inclusive.
            Defaults to ``False``.
        inplace (bool, optional): if write operations should be performed
            in-place. Defaults to ``False``.

    Note:
        This is equivalent to :py:class:`ScaledNearestTraceReducer` except rather than
        use a criterion based on the observation, the second argument of :py:meth:`fold`
        is a condition tensor.
    """

    def __init__(
        self,
        step_time: float,
        time_constant: float,
        amplitude: int | float | complex,
        scale: int | float | complex,
        *,
        duration: float = 0.0,
        inclusive: bool = False,
        inplace: bool = False,
    ):
        # call superclass constructor
        FoldReducer.__init__(self, step_time, duration, inclusive, inplace, 0)

        # reducer attributes
        self.time_constant = argtest.gt("time_constant", time_constant, 0, float)
        self.decay = math.exp(-self.dt / self.time_constant)
        self.amplitude = float(amplitude)
        self.scale = scale

    @property
    def dt(self) -> float:
        r"""Length of the simulation time step, in milliseconds.

        Args:
            value (float): new simulation time step length.

        Returns:
            float: length of the simulation time step.
        """
        return FoldReducer.dt.fget(self)

    @dt.setter
    def dt(self, value: float) -> None:
        FoldReducer.dt.fset(self, value)
        self.decay = exp(-self.dt / self.time_constant)

    def fold(
        self, obs: torch.Tensor, cond: torch.Tensor, state: torch.Tensor | None
    ) -> torch.Tensor:
        r"""Application of scaled nearest trace.

        Args:
            obs (torch.Tensor): observation to incorporate into state.
            cond (torch.Tensor): condition if observations match for the trace.
            state (torch.Tensor | None): state from the prior time step,
                ``None`` if no prior observations.

        Returns:
            torch.Tensor: state for the current time step.
        """
        return trace_nearest_scaled(
            obs,
            state,
            decay=self.decay,
            amplitude=self.amplitude,
            scale=self.scale,
            matchfn=partial(lambda o, c: c, c=cond),
        )

    def interpolate(
        self,
        prev_data: torch.Tensor,
        next_data: torch.Tensor,
        sample_at: torch.Tensor,
        step_time: float,
    ) -> torch.Tensor:
        r"""Exponential decay interpolation between observations.

        Args:
            prev_data (torch.Tensor): most recent observation prior to sample time.
            next_data (torch.Tensor): most recent observation subsequent to sample time.
            sample_at (torch.Tensor): relative time at which to sample data.
            step_time (float): length of time between the prior and
                subsequent observations.

        Returns:
            torch.Tensor: interpolated data at sample time.
        """
        return interp_expdecay(
            prev_data, next_data, sample_at, step_time, time_constant=self.time_constant
        )


class ConditionalCumulativeTraceReducer(FoldReducer):
    r"""Stores the trace over time, scaled by the input, considering all prior conditions.

    .. math::
        x(t) = x(t - \Delta t) \exp \left(-\frac{\Delta t}{\tau_x}\right)
        + (sh + A) \left[\lvert j^* \right]

    For the trace (state) :math:`x`, observation :math:`h`, and criterion :math:`j^*`.

    Args:
        step_time (float): length of the discrete step time, :math:`\Delta t`.
        time_constant (float): time constant of exponential decay, :math:`\tau_x`.
        amplitude (int | float | complex): value to add to trace for matching elements,
            :math:`A`.
        scale (int | float | complex): multiplicative scale for contributions to trace,
            :math:`s`.
        duration (float, optional): length of time over which results should be
            stored, in the same units as :math:`\Delta t`. Defaults to ``0.0``.
        inclusive (bool, optional): if the duration should be inclusive.
            Defaults to ``False``.
        inplace (bool, optional): if write operations should be performed
            in-place. Defaults to ``False``.

    Note:
        This is equivalent to :py:class:`ScaledCumulativeTraceReducer` except rather than
        use a criterion based on the observation, the second argument of :py:meth:`fold`
        is a condition tensor.
    """

    def __init__(
        self,
        step_time: float,
        time_constant: float,
        amplitude: int | float | complex,
        scale: int | float | complex,
        *,
        duration: float = 0.0,
        inclusive: bool = False,
        inplace: bool = False,
    ):
        # call superclass constructor
        FoldReducer.__init__(self, step_time, duration, inclusive, inplace, 0)

        # register state
        self.time_constant = argtest.gt("time_constant", time_constant, 0, float)
        self.decay = math.exp(-self.dt / self.time_constant)
        self.amplitude = float(amplitude)
        self.scale = scale

    @property
    def dt(self) -> float:
        r"""Length of the simulation time step, in milliseconds.

        Args:
            value (float): new simulation time step length.

        Returns:
            float: length of the simulation time step.
        """
        return FoldReducer.dt.fget(self)

    @dt.setter
    def dt(self, value: float) -> None:
        FoldReducer.dt.fset(self, value)
        self.decay = exp(-self.dt / self.time_constant)

    def fold(
        self, obs: torch.Tensor, cond: torch.Tensor, state: torch.Tensor | None
    ) -> torch.Tensor:
        r"""Application of scaled cumulative trace.

        Args:
            obs (torch.Tensor): observation to incorporate into state.
            cond (torch.Tensor): condition if observations match for the trace.
            state (torch.Tensor | None): state from the prior time step,
                ``None`` if no prior observations.

        Returns:
            torch.Tensor: state for the current time step.
        """
        return trace_cumulative_scaled(
            obs,
            state,
            decay=self.decay,
            amplitude=self.amplitude,
            scale=self.scale,
            matchfn=partial(lambda o, c: c, c=cond),
        )

    def interpolate(
        self,
        prev_data: torch.Tensor,
        next_data: torch.Tensor,
        sample_at: torch.Tensor,
        step_time: float,
    ) -> torch.Tensor:
        r"""Exponential decay interpolation between observations.

        Args:
            prev_data (torch.Tensor): most recent observation prior to sample time.
            next_data (torch.Tensor): most recent observation subsequent to sample time.
            sample_at (torch.Tensor): relative time at which to sample data.
            step_time (float): length of time between the prior and
                subsequent observations.

        Returns:
            torch.Tensor: interpolated data at sample time.
        """
        return interp_expdecay(
            prev_data, next_data, sample_at, step_time, time_constant=self.time_constant
        )
