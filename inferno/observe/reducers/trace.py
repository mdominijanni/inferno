from .base import FoldingReducer
import inferno
from inferno._internal import numeric_limit
from ... import (
    trace_nearest,
    trace_cumulative,
    trace_nearest_scaled,
    trace_cumulative_scaled,
)
from inferno.infernotypes import OneToOne
import math
import torch


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
        step_time (float): length of the discrete step time, :math:`\Delta t`.
        time_constant (float): time constant of exponential decay, :math:`\tau`.
        amplitude (int | float | complex): value to set trace to for matching elements,
            :math:`a`.
        target (int | float | bool | complex): target value test for when determining
            if an input is a match, :math:`f^*`.
        tolerance (int | float | None, optional): allowable absolute difference to
            still count as a match, :math:`\epsilon`. Defaults to None.
        history_len (float, optional): length of time over which results should be
            stored, in the same units as :math:`\Delta t`. Defaults to 0.0.
    """

    def __init__(
        self,
        step_time: float,
        time_constant: float,
        amplitude: int | float | complex,
        target: int | float | bool | complex,
        tolerance: int | float | None = None,
        *,
        history_len: float = 0.0,
    ):
        # call superclass constructor
        FoldingReducer.__init__(self, step_time, history_len)

        # reducer attributes
        self.time_constant, e = numeric_limit(
            "time_constant", time_constant, 0, "gt", float
        )
        if e:
            raise e
        self.decay = math.exp(-self.dt / self.time_constant)
        self.amplitude, e = numeric_limit("amplitude", amplitude, 0, "neq", None)
        if e:
            raise e
        self.target = target
        self.tolerance, e = (
            None,
            (
                None
                if tolerance is None
                else numeric_limit("tolerance", tolerance, 0, "gt", float)
            ),
        )
        if e:
            raise e

    @property
    def dt(self) -> float:
        r"""Length of the simulation time step, in milliseconds.

        Args:
            value (float): new simulation time step length.

        Returns:
            float: length of the simulation time step.
        """
        return FoldingReducer.dt.fget(self)

    @dt.setter
    def dt(self, value: float):
        FoldingReducer.dt.fset(self, value)
        self.decay = torch.tensor(inferno.exp(-self.dt / self.time_constant))

    def fold(self, obs: torch.Tensor, state: torch.Tensor | None) -> torch.Tensor:
        r"""Application of nearest trace.

        Args:
            obs (torch.Tensor): observation to incorporate into state.
            state (torch.Tensor | None): state from the prior time step,
                None if no prior observations.

        Returns:
            torch.Tensor: state for the current time step.
        """
        return trace_nearest(
            obs,
            state,
            decay=self.decay,
            amplitude=self.amplitude,
            target=self.target,
            tolerance=self.tolerance,
        )

    def initialize(self, inputs: torch.Tensor) -> torch.Tensor:
        r"""Setting of entire state history to zero.

        Args:
            inputs (torch.Tensor): empty tensor of state.

        Returns:
            torch.Tensor: filled state tensor.
        """
        return inputs.fill_(0)

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
            step_data (float): length of time between the prior and
                subsequent observations.

        Returns:
            torch.Tensor: interpolated data at sample time.
        """
        return inferno.interp_exp_decay(
            prev_data, next_data, sample_at, step_time, self.time_constant
        )


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
        step_time (float): length of the discrete step time, :math:`\Delta t`.
        time_constant (float): time constant of exponential decay, :math:`\tau`.
        amplitude (int | float | complex): value to add to trace for matching elements,
            :math:`a`.
        target (int | float | bool | complex): target value test for when determining
            if an input is a match, :math:`f^*`.
        tolerance (int | float | None, optional): allowable absolute difference to
            still count as a match, :math:`\epsilon`. Defaults to None.
        history_len (float, optional): length of time over which results should be
            stored, in the same units as :math:`\Delta t`. Defaults to 0.0.
    """

    def __init__(
        self,
        step_time: float,
        time_constant: float,
        amplitude: int | float | complex,
        target: int | float | bool | complex,
        tolerance: int | float | None = None,
        *,
        history_len: float = 0.0,
    ):
        # call superclass constructor
        FoldingReducer.__init__(self, step_time, history_len)

        # reducer attributes
        self.time_constant, e = numeric_limit(
            "time_constant", time_constant, 0, "gt", float
        )
        if e:
            raise e
        self.decay = math.exp(-self.dt / self.time_constant)
        self.amplitude, e = numeric_limit("amplitude", amplitude, 0, "neq", None)
        if e:
            raise e
        self.target = target
        self.tolerance, e = (
            None,
            (
                None
                if tolerance is None
                else numeric_limit("tolerance", tolerance, 0, "gt", float)
            ),
        )
        if e:
            raise e

    @property
    def dt(self) -> float:
        r"""Length of the simulation time step, in milliseconds.

        Args:
            value (float): new simulation time step length.

        Returns:
            float: length of the simulation time step.
        """
        return FoldingReducer.dt.fget(self)

    @dt.setter
    def dt(self, value: float):
        FoldingReducer.dt.fset(self, value)
        self.decay = torch.tensor(inferno.exp(-self.dt / self.time_constant))

    def fold(self, obs: torch.Tensor, state: torch.Tensor | None) -> torch.Tensor:
        r"""Application of cumulative trace.

        Args:
            obs (torch.Tensor): observation to incorporate into state.
            state (torch.Tensor | None): state from the prior time step,
                None if no prior observations.

        Returns:
            torch.Tensor: state for the current time step.
        """
        return trace_cumulative(
            obs,
            state,
            decay=self.decay,
            amplitude=self.amplitude,
            target=self.target,
            tolerance=self.tolerance,
        )

    def initialize(self, inputs: torch.Tensor) -> torch.Tensor:
        r"""Setting of entire state history to zero.

        Args:
            inputs (torch.Tensor): empty tensor of state.

        Returns:
            torch.Tensor: filled state tensor.
        """
        return inputs.fill_(0)

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
            step_data (float): length of time between the prior and
                subsequent observations.

        Returns:
            torch.Tensor: interpolated data at sample time.
        """
        return inferno.interp_exp_decay(
            prev_data, next_data, sample_at, step_time, self.time_constant
        )


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
        step_time (float): length of the discrete step time, :math:`\Delta t`.
        time_constant (float): time constant of exponential decay, :math:`\tau`.
        amplitude (int | float | complex): value to set trace to for matching elements,
            :math:`a`.
        scale (int | float | complex): multiplicative scale for contributions to trace,
            :math:`S`.
        criterion (OneToOne[torch.Tensor]): function to test if the input is considered
            a match for the purpose of tracing, :math:`K`.
        history_len (float, optional): length of time over which results should be
            stored, in the same units as :math:`\Delta t`. Defaults to 0.0.

    Note:
        The output of ``criterion`` must have a datatype (:py:class:`torch.dtype`) of
        :py:data:`torch.bool`.
    """

    def __init__(
        self,
        step_time: float,
        time_constant: float,
        amplitude: int | float | complex,
        scale: int | float | complex,
        criterion: OneToOne[torch.Tensor],
        *,
        history_len: float = 0.0,
    ):
        # call superclass constructor
        FoldingReducer.__init__(self, step_time, history_len)

        # reducer attributes
        self.time_constant, e = numeric_limit(
            "time_constant", time_constant, 0, "gt", float
        )
        if e:
            raise e
        self.decay = math.exp(-self.dt / self.time_constant)
        self.amplitude, e = numeric_limit("amplitude", amplitude, 0, "neq", None)
        if e:
            raise e
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
        return FoldingReducer.dt.fget(self)

    @dt.setter
    def dt(self, value: float):
        FoldingReducer.dt.fset(self, value)
        self.decay = torch.tensor(inferno.exp(-self.dt / self.time_constant))

    def fold(self, obs: torch.Tensor, state: torch.Tensor | None) -> torch.Tensor:
        r"""Application of scaled nearest trace.

        Args:
            obs (torch.Tensor): observation to incorporate into state.
            state (torch.Tensor | None): state from the prior time step,
                None if no prior observations.

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

    def initialize(self, inputs: torch.Tensor) -> torch.Tensor:
        r"""Setting of entire state history to zero.

        Args:
            inputs (torch.Tensor): empty tensor of state.

        Returns:
            torch.Tensor: filled state tensor.
        """
        return inputs.fill_(0)

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
            step_data (float): length of time between the prior and
                subsequent observations.

        Returns:
            torch.Tensor: interpolated data at sample time.
        """
        return inferno.interp_exp_decay(
            prev_data, next_data, sample_at, step_time, self.time_constant
        )


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
        step_time (float): length of the discrete step time, :math:`\Delta t`.
        time_constant (float): time constant of exponential decay, :math:`\tau`.
        amplitude (int | float | complex): value to add to trace for matching elements,
            :math:`a`.
        scale (int | float | complex): multiplicative scale for contributions to trace,
            :math:`S`.
        criterion (OneToOne[torch.Tensor]): function to test if the input is considered
            a match for the purpose of tracing, :math:`K`.
        history_len (float, optional): length of time over which results should be
            stored, in the same units as :math:`\Delta t`. Defaults to 0.0.

    Note:
        The output of ``criterion`` must have a datatype (:py:class:`torch.dtype`) of
        :py:data:`torch.bool`.
    """

    def __init__(
        self,
        step_time: float,
        time_constant: float,
        amplitude: int | float | complex,
        scale: int | float | complex,
        criterion: OneToOne[torch.Tensor],
        *,
        history_len: float = 0.0,
    ):
        # call superclass constructor
        FoldingReducer.__init__(self, step_time, history_len)

        # register state
        self.time_constant, e = numeric_limit(
            "time_constant", time_constant, 0, "gt", float
        )
        if e:
            raise e
        self.decay = math.exp(-self.dt / self.time_constant)
        self.amplitude, e = numeric_limit("amplitude", amplitude, 0, "neq", None)
        if e:
            raise e
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
        return FoldingReducer.dt.fget(self)

    @dt.setter
    def dt(self, value: float):
        FoldingReducer.dt.fset(self, value)
        self.decay = torch.tensor(inferno.exp(-self.dt / self.time_constant))

    def fold(self, obs: torch.Tensor, state: torch.Tensor | None) -> torch.Tensor:
        r"""Application of scaled cumulative trace.

        Args:
            obs (torch.Tensor): observation to incorporate into state.
            state (torch.Tensor | None): state from the prior time step,
                None if no prior observations.

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

    def initialize(self, inputs: torch.Tensor) -> torch.Tensor:
        r"""Setting of entire state history to zero.

        Args:
            inputs (torch.Tensor): empty tensor of state.

        Returns:
            torch.Tensor: filled state tensor.
        """
        return inputs.fill_(0)

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
            step_data (float): length of time between the prior and
                subsequent observations.

        Returns:
            torch.Tensor: interpolated data at sample time.
        """
        return inferno.interp_exp_decay(
            prev_data, next_data, sample_at, step_time, self.time_constant
        )
