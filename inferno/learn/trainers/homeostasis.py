from ...observe import CAReducer

from collections.abc import Sequence
from .. import IndependentCellTrainer
from ... import Module
from ..._internal import argtest
from ...neural import Cell
from ...observe import (
    StateMonitor,
)
import torch
from typing import Any, Callable, Literal


class LinearHomeostasis(IndependentCellTrainer):
    r"""Linear homeostatic regulation.

    When ``param == "weight"``:

    .. math::
        w(t + \Delta t) - w(t) = \lambda \frac{r^* - r}{r^*}

    When ``param == "bias"``:

    .. math::
        b(t + \Delta t) - b(t) = \frac{\lambda}{L} \sum{\frac{r^* - r}{r^*}}

    When ``param == "delay"``:

    .. math::
        d(t + \Delta t) - d(t) = -\lambda \frac{r^* - r}{r^*}

    Where:

    Times :math:`t` is the current time, :math:`\Delta t` is the
    duration of the simulation step, and :math:`L` is the number of outputs
    corresponding to each bias term (for connections with a trainable bias).

    Args:
        plasticity (float): learning rate for updates, :math:`\lambda`.
        target (float | None, optional): target rate for postsynaptic
            spikes, :math:`r^*`. Required to be specified on update when ``None``.
        param (Literal["weight", "bias", "delay"]): specified parameter to update.
        batch_reduction (Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] | None, optional):
            function to reduce updates over the batch dimension, :py:func:`torch.mean`
            when ``None``. Defaults to ``None``.

    Caution:
        The sign of :math:`\lambda` is reversed in the case of training ``"delay"``
        parameters. Generally, the sign of ``plasticity`` should be positive for the
        updater to behave in the manner expected, although this is not enforced.

    Note:
        Updates are averaaged along the "receptive" dimension, i.e. the individual
        spike rates corresponding to each parameter.

    Note:
        For the purposes of parameter bounding, the update term is split by clamping
        values. The sequence of operations follows:

        1. The receptive dimension is reduced.
        2. The updates are split into positive and negative parts.
        3. The batch dimension is reduced.

    See Also:
        For more details and references, visit
        :ref:`zoo/learning-other:Linear Homeostatic Plasticity` in the zoo.
    """

    def __init__(
        self,
        plasticity: float,
        target: float | None,
        param: Literal["weight", "bias", "delay"],
        batch_reduction: (
            Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] | None
        ) = None,
        **kwargs,
    ):
        # call superclass constructor
        IndependentCellTrainer.__init__(self, **kwargs)

        # default hyperparameters
        self.plasticity = float(plasticity)
        self.target = (
            argtest.gt("target", target, 0, float) if target is not None else None
        )
        self.param = argtest.oneof(
            "param", param, "weight", "bias", "delay", op=lambda x: x.lower()
        )
        self.batchreduce = batch_reduction if batch_reduction else torch.mean

    def _build_cell_state(self, **kwargs) -> Module:
        r"""Builds auxiliary state for a cell.

        Keyword arguments will override module-level hyperparameters.

        Returns:
            Module: state module.
        """
        state = Module()

        plasticity = kwargs.get("plasticity", self.plasticity)
        target = kwargs.get("target", self.target)
        param = kwargs.get("param", self.param)
        batch_reduction = kwargs.get("batch_reduction", self.batchreduce)

        state.plasticity = float(plasticity)
        if target is None:
            state.target = None
        elif isinstance(target, torch.Tensor):
            _ = argtest.gte(
                "target", target.amin().item(), 0, float, prefix="minimum element in "
            )
            state.register_buffer("target", target, persistent=False)
        else:
            state.target = argtest.gte("target", target, 0, float)
        state.param = argtest.oneof(
            "param", param, "weight", "bias", "delay", op=lambda x: x.lower()
        )
        state.batchreduce = (
            batch_reduction if (batch_reduction is not None) else torch.mean
        )

        return state

    def register_cell(
        self,
        name: str,
        cell: Cell,
        /,
        **kwargs: Any,
    ) -> IndependentCellTrainer.Unit:
        r"""Adds a cell with required state.

        Args:
            name (str): name of the cell to add.
            cell (Cell): cell to add.

        Keyword Args:
            plasticity (float): learning rate for updates, :math:`\lambda`.
            target (float | None, optional): target rate for postsynaptic
                spikes, :math:`r^*`. Required to be specified on update when ``None``.
            param (Literal["weight", "bias", "delay"]): specified parameter to update.
            batch_reduction (Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] | None):
                function to reduce updates over the batch dimension, :py:func:`torch.mean`
                when ``None``.

        Returns:
            IndependentCellTrainer.Unit: specified cell, auxiliary state, and monitors.

        Important:
            The ``target`` parameter can be specified as a tensor here, in which
            case it won't need to be passed in on each :py:meth:`forward` call but can
            still have a different value for each output. It must have the same shape as
            the output spikes of the ``cell.neuron``. The batch dimension can have a
            size of 1 regardless of the batch sie.

        Important:
            Any specified keyword arguments will override the default hyperparameters
            set on initialization. See :py:class:`DelayAdjustedSTDP` for details.
        """
        # add the cell with additional hyperparameters
        state = self._build_cell_state(**kwargs)
        cell, state = self.add_cell(name, cell, state, [state.param])

        # move the target and change the datatype to match
        if isinstance(state.target, torch.Tensor):
            state.target = state.target.to(
                device=getattr(cell.connection, state.param).device,
                dtype=getattr(cell.connection, state.param).dtype,
            )

        # postsynaptic spike monitor
        self.add_monitor(
            name,
            "spike_rate",
            "neuron.spike",
            StateMonitor.partialconstructor(
                reducer=CAReducer(
                    cell.connection.dt,
                    duration=0.0,
                    inclusive=True,
                ),
                as_prehook=False,
                train_update=True,
                eval_update=False,
                prepend=True,
            ),
            False,
            dt=cell.connection.dt,
        )

        return self.get_unit(name)

    def forward(
        self,
        target: float | torch.Tensor | None = None,
        cells: Sequence[str] | None = None,
    ) -> None:
        r"""Processes update for given layers based on current monitor stored data.

        Args:
            target (float | torch.Tensor | None): target rate for postsynaptic
                spikes, :math:`r^*`. Required to be specified on update if the default
                is ``None`` and will try to use the default when ``None``.
                Defaults to ``None``.
            cells (Sequence[str] | None): names of the cells to update, all cells if
                ``None``. Defaults to ``None``.

        .. admonition:: Shape
            :class: tensorshape

            ``target``:

            :math:`B \times N_0 \times \cdots` or :math:`1 \times N_0 \times \cdots`

            Where:
                * :math:`B` is the batch size.
                * :math:`N_0, \ldots` are the dimensions of the postsynaptic spikes.
        """
        # iterate through self
        for name, (cell, state, monitors) in zip(self.cells_, self):

            # skip if cell is not in a non-none training list
            if cells is not None and name not in cells:
                continue

            # skip if self or cell is not in training mode or has no updater
            if not cell.training or not self.training or not cell.updater:
                continue

            # get target rate
            if target is None:
                if state.target is None:
                    raise RuntimeError("'target' must be non-None if no default is set")
                else:
                    target = state.target

            # compute rate scaling term
            k = cell.connection.postsyn_receptive(
                (target - monitors["spike_rate"].peek()) / target
            ).mean(dim=-1)

            # compute update conditional on parameter
            if state.param == "weight":
                k = k * state.plasticity
                cell.updater.weight = (
                    state.batchreduce(k.clamp_min(0.0), 0),
                    state.batchreduce(k.clamp_max(0.0), 0),
                )
            elif state.param == "bias":
                k = k * state.plasticity
                cell.updater.bias = (
                    cell.connection.like_bias(state.batchreduce(k.clamp_min(0.0), 0)),
                    cell.connection.like_bias(state.batchreduce(k.clamp_max(0.0), 0)),
                )
            elif state.param == "delay":
                k = k * -state.plasticity
                cell.updater.delay = (
                    state.batchreduce(k.clamp_min(0.0), 0),
                    state.batchreduce(k.clamp_max(0.0), 0),
                )
            else:
                raise ValueError(
                    f'an invalid \'param\' ("{state.param}") was set for cell with name "{name}"'
                )
