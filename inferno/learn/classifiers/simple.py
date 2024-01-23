import einops as ein
from inferno import Module
from inferno._internal import numeric_limit, numeric_interval
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


class RateClassifier(Module):
    r"""Classifies spikes by per-class rates.

    Args:
        shape (tuple[int, ...] | int): shape of the group of neurons with
            their output being classified.
        num_classes (int): total number of possible classes.
        decay (float, optional): per-update amount by which previous results
            are scaled. Defaults to 1.0.
        proportional (float, optional): if logits should be computed with
            class-proportional rates. Defaults to True.
        reduction (Literal["sum", "mean"], optional): method by which non-reduced
            spikes should be reduced. Defaults to "sum".
    """

    def __init__(
        self,
        shape: tuple[int, ...] | int,
        num_classes: int,
        *,
        decay: float = 1.0,
        proportional: float = True,
        reduction: Literal["sum", "mean"] = "sum",
    ):
        # call superclass constructor
        Module.__init__(self)

        # validate parameters
        try:
            shape = numeric_limit("`shape`", shape, 0, "gt", int)
        except TypeError:
            shape = tuple(
                numeric_limit(f"`shape[{idx}]`", s, 0, "gt", int)
                for idx, s in enumerate(shape)
            )

        num_classes = num_classes("`num_classes`", num_classes, 0, "gt", int)

        if reduction.lower() not in ("sum", "mean"):
            raise ValueError(
                "`reduction` must be one of 'sum' or 'mean', "
                f"received '{reduction}.'"
            )

        # class attributes
        self.proportional = proportional
        self.reduction = reduction.lower()

        # register parameter
        self.register_parameter(
            "rates_", nn.Parameter(torch.zeros(*shape, num_classes).float(), False)
        )

        # register derived buffers
        self.register_buffer("assigns", torch.zeros(*shape).long())
        self.register_buffer("counts", torch.zeros(num_classes).long())
        self.register_buffer("props", torch.zeros(*shape, num_classes).float())

        self.decay = numeric_interval("`decay`", decay, 0, 1, "closed", float)

    @property
    def assignments(self) -> torch.Tensor:
        r"""Class assignments per-neuron.

        Args:
            value (torch.Tensor): new class assignments per-neuron.

        Returns:
            torch.Tensor: present class assignments per-neuron.

        Note:
            :py:attr:`occurances` is automatically recalculated on assignment.

        .. admonition:: Shape
            :class: tensorshape

            :math:`N_0 \times \cdots`

            Where:
                * :math:`N_0, \ldots` are the dimensions of the spikes being classified.
        """
        return self.assigns

    @assignments.setter
    def assignments(self, value: torch.Tensor) -> None:
        self.assigns = value
        self.counts = torch.bincount(self.assignments.view(-1), None, self.nclass)

    @property
    def occurances(self) -> torch.Tensor:
        r"""Number of assigned neurons per-class.

        Args:
            value (torch.Tensor): new number of assigned neurons per-class.

        Returns:
            torch.Tensor: present number of assigned neurons per-class.

        .. admonition:: Shape
            :class: tensorshape

            :math:`K`

            Where:
                * :math:`K` is the number of possible classes.
        """
        return self.counts

    @occurances.setter
    def occurances(self, value: torch.Tensor) -> None:
        self.counts = value

    @property
    def proportions(self) -> torch.Tensor:
        r"""Class-normalized spike rates.

        Args:
            value (torch.Tensor): new class-normalized spike rates.

        Returns:
            torch.Tensor: present class-normalized spike rates.

        Note:
            :py:attr:`assignments` is automatically recalculated on assignment.

        .. admonition:: Shape
            :class: tensorshape

            :math:`N_0 \times \cdots \times K`

            Where:
                * :math:`N_0, \ldots` are the dimensions of the spikes being classified.
                * :math:`K` is the number of possible classes.
        """
        return self.props

    @proportions.setter
    def proportions(self, value: torch.Tensor) -> None:
        self.props = value
        self.assignments = torch.argmax(self.proportions, dim=-1)

    @property
    def rates(self) -> torch.Tensor:
        r"""Computed per-class, per-neuron spike rates.

        Args:
            value (torch.Tensor): new computed per-class, per-neuron spike rates.

        Returns:
            torch.Tensor: present computed per-class, per-neuron spike rates.

        Note:
            :py:attr:`proportions` is automatically recalculated on assignment.

        .. admonition:: Shape
            :class: tensorshape

            :math:`N_0 \times \cdots \times K`

            Where:
                * :math:`N_0, \ldots` are the dimensions of the spikes being classified.
                * :math:`K` is the number of possible classes.
        """
        return self.rates_.data

    @rates.setter
    def rates(self, value: torch.Tensor) -> None:
        self.rates_.data = value
        self.proportions = F.normalize(self.rates, p=1, dim=-1)

    @property
    def isprop(self) -> bool:
        r"""If inference is weighted by class-average rates.

        Returns:
            bool: if inference is weighted by class-average rates.
        """
        return self.proportional

    @property
    def ndim(self) -> int:
        r"""Number of dimensions of the spikes being classified, excluding batch and time.

        Returns:
            tuple[int, ...]: number of dimensions of the spikes being classified
        """
        return self.assignments.ndim

    @property
    def nclass(self) -> int:
        r"""Number of possible classes

        Returns:
            int: number of possible classes.
        """
        return self.occurances.shape[0]

    @property
    def shape(self) -> tuple[int, ...]:
        r"""Shape of the spikes being classified, excluding batch and time.

        Returns:
            tuple[int, ...]: shape of spikes being classified.
        """
        return tuple(self.assignments.shape)

    def infer(
        self,
        spikes: torch.Tensor,
        proportional: bool,
        logits: bool,
    ) -> torch.Tensor:
        r"""Infers classes from reduced spikes

        Args:
            spikes (torch.Tensor): reduced spikes to classify.
            proportional (bool): if inference is weighted by class-average rates.
            logits (bool): if logits rather than class predictions should be returned.

        Returns:
            torch.Tensor: inferences, either logits or predictions.

        .. admonition:: Shape
            :class: tensorshape

            ``spikes``:

            :math:`B \times N_0 \times \cdots \times [T]`

            ``return (logits=False)``:

            :math:`B`

            ``return (logits=True)``:

            :math:`B \times K`

            Where:
                * :math:`B` is the batch size.
                * :math:`N_0, \ldots` are the dimensions of the spikes being classified.
                * :math:`K` is the number of possible classes.
        """
        # associations between neurons and classes
        assocs = F.one_hot(
            ein.rearrange(self.assignments, "... -> (...)"), self.nclass
        )
        if proportional:
            assocs = assocs * ein.rearrange(self.proportions, "... k -> (...) k")

        # compute logits
        logits = (
            torch.mm(
                ein.rearrange(spikes, "b ... -> b (...)"),
                assocs,
            )
            .div(self.occurances)
            .nan_to_num(0)
        )

        # return logits or predictions
        if logits:
            return logits
        else:
            return torch.argmax(logits, dim=1)

    def learn(
        self,
        spikes: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        r"""Updates stored rates from reduced spikes and labels

        Args:
            spikes (torch.Tensor): reduced spikes from which to update state.
            labels (torch.Tensor): ground-truth sample labels.

        .. admonition:: Shape
            :class: tensorshape

            ``spikes``:

            :math:`B \times N_0 \times \cdots`

            ``labels``:

            :math:`B`

            Where:
                * :math:`B` is the batch size.
                * :math:`N_0, \ldots` are the dimensions of the spikes being classified.
        """
        # number of instances per-class
        batchcls = torch.bincount(labels, None, self.nclass)

        # compute per-class scaled spike rates
        batchrates = (
            torch.zeros_like(self.rates)
            .scatter_add(
                dim=-1,
                index=labels.expand_like(self.rates),
                src=ein.rearrange(spikes, "b ... -> ... b"),
            )
            .div(batchcls)
            .nan_to_num(0)
        )

        # update rates, other properties update automatically
        self.rates = torch.where(
            batchcls.bool(), self.rates * self.decay + batchrates, self.rates
        )

    def forward(
        self,
        spikes: torch.Tensor,
        labels: torch.Tensor | None,
        logits: bool = False,
    ) -> torch.Tensor:
        r"""Performs inference and if labels are specified, updates state.

        Args:
            spikes (torch.Tensor): reduced spikes to classify.
            labels (torch.Tensor | None): ground-truth sample labels.
            logits (bool, optional): if logits rather than class predictions
                should be returned.. Defaults to False.

        Returns:
            torch.Tensor: inferences, either logits or predictions.

        .. admonition:: Shape
            :class: tensorshape

            ``spikes``:

            :math:`B \times N_0 \times \cdots \times [T]`

            ``labels``:

            :math:`B`

            ``return (logits=False)``:

            :math:`B`

            ``return (logits=True)``:

            :math:`B \times K`

            Where:
                * :math:`B` is the batch size.
                * :math:`N_0, \ldots` are the dimensions of the spikes being classified.
                * :math:`T` is the number of simulation steps over which spikes were gathered.
                * :math:`K` is the number of possible classes.
        """
        # reduce along input time dimension, if present, to generate spike counts
        if spikes.ndim == self.ndim + 2:
            spikes = ein.reduce(spikes, "b ... t -> b ...", self.reduction)

        # infer
        inferred = self.infer(spikes, proportional=self.isprop, logits=logits)

        # update state
        if labels is not None:
            self.learn(spikes, labels)

        # return inference
        return inferred
