import einops as ein
from inferno import Module
from inferno._internal import numeric_limit
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaxRateClassifier(Module):
    r"""Classifies spikes by maximum per-class rates.

    The classifier uses an internal parameter :py:attr:`rates` internally for other
    calculations. When learning, the existing rates are decayed, multiplying them by
    :math:`\exp{-\lambda b_k}` where :math:`b_k` is number of elements of class
    :math:`k` in the batch.

    Args:
        shape (tuple[int, ...] | int): shape of the group of neurons with
            their output being classified.
        num_classes (int): total number of possible classes.
        decay_rate (float): per-update amount by which previous results
            are scaled, :math:`\lambda`.
        proportional (float, optional): if logits should be computed with
            class-proportional rates. Defaults to True.
    """

    def __init__(
        self,
        shape: tuple[int, ...] | int,
        num_classes: int,
        *,
        decay: float = 1.0,
        proportional: float = True,
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

        num_classes = numeric_limit("`num_classes`", num_classes, 0, "gt", int)

        # class attributes
        self._proportional = proportional

        # register parameter
        self.register_parameter(
            "rates_", nn.Parameter(torch.zeros(*shape, num_classes).float(), False)
        )

        # register derived buffers
        self.register_buffer(
            "assignments_", torch.zeros(*shape).long(), persistent=False
        )
        self.register_buffer(
            "occurances_", torch.zeros(num_classes).long(), persistent=False
        )
        self.register_buffer(
            "proportions_", torch.zeros(*shape, num_classes).float(), persistent=False
        )

        # class attribute
        self.decay = numeric_limit("`decay`", decay, 0, "gte", float)

        # run after loading state_dict to recompute non-persistent buffers
        def sdhook(module, incompatible_keys) -> None:
            module.rates = module.rates

        self.register_load_state_dict_post_hook(sdhook)

    @property
    def assignments(self) -> torch.Tensor:
        r"""Class assignments per-neuron.

        The label, computed as the argument of the maximum of normalized rates
        (proportions), per neuron.

        Returns:
            torch.Tensor: present class assignments per-neuron.

        .. admonition:: Shape
            :class: tensorshape

            :math:`N_0 \times \cdots`

            Where:
                * :math:`N_0, \ldots` are the dimensions of the spikes being classified.
        """
        return self.assignments_

    @property
    def occurances(self) -> torch.Tensor:
        r"""Number of assigned neurons per-class.

        The number of neurons which are assigned to each label.

        Returns:
            torch.Tensor: present number of assigned neurons per-class.

        .. admonition:: Shape
            :class: tensorshape

            :math:`K`

            Where:
                * :math:`K` is the number of possible classes.
        """
        return self.occurances_

    @property
    def proportions(self) -> torch.Tensor:
        r"""Class-normalized spike rates.

        The rates :math:`L_1`-normalized such that for a given neuron, such that the
        normalized rates for it over the different classes sum to 1.

        Returns:
            torch.Tensor: present class-normalized spike rates.

        .. admonition:: Shape
            :class: tensorshape

            :math:`N_0 \times \cdots \times K`

            Where:
                * :math:`N_0, \ldots` are the dimensions of the spikes being classified.
                * :math:`K` is the number of possible classes.
        """
        return self.proportions_

    @property
    def rates(self) -> torch.Tensor:
        r"""Computed per-class, per-neuron spike rates.

        These are the raw rates
        :math:`\left(\frac{\text{# spikes}}{\text{# steps}}\right)`
        for each neuron, per class.

        Args:
            value (torch.Tensor): new computed per-class, per-neuron spike rates.

        Returns:
            torch.Tensor: present computed per-class, per-neuron spike rates.

        Note:
            The attributes :py:attr:`proportions`, :py:attr:`assignments`, and
            :py:attr:`occurances` are automatically recalculated on assignment.

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
        # rates are assigned directly
        self.rates_.data = value
        self.proportions_ = F.normalize(self.rates, p=1, dim=-1)
        self.assignments_ = torch.argmax(self.proportions, dim=-1)
        self.occurances_ = torch.bincount(self.assignments.view(-1), None, self.nclass)

    @property
    def proportional(self) -> bool:
        r"""If inference is weighted by class-average rates.

        Args:
            value (bool): if inference should be computed using class-average rates.

        Returns:
            bool: if inference is weighted by class-average rates.
        """
        return self._proportional

    @proportional.setter
    def proportional(self, value: bool) -> None:
        return self._proportional

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
        inputs: torch.Tensor,
        logits: bool,
    ) -> torch.Tensor:
        r"""Infers classes from reduced spikes

        Args:
            inputs (torch.Tensor): batch spike rates to classify.
            proportional (bool): if inference is weighted by class-average rates.
            logits (bool): if logits rather than class predictions should be returned.

        Returns:
            torch.Tensor: inferences, either logits or predictions.

        .. admonition:: Shape
            :class: tensorshape

            ``inputs``:

            :math:`B \times N_0 \times \cdots`

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
        if self.proportional:
            assocs = F.one_hot(self.assignments.view(-1), self.nclass) * ein.rearrange(
                self.proportions, "... k -> (...) k"
            )
        else:
            assocs = F.one_hot(self.assignments.view(-1), self.nclass).float()

        # compute logits
        ylogits = (
            torch.mm(
                ein.rearrange(inputs, "b ... -> b (...)"),
                assocs,
            )
            .div(self.occurances)
            .nan_to_num(nan=0, posinf=0)
        )

        # return logits or predictions
        if logits:
            return ylogits
        else:
            return torch.argmax(ylogits, dim=1)

    def learn(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        r"""Updates stored rates from reduced spikes and labels

        Args:
            inputs (torch.Tensor): batch spike rates from which to update state.
            labels (torch.Tensor): ground-truth sample labels.

        .. admonition:: Shape
            :class: tensorshape

            ``inputs``:

            :math:`B \times N_0 \times \cdots`

            ``labels``:

            :math:`B`

            Where:
                * :math:`B` is the batch size.
                * :math:`N_0, \ldots` are the dimensions of the spikes being classified.
        """
        # number of instances per-class
        clscounts = torch.bincount(labels, None, self.nclass)

        # compute per-class scaled spike rates
        rates = (
            torch.zeros_like(self.rates)
            .scatter_add_(
                dim=-1,
                index=labels.expand(*self.shape, -1),
                src=ein.rearrange(inputs, "b ... -> ... b"),
            )
            .div_(clscounts)
            .nan_to_num(nan=0, posinf=0)
        )

        # update rates, other properties update automatically
        self.rates = torch.exp(-self.decay * clscounts) * self.rates + rates

    def forward(
        self,
        spikes: torch.Tensor,
        labels: torch.Tensor | None,
        logits: bool = False,
    ) -> torch.Tensor | None:
        r"""Performs inference and if labels are specified, updates state.

        Args:
            spikes (torch.Tensor): spikes or spike rates to classify.
            labels (torch.Tensor | None): ground-truth sample labels.
            logits (bool, optional): if logits rather than class predictions
                should be returned.. Defaults to False.
            proportional (bool | None, optional): if not None, then it overrides the
                class inference behavior. Defaults to None.

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
            spikes = ein.reduce(
                spikes.to(dtype=self.rates.dtype), "b ... t -> b ...", "mean"
            )

        # inference
        inferred = self.infer(spikes, logits=logits)

        # update state
        if labels is not None:
            self.learn(spikes, labels)

        # return inference
        return inferred
