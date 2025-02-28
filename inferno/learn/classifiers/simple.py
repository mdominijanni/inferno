from ... import Module
from ..._internal import argtest
from collections.abc import Sequence
import einops as ein
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaxRateClassifier(Module):
    r"""Classifies spikes by maximum per-class rates.

    The classifier uses an internal parameter :py:attr:`rates` for other
    calculations. When learning, the existing rates are decayed, multiplying them by
    :math:`\exp (-\lambda b_k)` where :math:`b_k` is the number of elements of class
    :math:`k` in the batch.

    Each neuron is assigned a class based on its maximum normalized per-class rate (i.e.
    the class with which it fires most frequently, accounting for a non-uniform class
    distribution). Given a sample, the firing rate for each neuron is added to the
    class to which it is assigned. These per-class sample rates are divided by the number
    of neurons assigned to that class. The maximum of these unnormalized logits is the
    predicted class.

    Args:
        shape (Sequence[int] | int): shape of the group of neurons with
            their output being classified.
        num_classes (int): total number of possible classes.
        decay_rate (float): per-update amount by which previous results
            are scaled, :math:`\lambda`. Defaults to `0.0`.

    Note:
        The methods :py:meth:`regress`, :py:meth:`classify`, and :py:meth:`forward` take
        an argument ``proportional``. When ``True``, the contribution of each neuron's
        assigned class is weighted by relative affinity of that neuron for the
        corresponding class. For example, if half of the times a neuron spiked it did
        so on samples with its assigned class, the sample rate will be multiplied by
        :math:`\frac{1}{2}` rather than :math:`1`.
    """

    def __init__(
        self,
        shape: Sequence[int] | int,
        num_classes: int,
        *,
        decay: float = 0.0,
    ):
        # call superclass constructor
        Module.__init__(self)

        # validate parameters
        try:
            shape = (argtest.gt("shape", shape, 0, int),)
        except TypeError:
            if isinstance(shape, Sequence):
                shape = argtest.ofsequence("shape", shape, argtest.gt, 0, int)
            else:
                raise TypeError(
                    f"'shape' ({argtest._typename(type(shape))}) cannot be interpreted "
                    "as an integer or a sequence thereof"
                )

        num_classes = argtest.gt("num_classes", num_classes, 0, int)

        # register parameter
        self.register_parameter(
            "rates_", nn.Parameter(torch.zeros(*shape, num_classes).float(), False)
        )

        # register derived buffers
        self.register_buffer(
            "assignments_", torch.zeros(*shape).long(), persistent=False
        )
        self.register_buffer(
            "occurrences_", torch.zeros(num_classes).long(), persistent=False
        )
        self.register_buffer(
            "proportions_", torch.zeros(*shape, num_classes).float(), persistent=False
        )

        # class attribute
        self.decay = argtest.gte("decay", decay, 0, float)

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
    def occurrences(self) -> torch.Tensor:
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
        return self.occurrences_

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
            :py:attr:`occurrences` are automatically recalculated on assignment.

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
        self.occurrences_ = torch.bincount(self.assignments.view(-1), None, self.nclass)

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
        return self.occurrences.shape[0]

    @property
    def shape(self) -> tuple[int, ...]:
        r"""Shape of the spikes being classified, excluding batch and time.

        Returns:
            tuple[int, ...]: shape of spikes being classified.
        """
        return tuple(self.assignments.shape)

    def regress(self, inputs: torch.Tensor, proportional: bool = True) -> torch.Tensor:
        r"""Computes class logits from spike rates.

        Args:
            inputs (torch.Tensor): batched spike rates to classify.
            proportional (bool, optional): if inference is weighted by class-average
                rates. Defaults to ``True``.

        Returns:
            torch.Tensor: predicted logits.

        .. admonition:: Shape
            :class: tensorshape

            ``inputs``:

            :math:`B \times N_0 \times \cdots`

            ``return``:

            :math:`B \times K`

            Where:
                * :math:`B` is the batch size.
                * :math:`N_0, \ldots` are the dimensions of the spikes being classified.
                * :math:`K` is the number of possible classes.
        """
        # associations between neurons and classes
        if proportional:
            assocs = F.one_hot(self.assignments.view(-1), self.nclass) * ein.rearrange(
                self.proportions, "... k -> (...) k"
            )
        else:
            assocs = F.one_hot(self.assignments.view(-1), self.nclass).to(
                dtype=self.proportions_.dtype
            )

        # compute logits
        ylogits = (
            torch.mm(
                ein.rearrange(inputs, "b ... -> b (...)"),
                assocs,
            )
            .div(self.occurrences)
            .nan_to_num(nan=0, posinf=0)
        )

        # return logits or predictions
        return ylogits

    def classify(self, inputs: torch.Tensor, proportional: bool = True) -> torch.Tensor:
        r"""Computes class labels from spike rates.

        Args:
            inputs (torch.Tensor): batched spike rates to classify.
            proportional (bool, optional): if inference is weighted by class-average
                rates. Defaults to ``True``.

        Returns:
            torch.Tensor: predicted labels.

        .. admonition:: Shape
            :class: tensorshape

            ``inputs``:

            :math:`B \times N_0 \times \cdots`

            ``return``:

            :math:`B`

            Where:
                * :math:`B` is the batch size.
                * :math:`N_0, \ldots` are the dimensions of the spikes being classified.
        """
        return torch.argmax(self.regress(inputs, proportional), dim=1)

    def update(self, inputs: torch.Tensor, labels: torch.Tensor) -> None:
        r"""Updates stored rates from spike rates and labels.

        Args:
            inputs (torch.Tensor): batched spike rates from which to update state.
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
        clscounts = torch.bincount(labels, None, self.nclass).to(dtype=self.rates.dtype)

        # compute per-class scaled spike rates
        rates = (
            torch.scatter_add(
                torch.zeros_like(self.rates),
                dim=-1,
                index=labels.expand(*self.shape, -1),
                src=ein.rearrange(inputs, "b ... -> ... b"),
            )
            / clscounts
        ).nan_to_num(nan=0, posinf=0)

        # update rates, other properties update automatically
        self.rates = torch.exp(-self.decay * clscounts) * self.rates + rates

    def forward(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor | None,
        logits: bool | None = False,
        proportional: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
        r"""Performs inference and updates the classifier state.

        Args:
            inputs (torch.Tensor): spikes or spike rates to classify.
            labels (torch.Tensor | None): ground-truth sample labels.
            logits (bool | None, optional): if predicted class logits should be
                returned along with labels, inference is skipped if ``None``.
                Defaults to ``False``.
            proportional (bool, optional): if inference is weighted by class-average
                rates. Defaults to ``True``.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None: predicted class
            labels, with unnormalized logits if specified.

        .. admonition:: Shape
            :class: tensorshape

            ``inputs``:

            :math:`[T] \times B \times N_0 \times \cdots`

            ``labels``:

            :math:`B`

            ``return (logits=False)``:

            :math:`B`

            ``return (logits=True)``:

            :math:`(B, B \times K)`

            Where:
                * :math:`T` is the number of simulation steps over which spikes were gathered.
                * :math:`B` is the batch size.
                * :math:`N_0, \ldots` are the dimensions of the spikes being classified.
                * :math:`K` is the number of possible classes.

        Important:
            This method will always perform the inference step prior to updating the classifier.
        """
        # reduce along input time dimension, if present, to generate spike counts
        if inputs.ndim == self.ndim + 2:
            inputs = inputs.to(dtype=self.rates.dtype).mean(dim=0, keepdim=False)

        # inference
        if logits is None:
            res = None
        elif not logits:
            res = self.classify(inputs, proportional)
        else:
            res = self.regress(inputs, proportional)
            res = (torch.argmax(res, dim=1), res)

        # update
        if labels is not None:
            self.update(inputs, labels)

        # return inference
        return res
