from .. import functional as lf
from abc import ABC, abstractmethod
import torch


class WeightBounding(ABC):
    r"""Modifies update magnitudes with LTP/LTD to bound weights.

    Args:
        wmin (float | None): lower bound for weights.
        wmax (float | None): upper bound for weights.

    Raises:
        ValueError: wmax must be greater than wmin.
    """

    def __init__(self, wmin: float | None, wmax: float | None):
        # convert bounds to floating point and none if no bounding
        wmin = None if wmin is None else float(wmin)
        wmax = None if wmax is None else float(wmax)

        # check for valid bounds
        if wmin is not None and wmax is not None:
            if wmin >= wmax:
                raise ValueError(f"wmin` {wmin} must be less than `wmax` {wmax}.")

        # add attributes
        if wmin is not None:
            self.wmin_ = wmin
        if wmax is not None:
            self.wmax_ = wmax

    @property
    def hasmin(self) -> bool:
        r"""If a minimum bound can be applied.

        Returns:
            bool: if a minimum bound can be applied.
        """
        return self.wmin is not None

    @property
    def hasmax(self) -> bool:
        r"""If a maximum bound can be applied.

        Returns:
            bool: if a maximum bound can be applied.
        """
        return self.wmax is not None

    @property
    def wmin(self) -> float | None:
        r"""Minimum bound.

        Args:
            value (float): new minimum bound

        Returns:
            float | None: present minimum bound, if defined.

        Raises:
            ValueError: wmin must be less than wmax.

        Note:
            If a minimum bound is not defined, assigning to this will have no effect.
        """
        if hasattr(self, "wmin_"):
            return self.wmin_

    @wmin.setter
    def wmin(self, value: float) -> None:
        if not self.hasmin:
            pass
        elif not self.hasmax or value < self.wmax:
            self._wmin = float(value)
        else:
            raise ValueError(f"wmin` {value} must be less than `wmax` {self.wmax}.")

    @property
    def wmax(self) -> float | None:
        r"""Maximum bound.

        Args:
            value (float): new maximum bound

        Returns:
            float | None: present maximum bound, if defined.

        Raises:
            ValueError: wmax must be greater than wmin.

        Note:
            If a maximum bound is not defined, assigning to this will have no effect.
        """
        if hasattr(self, "wmax_"):
            return self._wmax

    @wmax.setter
    def wmax(self, value: float) -> None:
        if not self.hasmax:
            pass
        elif not self.hasmin or value > self.wmin:
            self._wmax = float(value)
        else:
            raise ValueError(f"`wmax` {value} must be greater than `wmin` {self.wmin}.")

    @abstractmethod
    def lower(
        self, weights: torch.Tensor, amplitude: float | torch.Tensor
    ) -> torch.Tensor:
        r"""Applies lower bounds to a weight update.

        Args:
            weights (torch.Tensor): model weights.
            amplitude (float | torch.Tensor): amplitude of the update without weight
            bounding (e.g. the learning rate).

        Returns:
            torch.Tensor: weight-bound update amplitudes.

        Raises:
            NotImplementedError: ``lower`` must be implemented by the subclass.

        Note:
            This has the signature of :py:class:`~lf.BindWeights` and can be
            passed in where a parameter of that type is required.
        """
        raise NotImplementedError(
            f"{type(self).__name__}(WeightBounding) must implement "
            "the method `lower`."
        )

    @abstractmethod
    def upper(
        self, weights: torch.Tensor, amplitude: float | torch.Tensor
    ) -> torch.Tensor:
        r"""Applies upper bounds to a weight update.

        Args:
            weights (torch.Tensor): model weights.
            amplitude (float | torch.Tensor): amplitude of the update without weight
            bounding (e.g. the learning rate).

        Returns:
            torch.Tensor: weight-bound update amplitudes.

        Raises:
            NotImplementedError: ``upper`` must be implemented by the subclass.

        Note:
            This has the signature of :py:class:`~lf.BindWeights` and can be
            passed in where a parameter of that type is required.
        """
        raise NotImplementedError(
            f"{type(self).__name__}(WeightBounding) must implement "
            "the method `upper`."
        )


class HardWeightDependence(WeightBounding):
    r"""Modifies weight updates using hard weight dependence.

    Args:
        wmin (float | None): lower bound of weights, :math:`w_\text{min}`.
        wmax (float | None): upper bound of weights, :math:`w_\text{max}`.
    """

    def __init__(self, wmin: float | None, wmax: float | None):
        # call superclass constructor
        WeightBounding.__init__(self, wmin, wmax)

    def lower(
        self, weights: torch.Tensor, amplitude: float | torch.Tensor
    ) -> torch.Tensor:
        r"""Applies lower bounds to a weight update.

        .. math::

            A_- = \Theta(w - w_\text{min}) \eta_-

        Where

        .. math::

            \Theta(x) =
            \begin{cases}
                1 &x \geq 0 \\
                0 & x < 0
            \end{cases}

        Args:
            weights (torch.Tensor): model weights, :math:`w`.
            amplitude (float | torch.Tensor): amplitude of the update excluding weight
                dependence, :math:`\eta_-`.

        Returns:
            torch.Tensor: torch.Tensor: amplitudes :math:`A_-` after min-bounding.

        Raises:
            RuntimeError: cannot apply lower bounds if the lower bound was not given.

        Note:
            This has the signature of :py:class:`~lf.BindWeights` and can be
            passed in where a parameter of that type is required.
        """
        if self.hasmin:
            return lf.wdep_hard_lower_bounding(weights, amplitude, self.wmin)
        else:
            raise RuntimeError("cannot apply `lower()` with no `wmin`.")

    def upper(
        self, weights: torch.Tensor, amplitude: float | torch.Tensor
    ) -> torch.Tensor:
        r"""Applies upper bounds to a weight update.

        .. math::

            A_+ = \Theta(w_\text{max} - w) \eta_+

        Where

        .. math::

            \Theta(x) =
            \begin{cases}
                1 &x \geq 0 \\
                0 & x < 0
            \end{cases}

        Args:
            weights (torch.Tensor): model weights, :math:`w`.
            amplitude (float | torch.Tensor): amplitude of the update excluding weight
                dependence, :math:`\eta_+`.

        Returns:
            torch.Tensor: torch.Tensor: amplitudes :math:`A_+` after max-bounding.

        Raises:
            RuntimeError: cannot apply upper bounds if the lower bound was not given.

        Note:
            This has the signature of :py:class:`~lf.BindWeights` and can be
            passed in where a parameter of that type is required.
        """
        if self.hasmax:
            return lf.wdep_hard_upper_bounding(weights, amplitude, self.wmax)
        else:
            raise RuntimeError("cannot apply `upper()` with no `wmax`.")


class SoftWeightDependence(WeightBounding):
    r"""Modifies weight updates using soft weight dependence.

    Args:
        wmin (float | None): lower bound of weights, :math:`w_\text{min}`.
        wmax (float | None): upper bound of weights, :math:`w_\text{max}`.
        minpow (float | None, optional): exponent of lower bound weight dependence,
            :math:`\mu_-`. Defaults to 1.0.
        maxpow (float | None, optional): exponent of upper bound weight dependence,
            :math:`\mu_+`. Defaults to 1.0.

    Raises:
        TypeError: if a wmin or wmax is specified, a corresponding minpow or maxpow
            must be specified as well.
    """

    def __init__(
        self,
        wmin: float | None,
        wmax: float | None,
        minpow: float | None = 1.0,
        maxpow: float | None = 1.0,
    ):
        # call superclass constructor
        WeightBounding.__init__(self, wmin, wmax)

        # convert powers to floating point and none if no bounding
        minpow = None if self.hasmin and minpow is None else float(minpow)
        maxpow = None if self.hasmax and maxpow is None else float(maxpow)

        # check that powers are defined
        if self.hasmin and minpow is None:
            raise TypeError("if `wmin` is not None, `minpow` cannot be none.")
        if self.hasmax and maxpow is None:
            raise TypeError("if `wmax` is not None, `maxpow` cannot be none.")

        # assign attributes
        self.minpow_ = minpow
        self.maxpow_ = maxpow

    @property
    def minpow(self) -> float | None:
        r"""Power of minimum bounding, :math:`\mu_-`.

        Args:
            value (float): new power of minimum bound.

        Returns:
            float | None: present power of minimum bound, if defined.

        Note:
            If a minimum bound is not defined, assigning to this will have no effect.
        """
        if self.hasmin:
            return self.minpow_

    @minpow.setter
    def minpow(self, value: float) -> None:
        if self.hasmin:
            self.minpow_ = float(value)

    @property
    def maxpow(self) -> float | None:
        r"""Power of maximum bounding, :math:`\mu_+`.

        Args:
            value (float): new power of maximum bound

        Returns:
            float | None: present power of maximum bound, if defined.

        Note:
            If a maximum bound is not defined, assigning to this will have no effect.
        """
        if self.hasmax:
            return self.maxpow_

    @maxpow.setter
    def maxpow(self, value: float) -> None:
        if self.hasmax:
            self.maxpow_ = float(value)

    def lower(
        self, weights: torch.Tensor, amplitude: float | torch.Tensor
    ) -> torch.Tensor:
        r"""Applies lower bounds to a weight update.

        .. math::

            A_- = (w - w_\text{min})^{\mu_-} \eta_-

        Args:
            weights (torch.Tensor): model weights, :math:`w`.
            amplitude (float | torch.Tensor): amplitude of the update excluding weight
                dependence, :math:`\eta_-`.

        Returns:
            torch.Tensor: torch.Tensor: amplitudes :math:`A_-` after min-bounding.

        Raises:
            RuntimeError: cannot apply lower bounds if the lower bound was not given.

        Note:
            This has the signature of :py:class:`~lf.BindWeights` and can be
            passed in where a parameter of that type is required.
        """
        if self.hasmin:
            return lf.wdep_hard_lower_bounding(
                weights, amplitude, self.wmin, self.minpow
            )
        else:
            raise RuntimeError("cannot apply `lower()` with no `wmin`.")

    def upper(
        self, weights: torch.Tensor, amplitude: float | torch.Tensor
    ) -> torch.Tensor:
        r"""Applies upper bounds to a weight update.

        .. math::

            A_+ = (w_\text{max} - w)^{\mu_+} \eta_+

        Args:
            weights (torch.Tensor): model weights, :math:`w`.
            amplitude (float | torch.Tensor): amplitude of the update excluding weight
                dependence, :math:`\eta_+`.

        Returns:
            torch.Tensor: torch.Tensor: amplitudes :math:`A_+` after max-bounding.

        Raises:
            RuntimeError: cannot apply upper bounds if the lower bound was not given.

        Note:
            This has the signature of :py:class:`~lf.BindWeights` and can be
            passed in where a parameter of that type is required.
        """
        if self.hasmax:
            return lf.wdep_soft_upper_bounding(
                weights, amplitude, self.wmax, self.maxpow
            )
        else:
            raise RuntimeError("cannot apply `upper()` with no `wmax`.")
