import torch


class Poisson:
    r"""Sampling from and properties of the Poisson distribution.

    The Poisson distribution is a discrete probability distribution used to express
    the probability of a given number of events, :math:`k`, occuring in a fixed amount
    of time, given an expected number of events, :math:`\lambda`.

    .. admonition:: Parameters

        :math:`\lambda \in (0, \infty)`, rate

    .. admonition:: Support

        :math:`k \in \mathbb{N}_0`, count
    """

    @staticmethod
    def sample(
        rate: torch.Tensor, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        r"""Samples random variates from a Poisson distribution.

        .. math::
            K \sim \text{Poisson}(\lambda)

        Args:
            rate (torch.Tensor): expected rate of occurances :math:`\lambda`.
            generator (torch.Generator | None, optional): pseudorandom number generator
                to use for sampling. Defaults to None.

        Returns:
            torch.Tensor: resulting random variates :math:`K`.

        Caution:
            This calls :py:func:`torch.poisson` which as of PyTorch 2.1 does not support
            computation on Metal Performance Shaders. Compensate accordingly.
        """
        return torch.poisson(rate, generator=generator)

    @staticmethod
    def pmf(support: torch.Tensor, rate: torch.Tensor) -> torch.Tensor:
        r"""Computes the probability mass function.

        .. math::
            P(K=k; \lambda) =\frac{\lambda^k e^{-k}}{k!}

        Args:
            support (torch.Tensor): number of occurances :math:`k`.
            rate (torch.Tensor): expected rate of occurances :math:`\lambda`.

        Returns:
            torch.Tensor: resulting point probabilities.
        """
        return torch.exp(Poisson.pmf(support, rate))

    @staticmethod
    def logpmf(support: torch.Tensor, rate: torch.Tensor) -> torch.Tensor:
        r"""Computes the natural logarithm of the probability mass function.

        .. math::
            \log P(K=k; \lambda) = k \log \lambda - \lambda - \log(k!)

        Args:
            support (torch.Tensor): number of occurances :math:`k`.
            rate (torch.Tensor): expected rate of occurances :math:`\lambda`.

        Returns:
            torch.Tensor: log of the resulting point probabilities.
        """
        return torch.special.xlogy(support, rate) - rate - torch.lgamma(rate + 1)

    @staticmethod
    def cdf(support: torch.Tensor, rate: torch.Tensor) -> torch.Tensor:
        r"""Computes the cumulative distribution function.

        .. math::
            P(K \leq k; \lambda) = e^{-\lambda} \sum_{j=0}^{\lfloor k \rfloor}
            \frac{\lambda^j}{j!}

        Args:
            support (torch.Tensor): number of occurances :math:`k`.
            rate (torch.Tensor): expected rate of occurances :math:`\lambda`.

        Returns:
            torch.Tensor: resulting cumulative probabilities.
        """
        return torch.special.gammaincc(torch.floor(support + 1), rate)

    @staticmethod
    def logcdf(support: torch.Tensor, rate: torch.Tensor) -> torch.Tensor:
        r"""Computes the natural logarithm of the cumulative distribution function.

        .. math::
            \log P(K \leq k; \lambda) = \log
            \frac{\Gamma (\lfloor k + 1 \rfloor, \lambda)}{\Gamma (\lambda)}

        Args:
            support (torch.Tensor): number of occurances :math:`k`.
            rate (torch.Tensor): expected rate of occurances :math:`\lambda`.

        Returns:
            torch.Tensor: log of the resulting cumulative probabilities.
        """
        return torch.special.gammaincc(torch.floor(support + 1), rate)
