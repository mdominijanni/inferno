import inferno
import math
import torch


class Poisson:
    r"""Sampling from and properties of the Poisson distribution.

    The Poisson distribution is a discrete probability distribution used to express
    the probability of a given number of events, :math:`k`, occuring in a fixed amount
    of time, given an expected number of events, :math:`\lambda`.

    .. admonition:: Parameters

        :math:`\lambda \in \mathbb{R}_+^*`, rate

    .. admonition:: Support

        :math:`k \in \mathbb{N}_0`, count
    """

    @staticmethod
    def sample(
        rate: torch.Tensor | float, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        r"""Samples random variates from a Poisson distribution.

        .. math::
            K \sim \text{Poisson}(\lambda)

        Args:
            rate (torch.Tensor | float): expected rate of occurances, :math:`\lambda`.
            generator (torch.Generator | None, optional): pseudorandom number generator
                to use for sampling. Defaults to None.

        Returns:
            torch.Tensor: resulting random variates, :math:`K`.

        Caution:
            This calls :py:func:`torch.poisson` which as of PyTorch 2.1 does not support
            computation on Metal Performance Shaders. Compensate accordingly.
        """
        if not isinstance(rate, torch.Tensor):
            rate = torch.tensor(rate).float()

        return torch.poisson(rate, generator=generator)

    @staticmethod
    def pmf(support: torch.Tensor | float, rate: torch.Tensor | float) -> torch.Tensor:
        r"""Computes the probability mass function.

        .. math::
            P(K=k; \lambda) = \frac{\lambda^k e^{-k}}{k!}

        Args:
            support (torch.Tensor | float): number of occurances, :math:`k`.
            rate (torch.Tensor | float): expected rate of occurances, :math:`\lambda`.

        Returns:
            torch.Tensor: resulting point probabilities.
        """
        return torch.exp(Poisson.logpmf(support, rate))

    @staticmethod
    def logpmf(
        support: torch.Tensor | float, rate: torch.Tensor | float
    ) -> torch.Tensor:
        r"""Computes the natural logarithm of the probability mass function.

        .. math::
            \log P(K=k; \lambda) = k \log \lambda - \lambda - \log(k!)

        Args:
            support (torch.Tensor | float): number of occurances, :math:`k`.
            rate (torch.Tensor | float): expected rate of occurances, :math:`\lambda`.

        Returns:
            torch.Tensor: log of the resulting point probabilities.
        """
        match (isinstance(support, torch.Tensor), isinstance(rate, torch.Tensor)):
            case (False, False):
                support = torch.tensor(support).float()
                rate = torch.tensor(rate).float()
            case (False, True):
                support = inferno.scalar(support, rate)
            case (True, False):
                rate = inferno.scalar(rate, support)

        return torch.special.xlogy(support, rate) - rate - torch.lgamma(rate + 1)

    @staticmethod
    def cdf(support: torch.Tensor | float, rate: torch.Tensor | float) -> torch.Tensor:
        r"""Computes the cumulative distribution function.

        .. math::
            P(K \leq k; \lambda) = e^{-\lambda} \sum_{j=0}^{\lfloor k \rfloor}
            \frac{\lambda^j}{j!}

        Args:
            support (torch.Tensor | float): number of occurances, :math:`k`.
            rate (torch.Tensor | float): expected rate of occurances, :math:`\lambda`.

        Returns:
            torch.Tensor: resulting cumulative probabilities.
        """
        match (isinstance(support, torch.Tensor), isinstance(rate, torch.Tensor)):
            case (False, False):
                support = torch.tensor(support).float()
                rate = torch.tensor(rate).float()
            case (False, True):
                support = inferno.scalar(support, rate)
            case (True, False):
                rate = inferno.scalar(rate, support)

        return torch.special.gammaincc(torch.floor(support + 1), rate)

    @staticmethod
    def logcdf(
        support: torch.Tensor | float, rate: torch.Tensor | float
    ) -> torch.Tensor:
        r"""Computes the natural logarithm of the cumulative distribution function.

        .. math::
            \log P(K \leq k; \lambda) = \log
            \frac{\Gamma (\lfloor k + 1 \rfloor, \lambda)}{\Gamma (\lambda)}

        Args:
            support (torch.Tensor | float): number of occurances, :math:`k`.
            rate (torch.Tensor | float): expected rate of occurances, :math:`\lambda`.

        Returns:
            torch.Tensor: log of the resulting cumulative probabilities.

        Note:
            At least one of ``support`` and ``rate`` must be a
            :py:class:`~torch.Tensor`.
        """
        return torch.log(Poisson.cdf(support, rate))

    @staticmethod
    def mean(rate: torch.Tensor | float) -> torch.Tensor:
        r"""Computes the expected value of the distribution.

        .. math::
            \text{E}[K \mid K \sim \text{Poisson}(\lambda)] = \lambda

        Args:
            rate (torch.Tensor | float): expected rate of occurances, :math:`\lambda`.

        Returns:
            torch.Tensor: mean of the distribution with given parameters.
        """
        if not isinstance(rate, torch.Tensor):
            rate = torch.tensor(rate).float()
        return rate

    @staticmethod
    def variance(rate: torch.Tensor | float) -> torch.Tensor:
        r"""Computes the variance of the distribution.

        .. math::
            \text{Var}[K \mid K \sim \text{Poisson}(\lambda)] = \lambda

        Args:
            rate (torch.Tensor | float): expected rate of occurances, :math:`\lambda`.

        Returns:
            torch.Tensor: variance of the distribution with given parameters.
        """
        if not isinstance(rate, torch.Tensor):
            rate = torch.tensor(rate).float()
        return rate


class Normal:
    r"""Sampling from and properties of the normal distribution.

    .. admonition:: Parameters

        :math:`\mu \in \mathbb{R}`, mean

        :math:`\sigma^2 \in \mathbb{R}_+^*`, variance

    .. admonition:: Support

        :math:`x \in \mathbb{R}`
    """

    @staticmethod
    def sample(
        loc: torch.Tensor | float,
        scale: torch.Tensor | float,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        r"""Samples random variates from a normal distribution.

        .. math::
            X \sim \mathcal{N}(\mu, \sigma^2)

        Args:
            loc (torch.Tensor | float): mean of the distribution, :math:`\mu`.
            scale (torch.Tensor | float): variance of the distribution, :math:`\sigma^2`.
            generator (torch.Generator | None, optional): pseudorandom number generator
                to use for sampling. Defaults to None.

        Returns:
            torch.Tensor: resulting random variates :math:`X`.
        """
        match (isinstance(loc, torch.Tensor), isinstance(scale, torch.Tensor)):
            case (False, False):
                loc = torch.tensor(loc).float()
                scale = torch.tensor(scale).float()
            case (False, True):
                loc = inferno.scalar(loc, scale)
            case (True, False):
                scale = inferno.scalar(scale, loc)

        return torch.normal(loc, torch.sqrt(scale), generator=generator)

    @staticmethod
    def pdf(
        support: torch.Tensor | float,
        loc: torch.Tensor | float,
        scale: torch.Tensor | float,
    ) -> torch.Tensor:
        r"""Computes the probability density function.

        .. math::
            P(X=x; \mu, \sigma^2) = \frac{1}{\sigma \sqrt{2 \pi}} \exp
            \left( - \frac{1}{2} \left( \frac{x - \mu}{\sigma} \right)^2 \right)

        Args:
            support (torch.Tensor | float): location of observation, :math:`x`.
            loc (torch.Tensor | float): mean of the distribution, :math:`\mu`.
            scale (torch.Tensor | float): variance of the distribution, :math:`\sigma^2`.

        Returns:
            torch.Tensor: resulting relative likelihoods.
        """
        match (
            isinstance(support, torch.Tensor),
            isinstance(loc, torch.Tensor),
            isinstance(scale, torch.Tensor),
        ):
            case (False, False, False):
                support = torch.tensor(support).float()
                loc = torch.tensor(loc).float()
                scale = torch.tensor(scale).float()
            case (False, False, True):
                support = inferno.scalar(support, scale)
                loc = inferno.scalar(loc, scale)
            case (False, True, False):
                support = inferno.scalar(support, loc)
                scale = inferno.scalar(scale, loc)
            case (False, True, True):
                support = inferno.scalar(support, loc)
            case (True, False, False):
                loc = inferno.scalar(loc, support)
                scale = inferno.scalar(scale, support)
            case (True, False, True):
                loc = inferno.scalar(loc, support)
            case (True, True, False):
                scale = inferno.scalar(scale, support)

        std = torch.sqrt(scale)
        return (1 / (std * math.sqrt(math.tau))) * torch.exp(
            -0.5 * ((support - loc) / scale) ** 2
        )

    @staticmethod
    def logpdf(
        support: torch.Tensor | float,
        loc: torch.Tensor | float,
        scale: torch.Tensor | float,
    ) -> torch.Tensor:
        r"""Computes the natural logarithm of the probability density function.

        .. math::
            \log P(X=x; \mu, \sigma^2) = - \log \sigma - \frac{1}{2} \left[ \log \pi
            - \left(\frac{\mu - x}{\sigma}\right)^2 \right]

        Args:
            support (torch.Tensor | float): location of observation, :math:`x`.
            loc (torch.Tensor | float): mean of the distribution, :math:`\mu`.
            scale (torch.Tensor | float): variance of the distribution, :math:`\sigma^2`.

        Returns:
            torch.Tensor: log of the resulting relative likelihoods.
        """
        return torch.log(Normal.pdf(support, loc, scale))

    @staticmethod
    def cdf(
        support: torch.Tensor | float,
        loc: torch.Tensor | float,
        scale: torch.Tensor | float,
    ) -> torch.Tensor:
        r"""Computes the cumulative distribution function.

        .. math::
            P(X \leq x; \mu, \sigma^2) = \frac{1}{2} \left[ 1 +
            \text{erf} \left( \frac{x - \mu}{\sigma \sqrt{2}} \right) \right]

        Args:
            support (torch.Tensor | float): location of observation, :math:`x`.
            loc (torch.Tensor | float): mean of the distribution, :math:`\mu`.
            scale (torch.Tensor | float): variance of the distribution, :math:`\sigma^2`.

        Returns:
            torch.Tensor: resulting cumulative probabilities.
        """
        match (
            isinstance(support, torch.Tensor),
            isinstance(loc, torch.Tensor),
            isinstance(scale, torch.Tensor),
        ):
            case (False, False, False):
                support = torch.tensor(support).float()
                loc = torch.tensor(loc).float()
                scale = torch.tensor(scale).float()
            case (False, False, True):
                support = inferno.scalar(support, scale)
                loc = inferno.scalar(loc, scale)
            case (False, True, False):
                support = inferno.scalar(support, loc)
                scale = inferno.scalar(scale, loc)
            case (False, True, True):
                support = inferno.scalar(support, loc)
            case (True, False, False):
                loc = inferno.scalar(loc, support)
                scale = inferno.scalar(scale, support)
            case (True, False, True):
                loc = inferno.scalar(loc, support)
            case (True, True, False):
                scale = inferno.scalar(scale, support)

        return 0.5 * (
            1 + torch.special.erf((support - loc) / (torch.sqrt(scale) * math.sqrt(2)))
        )

    @staticmethod
    def logcdf(
        support: torch.Tensor | float,
        loc: torch.Tensor | float,
        scale: torch.Tensor | float,
    ) -> torch.Tensor:
        r"""Computes the natural logarithm of the cumulative distribution function.

        .. math::
            \log P(X \leq x; \mu, \sigma^2) = \log \frac{1}{2} \left[ 1 +
            \text{erf} \left( \frac{x - \mu}{\sigma \sqrt{2}} \right) \right]

        Args:
            support (torch.Tensor | float): location of observation, :math:`x`.
            loc (torch.Tensor | float): mean of the distribution, :math:`\mu`.
            scale (torch.Tensor | float): variance of the distribution, :math:`\sigma^2`.

        Returns:
            torch.Tensor: log of the resulting cumulative probabilities.
        """
        return torch.log(Normal.cdf(support, loc, scale))

    @staticmethod
    def mean(loc: torch.Tensor | float) -> torch.Tensor:
        r"""Computes the expected value of the distribution.

        .. math::
            \text{E}[X \mid X \sim \mathcal{N}(\mu, \sigma^2)] = \mu

        Args:
            loc (torch.Tensor | float): mean of the distribution, :math:`\mu`.

        Returns:
            torch.Tensor: mean of the distribution with given parameters.
        """
        if not isinstance(loc, torch.Tensor):
            loc = torch.tensor(loc).float()
        return loc

    @staticmethod
    def variance(scale: torch.Tensor | float) -> torch.Tensor:
        r"""Computes the variance of the distribution.

        .. math::
            \text{Var}[X \mid X \sim \mathcal{N}(\mu, \sigma^2)] = \sigma^2

        Args:
            scale (torch.Tensor | float): variance of the distribution, :math:`\sigma^2`.

        Returns:
            torch.Tensor: variance of the distribution with given parameters.
        """
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale).float()
        return scale


class LogNormal:
    r"""Sampling from and properties of the log-normal distribution.

    The log-normal distribution is a continuous probability distribution derived
    from the normal distribution, specifically the log of the normal distribution.

    .. admonition:: Parameters

        :math:`\mu \in \mathbb{R}`, location

        :math:`\sigma \in \mathbb{R}_+^*`, scale

    .. admonition:: Support

        :math:`x \in \mathbb{R}_+^*`
    """

    @staticmethod
    def sample(
        loc: torch.Tensor | float,
        scale: torch.Tensor | float,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        r"""Samples random variates from a log-normal distribution.

        .. math::
            \log X \sim \mathcal{N}(\mu, \sigma)

        Args:
            loc (torch.Tensor): distribution location :math:`\mu`.
            scale (torch.Tensor): distribution scale :math:`\sigma`.
            generator (torch.Generator | None, optional): pseudorandom number generator
                to use for sampling. Defaults to None.

        Returns:
            torch.Tensor: resulting random variates :math:`X`.
        """
        return torch.exp(Normal.sample(loc, scale, generator=generator))

    @staticmethod
    def sample_mv(
        mean: torch.Tensor | float,
        variance: torch.Tensor | float,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        r"""Samples random variates with desired mean and stdev from a log-normal distribution.

        .. math::
            \log X \sim \mathcal{N}(\mu, \sigma)

        Where the parameters :math:`\mu` and :math:`\sigma^2` are computed as.

        .. math::

            \mu = \log \left[ \frac{\mu_X^2}{\sqrt{\mu_X^2 + \sigma_X^2}} \right]
            \qquad
            \sigma^2 = \log \left[ 1 + \frac{\sigma_X^2}{\mu_X^2} \right]

        Args:
            mean (torch.Tensor | float): desired sample mean, :math:`\mu_X`.
            variance (torch.Tensor | float): desired sample variance, :math:`\sigma_X^2`.
            generator (torch.Generator | None, optional): pseudorandom number generator
                to use for sampling. Defaults to None.

        Returns:
            torch.Tensor: resulting random variates :math:`X`.
        """
        match (isinstance(mean, torch.Tensor), isinstance(variance, torch.Tensor)):
            case (False, False):
                mean = torch.tensor(mean).float()
                variance = torch.tensor(variance).float()
            case (False, True):
                mean = inferno.scalar(mean, variance)
            case (True, False):
                variance = inferno.scalar(variance, mean)

        meansq = mean**2
        loc = torch.log(meansq / torch.sqrt(meansq + variance))
