from . import constraints
from .base import DiscreteDistribution, ContinuousDistribution
from .. import astensors
from functools import partial
import math
import torch


_astensorsfloat = partial(astensors, conversion=lambda x: torch.tensor(x).float())


class Poisson(DiscreteDistribution):
    r"""Sampling from and properties of the Poisson distribution.

    The Poisson distribution is a discrete probability distribution used to express
    the probability of a given number of events, :math:`k`, occurring in a fixed amount
    of time, given an expected number of events, :math:`\lambda`.

    .. admonition:: Parameters

        :math:`\lambda \in \mathbb{R}_+^*`, rate

    .. admonition:: Support

        :math:`k \in \mathbb{N}_0`, count
    """

    @classmethod
    def validate(
        cls,
        rate: torch.Tensor | float | None = None,
        support: torch.Tensor | float | None = None,
    ) -> dict[str, bool | None]:
        r"""Tests if the arguments are valid for a Poisson distribution.

        Args:
            rate (torch.Tensor | float | None, optional): expected rate of occurrences,
                :math:`\lambda`. Defaults to ``None``.
            support (torch.Tensor | float | None, optional): number of occurrences,
                :math:`k`. Defaults to ``None``.

        Returns:
            dict[str, torch.Tensor | bool | None]: argument name and if it is valid,
            returned as a tensor of dtype ``torch.bool`` if a non-scalar tensor
            is given, None if not given.

        Note:
            This considers a rate of zero valid, although strictly not true for
            Poisson distributions. A Poisson distribution with zero rate is the
            degenerate distribution.
        """
        return {
            "rate": constraints.nonnegreal(rate),
            "support": constraints.nonneginteger(support),
        }

    @classmethod
    def sample(
        cls, rate: torch.Tensor | float, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        r"""Samples random variates from a Poisson distribution.

        .. math::
            K \sim \text{Poisson}(\lambda)

        Args:
            rate (torch.Tensor | float): expected rate of occurrences, :math:`\lambda`.
            generator (torch.Generator | None, optional): pseudorandom number generator
                to use for sampling. Defaults to ``None``.

        Returns:
            torch.Tensor: resulting random variates, :math:`K`.

        Caution:
            This calls :py:func:`torch.poisson` which as of PyTorch 2.1 does not support
            computation on Metal Performance Shaders. Compensate accordingly.
        """
        rate = _astensorsfloat(rate)
        return torch.poisson(rate, generator=generator)

    @classmethod
    def pmf(
        cls, support: torch.Tensor | float, rate: torch.Tensor | float
    ) -> torch.Tensor:
        r"""Computes the probability mass function.

        .. math::
            P(K=k; \lambda) = \frac{\lambda^k e^{-k}}{k!}

        Args:
            support (torch.Tensor | float): number of occurrences, :math:`k`.
            rate (torch.Tensor | float): expected rate of occurrences, :math:`\lambda`.

        Returns:
            torch.Tensor: resulting point probabilities.
        """
        return torch.exp(Poisson.logpmf(support, rate))

    @classmethod
    def logpmf(
        cls, support: torch.Tensor | float, rate: torch.Tensor | float
    ) -> torch.Tensor:
        r"""Computes the natural logarithm of the probability mass function.

        .. math::
            \log P(K=k; \lambda) = k \log \lambda - \lambda - \log(k!)

        Args:
            support (torch.Tensor | float): number of occurrences, :math:`k`.
            rate (torch.Tensor | float): expected rate of occurrences, :math:`\lambda`.

        Returns:
            torch.Tensor: log of the resulting point probabilities.
        """
        support, rate = _astensorsfloat(support, rate)
        return torch.special.xlogy(support, rate) - rate - torch.lgamma(rate + 1)

    @classmethod
    def cdf(
        cls, support: torch.Tensor | float, rate: torch.Tensor | float
    ) -> torch.Tensor:
        r"""Computes the cumulative distribution function.

        .. math::
            P(K \leq k; \lambda) =
            \frac{\Gamma (\lfloor k + 1 \rfloor, \lambda)}{\Gamma (\lambda)}

        Args:
            support (torch.Tensor | float): number of occurrences, :math:`k`.
            rate (torch.Tensor | float): expected rate of occurrences, :math:`\lambda`.

        Returns:
            torch.Tensor: resulting cumulative probabilities.
        """
        support, rate = _astensorsfloat(support, rate)
        return torch.special.gammaincc(torch.floor(support + 1), rate)

    @classmethod
    def logcdf(
        cls, support: torch.Tensor | float, rate: torch.Tensor | float
    ) -> torch.Tensor:
        r"""Computes the natural logarithm of the cumulative distribution function.

        .. math::
            \log P(K \leq k; \lambda) = \log
            \frac{\Gamma (\lfloor k + 1 \rfloor, \lambda)}{\Gamma (\lambda)}

        Args:
            support (torch.Tensor | float): number of occurrences, :math:`k`.
            rate (torch.Tensor | float): expected rate of occurrences, :math:`\lambda`.

        Returns:
            torch.Tensor: log of the resulting cumulative probabilities.
        """
        return torch.log(cls.cdf(support, rate))

    @classmethod
    def mean(cls, rate: torch.Tensor | float) -> torch.Tensor:
        r"""Computes the expected value of the distribution.

        .. math::
            \text{E}[K \mid K \sim \text{Poisson}(\lambda)] = \lambda

        Args:
            rate (torch.Tensor | float): expected rate of occurrences, :math:`\lambda`.

        Returns:
            torch.Tensor: mean of the distribution with given parameters.
        """
        rate = _astensorsfloat(rate)
        return rate

    @classmethod
    def variance(cls, rate: torch.Tensor | float) -> torch.Tensor:
        r"""Computes the variance of the distribution.

        .. math::
            \text{Var}[K \mid K \sim \text{Poisson}(\lambda)] = \lambda

        Args:
            rate (torch.Tensor | float): expected rate of occurrences, :math:`\lambda`.

        Returns:
            torch.Tensor: variance of the distribution with given parameters.
        """
        rate = _astensorsfloat(rate)
        return rate


class Normal(ContinuousDistribution):
    r"""Sampling from and properties of the normal distribution.

    .. admonition:: Parameters

        :math:`\mu \in \mathbb{R}`, mean

        :math:`\sigma \in \mathbb{R}_+^*`, standard deviation

    .. admonition:: Support

        :math:`x \in \mathbb{R}`
    """

    @classmethod
    def validate(
        cls,
        loc: torch.Tensor | float | None = None,
        scale: torch.Tensor | float | None = None,
        support: torch.Tensor | float | None = None,
    ) -> dict[str, bool | None]:
        r"""Tests if the arguments are valid for a Normal distribution.

        Args:
            loc (torch.Tensor | float): mean of the distribution, :math:`\mu`.
            scale (torch.Tensor | float | None, optional): standard deviation of the
                distribution, :math:`\sigma`. Defaults to ``None``.
            support (torch.Tensor | float | None, optional): location of observation,
                :math:`x`. Defaults to ``None``.

        Returns:
            dict[str, torch.Tensor | bool | None]: argument name and if it is valid,
            returned as a tensor of dtype ``torch.bool`` if a non-scalar tensor
            is given, ``None`` if not given.
        """
        return {
            "loc": constraints.real(loc),
            "scale": constraints.posreal(scale),
            "support": constraints.real(support),
        }

    @classmethod
    def params_mv(
        cls, mean: torch.Tensor | float, variance: torch.Tensor | float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Computes parameters of the normal distribution targeting a given mean and variance.

        Computes the location :math:`\mu` and scale :math:`\sigma` as follows.

        .. math::

            \mu = \mu_X
            \qquad
            \sigma = \sqrt{\sigma_X^2}

        Args:
            mean (torch.Tensor | float): desired mean, :math:`\mu_X`.
            variance (torch.Tensor | float): desired variance, :math:`\sigma_X^2`.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: tuple of the corresponding
            ``loc`` and ``scale``.
        """
        mean, variance = _astensorsfloat(mean, variance)
        return mean, torch.sqrt(variance)

    @classmethod
    def sample(
        cls,
        loc: torch.Tensor | float,
        scale: torch.Tensor | float,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        r"""Samples random variates from a normal distribution.

        .. math::
            X \sim \mathcal{N}(\mu, \sigma)

        Args:
            loc (torch.Tensor | float): mean of the distribution, :math:`\mu`.
            scale (torch.Tensor | float): standard deviation of the distribution,
                :math:`\sigma`.
            generator (torch.Generator | None, optional): pseudorandom number generator
                to use for sampling. Defaults to ``None``.

        Returns:
            torch.Tensor: resulting random variates :math:`X`.
        """
        loc, scale = _astensorsfloat(loc, scale)
        return torch.normal(loc, scale, generator=generator)

    @classmethod
    def sample_mv(
        cls,
        mean: torch.Tensor | float,
        variance: torch.Tensor | float,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        r"""Samples random variates with desired mean and variance from a normal distribution.

        .. math::
            \log X \sim \mathcal{N}\left(\mu, \sqrt{\sigma_X^2}\right)

        Args:
            mean (torch.Tensor | float): target sample mean, :math:`\mu_X`.
            variance (torch.Tensor | float): target sample variance, :math:`\sigma_X^2`.
            generator (torch.Generator | None, optional): pseudorandom number generator
                to use for sampling. Defaults to ``None``.

        Returns:
            torch.Tensor: resulting random variates :math:`X`.
        """
        return cls.sample(*cls.params_mv(mean, variance), generator=generator)

    @classmethod
    def pdf(
        cls,
        support: torch.Tensor | float,
        loc: torch.Tensor | float,
        scale: torch.Tensor | float,
    ) -> torch.Tensor:
        r"""Computes the probability density function.

        .. math::
            P(X=x; \mu, \sigma) = \frac{1}{\sigma \sqrt{2 \pi}} \exp
            \left( - \frac{1}{2} \left( \frac{x - \mu}{\sigma} \right)^2 \right)

        Args:
            support (torch.Tensor | float): location of observation, :math:`x`.
            loc (torch.Tensor | float): mean of the distribution, :math:`\mu`.
            scale (torch.Tensor | float): standard deviation of the distribution,
                :math:`\sigma`.

        Returns:
            torch.Tensor: resulting relative likelihoods.
        """
        support, loc, scale = _astensorsfloat(support, loc, scale)
        return (1 / (scale * math.sqrt(math.tau))) * torch.exp(
            -0.5 * ((support - loc) / scale) ** 2
        )

    @classmethod
    def logpdf(
        cls,
        support: torch.Tensor | float,
        loc: torch.Tensor | float,
        scale: torch.Tensor | float,
    ) -> torch.Tensor:
        r"""Computes the natural logarithm of the probability density function.

        .. math::
            \log P(X=x; \mu, \sigma) = - \log \sigma - \frac{1}{2} \left[ \log 2 \pi
            + \left(\frac{\mu - x}{\sigma}\right)^2 \right]

        Args:
            support (torch.Tensor | float): location of observation, :math:`x`.
            loc (torch.Tensor | float): mean of the distribution, :math:`\mu`.
            scale (torch.Tensor | float): standard deviation of the distribution,
                :math:`\sigma`.

        Returns:
            torch.Tensor: log of the resulting relative likelihoods.
        """
        return torch.log(cls.pdf(support, loc, scale))

    @classmethod
    def cdf(
        cls,
        support: torch.Tensor | float,
        loc: torch.Tensor | float,
        scale: torch.Tensor | float,
    ) -> torch.Tensor:
        r"""Computes the cumulative distribution function.

        .. math::
            P(X \leq x; \mu, \sigma) = \frac{1}{2} \left[ 1 +
            \text{erf} \left( \frac{x - \mu}{\sigma \sqrt{2}} \right) \right]

        Args:
            support (torch.Tensor | float): location of observation, :math:`x`.
            loc (torch.Tensor | float): mean of the distribution, :math:`\mu`.
            scale (torch.Tensor | float): standard deviation of the distribution,
                :math:`\sigma`.

        Returns:
            torch.Tensor: resulting cumulative probabilities.
        """
        support, loc, scale = _astensorsfloat(support, loc, scale)
        return 0.5 * (1 + torch.special.erf((support - loc) / (scale * math.sqrt(2))))

    @classmethod
    def logcdf(
        cls,
        support: torch.Tensor | float,
        loc: torch.Tensor | float,
        scale: torch.Tensor | float,
    ) -> torch.Tensor:
        r"""Computes the natural logarithm of the cumulative distribution function.

        .. math::
            \log P(X \leq x; \mu, \sigma) = \log \frac{1}{2} \left[ 1 +
            \text{erf} \left( \frac{x - \mu}{\sigma \sqrt{2}} \right) \right]

        Args:
            support (torch.Tensor | float): location of observation, :math:`x`.
            loc (torch.Tensor | float): mean of the distribution, :math:`\mu`.
            scale (torch.Tensor | float): standard deviation of the distribution,
                :math:`\sigma`.

        Returns:
            torch.Tensor: log of the resulting cumulative probabilities.
        """
        return torch.log(cls.cdf(support, loc, scale))

    @classmethod
    def mean(cls, loc: torch.Tensor | float) -> torch.Tensor:
        r"""Computes the expected value of the distribution.

        .. math::
            \text{E}[X \mid X \sim \mathcal{N}(\mu, \sigma)] = \mu

        Args:
            loc (torch.Tensor | float): mean of the distribution, :math:`\mu`.

        Returns:
            torch.Tensor: mean of the distribution with given parameters.
        """
        loc = _astensorsfloat(loc)
        return loc

    @classmethod
    def variance(cls, scale: torch.Tensor | float) -> torch.Tensor:
        r"""Computes the variance of the distribution.

        .. math::
            \text{Var}[X \mid X \sim \mathcal{N}(\mu, \sigma)] = \sigma^2

        Args:
            scale (torch.Tensor | float): standard deviation of the distribution,
                :math:`\sigma`.

        Returns:
            torch.Tensor: variance of the distribution with given parameters.
        """
        scale = _astensorsfloat(scale)
        return scale**2


class LogNormal(ContinuousDistribution):
    r"""Sampling from and properties of the log-normal distribution.

    The log-normal distribution is a continuous probability distribution derived
    from the normal distribution, specifically the log of the normal distribution.

    .. admonition:: Parameters

        :math:`\mu \in \mathbb{R}`, location

        :math:`\sigma \in \mathbb{R}_+^*`, scale

    .. admonition:: Support

        :math:`x \in \mathbb{R}_+^*`
    """

    @classmethod
    def validate(
        cls,
        loc: torch.Tensor | float | None = None,
        scale: torch.Tensor | float | None = None,
        support: torch.Tensor | float | None = None,
    ) -> dict[str, bool | None]:
        r"""Tests if the arguments are valid for a Normal distribution.

        Args:
            loc (torch.Tensor | float | None, optional): location of the distribution,
                :math:`\mu`. Defaults to ``None``.
            scale (torch.Tensor | float | None, optional): scale of the distribution,
                :math:`\sigma`. Defaults to ``None``.
            support (torch.Tensor | float | None, optional): support of observation,
                :math:`x`. Defaults to ``None``.

        Returns:
            dict[str, torch.Tensor | bool | None]: argument name and if it is valid,
            returned as a tensor of dtype ``torch.bool`` if a non-scalar tensor
            is given, ``None`` if not given.
        """
        return {
            "loc": constraints.real(loc),
            "scale": constraints.posreal(scale),
            "support": constraints.posreal(support),
        }

    @classmethod
    def params_mv(
        cls, mean: torch.Tensor | float, variance: torch.Tensor | float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Computes parameters of the log-normal distribution targeting a given mean and variance.

        Computes the location :math:`\mu` and scale :math:`\sigma` as follows.

        .. math::

            \mu = \log \left[ \frac{\mu_X^2}{\sqrt{\mu_X^2 + \sigma_X^2}} \right]
            \qquad
            \sigma = \sqrt{\log \left[ 1 + \frac{\sigma_X^2}{\mu_X^2} \right]}

        Args:
            mean (torch.Tensor | float): desired mean, :math:`\mu_X`.
            variance (torch.Tensor | float): desired variance, :math:`\sigma_X^2`.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: tuple of the corresponding ``loc`` and ``scale``.
        """
        mean, variance = _astensorsfloat(mean, variance)

        meansq = mean**2
        loc = torch.log(meansq / torch.sqrt(meansq + variance))
        scale = torch.sqrt(torch.log(1 + variance / meansq))

        return loc, scale

    @classmethod
    def sample(
        cls,
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
                to use for sampling. Defaults to ``None``.

        Returns:
            torch.Tensor: resulting random variates :math:`X`.
        """
        return torch.exp(Normal.sample(loc, scale, generator=generator))

    @classmethod
    def sample_mv(
        cls,
        mean: torch.Tensor | float,
        variance: torch.Tensor | float,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        r"""Samples random variates with desired mean and variance from a log-normal distribution.

        .. math::
            \log X \sim \mathcal{N}
            \left(
            \log \left[ \frac{\mu_X^2}{\sqrt{\mu_X^2 + \sigma_X^2}} \right],
            \sqrt{\log \left[ 1 + \frac{\sigma_X^2}{\mu_X^2} \right]}
            \right)

        Args:
            mean (torch.Tensor | float): target sample mean, :math:`\mu_X`.
            variance (torch.Tensor | float): target sample variance, :math:`\sigma_X^2`.
            generator (torch.Generator | None, optional): pseudorandom number generator
                to use for sampling. Defaults to ``None``.

        Returns:
            torch.Tensor: resulting random variates :math:`X`.
        """
        return cls.sample(*cls.params_mv(mean, variance), generator=generator)

    @classmethod
    def pdf(
        cls,
        support: torch.Tensor | float,
        loc: torch.Tensor | float,
        scale: torch.Tensor | float,
    ) -> torch.Tensor:
        r"""Computes the probability density function.

        .. math::
            P(X=x; \mu, \sigma) = \frac{1}{x \sigma \sqrt{2 \pi}} \exp
            \left( - \left( \frac{\log x - \mu}{\sigma\sqrt{2}} \right)^2 \right)

        Args:
            support (torch.Tensor | float): location of observation, :math:`x`.
            loc (torch.Tensor | float): mean of the distribution, :math:`\mu`.
            scale (torch.Tensor | float): standard deviation of the distribution,
                :math:`\sigma`.

        Returns:
            torch.Tensor: resulting relative likelihoods.
        """
        return torch.exp(cls.logpdf(support, loc, scale))

    @classmethod
    def logpdf(
        cls,
        support: torch.Tensor | float,
        loc: torch.Tensor | float,
        scale: torch.Tensor | float,
    ) -> torch.Tensor:
        r"""Computes the natural logarithm of the probability density function.

        .. math::
            \log P(X=x; \mu, \sigma) = - \log \sigma - \log x - \frac{1}{2}
            \left[ \log 2 \pi
            + \left(\frac{\mu - \log x}{\sigma}\right)^2 \right]

        Args:
            support (torch.Tensor | float): location of observation, :math:`x`.
            loc (torch.Tensor | float): mean of the distribution, :math:`\mu`.
            scale (torch.Tensor | float): standard deviation of the distribution,
                :math:`\sigma`.

        Returns:
            torch.Tensor: log of the resulting relative likelihoods.
        """
        support, loc, scale = _astensorsfloat(support, loc, scale)

        logsupport = torch.log(support)
        return (
            -torch.log(scale)
            - logsupport
            - 0.5 * (math.log(math.tau) + ((loc - logsupport) / scale) ** 2)
        )

    @classmethod
    def cdf(
        cls,
        support: torch.Tensor | float,
        loc: torch.Tensor | float,
        scale: torch.Tensor | float,
    ) -> torch.Tensor:
        r"""Computes the cumulative distribution function.

        .. math::
            P(X \leq x; \mu, \sigma) = \frac{1}{2} \left[ 1 +
            \text{erf} \left( \frac{\log x - \mu}{\sigma \sqrt{2}} \right) \right]

        Args:
            support (torch.Tensor | float): location of observation, :math:`x`.
            loc (torch.Tensor | float): mean of the distribution, :math:`\mu`.
            scale (torch.Tensor | float): standard deviation of the distribution,
                :math:`\sigma`.

        Returns:
            torch.Tensor: resulting cumulative probabilities.
        """
        support, loc, scale = _astensorsfloat(support, loc, scale)
        return Normal.cdf(torch.log(support), loc, scale)

    @classmethod
    def logcdf(
        cls,
        support: torch.Tensor | float,
        loc: torch.Tensor | float,
        scale: torch.Tensor | float,
    ) -> torch.Tensor:
        r"""Computes the natural logarithm of the cumulative distribution function.

        .. math::
            \log P(X \leq x; \mu, \sigma) = \log \frac{1}{2} \left[ 1 +
            \text{erf} \left( \frac{\log x - \mu}{\sigma \sqrt{2}} \right) \right]

        Args:
            support (torch.Tensor | float): location of observation, :math:`x`.
            loc (torch.Tensor | float): mean of the distribution, :math:`\mu`.
            scale (torch.Tensor | float): standard deviation of the distribution,
                :math:`\sigma`.

        Returns:
            torch.Tensor: resulting cumulative probabilities.
        """
        return torch.log(cls.logcdf(support, loc, scale))

    @classmethod
    def mean(
        cls, loc: torch.Tensor | float, scale: torch.Tensor | float
    ) -> torch.Tensor:
        r"""Computes the expected value of the distribution.

        .. math::
            \text{E}[X \mid X \sim \text{LogNormal}(\mu, \sigma)] =
            \exp\left( \mu + \frac{\sigma^2}{2} \right)

        Args:
            loc (torch.Tensor | float): distribution location :math:`\mu`.
            scale (torch.Tensor | float): distribution scale :math:`\sigma`.

        Returns:
            torch.Tensor: mean of the distribution with given parameters.
        """
        loc, scale = _astensorsfloat(loc, scale)
        return torch.exp(loc + scale**2 / 2)

    @classmethod
    def variance(
        cls, loc: torch.Tensor | float, scale: torch.Tensor | float
    ) -> torch.Tensor:
        r"""Computes the variance of the distribution.

        .. math::
            \text{Var}[X \mid X \sim \text{LogNormal}(\mu, \sigma)] =
            \left( \exp\left( \sigma^2 \right) - 1 \right)
            \exp\left( 2 \mu + \sigma^2 \right)

        Args:
            loc (torch.Tensor): distribution location :math:`\mu`.
            scale (torch.Tensor): distribution scale :math:`\sigma`.

        Returns:
            torch.Tensor: variance of the distribution with given parameters.
        """
        loc, scale = astensors(loc, scale, conversion=lambda x: torch.tensor(x).float())
        scalesq = scale**2
        return torch.special.expm1(scalesq) * torch.exp(2 * loc + scalesq)
