from abc import ABC, abstractmethod
import torch


class Distribution(ABC):
    r"""Base class for representing probability distributions."""

    @classmethod
    @abstractmethod
    def validate(cls, *args, **kwargs) -> dict[str, torch.Tensor | bool | None]:
        r"""Tests if the arguments are valid for the distribution.

        Returns:
            dict[str, torch.Tensor | bool | None]: argument name and if it is valid,
            returned as a tensor of dtype ``torch.bool`` if a non-scalar tensor
            is given, ``None`` if not given.

        Raises:
            NotImplementedError: must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"the method 'validate' must be implemented by the subclass {cls.__name__}."
        )

    @classmethod
    @abstractmethod
    def sample(cls, *args, **kwargs) -> torch.Tensor:
        r"""Samples random variates from the distribution.

        Returns:
            torch.Tensor: resulting random variates.

        Raises:
            NotImplementedError: must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"the method 'sample' must be implemented by the subclass {cls.__name__}."
        )

    @classmethod
    @abstractmethod
    def mean(cls, *args, **kwargs) -> torch.Tensor:
        r"""Computes the expected value of the distribution.

        Returns:
            torch.Tensor: mean of the distribution with given parameters.

        Raises:
            NotImplementedError: must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"the method 'mean' must be implemented by the subclass {cls.__name__}."
        )

    @classmethod
    @abstractmethod
    def variance(cls, *args, **kwargs) -> torch.Tensor:
        r"""Computes the variance of the distribution.

        Returns:
            torch.Tensor: variance of the distribution with given parameters.

        Raises:
            NotImplementedError: must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"the method 'variance' must be implemented by the subclass {cls.__name__}."
        )


class DiscreteDistribution(Distribution):
    r"""Base class for representing discrete probability distributions."""

    @classmethod
    @abstractmethod
    def pmf(cls, *args, **kwargs) -> torch.Tensor:
        r"""Computes the probability mass function.

        Returns:
            torch.Tensor: resulting point probabilities.

        Raises:
            NotImplementedError: must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"the method 'pmf' must be implemented by the subclass {cls.__name__}."
        )

    @classmethod
    @abstractmethod
    def logpmf(cls, *args, **kwargs) -> torch.Tensor:
        r"""Computes the natural logarithm of the probability mass function.

        Returns:
            torch.Tensor: log of the resulting point probabilities.

        Raises:
            NotImplementedError: must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"the method 'logpmf' must be implemented by the subclass {cls.__name__}."
        )

    @classmethod
    @abstractmethod
    def cdf(cls, *args, **kwargs) -> torch.Tensor:
        r"""Computes the cumulative distribution function.

        Returns:
            torch.Tensor: resulting cumulative probabilities.

        Raises:
            NotImplementedError: must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"the method 'cdf' must be implemented by the subclass {cls.__name__}."
        )

    @classmethod
    @abstractmethod
    def logcdf(cls, *args, **kwargs) -> torch.Tensor:
        r"""Computes the natural logarithm of the cumulative distribution function.

        Returns:
            torch.Tensor: log of the resulting cumulative probabilities.

        Raises:
            NotImplementedError: must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"the method 'logcdf' must be implemented by the subclass {cls.__name__}."
        )


class ContinuousDistribution(Distribution):
    r"""Base class for representing continuous probability distributions."""

    @classmethod
    @abstractmethod
    def pdf(cls, *args, **kwargs) -> torch.Tensor:
        r"""Computes the probability density function.

        Returns:
            torch.Tensor: resulting relative likelihoods.

        Raises:
            NotImplementedError: must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"the method 'pmf' must be implemented by the subclass {cls.__name__}."
        )

    @classmethod
    @abstractmethod
    def logpdf(cls, *args, **kwargs) -> torch.Tensor:
        r"""Computes the natural logarithm of the probability density function.

        Returns:
            torch.Tensor: log of the resulting relative likelihoods.

        Raises:
            NotImplementedError: must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"the method 'logpmf' must be implemented by the subclass {cls.__name__}."
        )

    @classmethod
    @abstractmethod
    def cdf(cls, *args, **kwargs) -> torch.Tensor:
        r"""Computes the cumulative distribution function.

        Returns:
            torch.Tensor: resulting cumulative probabilities.

        Raises:
            NotImplementedError: must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"the method 'cdf' must be implemented by the subclass {cls.__name__}."
        )

    @classmethod
    @abstractmethod
    def logcdf(cls, *args, **kwargs) -> torch.Tensor:
        r"""Computes the natural logarithm of the cumulative distribution function.

        Returns:
            torch.Tensor: log of the resulting cumulative probabilities.

        Raises:
            NotImplementedError: must be implemented by the subclass.
        """
        raise NotImplementedError(
            f"the method 'logcdf' must be implemented by the subclass {cls.__name__}."
        )
