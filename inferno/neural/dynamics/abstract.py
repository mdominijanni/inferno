from abc import ABC, abstractmethod, abstractproperty
import math

import torch
import torch.nn as nn


class AbstractDynamics(nn.Module, ABC):
    """Abstract class for populations of neurons behaving according to a common neuronal dynamic.
    """
    def __init__(self):
        nn.Module.__init__(self)

    @abstractproperty
    def size(self) -> int:
        """Number of expected inputs, excluding the batch axis.

        Raises:
            NotImplementedError: :py:attr:`size` is abstract and must be implemented by the subclass.

        Returns:
            int: size of the population of neurons, excluding the batch axis.
        """
        raise NotImplementedError(f"'AbstractDynamics.size' is abstract, {type(self).__name__} must implement the 'size' property")

    @abstractproperty
    def shape(self) -> tuple[int, ...]:
        """Number of neurons along each tensor dimension, excluding the batch axis.

        Raises:
            NotImplementedError: :py:attr:`shape` is abstract and must be implemented by the subclass.

        Returns:
            tuple[int, ...]: shape of the population of neurons, excluding the batch axis.
        """
        raise NotImplementedError(f"'AbstractDynamics.shape' is abstract, {type(self).__name__} must implement the 'shape' property")

    @abstractproperty
    def batched_shape(self) -> tuple[int, ...]:
        """Number of neurons along each tensor dimension, including the batch axis.

        Raises:
            NotImplementedError: :py:attr:`batched_shape` is abstract and must be implemented by the subclass.

        Returns:
            tuple[int, ...]: shape of the population of neurons, including the batch axis.
        """
        raise NotImplementedError(f"'AbstractDynamics.batched_shape' is abstract, {type(self).__name__} must implement the 'batched_shape' property")

    @abstractproperty
    def batch_size(self) -> int:
        """Number of expected inputs along the batch axis.

        :getter: returns the current number of expected inputs along the batch axis.
        :setter: changes the number of expected inputs along the batch axis, invokes :py:meth:`clear`.
        :type: int

        Raises:
            NotImplementedError: :py:attr:`batch_size` is abstract and must be implemented by the subclass.
        """
        raise NotImplementedError(f"'AbstractDynamics.batch_size' is abstract, {type(self).__name__} must implement the 'batch_size' property")

    @abstractmethod
    @batch_size.setter
    def batch_size(self, batch_size: int) -> None:
        raise NotImplementedError(f"'AbstractDynamics.batch_size' is abstract, {type(self).__name__} must implement the 'batch_size' property")

    @abstractmethod
    def clear(self, **kwargs) -> None:
        """Reinitializes the neurons' states.

        Raises:
            NotImplementedError: :py:meth:`clear` is abstract and must be implemented by the subclass.
        """
        raise NotImplementedError(f"'AbstractDynamics.clear()' is abstract, {type(self).__name__} must implement the 'clear' method")


class ShapeMixin(object):
    """Mixin providing functionality for shape and size properties with Dynamics objects.

    Args:
        shape (tuple[int, ...] | int): shape of the population of neurons specified as the lengths along tensor dimensions.
        batch_size (int): number of separate inputs to be passed along the batch (:math:`0^\\text{th}`) axis.
        batch_dependent_parameters (tuple[str, ...]) str | None: list of names of :py:class:`nn.Parameter` attributes which need to be deleted and rebuilt on change in batch size.

    Raises:
        ValueError: `batch_size` must be a positive integer.
    """
    def __init__(self, shape: tuple[int, ...] | int, batch_size: int, batch_dependent_parameters: tuple[str, ...] | str | None):
        if int(batch_size) < 1:
            raise ValueError(f"batch size must be at least 1, received {int(batch_size)}")
        self._batch_size = int(batch_size)
        self._shape = tuple([shape]) if type(shape) is int else tuple(shape)
        self._size = math.prod(self._shape)
        if batch_dependent_parameters is None:
            self.batch_dependent_parameters = ()
        elif isinstance(batch_dependent_parameters, str):
            self.batch_dependent_parameters = (batch_dependent_parameters,)
        else:
            self.batch_dependent_parameters = tuple(batch_dependent_parameters)

    @property
    def size(self) -> int:
        """Number of expected inputs, excluding the batch axis.

        Returns:
            int: size of the population of neurons, excluding the batch axis.
        """
        return self._size

    @property
    def shape(self) -> tuple[int, ...]:
        """Number of neurons along each tensor dimension, excluding the batch axis.

        Returns:
            tuple[int, ...]: shape of the population of neurons, excluding the batch axis.
        """
        return self._shape

    @property
    def batched_shape(self) -> tuple[int, ...]:
        """Number of neurons along each tensor dimension, including the batch axis.

        Returns:
            tuple[int, ...]: shape of the population of neurons, including the batch axis.
        """
        return (self._batch_size,) + self._shape

    @property
    def batch_size(self) -> int:
        """Number of expected inputs along the batch axis.

        :getter: returns the current number of expected inputs along the batch axis.
        :setter: changes the number of expected inputs along the batch axis, invokes :py:meth:`clear`.
        :type: int
        """
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int) -> None:
        if batch_size != self._batch_size:
            if int(batch_size) < 1:
                raise ValueError(f"batch size must be at least 1, received {int(batch_size)}")
            self._batch_size = int(batch_size)
            for attr in self.batch_dependent_parameters:
                delattr(self, attr)
                self.register_parameter(attr, nn.Parameter(torch.empty(self.batched_shape, requires_grad=False), False))
            self.clear()
