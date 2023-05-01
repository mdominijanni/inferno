from abc import ABC, abstractmethod, abstractproperty

import torch
import torch.nn as nn


class AbstractConnection(nn.Module, ABC):
    """Abstract class for a weighted mapping from some input to a population of neurons.
    """
    def __init__(self):
        nn.Module.__init__(self)

    @abstractproperty
    def weight(self) -> torch.Tensor:
        """Connection weights for the connection object.

        :getter: returns the current connection weights.
        :setter: sets the current connection weights.
        :type: torch.Tensor

        Raises:
            NotImplementedError: :py:attr:`weight` is abstract and must be implemented by the subclass.
        """
        raise NotImplementedError(f"'AbstractConnection.weight' is abstract, {type(self).__name__} must implement the 'weight' property")

    @abstractmethod
    @weight.setter
    def weight(self, value: torch.Tensor) -> None:
        raise NotImplementedError(f"'AbstractConnection.weight' is abstract, {type(self).__name__} must implement the 'weight' property")

    @abstractmethod
    def reshape_outputs(self, outputs: torch.Tensor) -> torch.Tensor:
        """Reshapes a postsynaptic tensor, for dimensional compatibility with like presynaptic tensors.

        Used for updates by various learning algorithms, this output is shaped for compatibility with batch matrix multiplication operations.

        Args:
            outputs (torch.Tensor): like postsynaptic tensor to reshape.

        Raises:
            NotImplementedError: :py:meth:`reshape_outputs` is abstract and must be implemented by the subclass.

        Returns:
            torch.Tensor: reshaped form of the postsynaptic tensor.
        """
        raise NotImplementedError(f"'AbstractConnection.reshape_outputs()' is abstract, {type(self).__name__} must implement the 'reshape_outputs' method")

    @abstractmethod
    def reshape_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """Reshapes a presynaptic tensor, for dimensional compatibility with like postsynaptic tensors.

        Used for updates by various learning algorithms, this output is shaped for compatibility with batch matrix multiplication operations.

        Args:
            inputs (torch.Tensor): like presynaptic tensor to reshape.

        Raises:
            NotImplementedError: :py:meth:`reshape_inputs` is abstract and must be implemented by the subclass.

        Returns:
            torch.Tensor: reshaped form of the presynaptic tensor.
        """
        raise NotImplementedError(f"'AbstractConnection.reshape_inputs()' is abstract, {type(self).__name__} must implement the 'reshape_inputs' method")

    @abstractmethod
    def inputs_as_receptive_areas(
        self,
        inputs: torch.Tensor
    ) -> torch.Tensor:
        """Builds a tensor representing the receptive areas for each corresponding output.

        The receptive area is a generalization of the idea of `im2col` or `unfold` (albeit transposed).
        For a connection, let the batch size be :math:`B`, the number of outputs be :math:`N`, and the
        number of inputs which contributed to each output to be :math:`M`. The results in the following shape.

        .. math:
            B \\times N \\times M

        Where the values stored are the inputs, duplicated and reshaped as necessary.

        Args:
            inputs (torch.Tensor): inputs for which to build the receptive area.

        Raises:
            NotImplementedError: :py:meth:`inputs_as_receptive_areas` is abstract and must be implemented by the subclass.

        Returns:
            torch.Tensor: resulting tensor representing the receptive areas of the provided inputs on this connection.
        """
        raise NotImplementedError(f"'AbstractConnection.inputs_as_receptive_areas()' is abstract, {type(self).__name__} must implement the 'inputs_as_receptive_areas' method")

    @abstractmethod
    def update_weight_add(
        self,
        update: torch.Tensor,
        weight_min: float | None = None,
        weight_max: float | None = None,
    ) -> None:
        """Applies an additive update to the connection weights.

        Args:
            update (torch.Tensor): The update to apply.
            weight_min (float | None, optional): Minimum allowable weight values. Defaults to None.
            weight_max (float | None, optional): Maximum allowable weight values. Defaults to None.

        Raises:
            NotImplementedError: :py:meth:`update_weight_additive` is abstract and must be implemented by the subclass.
        """
        raise NotImplementedError(f"'AbstractConnection.update_weight_additive()' is abstract, {type(self).__name__} must implement the 'update_weight_additive' method")

    @abstractmethod
    def update_weight_potdep(
        self,
        add_update: torch.Tensor,
        sub_update: torch.Tensor,
        weight_min: float | None = None,
        weight_max: float | None = None,
        bounding_mode: str | None = None,
    ) -> None:
        """Applies both a potentiation and a depression update to the connection weights, allows for advanced weight bounding.

        Args:
            add_update (torch.Tensor): The potentiation update to apply.
            sub_update (torch.Tensor): The depression update to apply.
            weight_min (float | None, optional): Minimum allowable weight values. Defaults to None.
            weight_min (float | None, optional): Maximum allowable weight values. Defaults to None.
            bounding_mode (str | None, optional): Weight bounding to use. Defaults to None.

        .. note::
            There are three weight bounding modes, other than `None`. The bounding mode 'hard' multiplies the potentiation update term
            by :math:`Θ(w_{max} - w)` and the depression update term by :math:`Θ(w - w_{min})`, where :math:`Θ` is the heaviside step function.
            The bounding mode 'soft' multiplies the potentiation update term by :math:`w_{max} - w` and the depression update term by :math:`w - w_{min}`.
            The bounding mode 'clamp' sets any weight values greater than the specified maximum to the maximum, and any less than the specified minimum to the minimum.

        Raises:
            NotImplementedError: :py:meth:`update_weight_potdep` is abstract and must be implemented by the subclass.
        """
        raise NotImplementedError(f"'AbstractConnection.update_weight_potdep()' is abstract, {type(self).__name__} must implement the 'update_weight_potdep' method")


class AbstractDelayedConnection(AbstractConnection):
    """Abstract class for a weighted, time-delayed mapping from some input to a population of neurons.
    """
    def __init__(self):
        AbstractConnection.__init__(self)

    @abstractproperty
    def delay_max(self) -> int:
        """Maximum permissible delay.

        Returns:
            int: maximum delay to apply to any inputs, in number of time steps.
        """
        return self._delay_max

    @abstractproperty
    def delay(self) -> torch.Tensor:
        """Connection delays for the connection object.

        :getter: returns the current connection delays.
        :setter: sets the current connection delays.
        :type: torch.Tensor

        Raises:
            NotImplementedError: :py:attr:`delay` is abstract and must be implemented by the subclass.
        """
        raise NotImplementedError(f"'AbstractDelayedConnection.delay' is abstract, {type(self).__name__} must implement the 'delay' property")

    @abstractmethod
    @delay.setter
    def delay(self, value: torch.Tensor) -> None:
        raise NotImplementedError

    @abstractmethod
    def clear(self, **kwargs) -> None:
        """Reinitializes the input history.

        Raises:
            NotImplementedError: :py:meth:`clear` is abstract and must be implemented by the subclass.
        """
        raise NotImplementedError(f"'AbstractDelayedConnection.clear()' is abstract, {type(self).__name__} must implement the 'clear' method")

    @abstractmethod
    def delays_as_receptive_areas(self) -> torch.Tensor:
        """Returns the delays of the layer, stuctured by receptive area.

        The resulting shape of this will be as follows.

        .. math:
            B \\times N

        Where :math:`B` is the batch size and :math:`N` is the number of parameters.

        Returns:
            torch.Tensor: The learned delays, structured by its receptive area
        """
        raise NotImplementedError(f"'AbstractDelayedConnection.receptive_areas()' is abstract, {type(self).__name__} must implement the 'delays_as_receptive_area' method")

    @abstractmethod
    def update_delay_add(
        self,
        update: torch.Tensor,
    ) -> None:
        """Applies an additive update to the learned delays

        Args:
            update (torch.Tensor): The update to apply.

        Raises:
            NotImplementedError: :py:meth:`update_delay_add` is abstract and must be implemented by the subclass.
        """
        raise NotImplementedError(f"'AbstractDelayedConnection.update_delay_add()' is abstract, {type(self).__name__} must implement the 'update_delay_add' method")

    @abstractmethod
    def update_delay_ra(
        self,
        update: torch.Tensor
    ) -> None:
        """Applies an additive update, in the form of a receptive area, to the connection delays.

        Args:
            update (torch.Tensor): The update to apply.

        Raises:
            NotImplementedError: :py:meth:`update_delay_by_receptive_area` is abstract and must be implemented by the subclass.

        .. note:
            A receptive area matrix is one in which each row corresponds to a given output and the elements in that row correspond to the inputs
            which compose said output. For example, in a convolutional network, this corresponds to the transpose of the im2col matrix.
        """
        raise NotImplementedError(f"'AbstractDelayedConnection.update_delay_by_receptive_area()' is abstract, {type(self).__name__} must implement the 'update_delay' method")
