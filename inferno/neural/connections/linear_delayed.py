from functools import partial
from typing import Any, Callable

import torch
import torch.nn as nn

from einops._torch_specific import allow_ops_in_compiled_graph
allow_ops_in_compiled_graph()
from einops import rearrange as einrearrange

from inferno.neural.connections.abstract import AbstractDelayedConnection


class DenseDelayedConnection(AbstractDelayedConnection):
    """Connection object with trainable delays which provides an all-to-all connection structure between the inputs and the weights/delays.

    Args:
        input_size (int): number of elements of each input sample.
        output_size (int): number of elements of each output sample.
        weight_init (Callable[[torch.Tensor], Any], optional): initialization function for connection weights. Defaults to `partial(torch.nn.init.uniform_, a=0.0, b=1.0)`.
        delay_max (int): maximum delay to apply to any inputs, in number of time steps.
        delay_init (Callable[[torch.Tensor], Any], optional): initialization function for connection delays. Defaults to `partial(nn.init.constant_, val=0)`.

    Raises:
        ValueError: `input_size` must be a positive integer.
        ValueError: `output_size` must be a positive integer.
        ValueError: `delay_max` must be a non-negative integer.
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        delay_max: int,
        weight_init: Callable[[torch.Tensor], Any] = partial(nn.init.uniform_, a=0.0, b=1.0),
        delay_init: Callable[[torch.Tensor], Any] = partial(nn.init.constant_, val=0)
    ):
        # call superclass constructors
        AbstractDelayedConnection.__init__(self)

        # set input and output sizes
        if int(input_size) < 1:
            raise ValueError(f"parameter 'input_size' must be at least 1, received {input_size}")
        if int(output_size) < 1:
            raise ValueError(f"parameter 'output_size' must be at least 1, received {output_size}")

        # register weights
        self.register_parameter('_weight', nn.Parameter(torch.empty((int(output_size), int(input_size))), False))
        weight_init(self._weight)

        # register delays
        self.register_parameter('_delay', nn.Parameter(torch.empty((self.output_size, self.input_size), dtype=torch.int64), False))
        delay_init(self._delay)

        # set maximum delay
        if delay_max < 0:
            raise ValueError(f"parameter 'delay_max' must be at least 0, received {delay_max}")
        self._delay_max = int(delay_max)

        # register empty queue
        self.register_buffer('_queue', None)

    @property
    def input_size(self) -> int:
        """Number of expected inputs.

        Returns:
            int: number of elements of each input sample, excluding any batch dimensions.
        """
        return self.weight.shape[1]

    @property
    def output_size(self) -> int:
        """Number of generated outputs.

        Returns:
            int: number of elements of each output sample, excluding any batch dimensions.
        """
        return self.weight.shape[0]

    @property
    def weight(self) -> torch.Tensor:
        """Learnable weights for the connection object.

        Shape:
            :math:`N_\\text{out} \\times N_\\text{in},`
            where :math:`N_\\text{out}` is the number of outputs set at object construction and
            :math:`N_\\text{in}` is the number of inputs set at object construction.

        :getter: returns the current connection weights.
        :setter: sets the current connection weights.
        :type: torch.Tensor
        """
        return self._weight.data

    @weight.setter
    def weight(self, value):
        if isinstance(value, torch.Tensor):
            self._weight.data = value
        else:
            self._weight = value

    @property
    def delay(self) -> torch.Tensor:
        """Learnable delays for the connection object.

        Shape:
            :math:`N \\times N,`
            where :math:`N` is the number of inputs and the number of outputs as set at object construction.

        :getter: returns the current connection delays.
        :setter: sets the current connection delays.
        :type: torch.Tensor
        """
        return self._delay.data

    @delay.setter
    def delay(self, value):
        if isinstance(value, torch.Tensor):
            self._delay.data = value
        else:
            self._delay = value

    @property
    def delay_max(self) -> int:
        """Maximum permissible delay.

        Returns:
            int: maximum delay to apply to any inputs, in number of time steps.
        """
        return self._delay_max

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Applies a linear transformation, with temporal delays, from all inputs to all of the outputs.

        Shape:
            **Input:**

            :math:`B \\times N_\\text{in}^{(0)} \\times \\cdots,`
            where :math:`B` is the batch size and :math:`N_\\text{in}^{(0)}, \\ldots` are the input
            feature dimensions, and their product must be equal to the number of input features.

            **Output:**

            :math:`B \\times N_\\text{out},`
            where :math:`B` is the batch size and :math:`N_\\text{out}` is the number of output features.

        Args:
            inputs (torch.Tensor): tensor of values to which a linear transformation should be applied.

        Returns:
            torch.Tensor: tensor of values after the linear transformation was applied.
                """
        # function to rebuild queue
        def rebuild_queue():
            self._queue = torch.zeros((inputs.shape[0], self.delay_max + 1, self.input_size), dtype=self.weight.dtype)

        # check if queue exists
        if self._queue is None:
            rebuild_queue()
        # check if the queue has a different batch size
        elif self._queue.shape[0] != inputs.shape[0]:
            rebuild_queue()

        # roll the queue along the time dimension to send oldest to the front and then replace oldest (frontmost) in queue with current input
        self._queue = self._queue.roll(1, 1)
        self._queue[:, 0, :].copy_(inputs.view(inputs.shape[0], -1).to(dtype=self.weight.dtype))

        # select the required inputs
        res = torch.gather(self._queue, 1, self.delay.unsqueeze(0).tile(self._queue.shape[0], 1, 1))

        # elementwise multiplication with weights
        res.mul_(self.weight)

        # sum input for each neuron in the output
        res = res.sum(-1)

        # return computed result
        return res

    def clear(self, **kwargs) -> None:
        """Reinitializes the input history.
        """
        if self._queue is not None:
            self._queue.fill_(0)

    def reshape_outputs(self, outputs: torch.Tensor) -> torch.Tensor:
        """Reshapes a postsynaptic tensor, for dimensional compatibility with like presynaptic tensors.

        Used for updates by various learning algorithms, this output is shaped for compatibility with batch matrix multiplication operations.

        Shape:
            **Input:**

            :math:`B \\times N_\\text{out}^{(0)} \\times \\cdots,`
            where :math:`B` is the batch size and :math:`N_\\text{out}^{(0)}, \\ldots` are the
            output feature dimensions with a product equal to the number of output features.

            **Output:**

            :math:`B \\times N_\\text{out} \\times 1,`
            where :math:`B` is the batch size and :math:`N_\\text{out}` is the number of output feature.

        Args:
            outputs (torch.Tensor): like postsynaptic tensor to reshape.

        Returns:
            torch.Tensor: reshaped form of the postsynaptic tensor.
        """
        return einrearrange(outputs, '(b z) ... -> b (...) z', z=1)

    def reshape_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """Reshapes a presynaptic tensor, for dimensional compatibility with like postsynaptic tensors.

        Used for updates by various learning algorithms, this output is shaped for compatibility with batch matrix multiplication operations.

        Shape:
            **Input:**

            :math:`B \\times N_\\text{in}^{(0)} \\times \\cdots,`
            where :math:`B` is the batch size and :math:`N_\\text{in}^{(0)}, \\ldots` are the
            input feature dimensions with a product equal to the number of input features.

            **Output:**

            :math:`B \\times 1 \\times  N_\\text{in},`
            where :math:`B` is the batch size and :math:`N_\\text{in}` is the number of input features.

        Args:
            inputs (torch.Tensor): like presynaptic tensor to reshape.

        Returns:
            torch.Tensor: reshaped form of the presynaptic tensor.
        """
        return einrearrange(inputs, '(b z) ... -> b z (...)', z=1)

    def inputs_as_receptive_areas(
        self,
        inputs: torch.Tensor
    ) -> torch.Tensor:
        """Builds a tensor representing the receptive areas for each corresponding output.

        The receptive area representation arranges and replicates inputs to show which inputs contributed
        to each output.

        Shape:
            **Input:**

            :math:`B \\times N_\\text{out}^{(0)} \\times \\cdots,`
            where :math:`B` is the size of the batch dimension, :math:`N_\\text{out} \\times \\cdots` are
            the output feature dimensions with a product equal to the number of output features.

            **Output:**

            :math:`B \\times N_\\text{out} \\times N_\\text{in},`
            where :math:`B` is the size of the batch dimension, :math:`N_\\text{out}` is the
            number of outputs, and :math:`N_\\text{in}` is the number of inputs.

        Args:
            inputs (torch.Tensor): inputs for which to build the receptive area.

        Returns:
            torch.Tensor: resulting tensor representing the receptive areas of the provided inputs on this connection.
        """
        return einrearrange([inputs] * self.output_size, 'o b ... -> b o (...)')

    def reshape_weight_update(
        self,
        update: torch.Tensor,
        spatial_reduction: Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] | None = None
    ) -> torch.Tensor:
        """Reshapes an update from a learning method to be specific for the kind of layer to be like weights.

        Shape:
            **Input:**

            :math:`B \\times N_\\text{out} \\times N_\\text{in},`
            where :math:`B`, is the size of the batch dimension, :math:`N_\\text{out}` is
            the number of output features and :math:`N_\\text{in}` is the number of input features.

            **Output:**

            :math:`B \\times N_\\text{out} \\times N_\\text{in},`
            where :math:`B`, is the size of the batch dimension, :math:`N_\\text{out}` is
            the number of output features and :math:`N_\\text{in}` is the number of input features.

        Args:
            update (torch.Tensor): the update to reshape.
            space_reduction (Callable[[torch.Tensor, tuple): unused by `DenseConnection`. Defaults to `None`.

        Returns:
            torch.Tensor: the update tensor after reshaping and reduction.
        """
        return update

    def update_weight(
        self,
        update: torch.Tensor,
        weight_min: float | None = None,
        weight_max: float | None = None,
    ) -> None:
        """Applies an additive update to the connection weights.

        Shape:
            **Input:**

            :math:`N_\\text{out} \\times N_\\text{in},`
            where :math:`N_\\text{out}` is the number of outputs,
            and :math:`N_\\text{in}` is the number of inputs.

        Args:
            update (torch.Tensor): The update to apply.
            weight_min (float | None, optional): Minimum allowable weight values. Defaults to None.
            weight_max (float | None, optional): Maximum allowable weight values. Defaults to None.
        """
        self.weight.add_(update)
        if (weight_min is not None) or (weight_max is not None):
            self.weight.clamp_(min=weight_min, max=weight_max)

    def delays_as_receptive_areas(self) -> torch.Tensor:
        """Returns the delays of the layer, stuctured by receptive area.

        Shape:
            **Output:**

            :math:`B \\times N_\\text{out} \\times N_\\text{in},`
            where :math:`B` is the batch size and :math:`N_\\text{out}` is the number of outputs,
            and :math:`N_\\text{in}` is the number of inputs.

        Returns:
            torch.Tensor: The learned delays, structured by its receptive area
        """
        return self.delay.clone()

    def update_delay(
        self,
        update: torch.Tensor
    ) -> None:
        """Applies an additive update to the learned delays

        Args:
            update (torch.Tensor): The update to apply.
        """
        self.delay.add_(update)
        self.delay.clamp_(min=0, max=self.delay_max)

    def update_delay_ra(
        self,
        update: torch.Tensor
    ) -> None:
        """Applies an additive update, in the form of a receptive area, to the connection delays.

        Args:
            update (torch.Tensor): The update to apply.

        .. note:
            A receptive area matrix is one in which each row corresponds to a given output and the elements in that row correspond to the inputs
            which compose said output. For example, in a convolutional network, this corresponds to the transpose of the im2col matrix.
        """
        if torch.is_floating_point(update):
            self.delay.add_(torch.round(update).to(dtype=self.delay.dtype))
        else:
            self.delay.add_(update.to(dtype=self.delay.dtype))
        self.delay.clamp_(min=0, max=self.delay_max)


class DirectDelayedConnection(AbstractDelayedConnection):
    """Connection object with trainable delays which provides an one-to-one connection structure between the inputs and the weights/delays.

    Args:
        size (int): number of elements of each input/output sample.
        weight_init (Callable[[torch.Tensor], Any], optional): initialization function for connection weights. Defaults to `partial(torch.nn.init.uniform_, a=0.0, b=1.0)`.
        delay_max (int): maximum delay to apply to any inputs, in number of time steps.
        delay_init (Callable[[torch.Tensor], Any], optional): initialization function for connection delays. Defaults to `partial(nn.init.constant_, val=0)`.

    Raises:
        ValueError: `size` must be a positive integer.
        ValueError: `delay_max` must be a non-negative integer.
    """
    def __init__(
        self,
        size: int,
        delay_max: int,
        weight_init: Callable[[torch.Tensor], Any] = partial(nn.init.uniform_, a=0.0, b=1.0),
        delay_init: Callable[[torch.Tensor], Any] = partial(nn.init.constant_, val=0)
    ):
        # call superclass constructors
        AbstractDelayedConnection.__init__(self)

        # set size
        if int(size) < 1:
            raise ValueError(f"parameter 'size' must be at least 1, received {size}")

        # register weights
        self.register_parameter('_weight', nn.Parameter(torch.empty((int(size), int(size))), False))
        weight_init(self._weight)

        # masking matrix (diagonal is 1, rest is 0)
        self.register_buffer('_mask', torch.eye(self.size, dtype=self.weight.dtype, device=self.weight.device))

        # initial weight masking
        self.weight = self.weight * self._mask

        # register delays
        self.register_parameter('_delay', nn.Parameter(torch.empty((self.size, self.size), dtype=torch.int64), False))
        delay_init(self._delay)

        # set maximum delay
        if delay_max < 0:
            raise ValueError(f"parameter 'delay_max' must be at least 0, received {delay_max}")
        self._delay_max = int(delay_max)

        # register empty queue
        self.register_buffer('_queue', None)

        # register helpers
        self.register_buffer('_receptive_area_indices', torch.tensor([range(self.size)], dtype=torch.int64, device=self.weight.device))

    @property
    def size(self) -> int:
        """Number of expected inputs and generated outputs.

        Returns:
            int: number of elements of each input/output sample, excluding any batch dimensions.
        """
        return self.weight.shape[0]

    @property
    def weight(self) -> torch.Tensor:
        """Learnable weights for the connection object.

        Shape:
            :math:`N \\times N,`
            where :math:`N` is the number of inputs and the number of outputs as set at object construction.

        :getter: returns the current connection weights.
        :setter: sets the current connection weights.
        :type: torch.Tensor
        """
        return self._weight.data

    @weight.setter
    def weight(self, value):
        if isinstance(value, torch.Tensor):
            self._weight.data = value
        else:
            self._weight = value

    @property
    def delay(self) -> torch.Tensor:
        """Learnable delays for the connection object.

        Shape:
            :math:`N \\times N,`
            where :math:`N` is the number of inputs and the number of outputs as set at object construction.

        :getter: returns the current connection delays.
        :setter: sets the current connection delays.
        :type: torch.Tensor
        """
        return self._delay.data

    @delay.setter
    def delay(self, value):
        if isinstance(value, torch.Tensor):
            self._delay.data = value
        else:
            self._delay = value

    @property
    def delay_max(self) -> int:
        """Maximum permissible delay.

        Returns:
            int: maximum delay to apply to any inputs, in number of time steps.
        """
        return self._delay_max

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Applies a linear transformation, with temporal delays, from one input to one of the outputs.

        Shape:
            **Input:**

            :math:`B \\times N^{(0)} \\times \\cdots,`
            where :math:`B` is the batch size and :math:`N^{(0)}, \\ldots` are the feature dimensions,
            and their product must be equal to the number of features.

            **Output:**

            :math:`B \\times N,`
            where :math:`B` is the batch size and :math:`N` is the number of features.

        Args:
            inputs (torch.Tensor): tensor of values to which a linear transformation should be applied.

        Returns:
            torch.Tensor: tensor of values after the linear transformation was applied.
        """
        # mask weights and delays
        self.weight.mul_(self._mask)
        self.delay.mul_(self._mask.to(dtype=self.delay.dtype))

        # function to rebuild queue
        def rebuild_queue():
            self._queue = torch.zeros((inputs.shape[0], self.delay_max + 1, self.input_size), dtype=self.weight.dtype)

        # check if queue exists
        if self._queue is None:
            rebuild_queue()
        # check if the queue has a different batch size
        elif self._queue.shape[0] != inputs.shape[0]:
            rebuild_queue()

        # roll the queue along the time dimension to send oldest to the front and then replace oldest (frontmost) in queue with current input
        self._queue = self._queue.roll(1, 1)
        self._queue[:, 0, :].copy_(inputs.view(inputs.shape[0], -1).to(dtype=self.weight.dtype))

        # select the required inputs
        res = torch.gather(self._queue, 1, self.delay.unsqueeze(0).tile(self._queue.shape[0], 1, 1))

        # elementwise multiplication with weights
        res.mul_(self.weight)

        # sum input for each neuron in the output
        res = res.sum(-1)

        # return computed result
        return res

    def clear(self, **kwargs) -> None:
        """Reinitializes the input history.
        """
        if self._queue is not None:
            self._queue.fill_(0)

    def reshape_outputs(self, outputs: torch.Tensor) -> torch.Tensor:
        """Reshapes a postsynaptic tensor, for dimensional compatibility with like presynaptic tensors.

        Used for updates by various learning algorithms, this output is shaped for compatibility with batch matrix multiplication operations.

        Shape:
            **Input:**

            :math:`B \\times N^{(0)} \\times \\cdots,`
            where :math:`B` is the batch size and :math:`N^{(0)}, \\ldots` are the
            feature dimensions with a product equal to the number of features.

            **Output:**

            :math:`B \\times N \\times 1,`
            where :math:`B` is the batch size and :math:`N` is the number of features.

        This function assumes a temporal dimension and will generate a postsynaptic intermediate valid for Hadamard product operations.

        Args:
            outputs (torch.Tensor): like postsynaptic tensor to reshape.

        Returns:
            torch.Tensor: reshaped form of the postsynaptic tensor.
        """
        return einrearrange(outputs, '(b z) ... -> b (...) z', z=1)

    def reshape_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """Reshapes a presynaptic tensor, for dimensional compatibility with like postsynaptic tensors.

        Used for updates by various learning algorithms, this output is shaped for compatibility with batch matrix multiplication operations.

        Shape:
            **Input:**

            :math:`B \\times N^{(0)} \\times \\cdots,`
            where :math:`B` is the batch size and :math:`N^{(0)}, \\ldots` are the
            feature dimensions with a product equal to the number of features.

            **Output:**

            :math:`B \\times N \\times 1,`
            where :math:`B` is the batch size and :math:`N` is the number of features.

        Args:
            inputs (torch.Tensor): like presynaptic tensor to reshape.

        Returns:
            torch.Tensor: reshaped form of the presynaptic tensor.
        """
        return einrearrange(inputs, '(b z) ... -> b z (...)', z=1)

    def inputs_as_receptive_areas(
        self,
        inputs: torch.Tensor
    ) -> torch.Tensor:
        """Builds a tensor representing the receptive areas for each corresponding output.

        The receptive area representation arranges and replicates inputs to show which inputs contributed
        to each output.

        Shape:
            **Input:**

            :math:`B \\times N^{(0)} \\times \\cdots,`
            where :math:`B` is the batch size and :math:`N^{(0)} \\times \\ldots` are the
            feature dimensions with a product equal to the number of features.

            **Output:**

            :math:`B \\times N \\times 1,`
            where :math:`B` is the batch size and :math:`N` is the number of features.

        Args:
            inputs (torch.Tensor): inputs for which to build the receptive area.

        Returns:
            torch.Tensor: resulting tensor representing the receptive areas of the provided inputs on this connection.
        """
        return einrearrange(inputs, '(b z) ... -> b (...) z', z=1)

    def reshape_weight_update(
        self,
        update: torch.Tensor,
        spatial_reduction: Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] | None = None
    ) -> torch.Tensor:
        """Reshapes an update from a learning method to be specific for the kind of layer to be like weights.

        Shape:
            **Input:**

            :math:`B \\times N,`
            where :math:`B`, is the size of the batch dimension and :math:`N` is the number of features.

            **Output:**

            :math:`B \\times N \\times N,`
            where :math:`B`, is the size of the batch dimension and :math:`N` is the number of features.

        Args:
            update (torch.Tensor): the update to reshape.
            space_reduction (Callable[[torch.Tensor, tuple): unused by `DirectConnection`. Defaults to `None`.

        Returns:
            torch.Tensor: the update tensor after reshaping and reduction.
        """
        return update

    def update_weight(
        self,
        update: torch.Tensor,
        weight_min: float | None = None,
        weight_max: float | None = None,
    ) -> None:
        """Applies an additive update to the connection weights.

        Shape:
            **Input:**

            :math:`N \\times N,`
            where :math:`N` is the number of features

        Args:
            update (torch.Tensor): The update to apply.
            weight_min (float | None, optional): Minimum allowable weight values. Defaults to None.
            weight_max (float | None, optional): Maximum allowable weight values. Defaults to None.
        """
        self.weight.add_(update)
        if (weight_min is not None) or (weight_max is not None):
            self.weight.clamp_(min=weight_min, max=weight_max)
        self.weight.mul_(self._mask)

    def delays_as_receptive_areas(self) -> torch.Tensor:
        """Returns the delays of the layer, stuctured by receptive area.

        Shape:
            **Output:**

            :math:`B \\times N \\times 1,`
            where :math:`B` is the batch size and :math:`N` is the number of parameters.

        Returns:
            torch.Tensor: The learned delays, structured by its receptive area
        """
        return torch.masked_select(self.delay.data, self._mask.bool()).view(self.size, 1)

    def update_delay(
        self,
        update: torch.Tensor,
    ) -> None:
        """Applies an additive update to the learned delays

        Args:
            update (torch.Tensor): The update to apply.
        """
        self.delay.add_(update)
        self.delay.clamp_(min=0, max=self.delay_max)

    def update_delay_ra(
        self,
        update: torch.Tensor,
    ) -> None:
        """Applies an additive update, in the form of a receptive area, to the connection delays.

        Args:
            update (torch.Tensor): The update to apply.

        .. note:
            A receptive area matrix is one in which each row corresponds to a given output and the elements in that row correspond to the inputs
            which compose said output. For example, in a convolutional network, this corresponds to the transpose of the im2col matrix.
        """
        if torch.is_floating_point(update):
            self.delay.add_(torch.zeros_like(self.delay).scatter_(0, self._receptive_area_indices, torch.round(update).to(dtype=self.delay.dtype).t()))
        else:
            self.delay.add_(torch.zeros_like(self.delay).scatter_(0, self._receptive_area_indices, update.to(dtype=self.delay.dtype).t()))
        self.delay.clamp_(min=0, max=self.delay_max)
        self.delay.mul_(self._mask.to(dtype=self.delay.dtype))


class LateralDelayedConnection(AbstractDelayedConnection):
    """Connection object with trainable delays which provides an all-to-all-but-one connection structure between the inputs and the weights/delays.

    Args:
        size (int): number of elements of each input/output sample.
        weight_init (Callable[[torch.Tensor], Any], optional): initialization function for connection weights. Defaults to `partial(torch.nn.init.uniform_, a=0.0, b=1.0)`.
        delay_max (int): maximum delay to apply to any inputs, in number of time steps.
        delay_init (Callable[[torch.Tensor], Any], optional): initialization function for connection delays. Defaults to `partial(nn.init.constant_, val=0)`.

    Raises:
        ValueError: `size` must be a positive integer.
    ValueError: `delay_max` must be a non-negative integer.
    """
    def __init__(
        self,
        size: int,
        delay_max: int,
        weight_init: Callable[[torch.Tensor], Any] = partial(nn.init.uniform_, a=0.0, b=1.0),
        delay_init: Callable[[torch.Tensor], Any] = partial(nn.init.constant_, val=0)
    ):
        # call superclass constructors
        AbstractDelayedConnection.__init__(self)

        # set size
        if int(size) < 1:
            raise ValueError(f"parameter 'size' must be at least 1, received {size}")

        # register weights
        self.register_parameter('_weight', nn.Parameter(torch.empty((int(size), int(size))), False))
        weight_init(self._weight)

        # masking matrix (diagonal is 0, rest is 1)
        self.register_buffer('_mask', torch.abs(torch.eye(self.size, dtype=self.weight.dtype, device=self.weight.device) - 1))

        # initial weight masking
        self.weight = self.weight * self._mask

        # register delays
        self.register_parameter('_delay', nn.Parameter(torch.empty((self.size, self.size), dtype=torch.int64), False))
        delay_init(self._delay)

        # set maximum delay
        if delay_max < 0:
            raise ValueError(f"parameter 'delay_max' must be at least 0, received {delay_max}")
        self._delay_max = int(delay_max)

        # register empty queue
        self.register_buffer('_queue', None)

        # register helpers
        self.register_buffer('_receptive_area_indices',
            (torch.ones((self.size, self.size - 1), dtype=torch.int64, device=self.weight.device)
            * torch.tensor([range(1, self.size)], dtype=torch.int64, device=self.weight.device)).t()
            - torch.triu(torch.ones((self.size - 1, self.size), dtype=torch.int64, device=self.weight.device), diagonal=1))

    @property
    def size(self) -> int:
        """Number of expected inputs and generated outputs.

        Returns:
            int: number of elements of each input/output sample, excluding any batch dimensions.
        """
        return self.weight.shape[0]

    @property
    def weight(self) -> torch.Tensor:
        """Learnable weights for the connection object.

        Shape:
            :math:`N \\times N,`
            where :math:`N` is the number of inputs and the number of outputs as set at object construction.

        :getter: returns the current connection weights.
        :setter: sets the current connection weights.
        :type: torch.Tensor
        """
        return self._weight.data

    @weight.setter
    def weight(self, value):
        if isinstance(value, torch.Tensor):
            self._weight.data = value
        else:
            self._weight = value

    @property
    def delay(self) -> torch.Tensor:
        """Learnable delays for the connection object.

        Shape:
            :math:`N \\times N,`
            where :math:`N` is the number of inputs and the number of outputs as set at object construction.

        :getter: returns the current connection delays.
        :setter: sets the current connection delays.
        :type: torch.Tensor
        """
        return self._delay.data

    @delay.setter
    def delay(self, value):
        if isinstance(value, torch.Tensor):
            self._delay.data = value
        else:
            self._delay = value

    @property
    def delay_max(self) -> int:
        """Maximum permissible delay.

        Returns:
            int: maximum delay to apply to any inputs, in number of time steps.
        """
        return self._delay_max

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Applies a linear transformation, with temporal delays, from one input to one of the outputs.

        Shape:
            **Input:**

            :math:`B \\times N^{(0)} \\times \\cdots,`
            where :math:`B` is the batch size and :math:`N^{(0)}, \\ldots` are the feature dimensions,
            and their product must be equal to the number of features.

            **Output:**

            :math:`B \\times N,`
            where :math:`B` is the batch size and :math:`N` is the number of features.

        Args:
            inputs (torch.Tensor): tensor of values to which a linear transformation should be applied.

        Returns:
            torch.Tensor: tensor of values after the linear transformation was applied.
        """
        # mask weights and delays
        self.weight.mul_(self._mask)
        self.delay.mul_(self._mask.to(dtype=self.delay.dtype))

        # function to rebuild queue
        def rebuild_queue():
            self._queue = torch.zeros((inputs.shape[0], self.delay_max + 1, self.input_size), dtype=self.weight.dtype)

        # check if queue exists
        if self._queue is None:
            rebuild_queue()
        # check if the queue has a different batch size
        elif self._queue.shape[0] != inputs.shape[0]:
            rebuild_queue()

        # roll the queue along the time dimension to send oldest to the front and then replace oldest (frontmost) in queue with current input
        self._queue = self._queue.roll(1, 1)
        self._queue[:, 0, :].copy_(inputs.view(inputs.shape[0], -1).to(dtype=self.weight.dtype))

        # select the required inputs
        res = torch.gather(self._queue, 1, self.delay.unsqueeze(0).tile(self._queue.shape[0], 1, 1))

        # elementwise multiplication with weights
        res.mul_(self.weight)

        # sum input for each neuron in the output
        res = res.sum(-1)

        # return computed result
        return res

    def clear(self, **kwargs) -> None:
        """Reinitializes the input history.
        """
        if self._queue is not None:
            self._queue.fill_(0)

    def reshape_outputs(self, outputs: torch.Tensor) -> torch.Tensor:
        """Reshapes a postsynaptic tensor, for dimensional compatibility with like presynaptic tensors.

        Used for updates by various learning algorithms, this output is shaped for compatibility with batch matrix multiplication operations.

        Shape:
            **Input:**

            :math:`B \\times N^{(0)} \\times \\cdots,`
            where :math:`B` is the batch size and :math:`N^{(0)}, \\ldots` are the
            feature dimensions with a product equal to the number of features.

            **Output:**

            :math:`B \\times N \\times 1,`
            where :math:`B` is the batch size and :math:`N` is the number of features.

        This function assumes a temporal dimension and will generate a postsynaptic intermediate valid for Hadamard product operations.

        Args:
            outputs (torch.Tensor): like postsynaptic tensor to reshape.

        Returns:
            torch.Tensor: reshaped form of the postsynaptic tensor.
        """
        return einrearrange(outputs, '(b z) ... -> b (...) z', z=1)

    def reshape_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """Reshapes a presynaptic tensor, for dimensional compatibility with like postsynaptic tensors.

        Used for updates by various learning algorithms, this output is shaped for compatibility with batch matrix multiplication operations.

        Shape:
            **Input:**

            :math:`B \\times N^{(0)} \\times \\cdots,`
            where :math:`B` is the batch size and :math:`N^{(0)}, \\ldots` are the
            feature dimensions with a product equal to the number of features.

            **Output:**

            :math:`B \\times N \\times 1,`
            where :math:`B` is the batch size and :math:`N` is the number of features.

        Args:
            inputs (torch.Tensor): like presynaptic tensor to reshape.

        Returns:
            torch.Tensor: reshaped form of the presynaptic tensor.
        """
        return einrearrange(inputs, '(b z) ... -> b z (...)', z=1)

    def inputs_as_receptive_areas(
        self,
        inputs: torch.Tensor
    ) -> torch.Tensor:
        """Builds a tensor representing the receptive areas for each corresponding output.

        The receptive area representation arranges and replicates inputs to show which inputs contributed
        to each output.

        Shape:
            **Input:**

            :math:`B \\times N,`
            where :math:`B` is the size of the batch dimension, :math:`N` is the
            number of features.

            **Output:**

            :math:`B \\times N \\times N - 1,`
            where :math:`B` is the size of the batch dimension, :math:`\\times N` is the
            number of outputs, and :math:`\\times N_\\text{in}` is the number of inputs.

        Args:
            inputs (torch.Tensor): inputs for which to build the receptive area.

        Returns:
            torch.Tensor: resulting tensor representing the receptive areas of the provided inputs on this connection.
        """
        return einrearrange(einrearrange([inputs] * self.size, 'n b ... -> b n (...)')
            .masked_select(self._mask.unsqueeze(0).bool()), '(b n m) -> b n m', n=self.size, m=(self.size - 1))

    def reshape_weight_update(
        self,
        update: torch.Tensor,
        spatial_reduction: Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] | None = None
    ) -> torch.Tensor:
        """Reshapes an update from a learning method to be specific for the kind of layer to be like weights.

        Shape:
            **Input:**

            :math:`B \\times N,`
            where :math:`B`, is the size of the batch dimension and :math:`N` is the number of features.

            **Output:**

            :math:`B \\times N \\times N,`
            where :math:`B`, is the size of the batch dimension and :math:`N` is the number of features.

        Args:
            update (torch.Tensor): the update to reshape.
            space_reduction (Callable[[torch.Tensor, tuple): unused by `DirectConnection`. Defaults to `None`.

        Returns:
            torch.Tensor: the update tensor after reshaping and reduction.
        """
        return update

    def update_weight(
        self,
        update: torch.Tensor,
        weight_min: float | None = None,
        weight_max: float | None = None,
    ) -> None:
        """Applies an additive update to the connection weights.

        Shape:
            **Input:**

            :math:`N \\times N,`
            where :math:`N` is the number of features

        Args:
            update (torch.Tensor): The update to apply.
            weight_min (float | None, optional): Minimum allowable weight values. Defaults to None.
            weight_max (float | None, optional): Maximum allowable weight values. Defaults to None.
        """
        self.weight.add_(update)
        if (weight_min is not None) or (weight_max is not None):
            self.weight.clamp_(min=weight_min, max=weight_max)
        self.weight.mul_(self._mask)

    def delays_as_receptive_areas(self) -> torch.Tensor:
        """Returns the delays of the layer, stuctured by receptive area.

        Shape:
            **Output:**

            :math:`B \\times N \\times N - 1,`
            where :math:`B` is the batch size and :math:`N` is the number of parameters.

        Returns:
            torch.Tensor: The learned delays, structured by its receptive area
        """
        return self.delay.clone()

    def update_delay(
        self,
        update: torch.Tensor,
    ) -> None:
        """Applies an additive update to the learned delays

        Args:
            update (torch.Tensor): The update to apply.
        """
        self.delay.add_(update)
        self.delay.clamp_(min=0, max=self.delay_max)

    def update_delay_ra(
        self,
        update: torch.Tensor
    ) -> None:
        """Applies an additive update, in the form of a receptive area, to the connection delays.

        Args:
            update (torch.Tensor): The update to apply.

        .. note:
            A receptive area matrix is one in which each row corresponds to a given output and the elements in that row correspond to the inputs
            which compose said output. For example, in a convolutional network, this corresponds to the transpose of the im2col matrix.
        """
        if torch.is_floating_point(update):
            self.delay.add_(torch.zeros_like(self.delay).scatter_(0, self._receptive_area_indices, torch.round(update).to(dtype=self.delay.dtype).t()).t())
        else:
            self.delay.add_(torch.zeros_like(self.delay).scatter_(0, self._receptive_area_indices, update.to(dtype=self.delay.dtype).t()).t())
        self.delay.clamp_(min=0, max=self.delay_max)
        self.delay.mul_(self._mask.to(dtype=self.delay.dtype))
