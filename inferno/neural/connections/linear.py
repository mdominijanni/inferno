from functools import partial
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from inferno.neural.connections.abstract import AbstractConnection


class DenseConnection(AbstractConnection):
    """Connection object which provides an all-to-all connection structure between the inputs and the weights.

    Args:
        input_size (int): number of elements of each input sample.
        output_size (int): number of elements of each output sample.
        weight_init (Callable[[torch.Tensor], Any], optional): initialization function for connection weights. Defaults to `partial(torch.nn.init.uniform_, a=0.0, b=1.0)`.

    Raises:
        ValueError: `input_size` must be a positive integer.
        ValueError: `output_size` must be a positive integer.
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        weight_init: Callable[[torch.Tensor], Any] = partial(torch.nn.init.uniform_, a=0.0, b=1.0)
    ):
        # call superclass constructor
        AbstractConnection.__init__(self)

        # set input and output sizes
        if int(input_size) < 1:
            raise ValueError(f"parameter 'input_size' must be at least 1, received {input_size}")
        if int(output_size) < 1:
            raise ValueError(f"parameter 'output_size' must be at least 1, received {output_size}")

        # register weights
        self.register_parameter('_weight', nn.Parameter(torch.empty((int(output_size), int(input_size))), False))
        weight_init(self._weight)

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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Applies a linear transformation from all inputs to all of the outputs.

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
        return F.linear(inputs.view(inputs.shape[0], -1).to(dtype=self.weight.dtype), self.weight)

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
        outputs = outputs.view(outputs.shape[0], -1)  # B x * -> B x N
        outputs = outputs.unsqueeze(-1)               # B x N -> B x N x 1
        return outputs

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
        inputs = inputs.view(inputs.shape[0], -1)  # B x * -> B x N
        inputs = inputs.unsqueeze(1)               # B x N -> B x 1 x N
        return inputs

    def inputs_as_receptive_areas(
        self,
        inputs: torch.Tensor
    ) -> torch.Tensor:
        """Builds a tensor representing the receptive areas for each corresponding output.

        The receptive area representation arranges and replicates inputs to show which inputs contributed
        to each output.

        Shape:
            **Input:**

            :math:`B \\times N_\\text{out},`
            where :math:`B` is the size of the batch dimension, :math:`N_\\text{out}` is the
            number of outputs.

            **Output:**

            :math:`B \\times N_\\text{out} \\times N_\\text{in},`
            where :math:`B` is the size of the batch dimension, :math:`N_\\text{out}` is the
            number of outputs, and :math:`N_\\text{in}` is the number of inputs.

        Args:
            inputs (torch.Tensor): inputs for which to build the receptive area.

        Returns:
            torch.Tensor: resulting tensor representing the receptive areas of the provided inputs on this connection.
        """
        return torch.stack([inputs.view(inputs.shape[0], -1)] * self.output_size, dim=-2)

    def update_weight_add(
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

    def update_weight_pd(
        self,
        add_update: torch.Tensor,
        sub_update: torch.Tensor,
        weight_min: float | None = None,
        weight_max: float | None = None,
        bounding_mode: str | None = None,
    ) -> None:
        """Applies both a potentiation and a depression update to the connection weights, allows for advanced weight bounding.

        Shape:
            **Input:**

            :math:`N_\\text{out} \\times N_\\text{in},`
            where :math:`N_\\text{out}` is the number of outputs,
            and :math:`N_\\text{in}` is the number of inputs.

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
            ValueError: an unsupported value for the parameter `bounding_mode` was specified
        """
        wb_mode = str(bounding_mode).lower()
        match wb_mode:
            case 'hard':
                if weight_min is not None:
                    dep_mult = torch.heaviside(self.weight - weight_min, torch.zeros(1, dtype=sub_update.dtype, device=sub_update.device))
                else:
                    dep_mult = 1.0
                if weight_max is not None:
                    pot_mult = torch.heaviside(weight_max - self.weight, torch.zeros(1, dtype=sub_update.dtype, device=sub_update.device))
                else:
                    pot_mult = 1.0
                self.weight.sub_(sub_update * dep_mult)
                self.weight.add_(add_update * pot_mult)
            case 'soft':
                if weight_min is not None:
                    dep_mult = self.weight - weight_min
                else:
                    dep_mult = 1.0
                if weight_max is not None:
                    pot_mult = weight_max - self.weight
                else:
                    pot_mult = 1.0
                self.weight.sub_(sub_update * dep_mult)
                self.weight.add_(add_update * pot_mult)
            case 'clamp':
                self.weight.add_(add_update - sub_update)
                self.weight.clamp_(min=weight_min, max=weight_max)
            case 'none':
                self.weight.add_(add_update - sub_update)
            case _:
                raise ValueError(f"invalid 'bounding_mode' specified, must be None, 'hard', 'soft', or 'clamp', received {bounding_mode}")


class DirectConnection(AbstractConnection):
    """Connection object which provides an one-to-one connection structure between the inputs and the weights.

    Args:
        size (int): number of elements of each input/output sample.
        weight_init (Callable[[torch.Tensor], Any], optional): initialization function for connection weights. Defaults to `partial(torch.nn.init.uniform_, a=0.0, b=1.0)`.

    Raises:
        ValueError: `size` must be a positive integer.
    """
    def __init__(
        self,
        size: int,
        weight_init: Callable[[torch.Tensor], Any] = partial(torch.nn.init.uniform_, a=0.0, b=1.0)
    ):
        # call superclass constructor
        AbstractConnection.__init__(self)

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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Applies a linear transformation from one input to one output.

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
        self.weight = self.weight * self._mask
        return F.linear(inputs.view(inputs.shape[0], -1).to(dtype=self.weight.dtype), self.weight)

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
        outputs = outputs.view(outputs.shape[0], -1)  # B x * -> B x N
        outputs = outputs.unsqueeze(-1)               # B x N -> B x N x 1
        return outputs

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
        inputs = inputs.view(inputs.shape[0], -1)  # B x * -> B x N
        inputs = inputs.unsqueeze(1)               # B x N -> B x 1 x N
        return inputs

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

            :math:`B \\times N \\times 1,`
            where :math:`B` is the size of the batch dimension and :math:`\\times N` is the
            number of outputs.

        Args:
            inputs (torch.Tensor): inputs for which to build the receptive area.

        Returns:
            torch.Tensor: resulting tensor representing the receptive areas of the provided inputs on this connection.
        """
        return torch.stack([inputs.view(inputs.shape[0], -1)] * self.size, dim=-2)\
            .masked_select(self._mask[(None,) * (inputs.ndim - 1) + (...,)].bool())\
            .view(list(inputs.shape) + [1])

    def update_weight_add(
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

    def update_weight_pd(
        self,
        add_update: torch.Tensor,
        sub_update: torch.Tensor,
        weight_min: float | None = None,
        weight_max: float | None = None,
        bounding_mode: str | None = None,
    ) -> None:
        """Applies both a potentiation and a depression update to the connection weights.

        Shape:
            **Input:**

            :math:`N \\times N,`
            where :math:`N` is the number of features

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
            ValueError: an unsupported value for the parameter `bounding_mode` was specified
        """
        wb_mode = str(bounding_mode).lower()
        match wb_mode:
            case 'hard':
                if weight_min is not None:
                    dep_mult = torch.heaviside(self.weight - weight_min, torch.zeros(1, dtype=sub_update.dtype, device=sub_update.device))
                else:
                    dep_mult = 1.0
                if weight_max is not None:
                    pot_mult = torch.heaviside(weight_max - self.weight, torch.zeros(1, dtype=sub_update.dtype, device=sub_update.device))
                else:
                    pot_mult = 1.0
                self.weight.sub_(sub_update * dep_mult)
                self.weight.add_(add_update * pot_mult)
                self.weight.mul_(self._mask)
            case 'soft':
                if weight_min is not None:
                    dep_mult = self.weight - weight_min
                else:
                    dep_mult = 1.0
                if weight_max is not None:
                    pot_mult = weight_max - self.weight
                else:
                    pot_mult = 1.0
                self.weight.sub_(sub_update * dep_mult)
                self.weight.add_(add_update * pot_mult)
                self.weight.mul_(self._mask)
            case 'clamp':
                self.weight.add_(add_update - sub_update)
                self.weight.clamp_(min=weight_min, max=weight_max)
                self.weight.mul_(self._mask)
            case 'none':
                self.weight.add_(add_update - sub_update)
                self.weight.mul_(self._mask)
            case _:
                raise ValueError(f"invalid 'bounding_mode' specified, must be None, 'hard', 'soft', or 'clamp', received {bounding_mode}")


class LateralConnection(AbstractConnection):
    """Connection object which provides an all-to-all-but-one connection structure between the inputs and the weights.

    Args:
        size (int): number of elements of each input/output sample.
        weight_init (Callable[[torch.Tensor], Any], optional): initialization function for connection weights. Defaults to `partial(torch.nn.init.uniform_, a=0.0, b=1.0)`.

    Raises:
        ValueError: `size` must be a positive integer.
    """
    def __init__(
        self,
        size: int,
        weight_init: Callable[[torch.Tensor], Any] = partial(torch.nn.init.uniform_, a=0.0, b=1.0)
    ):
        # call superclass constructor
        AbstractConnection.__init__(self)

        # set size
        if int(size) < 1:
            raise ValueError(f"parameter 'size' must be at least 1, received {size}")

        # register weights
        self.register_parameter('_weight', nn.Parameter(torch.empty((int(size), int(size))), False))
        weight_init(self._weight)

        # masking matrix (diagonal is 0, rest is 1)
        weight_init('_mask', torch.abs(torch.eye(self.size, dtype=self.weight.dtype, device=self.weight.device) - 1))

    @property
    def size(self) -> int:
        """Number of expected inputs and generated outputs.

        Returns:
            int: number of elements of each input/output sample, excluding any batch dimensions.
        """
        return self._size

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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Applies a linear transformation from all inputs to all-but-one outputs.

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
        self.weight = self.weight * self._mask
        return F.linear(inputs.view(inputs.shape[0], -1).to(dtype=self.weight.dtype), self.weight)

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
        outputs = outputs.view(outputs.shape[0], -1)  # B x * -> B x N
        outputs = outputs.unsqueeze(-1)               # B x N -> B x N x 1
        return outputs

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
        inputs = inputs.view(inputs.shape[0], -1)  # B x * -> B x N
        inputs = inputs.unsqueeze(1)               # B x N -> B x 1 x N
        return inputs

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
        return torch.stack([inputs] * self.size, dim=-2)\
            .masked_select(self._mask[(None,) * (inputs.ndim - 1) + (...,)].bool())\
            .view(list(inputs.shape) + [self.size - 1])

    def update_weight_add(
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

    def update_weight_pd(
        self,
        add_update: torch.Tensor,
        sub_update: torch.Tensor,
        weight_min: float | None = None,
        weight_max: float | None = None,
        bounding_mode: str | None = None,
    ) -> None:
        """Applies both a potentiation and a depression update to the connection weights.

        Shape:
            **Input:**

            :math:`N \\times N,`
            where :math:`N` is the number of features

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
            ValueError: an unsupported value for the parameter `bounding_mode` was specified
        """
        wb_mode = str(bounding_mode).lower()
        match wb_mode:
            case 'hard':
                if weight_min is not None:
                    dep_mult = torch.heaviside(self.weight - weight_min, torch.zeros(1, dtype=sub_update.dtype, device=sub_update.device))
                else:
                    dep_mult = 1.0
                if weight_max is not None:
                    pot_mult = torch.heaviside(weight_max - self.weight, torch.zeros(1, dtype=sub_update.dtype, device=sub_update.device))
                else:
                    pot_mult = 1.0
                self.weight.sub_(sub_update * dep_mult)
                self.weight.add_(add_update * pot_mult)
                self.weight.mul_(self._mask)
            case 'soft':
                if weight_min is not None:
                    dep_mult = self.weight - weight_min
                else:
                    dep_mult = 1.0
                if weight_max is not None:
                    pot_mult = weight_max - self.weight
                else:
                    pot_mult = 1.0
                self.weight.sub_(sub_update * dep_mult)
                self.weight.add_(add_update * pot_mult)
                self.weight.mul_(self._mask)
            case 'clamp':
                self.weight.add_(add_update - sub_update)
                self.weight.clamp_(min=weight_min, max=weight_max)
                self.weight.mul_(self._mask)
            case 'none':
                self.weight.add_(add_update - sub_update)
                self.weight.mul_(self._mask)
            case _:
                raise ValueError(f"invalid 'bounding_mode' specified, must be None, 'hard', 'soft', or 'clamp', received {bounding_mode}")
