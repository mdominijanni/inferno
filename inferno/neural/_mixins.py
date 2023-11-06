import torch


class VoltageMixin:
    @property
    def voltage(self) -> torch.Tensor:
        r"""Membrane voltages of the neurons, in millivolts.

        Args:
            value (torch.Tensor): new membrane voltages.

        Returns:
            torch.Tensor: current membrane voltages.
        """
        return self.voltages.data

    @voltage.setter
    def voltage(self, value: torch.Tensor):
        self.voltages.data[:] = value


class RefractoryMixin:
    @property
    def spike(self) -> torch.Tensor:
        r"""Which neurons generated an action potential on the last simulation step.

        Returns:
            torch.Tensor: if the correspond neuron generated an action potential last step.
        """
        return self.refracs.data == self.refrac_t

    @property
    def refrac(self) -> torch.Tensor:
        r"""Remaining refractory periods of the neurons, in milliseconds.

        Returns:
            torch.Tensor: current remaining refractory periods.
        """
        return self.refracs.data


class ConnectionParameterMixin:
    @property
    def weight(self) -> torch.Tensor:
        r"""Learnable weights of the connection.

        Args:
            value (torch.Tensor): new weights.

        Returns:
            torch.Tensor: current weights.
        """
        return self.weights.data

    @weight.setter
    def weight(self, value: torch.Tensor):
        self.weights.data = value

    @property
    def bias(self) -> torch.Tensor | None:
        r"""Learnable biases of the connection.

        Args:
            value (torch.Tensor): new biases.

        Returns:
            torch.Tensor | None: current biases, if the connection has any.

        Raises:
            RuntimeError: ``bias`` cannot be set on a connection without learnable biases.
        """
        if self.biases is not None:
            return self.biases.data

    @bias.setter
    def bias(self, value: torch.Tensor):
        if self.biases is not None:
            self.biases.data = value
        else:
            raise RuntimeError(
                f"cannot set `bias` on a {type(self).__name__} without trainable biases"
            )

    @property
    def delay(self) -> torch.Tensor | None:
        r"""Learnable delays of the connection.

        Args:
            value (torch.Tensor): new delays.

        Returns:
            torch.Tensor | None: current delays, if the connection has any.

        Raises:
            RuntimeError: ``delay`` cannot be set on a connection without learnable delays.
        """
        if self.delays is not None:
            return self.delays.data

    @delay.setter
    def delay(self, value: torch.Tensor):
        if self.delays is not None:
            self.delays.data = value
        else:
            raise RuntimeError(
                f"cannot set `delay` on a {type(self).__name__} without trainable delays"
            )
