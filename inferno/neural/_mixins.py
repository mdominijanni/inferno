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
