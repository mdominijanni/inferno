from .. import StateHook, normalize
from .._internal import argtest, rgetattr, rsetattr
import torch
import torch.nn as nn


class Normalization(StateHook):
    r"""Normalizes attribute of registered module on call.

    Args:
        module (nn.Module): module to which the hook should be registered.
        attr (str): fully-qualified string name of attribute to normalize.
        order (int | float): order of :math:`p`-norm by which to normalize.
        scale (float | complex): desired :math:`p`-norm of elements along
            specified dimensions.
        dim (int | tuple[int] | None): dimension(s) along which to normalize,
            all dimensions if ``None``.
        epsilon (float, optional): value added to the denominator in case of
            zero-valued norms. Defaults to ``1e-12``.
        train_update (bool, optional): if normalization should be performed when
            hooked module is in train mode. Defaults to ``True``.
        eval_update (bool, optional): if normalization should be performed when
            hooked module is in eval mode. Defaults to ``True``.
        as_prehook (bool, optional): if normalization should occur before
            :py:meth:`~torch.nn.Module.forward` is. Defaults to ``False``.
        prepend (bool, optional): if normalization should occur before other hooks
            previously registered to the hooked module. Defaults to ``False``.
        always_call (bool, optional): if the hook should be run even if an exception
            occurs, only applies when ``as_prehook`` is False. Defaults to ``False``.
    """

    def __init__(
        self,
        module: nn.Module,
        attr: str,
        order: int | float,
        scale: float | complex,
        dim: int | tuple[int, ...] | None,
        epsilon: float = 1e-12,
        *,
        train_update: bool = True,
        eval_update: bool = True,
        as_prehook: bool = False,
        prepend: bool = False,
        always_call: bool = False,
    ):
        # sanity check arguments
        self.attribute = argtest.nestedidentifier("attr", attr)
        self.order = argtest.neq("order", order, 0, None)
        self.scale = argtest.neq("scale", scale, 0, None)
        self.dim = argtest.dimensions("dim", dim, None, None, permit_none=True)
        self.eps = float(epsilon)

        # call superclass constructor
        StateHook.__init__(
            self,
            module,
            train_update=train_update,
            eval_update=eval_update,
            as_prehook=as_prehook,
            prepend=prepend,
            always_call=always_call,
        )

    def hook(self, module: nn.Module) -> None:
        r"""Function to be called on the registered module's call.

        Args:
            module (nn.Module): registered module.
        """
        rsetattr(
            module,
            self.attribute,
            normalize(
                rgetattr(self.module, self.attribute),
                self.order,
                self.scale,
                self.dim,
                epsilon=self.eps,
            ),
        )


class Clamping(StateHook):
    r"""Clamps attribute of registered module on call.

    Args:
        module (nn.Module): module to which the hook should be registered.
        attr (str): fully-qualified string name of attribute to clamp.
        min (int | float | None, optional): inclusive lower-bound of the clamped range.
            Defaults to ``None``.
        max (int | float | None, optional): inclusive upper-bound of the clamped range.
            Defaults to ``None``.
        train_update (bool, optional): if normalization should be performed when
            hooked module is in train mode. Defaults to ``True``.
        eval_update (bool, optional): if normalization should be performed when
            hooked module is in eval mode. Defaults to ``True``.
        as_prehook (bool, optional): if normalization should occur before
            :py:meth:`~torch.nn.Module.forward` is. Defaults to ``False``.
        prepend (bool, optional): if normalization should occur before other hooks
            previously registered to the hooked module. Defaults to ``False``.
        always_call (bool, optional): if the hook should be run even if an exception
            occurs, only applies when ``as_prehook`` is False. Defaults to ``False``.
    """

    def __init__(
        self,
        module: nn.Module,
        attr: str,
        min: int | float | None = None,
        max: int | float | None = None,
        *,
        train_update: bool = True,
        eval_update: bool = True,
        as_prehook: bool = False,
        prepend: bool = False,
        always_call: bool = False,
    ):
        # sanity check arguments
        self.attribute = argtest.nestedidentifier("attr", attr)
        self.clampmin, self.clampmax = argtest.onedefined(("min", min), ("max", max))
        if min is not None and max is not None:
            _ = argtest.gt("max", max, min, None, limit_name="min")

        # call superclass constructor
        StateHook.__init__(
            self,
            module,
            train_update=train_update,
            eval_update=eval_update,
            as_prehook=as_prehook,
            prepend=prepend,
            always_call=always_call,
        )

    def hook(self, module: nn.Module) -> None:
        r"""Function to be called on the registered module's call.

        Args:
            module (nn.Module): registered module.
        """
        rsetattr(
            module,
            self.attribute,
            torch.clamp(
                rgetattr(self.module, self.attribute),
                min=self.clampmin,
                max=self.clampmax,
            ),
        )
