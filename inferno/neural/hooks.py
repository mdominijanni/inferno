from inferno import StateHook, normalize
from inferno._internal import numeric_limit, rgetattr, rsetattr
import torch
import torch.nn as nn


class Normalization(StateHook):
    r"""Normalizes attribute of registered module on call.

    Args:
        module (nn.Module): module to which the hook should be registered.
        name (str): fully-qualified string name of attribute to normalize.
        order (int | float): order of :math:`p`-norm by which to normalize.
        scale (float | complex): desired :math:`p`-norm of elements along
            specified dimensions.
        dims (int | tuple[int] | None): dimensions along which to normalize,
            all dimensions if None.
        train_update (bool, optional): if normalization should be performed when
            hooked module is in train mode. Defaults to True.
        eval_update (bool, optional): if normalization should be performed when
            hooked module is in eval mode. Defaults to True.
        as_prehook (bool, optional): if normalization should occur before
            :py:meth:`~torch.nn.Module.forward` is. Defaults to False.
        prepend (bool, optional): if normalization should occur before other hooks
            previously registered to the hooked module. Defaults to False.
        always_call (bool, optional): if the hook should be run even if an exception
            occurs, only applies when ``as_prehook`` is False. Defaults to False.
    """

    def __init__(
        self,
        module: nn.Module,
        name: str,
        order: int | float,
        scale: float | complex,
        dims: int | tuple[int] | None,
        *,
        train_update: bool = True,
        eval_update: bool = True,
        as_prehook: bool = False,
        prepend: bool = False,
        always_call: bool = False,
    ):
        # sanity check arguments
        _ = numeric_limit("`order`", order, 0, "neq", None)
        _ = numeric_limit("`scale`", scale, 0, "neq", None)

        # inner hook function
        def hook(module):
            rsetattr(
                module, name, normalize(rgetattr(module, name), order, scale, dims)
            )

        # call superclass constructor
        StateHook.__init__(
            self,
            hook,
            module,
            train_update=train_update,
            eval_update=eval_update,
            as_prehook=as_prehook,
            prepend=prepend,
            always_call=always_call,
        )


class Clamping(StateHook):
    """Clamps attribute of registered module on call.

    Args:
        module (nn.Module): module to which the hook should be registered.
        name (str): fully-qualified string name of attribute to normalize.
        min (int | float | None, optional): inclusive lower-bound of the clamped range.
            Defaults to None.
        max (int | float | None, optional): inclusive upper-bound of the clamped range.
            Defaults to None.
        train_update (bool, optional): if normalization should be performed when
            hooked module is in train mode. Defaults to True.
        eval_update (bool, optional): if normalization should be performed when
            hooked module is in eval mode. Defaults to True.
        as_prehook (bool, optional): if normalization should occur before
            :py:meth:`~torch.nn.Module.forward` is. Defaults to False.
        prepend (bool, optional): if normalization should occur before other hooks
            previously registered to the hooked module. Defaults to False.
        always_call (bool, optional): if the hook should be run even if an exception
            occurs, only applies when ``as_prehook`` is False. Defaults to False.
    """

    def __init__(
        self,
        module: nn.Module,
        name: str,
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
        if min is None and max is None:
            raise TypeError("`min` and `max` cannot both be None.")
        if min is not None and max is not None and min >= max:
            raise ValueError(
                f"received `max` of {max} not greater than `min` of {min}."
            )

        # inner hook function
        def hook(module):
            rsetattr(
                module, name, torch.clamp(rgetattr(module, name), min=min, max=max)
            )

        # call superclass constructor
        StateHook.__init__(
            self,
            hook,
            module,
            train_update=train_update,
            eval_update=eval_update,
            as_prehook=as_prehook,
            prepend=prepend,
            always_call=always_call,
        )
