from inferno import Hook, normalize
from inferno._internal import rgetattr, rsetattr
import torch
import torch.nn as nn


class Normalization(Hook):
    r"""Normalizes attribute of registered module on call.

    Args:
        name (str): fully-qualified string name of attribute to normalize.
        order (int | float): order of :math:`p`-norm by which to normalize.
        scale (float | complex): desired :math:`p`-norm of elements along specified dimensions.
        dims (int | tuple[int] | None): dimensions along which to normalize, all dimensions if None.
        train_update (bool, optional): if normalization should be performed when module is
            in train mode. Defaults to True.
        eval_update (bool, optional): if normalization should be performed when module is
            in eval mode. Defaults to True.
        as_prehook (bool, optional): if normalization should occur before
            :py:meth:`~torch.nn.Module.forward` is. Defaults to False.
        prepend (bool, optional): if normalization should occur before other hooks registered
            to the module. Defaults to False.
        always_call (bool, optional): if the hook should be run even if an exception occurs,
            only applies when ``as_prehook`` is False. Defaults to False.
        module (nn.Module | None, optional): module to which the hook should be registered. Defaults to None.
    """

    def __init__(
        self,
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
        module: nn.Module | None = None,
    ):
        # sanity check arguments
        if order == 0:
            raise ValueError(f"order must be non-zero, received {order}.")
        if scale == 0:
            raise ValueError(f"scale must be non-zero, received {scale}.")

        # configure as forward prehook
        if as_prehook:

            def norm_hook(module, args):
                rsetattr(
                    module, name, normalize(rgetattr(module, name), order, scale, dims)
                )

            Hook.__init__(
                self,
                norm_hook,
                None,
                prehook_kwargs={"prepend": prepend},
                train_update=train_update,
                eval_update=eval_update,
                module=module,
            )

        # configure as forward posthook
        else:

            def norm_hook(module, args, output):
                rsetattr(
                    module, name, normalize(rgetattr(module, name), order, scale, dims)
                )

            Hook.__init__(
                self,
                None,
                norm_hook,
                posthook_kwargs={"prepend": prepend, "always_call": always_call},
                train_update=train_update,
                eval_update=eval_update,
                module=module,
            )


class Clamping(Hook):
    """Clamps attribute of registered module on call.

    Args:
        name (str): fully-qualified string name of attribute to normalize.
        min (int | float | None, optional): inclusive lower-bound of the clamped range. Defaults to None.
        max (int | float | None, optional): inclusive upper-bound of the clamped range. Defaults to None.
        train_update (bool, optional): if normalization should be performed when module is
            in train mode. Defaults to True.
        eval_update (bool, optional): if normalization should be performed when module is
            in eval mode. Defaults to True.
        as_prehook (bool, optional): if normalization should occur before
            :py:meth:`~torch.nn.Module.forward` is. Defaults to False.
        prepend (bool, optional): if normalization should occur before other hooks registered
            to the module. Defaults to False.
        always_call (bool, optional): if the hook should be run even if an exception occurs,
            only applies when ``as_prehook`` is False. Defaults to False.
        module (nn.Module | None, optional): module to which the hook should be registered. Defaults to None.

    Raises:
        TypeError: at least one of ``min`` and ``max`` must not be None.
        ValueError: ``min`` must be strictly less than ``max``.
    """

    def __init__(
        self,
        name: str,
        min: int | float | None = None,
        max: int | float | None = None,
        *,
        train_update: bool = True,
        eval_update: bool = True,
        as_prehook: bool = False,
        prepend: bool = False,
        always_call: bool = False,
        module: nn.Module | None = None,
    ):
        # sanity check arguments
        if min is None and max is None:
            raise TypeError("min and max cannot both be None.")
        if min is not None and max is not None and min >= max:
            raise ValueError(
                f"received max of {max} does not exceed received min of {min}."
            )

        # configure as forward prehook
        if as_prehook:

            def norm_hook(module, args):
                rsetattr(
                    module, name, torch.clamp(rgetattr(module, name), min=min, max=max)
                )

            Hook.__init__(
                self,
                norm_hook,
                None,
                prehook_kwargs={"prepend": prepend},
                train_update=train_update,
                eval_update=eval_update,
                module=module,
            )

        # configure as forward posthook
        else:

            def norm_hook(module, args, output):
                rsetattr(
                    module, name, torch.clamp(rgetattr(module, name), min=min, max=max)
                )

            Hook.__init__(
                self,
                None,
                norm_hook,
                posthook_kwargs={"prepend": prepend, "always_call": always_call},
                train_update=train_update,
                eval_update=eval_update,
                module=module,
            )
