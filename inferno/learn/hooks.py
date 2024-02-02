from inferno import RemoteHook, normalize
from inferno._internal import numeric_limit
from inferno.neural import Layer
import torch
import torch.nn as nn


class WeightNormalization(RemoteHook):
    r"""Normalizes layer weights on module call.

    Args:

        module (~torch.nn.Module | None, optional): module to which the hook
            should be registered. Defaults to None.
        layer_train_update (bool, optional): if weights should be normalized for layers
            in train mode. Defaults to True.
        layer_eval_update (bool, optional): if weights should be normalized for layers
            in eval mode. Defaults to False.
        as_prehook (bool, optional): if the hook should be run prior to the triggering
            module's :py:meth:`~torch.nn.Module.forward` call. Defaults to False.
        prepend (bool, optional): if normalization should occur before other hooks
            registered to the triggering module. Defaults to False.
        always_call (bool, optional): if normalization should occur even if an
            exception occurs in module call. Defaults to True.

    Note:
        Initial layers share the same set of parameters (``order``, ``scale``, and
        ``dims``). More can be added with :py:meth:`add_inner` with distinct
        parameters.
    """

    def __init__(
        self,
        module: nn.Module,
        layer_train_update: bool = True,
        layer_eval_update: bool = False,
        *,
        as_prehook: bool = False,
        prepend: bool = False,
        always_call: bool = False,
    ):
        def hook(layer, *, order, scale, dims):
            layer.connection.weight = normalize(
                layer.connection.weight, order, scale, dims
            )

        RemoteHook.__init__(
            self,
            hook,
            module,
            layer_train_update,
            layer_eval_update,
            as_prehook=as_prehook,
            prepend=prepend,
            always_call=always_call,
        )

    def add_inner(
        self,
        *layers: Layer,
        order: int | float,
        scale: float | complex,
        dims: int | tuple[int] | None,
    ) -> None:
        r"""Adds layers with connection weights to normalize.

        Args:
            *layers (Layers): additional layers for which normalization
                should be performed.
            order (int | float): order of :math:`p`-norm by which to normalize.
            scale (float | complex): desired :math:`p`-norm of elements along
                specified dimensions.
            dims (int | tuple[int] | None): dimensions along which to normalize,
                all dimensions if None.
        """
        _, e = numeric_limit("order", order, 0, "neq", None)
        if e:
            raise e
        _, e = numeric_limit("scale", scale, 0, "neq", None)
        if e:
            raise e
        RemoteHook.add_inner(self, *layers, order=order, scale=scale, dims=dims)


class WeightClamping(RemoteHook):
    r"""Clamps layer weights on module call.

    Args:

        module (~torch.nn.Module | None, optional): module to which the hook
            should be registered. Defaults to None.
        layer_train_update (bool, optional): if weights should be normalized for layers
            in train mode. Defaults to True.
        layer_eval_update (bool, optional): if weights should be normalized for layers
            in eval mode. Defaults to False.
        as_prehook (bool, optional): if the hook should be run prior to the triggering
            module's :py:meth:`~torch.nn.Module.forward` call. Defaults to False.
        prepend (bool, optional): if normalization should occur before other hooks
            registered to the triggering module. Defaults to False.
        always_call (bool, optional): if normalization should occur even if an
            exception occurs in module call. Defaults to True.

    Note:
        Initial layers share the same set of parameters (``order``, ``scale``, and
        ``dims``). More can be added with :py:meth:`add_inner` with distinct
        parameters.
    """

    def __init__(
        self,
        module: nn.Module,
        layer_train_update: bool = True,
        layer_eval_update: bool = False,
        *,
        as_prehook: bool = False,
        prepend: bool = False,
        always_call: bool = False,
    ):
        def hook(layer, *, wmin, wmax):
            layer.connection.weight = torch.clamp(
                layer.connection.weight, min=wmin, max=wmax
            )

        RemoteHook.__init__(
            self,
            hook,
            module,
            layer_train_update,
            layer_eval_update,
            as_prehook=as_prehook,
            prepend=prepend,
            always_call=always_call,
        )

    def add_inner(
        self,
        *layers: Layer,
        wmin: int | float | None = None,
        wmax: int | float | None = None,
    ) -> None:
        r"""Adds layers with connection weights to clamp.

        Args:
            *layers (Layers): additional layers for which clamping
                should be performed.
            wmin (int | float | None, optional): inclusive lower-bound of the
                clamped range. Defaults to None.
            wmax (int | float | None, optional): inclusive upper-bound of the
                clamped range. Defaults to None.
        """
        if wmin is None and wmax is None:
            raise TypeError("`wmin` and `wmax` cannot both be None.")
        if wmin is not None and wmax is not None and wmin >= wmax:
            raise ValueError(
                f"received `wmax` of {wmax} not greater than `wmin` of {wmin}."
            )

        RemoteHook.add_inner(self, *layers, wmin=wmin, wmax=wmax)
