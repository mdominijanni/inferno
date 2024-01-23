from inferno import Hook, Module, normalize
from inferno._internal import numeric_limit
from inferno.neural import Layer
import torch


class WeightNormalization(Hook):
    r"""Normalizes layer weights on module call.

    Args:
        *layers (Layers): layers for which normalization should be performed.
        order (int | float): order of :math:`p`-norm by which to normalize.
        scale (float | complex): desired :math:`p`-norm of elements along
            specified dimensions.
        dims (int | tuple[int] | None): dimensions along which to normalize,
            all dimensions if None.
        train_update (bool, optional): if weights should be normalized for layers
            in train mode. Defaults to True.
        eval_update (bool, optional): if weights should be normalized for layers
            in eval mode. Defaults to False.
        prepend (bool, optional): if normalization should occur before other hooks
            registered to the triggering module. Defaults to False.
        always_call (bool, optional): if normalization should occur even if an
            exception occurs in module call. Defaults to True.
        module (Module | None, optional): module to which the hook
            should be registered. Defaults to None.
    """
    def __init__(
        self,
        *layers: Layer,
        order: int | float,
        scale: float | complex,
        dims: int | tuple[int] | None,
        train_update: bool = True,
        eval_update: bool = False,
        prepend: bool = False,
        always_call: bool = False,
        module: Module | None = None,
    ):
        # sanity check arguments
        _ = numeric_limit("`order`", order, 0, "neq", None)
        _ = numeric_limit("`scale`", scale, 0, "neq", None)
        if not train_update and not eval_update:
            raise RuntimeError(
                "at least one of `train_update` and `eval_update` must be True."
            )

        # create closure with layers for hook function
        if train_update and eval_update:

            def norm_hook(module, args, output):  # noqa:F811
                for layer in layers:
                    layer.connection.weight = normalize(
                        layer.connection.weight, order, scale, dims
                    )

        if train_update and not eval_update:

            def norm_hook(module, args, output):  # noqa:F811
                for layer in layers:
                    if layer.training:
                        layer.connection.weight = normalize(
                            layer.connection.weight, order, scale, dims
                        )

        if not train_update and eval_update:

            def norm_hook(module, args, output):  # noqa:F811
                for layer in layers:
                    if not layer.training:
                        layer.connection.weight = normalize(
                            layer.connection.weight, order, scale, dims
                        )

        # construct hook
        Hook.__init__(
            self,
            None,
            norm_hook,
            posthook_kwargs={"prepend": prepend, "always_call": always_call},
            train_update=True,
            eval_update=True,
            module=module,
        )


class WeightClamping(Hook):
    r"""Clamps layer weights on module call.

    Args:
        *layers (Layers): layers for which normalization should be performed.
        wmin (int | float | None, optional): inclusive lower-bound of the clamped range.
            Defaults to None.
        wmax (int | float | None, optional): inclusive upper-bound of the clamped range.
            Defaults to None.
        train_update (bool, optional): if weights should be normalized for layers
            in train mode. Defaults to True.
        eval_update (bool, optional): if weights should be normalized for layers
            in eval mode. Defaults to False.
        prepend (bool, optional): if normalization should occur before other hooks
            registered to the triggering module. Defaults to False.
        always_call (bool, optional): if normalization should occur even if an
            exception occurs in module call. Defaults to True.
        module (Module | None, optional): module to which the hook
            should be registered. Defaults to None.
    """
    def __init__(
        self,
        *layers: Layer,
        wmin: int | float | None = None,
        wmax: int | float | None = None,
        train_update: bool = True,
        eval_update: bool = False,
        prepend: bool = False,
        always_call: bool = False,
        module: Module | None = None,
    ):
        # sanity check arguments
        if wmin is None and wmax is None:
            raise TypeError("`wmin` and `wmax` cannot both be None.")
        if wmin is not None and wmax is not None and wmin >= wmax:
            raise ValueError(
                f"received `wmax` of {wmax} not greater than `wmin` of {wmin}."
            )
        if not train_update and not eval_update:
            raise RuntimeError(
                "at least one of `train_update` and `eval_update` must be True."
            )

        # create closure with layers for hook function
        if train_update and eval_update:

            def clamp_hook(module, args, output):  # noqa:F811
                for layer in layers:
                    layer.connection.weight = torch.clamp(
                        layer.connection.weight, min=wmin, max=wmax
                    )

        if train_update and not eval_update:

            def clamp_hook(module, args, output):  # noqa:F811
                for layer in layers:
                    if layer.training:
                        layer.connection.weight = torch.clamp(
                            layer.connection.weight, min=wmin, max=wmax
                        )

        if not train_update and eval_update:

            def clamp_hook(module, args, output):  # noqa:F811
                for layer in layers:
                    if not layer.training:
                        layer.connection.weight = torch.clamp(
                            layer.connection.weight, min=wmin, max=wmax
                        )

        # construct hook
        Hook.__init__(
            self,
            None,
            clamp_hook,
            posthook_kwargs={"prepend": prepend, "always_call": always_call},
            train_update=True,
            eval_update=True,
            module=module,
        )
