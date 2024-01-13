from inferno import Hook, Module, normalize
from inferno.neural import Layer
import torch


class WeightNormalization(Hook):
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
        """Normalizes layer weights on module call.

        Args:
            *layers (Layers): layers for which normalization should be performed.
            order (int | float): order of :math:`p`-norm by which to normalize.
            scale (float | complex): desired :math:`p`-norm of elements along specified dimensions.
            dims (int | tuple[int] | None): dimensions along which to normalize, all dimensions if None.
            train_update (bool, optional): if weights should be normalized for layers in train mode. Defaults to True.
            eval_update (bool, optional): if weights should be normalized for layers in eval mode. Defaults to False.
            prepend (bool, optional): if normalization should occur before other hooks registered
                to the triggering module. Defaults to False.
            always_call (bool, optional): if normalization should occur even if an exception occurs
                in module call. Defaults to False.
            module (Module | None, optional): module to which the hook should be registered. Defaults to None.

        Raises:
            ValueError: ``order`` must be non-zero.
            ValueError: ``scale`` must be non-zero.
            RuntimeError: at least one of ``train_update`` and ``eval_update`` must be True.
        """
        # sanity check arguments
        if order == 0:
            raise ValueError(f"order must be non-zero, received {order}.")
        if scale == 0:
            raise ValueError(f"scale must be non-zero, received {scale}.")
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
    def __init__(
        self,
        *layers: Layer,
        min: int | float | None = None,
        max: int | float | None = None,
        train_update: bool = True,
        eval_update: bool = False,
        prepend: bool = False,
        always_call: bool = False,
        module: Module | None = None,
    ):
        """Clamps layer weights on module call.

        Args:
            *layers (Layers): layers for which normalization should be performed.
            min (int | float | None, optional): inclusive lower-bound of the clamped range. Defaults to None.
            max (int | float | None, optional): inclusive upper-bound of the clamped range. Defaults to None.
            train_update (bool, optional): if weights should be normalized for layers in train mode. Defaults to True.
            eval_update (bool, optional): if weights should be normalized for layers in eval mode. Defaults to False.
            prepend (bool, optional): if normalization should occur before other hooks registered
                to the triggering module. Defaults to False.
            always_call (bool, optional): if normalization should occur even if an exception occurs
                in module call. Defaults to False.
            module (Module | None, optional): module to which the hook should be registered. Defaults to None.

        Raises:
            TypeError: at least one of ``min`` and ``max`` must not be None.
            ValueError: ``min`` must be strictly less than ``max``.
            RuntimeError: at least one of ``train_update`` and ``eval_update`` must be True.
        """
        # sanity check arguments
        if min is None and max is None:
            raise TypeError("min and max cannot both be None.")
        if min is not None and max is not None and min >= max:
            raise ValueError(
                f"received max of {max} does not exceed received min of {min}."
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
                        layer.connection.weight, min=min, max=max
                    )

        if train_update and not eval_update:

            def clamp_hook(module, args, output):  # noqa:F811
                for layer in layers:
                    if layer.training:
                        layer.connection.weight = torch.clamp(
                            layer.connection.weight, min=min, max=max
                        )

        if not train_update and eval_update:

            def clamp_hook(module, args, output):  # noqa:F811
                for layer in layers:
                    if not layer.training:
                        layer.connection.weight = torch.clamp(
                            layer.connection.weight, min=min, max=max
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
