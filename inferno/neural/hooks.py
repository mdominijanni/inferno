from inferno import RemoteHook, StateHook, normalize
from inferno._internal import argtest, rgetattr, rsetattr
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
            all dimensions if None.
        epsilon (float, optional): value added to the demoninator in case of
            zero-valued norms. Defaults to 1e-12.
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
    """Clamps attribute of registered module on call.

    Args:
        module (nn.Module): module to which the hook should be registered.
        attr (str): fully-qualified string name of attribute to clamp.
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
            _ = argtest.lte("max", max, min, None, limit_name="min")

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


class RemoteNormalization(RemoteHook):
    r"""Normalizes module parameters on another module's call.

    Args:
        train_update (bool, optional): if weights should be normalized for layers
            in train mode. Defaults to True.
        eval_update (bool, optional): if weights should be normalized for layers
            in eval mode. Defaults to False.
        as_prehook (bool, optional): if the hook should be run prior to the triggering
            module's :py:meth:`~torch.nn.Module.forward` call. Defaults to False.
        prepend (bool, optional): if normalization should occur before other hooks
            registered to the triggering module. Defaults to False.
        always_call (bool, optional): if normalization should occur even if an
            exception occurs in module call. Defaults to True.

    Keyword Args:
        attr (str): default fully-qualified string name of attribute to normalize.
        order (int | float): default order of the :math:`p`-norm by which to normalize.
        scale (float | complex): default value of the :math:`p`-norm along dimensions
            after normalization.
        dim (int | tuple[int, ...] | None): default dimension(s) along which to
            normalize, all dimensions if None.
        epsilon (float): default value added to the demoninator in case of zero-valued
            norms, when not specified a value of 1e-12 is used.

    Important:
        The keyword arguments ``attr``, ``order``, ``scale``, ``dims`, and ``epsilon`
        define the default values for added modules. If any are not specified, those
        which are not must be specified on every :py:meth:`add_inner` call.
    """

    def __init__(
        self,
        train_update: bool = True,
        eval_update: bool = False,
        *,
        as_prehook: bool = False,
        prepend: bool = False,
        always_call: bool = False,
        **kwargs,
    ):
        # tests for and sets default normalization parameters
        if "attr" in kwargs:
            self.default_attr = argtest.nestedidentifier(
                "attr", kwargs["attr"], prefix="keyword argument "
            )
        if "order" in kwargs:
            self.default_order = argtest.neq(
                "order", kwargs["order"], 0, None, prefix="keyword argument "
            )
        if "scale" in kwargs:
            self.default_scale = argtest.neq(
                "scale", kwargs["scale"], 0, None, prefix="keyword argument "
            )
        if "dim" in kwargs:
            self.default_dim = argtest.dimensions(
                "dim",
                kwargs["dim"],
                None,
                None,
                permit_none=True,
                wrap_output=False,
                prefix="keyword argument ",
            )
        if "epsilon" in kwargs:
            self.default_eps = float(kwargs["epsilon"])
        else:
            self.default_eps = 1e-12

        RemoteHook.__init__(
            self,
            train_update,
            eval_update,
            as_prehook=as_prehook,
            prepend=prepend,
            always_call=always_call,
        )

    def innerhook(
        self,
        module: nn.Module,
        *,
        attr: str,
        order: int | float,
        scale: float | complex,
        dim: int | tuple[int, ...] | None,
        eps: float,
    ) -> None:
        r"""Function to be called on the inner modules when the registered module is called.

        Args:
            module (nn.Module): inner module.

        Keyword Args:
            attr (str): fully-qualified string name of attribute to normalize.
            order (int | float): order of :math:`p`-norm by which to normalize.
            scale (float | complex): desired :math:`p`-norm of elements along
                specified dimensions.
            dim (int | tuple[int] | None): dimension(s) along which to normalize,
                all dimensions if None.
            epsilon (float): value added to the demoninator in case of
                zero-valued norms.
        """
        rsetattr(
            module,
            attr,
            normalize(
                rgetattr(module, attr),
                order,
                scale,
                dim,
                epsilon=eps,
            ),
        )

    def add_inner(self, *modules: nn.Module, **kwargs) -> None:
        r"""Adds layers with connection weights to normalize.

        Args:
            *modules (~torch.nn.Module): additional modules on which normalization
                should be performed.

        Keyword Args:
            attr (str): fully-qualified string name of attribute to normalize.
            order (int | float): order of the :math:`p`-norm by which to normalize.
            scale (float | complex): value of the :math:`p`-norm along dimensions
                after normalization.
            dim (int | tuple[int, ...] | None): dimension(s) along which to
                normalize, all dimensions if None.
            epsilon (float): default value added to the demoninator in case of
                zero-valued norms, when not specified a value of 1e-12 is used.

        Important:
            If any of the above keyword argument were not given a default on
            construction, they must be specified here. If they were specified and
            another value is given here, it will override the default for only this
            call of ``add_inner``.
        """
        if "attr" in kwargs:
            attr = argtest.nestedidentifier(
                "attr", kwargs["attr"], prefix="keyword argument "
            )
        elif hasattr(self, "default_attr"):
            attr = self.default_attr
        else:
            raise RuntimeError("'attr' was not specified and no default was provided")

        if "order" in kwargs:
            order = argtest.neq(
                "order", kwargs["order"], 0, None, prefix="keyword argument "
            )
        elif hasattr(self, "default_order"):
            order = self.default_order
        else:
            raise RuntimeError("'order' was not specified and no default was provided")

        if "scale" in kwargs:
            scale = argtest.neq(
                "scale", kwargs["scale"], 0, None, prefix="keyword argument "
            )
        elif hasattr(self, "default_scale"):
            scale = self.default_scale
        else:
            raise RuntimeError("'scale' was not specified and no default was provided")

        if "dim" in kwargs:
            dim = argtest.dimensions(
                "dim",
                kwargs["dim"],
                None,
                None,
                permit_none=True,
                wrap_output=False,
                prefix="keyword argument ",
            )
        elif hasattr(self, "default_dim"):
            dim = self.default_dim
        else:
            raise RuntimeError("'dim' was not specified and no default was provided")

        if "epsilon" in kwargs:
            eps = float(kwargs["epsilon"])
        elif hasattr(self, "default_eps"):
            eps = self.default_eps
        else:
            raise RuntimeError("'eps' was not specified and no default was provided")

        RemoteHook.add_inner(
            self, modules, attr=attr, order=order, scale=scale, dim=dim, eps=eps
        )

    def remove_inner(self, *modules: nn.Module) -> None:
        r"""Removes inner modules from the hook.

        Args:
            *modules (~torch.nn.Module): inner module(s) to remove.
        """
        RemoteHook.remove_inner(modules)


class RemoteClamping(RemoteHook):
    r"""Clamps module parameters on another module's call.

    Args:
        train_update (bool, optional): if weights should be normalized for layers
            in train mode. Defaults to True.
        eval_update (bool, optional): if weights should be normalized for layers
            in eval mode. Defaults to False.
        as_prehook (bool, optional): if the hook should be run prior to the triggering
            module's :py:meth:`~torch.nn.Module.forward` call. Defaults to False.
        prepend (bool, optional): if normalization should occur before other hooks
            registered to the triggering module. Defaults to False.
        always_call (bool, optional): if normalization should occur even if an
            exception occurs in module call. Defaults to True.

    Keyword Args:
        attr (str): default fully-qualified string name of attribute to normalize.
        min (int | float | None): default inclusive lower-bound of the clamped range.
            Defaults to None.
        max (int | float | None): default inclusive upper-bound of the clamped range.
            Defaults to None.

    Important:
        The keyword arguments ``attr``, ``min``, and ``max`` define
        the default values for added modules. If any are not specified, those which
        are not must be specified on every :py:meth:`add_inner` call.
    """

    def __init__(
        self,
        train_update: bool = True,
        eval_update: bool = False,
        *,
        as_prehook: bool = False,
        prepend: bool = False,
        always_call: bool = False,
        **kwargs,
    ):
        # tests for and sets default normalization parameters
        if "attr" in kwargs:
            self.default_attr = argtest.nestedidentifier(
                "attr", kwargs["attr"], prefix="keyword argument "
            )
        if "min" in kwargs:
            self.default_min = kwargs["min"]
        if "max" in kwargs:
            self.default_max = kwargs["max"]

        RemoteHook.__init__(
            self,
            train_update,
            eval_update,
            as_prehook=as_prehook,
            prepend=prepend,
            always_call=always_call,
        )

    def innerhook(
        self,
        module: nn.Module,
        *,
        attr: str,
        min: int | float | None,
        max: int | float | None,
    ) -> None:
        r"""Function to be called on the inner modules when the registered module is called.

        Args:
            module (nn.Module): inner module.

        Keyword Args:
            attr (str): fully-qualified string name of attribute to clamp.
            min (int | float | None): inclusive lower-bound of the clamped range.
            max (int | float | None): inclusive upper-bound of the clamped range.
        """
        rsetattr(
            module,
            attr,
            torch.clamp(
                rgetattr(module, attr),
                min=min,
                max=max,
            ),
        )

    def add_inner(self, *modules: nn.Module, **kwargs) -> None:
        r"""Adds layers with connection weights to clamp.

        Args:
            *modules (~torch.nn.Module): additional modules on which clamping
                should be performed.

        Keyword Args:
            attr (str): fully-qualified string name of attribute to clamp.
            min (int | float | None): inclusive lower-bound of the clamped range.
            max (int | float | None): inclusive upper-bound of the clamped range.

        Important:
            If any of the above keyword argument were not given a default on
            construction, they must be specified here. If they were specified and
            another value is given here, it will override the default for only this
            call of ``add_inner``.
        """
        if "attr" in kwargs:
            attr = argtest.nestedidentifier(
                "attr", kwargs["attr"], prefix="keyword argument "
            )
        elif hasattr(self, "default_attr"):
            attr = self.default_attr
        else:
            raise RuntimeError("'attr' was not specified and no default was provided")

        if "min" in kwargs:
            min = kwargs["min"]
        elif hasattr(self, "default_min"):
            min = self.default_min
        else:
            raise RuntimeError("'min' was not specified and no default was provided")

        if "max" in kwargs:
            max = kwargs["max"]
        elif hasattr(self, "default_max"):
            max = self.default_max
        else:
            raise RuntimeError("'max' was not specified and no default was provided")

        _ = argtest.onedefined(
            ("min", min),
            ("max", max),
            prefix="between defaults and values specified on-call, ",
        )

        if min is not None and max is not None:
            _ = argtest.lte(
                "max",
                max,
                min,
                None,
                limit_name="min",
                prefix="between defaults and values specified on-call, ",
            )

        RemoteHook.add_inner(self, modules, attr=attr, min=min, max=max)

    def remove_inner(self, *modules: nn.Module) -> None:
        r"""Removes inner modules from the hook.

        Args:
            *modules (~torch.nn.Module): inner module(s) to remove.
        """
        RemoteHook.remove_inner(modules)
