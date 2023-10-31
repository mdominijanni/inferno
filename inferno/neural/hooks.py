from typing import Any
import warnings

import torch
import torch.nn as nn

from inferno.common import PreHookable
from inferno._internal import get_nested_attr


class ParameterNormalization(PreHookable):
    """Normalizes a specified attribute of a PyTorch module before :py:meth:`torch.nn.Module.forward` on :py:class:`torch.nn.Module` call.

    `ParameterNormalization` applies the :py:func:`torch.linalg.vector_norm` operation to an attribute (`name`) contained within a :py:class:`torch.nn.Module` object
    (`module`). The attribute is retrieved via :py:func:`getattr`. It is applied by registering the operation with :py:meth:`torch.nn.Module.register_forward_pre_hook`.
    The `dims` and `norm` arguments of the constructor correspond with the `dim` and `max` arguments of :py:func:`torch.linalg.vector_norm` respectively. This performs the
    following operation.

    .. math::
        x_i = \\frac{\\text{scale}}{\\lVert \\mathbf{x} \\rVert_\\text{norm}}x_i

    When `dims` is `None`, the attribute tensor will be flattened before the norm is computed. If `dims` is an `int` or a `tuple` thereof, then the norm is computed over
    those dimensions and the remainder of the dimensions will be treated as batch dimensions.

    The parameter `norm` controls the :math:`p` of the :math:`p`-norm, where :math:`p \\in [-\\infty,0) \\cup (0,\\infty]`. The for :math:`p=-\\infty` and :math:`p=\\infty`,
    the :math:`p`-norms are then calculated in the following ways.

    .. math::
        \\lVert \\mathbf{x} \\rVert_{-\\infty} &= \\text{min}\\left(\\left|\\mathbf{x}\\right|\\right) \\\\
        \\lVert \\mathbf{x} \\rVert_{\\infty} &= \\text{max}\\left(\\left|\\mathbf{x}\\right|\\right)

    When :math:`p \\in \\mathbb{R} \\setminus \\{0\\}`, the :math:`p`-norm is instead computed in the following manner.

    .. math::
        \\lVert \\mathbf{x} \\rVert_p = \\left(\\sum_i \\left|x_i\\right|^p \\right)^\\frac{1}{p}

    .. note::
        This is usually used in such a way that normalization on a connection's weights are performed per-target, which varies by connection type.
        For linear connection types, this would involve normalizing the last dimension of the weights (i.e. the dimensions should target the input representations).

    .. note::
        Attribute with name `name` in :py:class:`torch.nn.Module` `module` must be a kind of tensor. This includes both instances of :py:class:`torch.Tensor` and
        :py:class:`torch.nn.parameter.Parameter`.

    .. note::
        In order to toggle a :py:class:`torch.nn.Module` between training and evaluation mode, the methods :py:meth:`torch.nn.Module.train` and :py:meth:`torch.nn.Module.eval`
        can be called, respectively.

    Args:
        name (str): name of a :py:class:`torch.Tensor` attribute contained within a :py:class:`torch.nn.Module` for which parameter normalization should be performed.
        dims (int | tuple[int] | None, optional): dimension(s) over which to normalize. Defaults to `None`.
        norm (float | int, optional): the :math:`p`-norm to normalize to, see above for more details. Defaults to `1`.
        scale (float, optional): the desired value of the vector norm. Defaults to `1.0`.
        train_update (bool, optional): specify if parameter clamping should occur when module is in training mode. Defaults to `True`.
        eval_update (bool, optional): specify if parameter clamping should occur when module is in evaluation mode. Defaults to `False`.
        module (torch.nn.Module | None, optional): PyTorch module to which this will be registered as a hook. Defaults to `None`.

    Raises:
        ValueError: `norm` must be non-zero.
        ValueError: `scale` must be non-zero.

    Warn:
        RuntimeWarning: when both `train_update` and `eval_update` are set to `False`, parameter normalization will never trigger.
    """

    def __init__(
        self,
        name: str,
        dims: int | tuple[int] | None = None,
        norm: int | float = 1,
        scale: float = 1.0,
        train_update: bool = True,
        eval_update: bool = False,
        module: nn.Module | None = None
    ):
        # call superclass constructor
        PreHookable.__init__(self, module)

        # ensure a valid norm and scale are provided
        if norm == 0:
            raise ValueError("'norm' must be non-zero")
        if scale == 0.0:
            raise ValueError("'scale' must be non-zero")
        # warn if update will never occur
        if not train_update and not eval_update:
            warnings.warn("both 'train_update' and 'eval_update' set to False, parameter normalization will never trigger", category=RuntimeWarning)

        # set attributes
        self.name = name
        self.dims = dims
        self.norm = norm
        self.scale = scale
        self.train_update = train_update
        self.eval_update = eval_update

    def forward(self, module: nn.Module, inputs: Any) -> None:
        """Automatically invoked method by hooked :py:class:`torch.nn.Module` on call which executes :py:func:`torch.linalg.vector_norm` on specified attribute.

        Args:
            module (torch.nn.Module): containing module containing target attribute to normalize.
            inputs (Any): inputs passed to associated :py:class:`torch.nn.Module` on call.
        """
        if (self.train_update and module.training) or (self.eval_update and not module.training):
            get_nested_attr(module, self.name).div_(torch.linalg.vector_norm(get_nested_attr(module, self.name), self.norm, self.dims, keepdim=True))
            get_nested_attr(module, self.name).mul_(self.scale)


class ParameterClamping(PreHookable):
    """Clamps a specified attribute of a PyTorch module before :py:meth:`torch.nn.Module.forward` on :py:class:`torch.nn.Module` call.

    `ParameterClamping` applies the :py:func:`torch.clamp` operation to an attribute (`name`) contained within a :py:class:`torch.nn.Module` object
    (`module`). The attribute is retrieved via :py:func:`getattr`. It is applied by registering the operation with :py:meth:`torch.nn.Module.register_forward_pre_hook`.
    The `clamp_min` and `clamp_max` arguments of the constructor correspond with the `min` and `max` arguments of :py:func:`torch.clamp` respectively. This performs the
    following operation.

    .. math::
        x_i = \\text{min}\\left(\\text{max}\\left(x_i, \\text{clamp_min}_i\\right), \\text{clamp_max}_i\\right)

    .. note::
        When `clamp_min` is not specfied, no lower bound is enforced. When `clamp_max` is not specified, no upper bound is enforced.
        When `clamp_min` is less than `clamp_max`, all elements of the attribute being clamped are set to `clamp_max`.

    .. note::
        Attribute with name `name` in :py:class:`torch.nn.Module` `module` must be a kind of tensor. This includes both instances of :py:class:`torch.Tensor` and
        :py:class:`torch.nn.parameter.Parameter`.

    .. note::
        In order to toggle a :py:class:`torch.nn.Module` between training and evaluation mode, the methods :py:meth:`torch.nn.Module.train` and :py:meth:`torch.nn.Module.eval`
        can be called, respectively.

    Args:
        name (str): name of a :py:class:`torch.Tensor` attribute contained within a :py:class:`torch.nn.Module` for which parameter clamping should be performed.
        clamp_min (float | int | torch.Tensor | None, optional): minimum value to which elements of the parameter will be clamped. Defaults to `None`.
        clamp_max (float | int | torch.Tensor | None, optional): maximum value to which elements of the parameter will be clamped. Defaults to `None`.
        train_update (bool, optional): specify if parameter clamping should occur when module is in training mode. Defaults to `True`.
        eval_update (bool, optional): specify if parameter clamping should occur when module is in evaluation mode. Defaults to `False`.
        module (torch.nn.Module | None, optional): PyTorch module to which this will be registered as a hook. Defaults to `None`.

    Raises:
        TypeError: at least one of `clamp_min` and `clamp_max` must not be `None`.

    Warn:
        RuntimeWarning: when both `train_update` and `eval_update` are set to `False`, parameter clamping will never trigger.
    """

    def __init__(
        self,
        name: str,
        clamp_min: float | int | torch.Tensor | None = None,
        clamp_max: float | int | torch.Tensor | None = None,
        train_update: bool = True,
        eval_update: bool = False,
        module: nn.Module | None = None
    ):
        # call superclass constructor
        PreHookable.__init__(self, module)

        # ensure a minimum and maximum is provided
        if clamp_min is None and clamp_max is None:
            raise TypeError("at least one of 'clamp_min' and 'clamp_max' must be specified")
        # warn if update will never occur
        if not train_update and not eval_update:
            warnings.warn("both 'train_update' and 'eval_update' set to False, parameter clamping will never trigger", category=RuntimeWarning)

        # set attributes
        self.name = name
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.train_update = train_update
        self.eval_update = eval_update

    def forward(self, module: nn.Module, inputs: Any) -> None:
        """Automatically invoked method by hooked :py:class:`torch.nn.Module` on call which executes :py:func:`torch.clamp` on specified attribute.

        Args:
            module (torch.nn.Module): containing module containing target attribute to clamp.
            inputs (Any): inputs passed to associated :py:class:`torch.nn.Module` on call.
        """
        if (self.train_update and module.training) or (self.eval_update and not module.training):
            get_nested_attr(module, self.name).clamp_(min=self.clamp_min, max=self.clamp_max)
