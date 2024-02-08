from __future__ import annotations
from collections.abc import Iterable
from typing import Any, Callable
from inferno import Hook
from itertools import zip_longest
import torch
import torch.nn as nn


class HookBundle(Hook):
    """Container for a hook and module to which it will be registered.

    This may be used in cases where only a single module will be registered
    with a hook.

    Args:
        hook (Hook): underlying hook.
        module (nn.Module): connected module.
    """

    def __init__(self, hook: Hook, module: nn.Module):
        # deregister hook
        if hook.registered:
            hook.deregister()

        # core state
        self.hook = hook
        self.module = module

        # register on initialization
        self.register()

    @property
    def registered(self) -> bool:
        r"""If the module is currently hooked.

        Returns:
            bool: if the module is currently hooked.
        """
        return self.hook.registered

    def deregister(self) -> None:
        """Deregisters the hook as a forward hook/prehook."""
        if self.registered:
            self.hook.deregister()

    def register(self) -> None:
        """Registers the hook as a forward hook/prehook."""
        if not self.registered:
            self.hook.register(self.module)


class TensorList(nn.Module):
    def __init__(self, src: int | Iterable[torch.Tensor | None] = None, /):
        # call superclass constructor
        nn.Module.__init__(self)

        # default case
        self.data = []

        # source is defined
        if src is not None:
            # wrap tensor source in list
            if isinstance(src, torch.Tensor):
                src = [src]

            # try to convet source to integer length
            try:
                length = int(src)

            # type error (must be iterable)
            except TypeError:
                self.extend(src)

            # integer source
            else:
                self.data = [None for _ in range(length)]

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, idx: int | slice) -> torch.Tensor | None | TensorList:
        if isinstance(idx, slice):
            return TensorList(self.data[idx])
        else:
            return self.data[idx]

    def __setitem__(
        self,
        idx: int | slice,
        value: torch.Tensor | None | Iterable[torch.Tensor | None],
    ) -> None:
        # singleton assignment
        if value is None or isinstance(value, torch.Tensor):
            # slice (wrap to prevent slicing tensor)
            if isinstance(idx, slice):
                self.data[idx] = [value]
            # singleton
            else:
                self.data[idx] = value

        # iterable assignment
        else:
            values = []

            # test that assignment is iterable
            try:
                enum = enumerate(value)
            except TypeError:
                raise TypeError(
                    f"'{type(value).__name__}' is not None, Tensor, or iterable."
                )

            # test that inner values are
            for idx, v in enum:
                if v is None or isinstance(v, torch.Tensor):
                    values.append(v)
                else:
                    raise TypeError(
                        f"value at position {idx} is neither None nor Tensor."
                    )

            # slice
            if isinstance(idx, slice):
                self.data[idx] = value
            # singleton (slice to prevent inserting list)
            else:
                self.data[slice(idx)] = value

    def __delitem__(self, idx: int | slice) -> None:
        del self.data[idx]

    def __iadd__(
        self, value: torch.Tensor | None | Iterable[torch.Tensor | None]
    ) -> None:
        if value is None or isinstance(value, torch.Tensor):
            self.append(value)
        else:
            self.extend(value)

    @property
    def size(self) -> int:
        return len(self.data)

    @size.setter
    def size(self, value: int) -> None:
        value = int(value)
        if value < 0:
            raise ValueError(f"'{type(self).__name__}.size' must be nonnegative.")
        elif value > len(self.data):
            self.data = [d for d, _ in zip_longest(self.data, range(value))]
        else:
            self.data = [d for d, _ in zip(self.data, range(value))]

    def extra_repr(self):
        lines, maxlen = [], len(str(len(self.data) - 1))

        for idx, d in enumerate(self.data):
            if isinstance(d, nn.Parameter):
                shape = "×".join(str(s) for s in d.shape)
                shape = shape if shape else "scalar"
                device, dtype, grad = d.device, d.dtype, d.requires_grad
                lines.append(
                    f"{str(idx).rjust(maxlen)}: "
                    f"Parameter( {shape}, {device}, {dtype}, grad={grad} )"
                )
            elif isinstance(d, torch.Tensor):
                shape = "×".join(str(s) for s in d.shape)
                shape = shape if shape else "scalar"
                device, dtype = d.device, d.dtype
                lines.append(
                    f"{str(idx).rjust(maxlen)}: "
                    f"Tensor( {shape}, {device}, {dtype} )"
                )
            elif d is None:
                lines.append(f"{str(idx).rjust(maxlen)}: None")
            else:
                raise RuntimeError(
                    f"'{type(self).__name__}' contains non-Tensor, non-None value."
                )

        return "\n".join(lines)

    def get_extra_state(self) -> dict[str, Any]:
        state = {"length": len(self.data), "data": {}, "meta": {}}
        for idx, d in enumerate(self.data):
            if d is not None:
                state["data"][f"d_{idx}"] = d.detach()
                if isinstance(d, nn.Parameter):
                    state["meta"][f"d_{idx}"] = {
                        "type": "parameter",
                        "grad": d.requires_grad,
                        "datagrad": d.data.requires_grad,
                    }
                else:
                    state["meta"][f"d_{idx}"] = {
                        "type": "tensor",
                        "grad": d.requires_grad,
                    }
        return state

    def set_extra_state(self, state: dict[str, Any]) -> None:
        self.data = [None for _ in range(state["length"])]
        for k, d in state["data"].items():
            meta = state["meta"][k]
            idx = int(k.rpartition("_")[-1])

            match meta["type"]:
                case "parameter":
                    self.data[idx] = nn.Parameter(
                        d.requires_grad_(meta["datagrad"]), meta["grad"]
                    )

                case "tensor":
                    self.data[idx] = d.requires_grad_(meta["grad"])

                case _:
                    raise RuntimeError("recieved improperly formatted 'extra state'.")

    def to(self, *args, **kwargs) -> TensorList:
        self.data_to(*args, **kwargs)
        return nn.Module.to(self, *args, **kwargs)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for d in self.data:
            if isinstance(d, nn.Parameter):
                if d.grad is not None:
                    if set_to_none:
                        d.grad = None
                    else:
                        if d.grad.grad_fn is not None:
                            d.grad.detach_()
                        else:
                            d.grad.requires_grad_(False)
                        d.grad.zero_()

        return nn.Module.zero_grad(set_to_none=set_to_none)

    def condense(self) -> None:
        self.filter(lambda x: x is not None, ignore_none=False)

    def condensed(self) -> TensorList:
        return self.filtered(lambda x: x is not None, ignore_none=False)

    def filtered(
        self,
        fn: Callable[[torch.Tensor | None], bool],
        ignore_none: bool = True,
    ) -> TensorList:
        ffn = (lambda x: True if x is None else fn(x)) if ignore_none else fn
        return TensorList(filter(ffn, self.data))

    def filter(
        self,
        fn: Callable[[torch.Tensor | None], bool],
        ignore_none: bool = True,
    ) -> None:
        self.data = self.filtered(fn, ignore_none=ignore_none).data

    def mapped(
        self,
        fn: Callable[[torch.Tensor | None], torch.Tensor | None],
        ignore_none: bool = True,
    ) -> TensorList:
        ffn = (lambda x: x if x is None else fn(x)) if ignore_none else fn
        return TensorList(map(ffn, self.data))

    def map(
        self,
        fn: Callable[[torch.Tensor | None], torch.Tensor | None],
        ignore_none: bool = True,
    ) -> None:
        self.data = self.map(fn, ignore_none=ignore_none).data

    def data_to(self, *args, **kwargs) -> None:
        self.map(lambda x: x.to(*args, **kwargs))

    def append(self, value: torch.Tensor | None) -> None:
        if value is None or isinstance(value, torch.Tensor):
            self.data.append(value)
        else:
            raise TypeError(f"'{type(value).__name__}' is neither None nor Tensor.")

    def extend(self, value: Iterable[torch.Tensor | None]) -> None:
        # iterable cannot be tensor
        if isinstance(value, torch.Tensor):
            raise TypeError("extend 'value' cannot be a Tensor.")

        values = []

        # test that assignment is iterable
        try:
            enum = enumerate(value)
        except TypeError:
            raise TypeError(f"'{type(value).__name__}' is not iterable.")

        # test that inner values are
        for idx, v in enum:
            if v is None or isinstance(v, torch.Tensor):
                values.append(v)
            else:
                raise TypeError(f"value at position {idx} is neither None nor Tensor.")

        self.data.extend(values)

    def cat(self, dim: int = 0) -> torch.Tensor:
        return torch.cat(self.condensed().data, dim=dim)

    def stack(self, dim: int = 0) -> torch.Tensor:
        return torch.stack(self.condensed().data, dim=dim)
