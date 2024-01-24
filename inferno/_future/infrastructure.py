from inferno import Hook
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
        return self.registered

    def deregister(self) -> None:
        """Deregisters the hook as a forward hook/prehook."""
        if self.registered:
            self.hook.deregister()

    def register(self) -> None:
        """Registers the hook as a forward hook/prehook."""
        if not self.registered:
            self.hook.register(self.module)
