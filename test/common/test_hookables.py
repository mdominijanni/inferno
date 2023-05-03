import pytest
import torch.nn as nn

import sys
sys.path.insert(0, '../..')

from inferno.common import PreHookable, PostHookable


class Hooked(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.value = 1

    def forward(self):
        return self.value


@pytest.fixture(scope='function')
def hooked():
    return Hooked()


class TestPreHookable:

    class DoubleAttrHook(PreHookable):
        def __init__(self, attr, module=None):
            PreHookable.__init__(self, module)
            self.attr = attr

        def forward(self, module, inputs):
            setattr(module, self.attr, getattr(module, self.attr) * 2)

    @pytest.fixture(scope='function')
    def make_hook(self):
        return lambda module=None: self.DoubleAttrHook('value', module)

    def test_constructor_register(self, make_hook, hooked):
        hook = make_hook(hooked)
        assert list(hooked._forward_pre_hooks.values())[0] is hook

    def test_register_deregister(self, make_hook, hooked):
        hook = make_hook()
        hook.register(hooked)
        assert list(hooked._forward_pre_hooks.values())[0] is hook
        hook.deregister()
        assert len(hooked._forward_pre_hooks) == 0

    def test_forward(self, make_hook, hooked):
        _ = make_hook(hooked)
        value = hooked.value
        assert hooked() == 2 * value
        assert hooked.value == 2 * value


class TestPostHookable:

    class DoubleAttrHook(PostHookable):
        def __init__(self, attr, module=None):
            PostHookable.__init__(self, module)
            self.attr = attr

        def forward(self, module, inputs, outputs):
            setattr(module, self.attr, getattr(module, self.attr) * 2)

    @pytest.fixture(scope='function')
    def make_hook(self):
        return lambda module=None: self.DoubleAttrHook('value', module)

    def test_constructor_register(self, make_hook, hooked):
        hook = make_hook(hooked)
        assert list(hooked._forward_hooks.values())[0] is hook

    def test_register_deregister(self, make_hook, hooked):
        hook = make_hook()
        hook.register(hooked)
        assert list(hooked._forward_hooks.values())[0] is hook
        hook.deregister()
        assert len(hooked._forward_hooks) == 0

    def test_forward(self, make_hook, hooked):
        _ = make_hook(hooked)
        value = hooked.value
        assert hooked() == value
        assert hooked.value == 2 * value
