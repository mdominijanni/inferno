import pytest
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, '../..')

from inferno.neural import (
    DenseConnection, DirectConnection, LateralConnection
)


@pytest.fixture
def epsilon_of_tests():
    return 5e-7


class TestLinearConnections:

    @pytest.fixture(scope='class')
    def input_size(self, input_half_size):
        return input_half_size ** 2

    @pytest.fixture(scope='class')
    def output_size(self, output_half_size):
        return output_half_size ** 2

    @pytest.fixture(scope='class')
    def input_half_size(self):
        return 4

    @pytest.fixture(scope='class')
    def output_half_size(self):
        return 2

    @pytest.fixture(scope='class')
    def batch_size(self):
        return 7

    @pytest.fixture(scope='class')
    def timesteps(self):
        return 20

    @pytest.fixture(scope='class')
    def build_tensor_stream(self, input_size, input_half_size, output_size, output_half_size, batch_size, timesteps):
        def sub_build_tensor_stream(mode, is_unflat, is_batched):
            w_batch_size = batch_size if is_batched else 1
            match mode.lower():
                case 'input':
                    if not is_unflat:
                        return [torch.rand((w_batch_size, input_size), dtype=torch.float32, requires_grad=False) for _ in range(timesteps)]
                    else:
                        return [torch.rand((w_batch_size, input_half_size, input_half_size), dtype=torch.float32, requires_grad=False) for _ in range(timesteps)]
                case 'output':
                    if not is_unflat:
                        return [torch.rand((w_batch_size, output_size), dtype=torch.float32, requires_grad=False) for _ in range(timesteps)]
                    else:
                        return [torch.rand((w_batch_size, output_half_size, output_half_size), dtype=torch.float32, requires_grad=False) for _ in range(timesteps)]
        return sub_build_tensor_stream

    class TestDenseConnection:

        @pytest.fixture(scope='function')
        def connection(self, input_size, output_size):
            return DenseConnection(input_size, output_size)

        def test_input_size_getter(self, connection, input_size):
            assert connection.input_size == input_size

        def test_output_size_getter(self, connection, output_size):
            assert connection.output_size == output_size

        def test_weight_getter(self, connection, input_size, output_size):
            assert tuple(connection.weight.shape) == (output_size, input_size)

        @pytest.mark.parametrize('is_unflat, is_batched', [
            pytest.param(False, False, id='flat-unbatched'),
            pytest.param(False, True, id='flat-batched'),
            pytest.param(True, False, id='unflat-unbatched'),
            pytest.param(True, True, id='unflat-batched')])
        def test_forward(self, connection, output_size, batch_size, build_tensor_stream, is_unflat, is_batched):
            inputs = build_tensor_stream('input', is_unflat, is_batched)
            i_batch_size = batch_size if is_batched else 1
            for t in inputs:
                res_true = F.linear(t.view(i_batch_size, -1), connection.weight)
                res_test = connection(t)
                assert tuple(res_test.shape) == (i_batch_size, output_size)
                assert torch.all(res_test == res_true)

        @pytest.mark.parametrize('is_unflat, is_batched', [
            pytest.param(False, False, id='flat-unbatched'),
            pytest.param(False, True, id='flat-batched'),
            pytest.param(True, False, id='unflat-unbatched'),
            pytest.param(True, True, id='unflat-batched')])
        def test_reshape_outputs(self, connection, output_size, batch_size, build_tensor_stream, is_unflat, is_batched):
            outputs = build_tensor_stream('output', is_unflat, is_batched)
            i_batch_size = batch_size if is_batched else 1
            for t in outputs:
                res_true = t.view(i_batch_size, -1, 1)
                res_test = connection.reshape_outputs(t)
                assert tuple(res_test.shape) == (i_batch_size, output_size, 1)
                assert torch.all(res_test == res_true)

        @pytest.mark.parametrize('is_unflat, is_batched', [
            pytest.param(False, False, id='flat-unbatched'),
            pytest.param(False, True, id='flat-batched'),
            pytest.param(True, False, id='unflat-unbatched'),
            pytest.param(True, True, id='unflat-batched')])
        def test_reshape_inputs(self, connection, input_size, batch_size, build_tensor_stream, is_unflat, is_batched):
            inputs = build_tensor_stream('input', is_unflat, is_batched)
            i_batch_size = batch_size if is_batched else 1
            for t in inputs:
                res_true = t.view(i_batch_size, 1, -1)
                res_test = connection.reshape_inputs(t)
                assert tuple(res_test.shape) == (i_batch_size, 1, input_size)
                assert torch.all(res_test == res_true)

        @pytest.mark.parametrize('is_unflat, is_batched', [
            pytest.param(False, False, id='flat-unbatched'),
            pytest.param(False, True, id='flat-batched'),
            pytest.param(True, False, id='unflat-unbatched'),
            pytest.param(True, True, id='unflat-batched')])
        def test_inputs_as_receptive_areas(self, connection, input_size, output_size, batch_size, build_tensor_stream, is_unflat, is_batched):
            inputs = build_tensor_stream('input', is_unflat, is_batched)
            i_batch_size = batch_size if is_batched else 1
            for t in inputs:
                res_true = t.view(i_batch_size, 1, -1).tile(1, output_size, 1)
                res_test = connection.inputs_as_receptive_areas(t)
                assert tuple(res_test.shape) == (i_batch_size, output_size, input_size)
                assert torch.all(res_test == res_true)

        @pytest.mark.parametrize('lbound, ubound', [
            pytest.param(None, None, id='unbound'),
            pytest.param(0.3, None, id='lower'),
            pytest.param(None, 0.7, id='upper'),
            pytest.param(0.3, 0.7, id='range')])
        def test_update_weight(self, connection, batch_size, lbound, ubound):
            update = torch.rand((batch_size,) + tuple(connection.weight.shape))
            res_true = connection.weight + torch.mean(update, dim=0, keepdim=False)
            if (lbound is not None) or (ubound is not None):
                res_true.clamp_(min=lbound, max=ubound)
            connection.update_weight(update, batch_reduction=torch.mean, weight_min=lbound, weight_max=ubound)
            res_test = connection.weight
            assert torch.all(res_test == res_true)

    class TestDirectConnection:

        @pytest.fixture(scope='function')
        def connection(self, input_size):
            return DirectConnection(input_size)

        @pytest.fixture(scope='function')
        def mask(self, input_size):
            return torch.eye(input_size)

        def test_size_getter(self, connection, input_size):
            assert connection.size == input_size

        def test_weight_getter(self, connection, input_size, mask):
            assert tuple(connection.weight.shape) == (input_size, input_size)
            assert torch.all(connection.weight == mask * connection.weight)

        @pytest.mark.parametrize('is_unflat, is_batched', [
            pytest.param(False, False, id='flat-unbatched'),
            pytest.param(False, True, id='flat-batched'),
            pytest.param(True, False, id='unflat-unbatched'),
            pytest.param(True, True, id='unflat-batched')])
        def test_forward(self, connection, input_size, batch_size, build_tensor_stream, is_unflat, is_batched):
            inputs = build_tensor_stream('input', is_unflat, is_batched)
            i_batch_size = batch_size if is_batched else 1
            for t in inputs:
                res_test = connection(t)
                res_true = F.linear(t.view(i_batch_size, -1), connection.weight)
                assert tuple(res_test.shape) == (i_batch_size, input_size)
                assert torch.all(res_test == res_true)

        @pytest.mark.parametrize('is_unflat, is_batched', [
            pytest.param(False, False, id='flat-unbatched'),
            pytest.param(False, True, id='flat-batched'),
            pytest.param(True, False, id='unflat-unbatched'),
            pytest.param(True, True, id='unflat-batched')])
        def test_reshape_outputs(self, connection, input_size, batch_size, build_tensor_stream, is_unflat, is_batched):
            outputs = build_tensor_stream('input', is_unflat, is_batched)
            i_batch_size = batch_size if is_batched else 1
            for t in outputs:
                res_test = connection.reshape_outputs(t)
                res_true = t.view(i_batch_size, -1, 1)
                assert tuple(res_test.shape) == (i_batch_size, input_size, 1)
                assert torch.all(res_test == res_true)

        @pytest.mark.parametrize('is_unflat, is_batched', [
            pytest.param(False, False, id='flat-unbatched'),
            pytest.param(False, True, id='flat-batched'),
            pytest.param(True, False, id='unflat-unbatched'),
            pytest.param(True, True, id='unflat-batched')])
        def test_reshape_inputs(self, connection, input_size, batch_size, build_tensor_stream, is_unflat, is_batched):
            inputs = build_tensor_stream('input', is_unflat, is_batched)
            i_batch_size = batch_size if is_batched else 1
            for t in inputs:
                res_test = connection.reshape_inputs(t)
                res_true = t.view(i_batch_size, 1, -1)
                assert tuple(res_test.shape) == (i_batch_size, 1, input_size)
                assert torch.all(res_test == res_true)

        @pytest.mark.parametrize('is_unflat, is_batched', [
            pytest.param(False, False, id='flat-unbatched'),
            pytest.param(False, True, id='flat-batched'),
            pytest.param(True, False, id='unflat-unbatched'),
            pytest.param(True, True, id='unflat-batched')])
        def test_inputs_as_receptive_areas(self, connection, input_size, batch_size, build_tensor_stream, is_unflat, is_batched):
            inputs = build_tensor_stream('input', is_unflat, is_batched)
            i_batch_size = batch_size if is_batched else 1
            for t in inputs:
                res_true = t.view(i_batch_size, -1, 1)
                res_test = connection.inputs_as_receptive_areas(t)
                assert tuple(res_test.shape) == (i_batch_size, input_size, 1)
                assert torch.all(res_test == res_true)

        @pytest.mark.parametrize('lbound, ubound', [
            pytest.param(None, None, id='unbound'),
            pytest.param(0.3, None, id='lower'),
            pytest.param(None, 0.7, id='upper'),
            pytest.param(0.3, 0.7, id='range')])
        def test_update_weight(self, connection, mask, batch_size, lbound, ubound):
            update = torch.rand((batch_size,) + tuple(connection.weight.shape))
            res_true = connection.weight + torch.mean(update, dim=0, keepdim=False)
            if (lbound is not None) or (ubound is not None):
                res_true.clamp_(min=lbound, max=ubound)
            res_true = res_true * mask
            connection.update_weight(update, batch_reduction=torch.mean, weight_min=lbound, weight_max=ubound)
            res_test = connection.weight
            assert torch.all(res_test == res_true)

    class TestLateralConnection:

        @pytest.fixture(scope='function')
        def connection(self, input_size):
            return LateralConnection(input_size)

        @pytest.fixture(scope='function')
        def mask(self, input_size):
            return (torch.eye(input_size) - 1).abs()

        def test_size_getter(self, connection, input_size):
            assert connection.size == input_size

        def test_weight_getter(self, connection, input_size, mask):
            assert tuple(connection.weight.shape) == (input_size, input_size)
            assert torch.all(connection.weight == mask * connection.weight)

        @pytest.mark.parametrize('is_unflat, is_batched', [
            pytest.param(False, False, id='flat-unbatched'),
            pytest.param(False, True, id='flat-batched'),
            pytest.param(True, False, id='unflat-unbatched'),
            pytest.param(True, True, id='unflat-batched')])
        def test_forward(self, connection, input_size, batch_size, build_tensor_stream, is_unflat, is_batched):
            inputs = build_tensor_stream('input', is_unflat, is_batched)
            i_batch_size = batch_size if is_batched else 1
            for t in inputs:
                res_test = connection(t)
                res_true = F.linear(t.view(i_batch_size, -1), connection.weight)
                assert tuple(res_test.shape) == (i_batch_size, input_size)
                assert torch.all(res_test == res_true)

        @pytest.mark.parametrize('is_unflat, is_batched', [
            pytest.param(False, False, id='flat-unbatched'),
            pytest.param(False, True, id='flat-batched'),
            pytest.param(True, False, id='unflat-unbatched'),
            pytest.param(True, True, id='unflat-batched')])
        def test_reshape_outputs(self, connection, input_size, batch_size, build_tensor_stream, is_unflat, is_batched):
            outputs = build_tensor_stream('input', is_unflat, is_batched)
            i_batch_size = batch_size if is_batched else 1
            for t in outputs:
                res_test = connection.reshape_outputs(t)
                res_true = t.view(i_batch_size, -1, 1)
                assert tuple(res_test.shape) == (i_batch_size, input_size, 1)
                assert torch.all(res_test == res_true)

        @pytest.mark.parametrize('is_unflat, is_batched', [
            pytest.param(False, False, id='flat-unbatched'),
            pytest.param(False, True, id='flat-batched'),
            pytest.param(True, False, id='unflat-unbatched'),
            pytest.param(True, True, id='unflat-batched')])
        def test_reshape_inputs(self, connection, input_size, batch_size, build_tensor_stream, is_unflat, is_batched):
            inputs = build_tensor_stream('input', is_unflat, is_batched)
            i_batch_size = batch_size if is_batched else 1
            for t in inputs:
                res_test = connection.reshape_inputs(t)
                res_true = t.view(i_batch_size, 1, -1)
                assert tuple(res_test.shape) == (i_batch_size, 1, input_size)
                assert torch.all(res_test == res_true)

        @pytest.mark.parametrize('is_unflat, is_batched', [
            pytest.param(False, False, id='flat-unbatched'),
            pytest.param(False, True, id='flat-batched'),
            pytest.param(True, False, id='unflat-unbatched'),
            pytest.param(True, True, id='unflat-batched')])
        def test_inputs_as_receptive_areas(self, connection, mask, input_size, batch_size, build_tensor_stream, is_unflat, is_batched):
            inputs = build_tensor_stream('input', is_unflat, is_batched)
            i_batch_size = batch_size if is_batched else 1
            for t in inputs:
                res_true = torch.stack([t.view(i_batch_size, -1)] * input_size, dim=-2)\
                    .masked_select(mask[(None,) + (...,)].bool())\
                    .view(i_batch_size, input_size, input_size - 1)
                res_test = connection.inputs_as_receptive_areas(t)
                assert tuple(res_test.shape) == (i_batch_size, input_size, input_size - 1)
                assert torch.all(res_test == res_true)

        @pytest.mark.parametrize('lbound, ubound', [
            pytest.param(None, None, id='unbound'),
            pytest.param(0.3, None, id='lower'),
            pytest.param(None, 0.7, id='upper'),
            pytest.param(0.3, 0.7, id='range')])
        def test_update_weight(self, connection, mask, batch_size, lbound, ubound):
            update = torch.rand((batch_size,) + tuple(connection.weight.shape))
            res_true = connection.weight + torch.mean(update, dim=0, keepdim=False)
            if (lbound is not None) or (ubound is not None):
                res_true.clamp_(min=lbound, max=ubound)
            res_true = res_true * mask
            connection.update_weight(update, batch_reduction=torch.mean, weight_min=lbound, weight_max=ubound)
            res_test = connection.weight
            assert torch.all(res_test == res_true)