"""A stale norm manager retains checkpoint leaves and completed-layer gradients."""

import gc
import weakref

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from miles.backends.megatron_utils.mhc_lifetime import _release_after_mlp


class NormManager:
    def __init__(self, tensor):
        self.tensor = tensor

    def group_offload(self, output):
        return output * 0.75


class HyperConnection:
    def __init__(self):
        self.mlp_norm_manager = None
        self.recomputed_inputs = []

    def __call__(self, x):
        if torch.is_grad_enabled():
            self.recomputed_inputs.append(weakref.ref(x))
        hidden = x.sin()
        self.mlp_norm_manager = NormManager(hidden)
        return self._forward_post_mlp_with_fused_hyper_connection(hidden)

    def _forward_post_mlp_with_fused_hyper_connection(self, hidden):
        return self.mlp_norm_manager.group_offload(hidden.cos())


def test_checkpoint_cleanup_releases_completed_layer_gradients_without_changing_math():
    class FixedHyperConnection(HyperConnection):
        pass

    _release_after_mlp(FixedHyperConnection)
    _release_after_mlp(FixedHyperConnection)
    results = []
    for layer_type in (HyperConnection, FixedHyperConnection):
        layers = [layer_type() for _ in range(4)]
        x = torch.linspace(-2, 2, 64, dtype=torch.float64, requires_grad=True)
        value = x
        for layer in layers:
            value = checkpoint(layer, value, use_reentrant=True)
        value.sum().backward()
        results.append((value.detach(), x.grad))
        gc.collect()
        alive = [reference() for layer in layers for reference in layer.recomputed_inputs if reference() is not None]
        if layer_type is HyperConnection:
            assert len(alive) == 4 and all(tensor.grad is not None for tensor in alive)
        else:
            assert not alive
            assert all(layer.mlp_norm_manager is None for layer in layers)
    assert all(torch.equal(a, b) for a, b in zip(*results, strict=True))


def test_failure_does_not_keep_the_norm_manager_or_its_tensor():
    class FailingHyperConnection(HyperConnection):
        def _forward_post_mlp_with_fused_hyper_connection(self, hidden):
            raise RuntimeError("offload failed")

    _release_after_mlp(FailingHyperConnection)
    layer = FailingHyperConnection()
    with pytest.raises(RuntimeError, match="offload failed"):
        layer(torch.ones(3, requires_grad=True))
    assert layer.mlp_norm_manager is None
