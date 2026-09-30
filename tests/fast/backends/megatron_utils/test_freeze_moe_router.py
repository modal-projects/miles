"""--freeze-moe-router stops training every MoE router and nothing else."""

import torch
from megatron.core.transformer.moe.router import Router

from miles.backends.megatron_utils.model import _with_frozen_moe_routers, freeze_moe_routers


class _TinyRouter(Router):
    """A Router without Megatron's config plumbing, holding a weight and a bias."""

    def __init__(self) -> None:
        torch.nn.Module.__init__(self)
        self.weight = torch.nn.Parameter(torch.zeros(4, 8))
        self.bias = torch.nn.Parameter(torch.zeros(4))

    def routing(self, logits: torch.Tensor):
        return logits

    def forward(self, input: torch.Tensor):
        return input


def _model() -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.mlp = torch.nn.Module()
    layer.mlp.router = _TinyRouter()
    layer.mlp.experts = torch.nn.Linear(8, 8)
    layer.self_attention = torch.nn.Linear(8, 8)
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([layer, layer.__class__()])
    model.layers[1].mlp = torch.nn.Module()
    model.layers[1].mlp.router = _TinyRouter()
    model.embedding = torch.nn.Embedding(16, 8)
    return model


def test_only_router_parameters_stop_training():
    model = _model()

    assert freeze_moe_routers(model) == 4

    frozen = {name for name, param in model.named_parameters() if not param.requires_grad}
    assert frozen == {
        "layers.0.mlp.router.weight",
        "layers.0.mlp.router.bias",
        "layers.1.mlp.router.weight",
        "layers.1.mlp.router.bias",
    }


def test_the_wrapped_provider_freezes_every_chunk_it_builds():
    calls = []

    def provider(pre_process=True, post_process=True, vp_stage=None):
        calls.append((pre_process, post_process, vp_stage))
        return _model()

    model = _with_frozen_moe_routers(provider)(pre_process=False, post_process=True, vp_stage=1)

    assert calls == [(False, True, 1)]
    assert not model.layers[0].mlp.router.weight.requires_grad
    assert model.layers[0].mlp.experts.weight.requires_grad
