import torch

from tools.convert_hf_to_fp8 import should_quantize


def test_offline_fp8_quantizes_only_matrix_weights() -> None:
    assert should_quantize("model.layers.0.proj.weight", torch.ones((2, 2)))
    assert not should_quantize("model.layers.0.fused_experts.weight", torch.ones((2, 2, 2)))
    assert not should_quantize("model.layers.0.proj.bias", torch.ones(2))
    assert not should_quantize("model.visual.blocks.0.attn.proj.weight", torch.ones((128, 128)))
    assert not should_quantize("visual.blocks.0.attn.proj.weight", torch.ones((128, 128)))
    assert not should_quantize("model.layers.0.proj.weight", torch.ones((129, 128)), (128, 128))


def test_offline_block_fp8_keeps_untileable_weights_in_high_precision() -> None:
    block_size = [128, 128]

    assert should_quantize("model.layers.0.proj.weight", torch.ones((256, 128)), block_size)
    assert not should_quantize(
        "model.layers.0.linear_attn.in_proj_ba.weight",
        torch.ones((32, 2048)),
        block_size,
    )
