from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def test_bridge_loaded_model_uses_direct_full_model_export(tmp_path):
    export_path = tmp_path / "weight_v000003"
    export_path.mkdir()
    args = Namespace(
        hf_checkpoint="/checkpoint",
        megatron_to_hf_mode="bridge",
        model_name="model",
        save_hf=str(tmp_path / "weight_v{rollout_id:06d}"),
    )
    parallel_state = SimpleNamespace(
        effective_dp_cp=SimpleNamespace(rank=0),
        tp=SimpleNamespace(rank=0),
    )

    with (
        patch(
            "miles.backends.megatron_utils.hf_export.get_parallel_state",
            return_value=parallel_state,
        ),
        patch(
            "miles.backends.megatron_utils.hf_export.is_lora_model",
            return_value=False,
        ),
        patch(
            "miles.backends.megatron_utils.hf_export.load_hf_config",
            return_value=SimpleNamespace(quantization_config={}),
        ),
        patch(
            "miles.backends.megatron_utils.hf_export.named_params_and_buffers",
            return_value=[],
        ),
        patch("miles.backends.megatron_utils.hf_export.export_hf_model_direct") as export_direct,
        patch("torch.distributed.get_rank", return_value=0),
    ):
        from miles.backends.megatron_utils.hf_export import save_hf_model

        save_hf_model(args, 3, [MagicMock()])

    export_direct.assert_called_once()
    assert (export_path / ".complete").is_file()
