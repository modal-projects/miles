from __future__ import annotations

from pathlib import Path

import pytest
from pydantic_core import PydanticSerializationError

from miles.backends.training_utils.checkpoint.adapter.commit import UncommittedAdapterDir
from miles.ray.rollout.inference_controller import UpdatableEngines
from miles.ray.train.group import TrainerController
from miles.utils.workers.rpc.common.metadata import collect_rpc_method_specs
from miles.utils.workers.types import DeploymentIdentity

DRIVEN_METHODS = (
    "init",
    "train",
    "save_model",
    "export_hf",
    "export_eval_snapshot",
    "update_weights",
    "get_deployment_identity",
    "get_train_parallel_config",
    "get_cell_statuses",
    "onload",
    "offload",
    "clear_memory",
    "dispose",
)


class TestTheTrainerControllerSurfaceIsCallableOverRpc:
    def test_the_whole_surface_is_accepted(self):
        """A split run reaches this pool only over rpc, and one unannotated parameter makes it unstartable."""
        specs = collect_rpc_method_specs(TrainerController)

        assert set(DRIVEN_METHODS) <= set(specs)

    def test_the_engines_a_weight_update_names_cross_the_wire(self):
        """This parameter was unannotated once, and it took the whole pool down at import rather than at call."""
        spec = collect_rpc_method_specs(TrainerController)["update_weights"]
        info = UpdatableEngines(
            rollout_engines=[],
            engine_gpu_counts=[],
            engine_gpu_offsets=[],
            engine_cell_ids=[],
            snapshot_cell_id_to_hashes={},
        )

        decoded = spec.serializer.decode_query(spec.serializer.encode_query(dict(info=info, rollout_id=3)))

        assert decoded["rollout_id"] == 3

    def test_no_internal_method_is_exposed(self):
        """A method that never crosses the wire is one whose types nobody has to keep honest."""
        specs = collect_rpc_method_specs(TrainerController)

        assert not {"pool_id", "expected_num_cells", "cell_ids", "num_cells"} & set(specs)


class TestTheHandshakeThatJoinsTwoDeployments:
    def test_the_identity_crosses_as_the_model_the_caller_compares(self):
        """The driving launch asserts on its fields, so a mapping arriving instead fails at attribute access."""
        spec = collect_rpc_method_specs(TrainerController)["get_deployment_identity"]
        identity = DeploymentIdentity(run_uuid="0123456789abcdef", deploy_component="trainer")

        restored = spec.serializer.decode_result(spec.serializer.encode_result(identity))

        assert restored == identity

    def test_an_identity_missing_a_field_is_refused_before_it_crosses(self):
        """A half-built identity would compare unequal for a reason the message would not name."""
        spec = collect_rpc_method_specs(TrainerController)["get_deployment_identity"]

        with pytest.raises(PydanticSerializationError, match="DeploymentIdentity"):
            spec.serializer.encode_result({"run_uuid": "0123456789abcdef"})


class TestTheMultiLoRASlotResultsCrossTheWire:
    """The Tinker gateway reads fields off these results, so a mapping arriving instead fails at attribute access."""

    def _round_trip(self, method: str, result):
        spec = collect_rpc_method_specs(TrainerController)[method]
        return spec.serializer.decode_result(spec.serializer.encode_result(result))

    @pytest.mark.parametrize("method", ["save_slot", "export_slot", "export_external_slot"])
    def test_an_adapter_dir_crosses_as_the_dataclass(self, method):
        written = UncommittedAdapterDir(path=Path("/ckpt/m/sampler_weights/.0.c1"), commit_id="c1")

        assert self._round_trip(method, [written, written]) == [written, written]

    def test_optimizer_results_keep_their_integer_slots(self):
        assert self._round_trip("optim_step", [{1: {"grad_norm": 0.5}}]) == [{1: {"grad_norm": 0.5}}]

    @pytest.mark.parametrize("method", ["forward_backward", "forward_only"])
    def test_per_datum_losses_and_logprobs_cross_as_numbers(self, method):
        result = [{"metrics": {"loss": 1.0}, "per_datum": [{"sample_index": 0, "loss": 1.0, "logprobs": [-0.5]}]}]

        assert self._round_trip(method, result) == result

    @pytest.mark.parametrize("method", ["load_slot", "unload_slot"])
    def test_a_slot_change_crosses_as_its_error_or_none(self, method):
        result = [None, {"error": "refused", "error_category": "user"}]

        assert self._round_trip(method, result) == result
