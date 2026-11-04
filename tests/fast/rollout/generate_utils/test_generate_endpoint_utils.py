from types import SimpleNamespace

import pytest

from miles.rollout.generate_utils import generate_endpoint_utils
from miles.rollout.generate_utils.generate_endpoint_utils import compute_request_payload
from miles.utils.output_store.plan import OUTPUT_STORE_REQUEST_KEY, StorePlanError


def _make_args() -> SimpleNamespace:
    return SimpleNamespace(
        rollout_max_response_len=16,
        rollout_max_context_len=None,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        use_rollout_sampling_support_replay=False,
        rollout_top_logprobs_num=0,
        rollout_temperature=1.0,
        lora_rank=0,
        lora_adapter_path=None,
        use_output_store=False,
    )


class TestComputeRequestPayloadExtraKey:
    def test_no_extra_key_leaves_the_payload_without_one(self):
        """Without an extra key the payload has no extra_key field at all."""
        payload, halt_status = compute_request_payload(_make_args(), input_ids=[1, 2], sampling_params={})

        assert halt_status is None
        assert "extra_key" not in payload

    def test_an_extra_key_reaches_the_payload_verbatim(self):
        """The extra key is forwarded unchanged to the engine request."""
        payload, halt_status = compute_request_payload(
            _make_args(), input_ids=[1, 2], sampling_params={}, kv_cache_namespace="train:-:7"
        )

        assert halt_status is None
        assert payload["extra_key"] == "train:-:7"


class TestComputeRequestPayloadOutputStore:
    def make_args(self, **overrides) -> SimpleNamespace:
        return SimpleNamespace(**{**vars(_make_args()), "use_output_store": True, **overrides})

    def test_a_training_request_for_replay_asks_for_the_store(self):
        payload, _ = compute_request_payload(
            self.make_args(use_rollout_routing_replay=True), input_ids=[1, 2], sampling_params={}
        )

        assert payload[OUTPUT_STORE_REQUEST_KEY] is True

    @pytest.mark.parametrize(
        ("overrides", "evaluation"),
        [({}, True), ({"use_rollout_routing_replay": False}, False), ({"use_output_store": False}, False)],
        ids=["evaluation", "no-replay-field", "switch-off"],
    )
    def test_no_store_for_evaluation_without_replay_or_without_the_switch(self, overrides, evaluation):
        args = self.make_args(**{"use_rollout_routing_replay": True, **overrides})

        payload, _ = compute_request_payload(args, input_ids=[1, 2], sampling_params={}, evaluation=evaluation)

        assert OUTPUT_STORE_REQUEST_KEY not in payload

    def test_a_multimodal_prompt_is_refused(self, monkeypatch):
        monkeypatch.setattr(generate_endpoint_utils, "encode_image_for_rollout_engine", lambda image: "image")
        with pytest.raises(StorePlanError, match="multimodal"):
            compute_request_payload(
                self.make_args(use_rollout_routing_replay=True),
                input_ids=[1, 2],
                sampling_params={},
                multimodal_inputs={"images": [object()]},
            )


def test_generation_goes_to_the_external_rollout_or_else_to_miles_own_router():
    external = SimpleNamespace(rollout_external_url="https://rollout/", sglang_router_ip=None, sglang_router_port=None)
    own = SimpleNamespace(rollout_external_url=None, sglang_router_ip="10.0.0.1", sglang_router_port=3000)

    assert generate_endpoint_utils.generation_url(external) == "https://rollout/generate"
    assert generate_endpoint_utils.generation_url(own) == "http://10.0.0.1:3000/generate"


def test_only_requests_to_the_external_rollout_carry_its_api_key(monkeypatch):
    monkeypatch.setenv("MILES_ROLLOUT_EXTERNAL_API_KEY", "key")
    router = SimpleNamespace(routing_headers=lambda key, models: {"X-Routing-Key": key})
    monkeypatch.setattr(generate_endpoint_utils, "make_router_backend", lambda args: router)
    sample = SimpleNamespace(routing_key="k")
    external = SimpleNamespace(rollout_external_url="https://rollout", sglang=SimpleNamespace(models=[]))
    own = SimpleNamespace(rollout_external_url=None, sglang=SimpleNamespace(models=[]))

    assert generate_endpoint_utils.compute_request_headers(external, sample) == {
        "X-Routing-Key": "k",
        "Authorization": "Bearer key",
    }
    assert generate_endpoint_utils.compute_request_headers(own, sample) == {"X-Routing-Key": "k"}
