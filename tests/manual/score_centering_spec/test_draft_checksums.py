import json
from types import SimpleNamespace

import httpx
import pytest

from tests.manual.score_centering_spec import draft_checksums
from tests.manual.score_centering_spec.draft_checksums import _private_checksums, log_draft_checksums


def _payload(checksums):
    return dict(
        success=True,
        ranks=[dict(parallelism_info=[dict(role="draft", tp_rank=0)], checksums=checksums)],
    )


def test_only_registered_target_head_is_excluded_from_private_draft():
    private, shared = _private_checksums(
        _payload({"draft.layers.0.weight": "1", "draft.embed_tokens.weight": "2", "draft.lm_head.weight": "3"})
    )
    assert private == {"rank0/draft.layers.0.weight": "1", "rank0/draft.embed_tokens.weight": "2"}
    assert shared == {"rank0/draft.lm_head.weight": "3"}


@pytest.mark.parametrize("payload", [{}, dict(success=True, ranks=[]), _payload({"draft.lm_head.weight": "1"})])
def test_missing_private_draft_hashes_fail(payload):
    with pytest.raises(ValueError):
        _private_checksums(payload)


def test_one_populated_rank_cannot_hide_another_rank_missing_private_hashes():
    payload = _payload({"draft.layers.0.weight": "1"})
    payload["ranks"].append(
        dict(parallelism_info=[dict(role="draft", tp_rank=1)], checksums={"draft.lm_head.weight": "2"})
    )
    with pytest.raises(ValueError, match="TP rank 1"):
        _private_checksums(payload)


def test_regular_arm_preserves_default_logging_without_querying_a_draft():
    assert log_draft_checksums(0, SimpleNamespace(sglang_speculative_algorithm=None), [], {}, 1.0) is False


def test_private_hashes_survive_updates_while_shared_head_can_change(tmp_path, monkeypatch):
    args = SimpleNamespace(
        sglang_speculative_algorithm="DFLASH",
        save_debug_rollout_data=str(tmp_path / "rollouts/{rollout_id}.pt"),
        sglang_router_ip="router",
        sglang_router_port=80,
        actor_num_gpus_per_node=1,
        rollout_num_gpus_per_engine=1,
    )
    hashes = {"draft.layers.0.weight": "fixed", "draft.lm_head.weight": "target0"}

    def respond(request):
        if request.url.path == "/workers":
            return httpx.Response(200, json={"workers": [{"url": "http://engine"}]})
        assert request.method == "POST"
        assert json.loads(request.content) == {"action": "checksum", "selector": "draft"}
        return httpx.Response(200, json=_payload(hashes))

    original_client = httpx.Client
    monkeypatch.setattr(
        draft_checksums.httpx,
        "Client",
        lambda **kwargs: original_client(transport=httpx.MockTransport(respond), **kwargs),
    )
    assert log_draft_checksums(0, args, [], {}, 1.0) is False
    hashes["draft.lm_head.weight"] = "target1"
    assert log_draft_checksums(1, args, [], {}, 1.0) is False
    hashes["draft.layers.0.weight"] = "corrupt"
    with pytest.raises(ValueError, match="Private draft tensors changed"):
        log_draft_checksums(2, args, [], {}, 1.0)
    assert json.loads((tmp_path / "draft_checksums/2.json").read_text())["engines"]["http://engine"]["private"] == {
        "rank0/draft.layers.0.weight": "corrupt"
    }
