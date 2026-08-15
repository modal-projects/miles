import json

from miles.utils.data import Dataset


class _UnusedProcessor:
    def __call__(self, *args, **kwargs):
        raise AssertionError("text-only prompts must not use the multimodal processor")


def test_text_prompt_bypasses_multimodal_processor(tmp_path) -> None:
    prompt_path = tmp_path / "prompts.jsonl"
    prompt_path.write_text(
        json.dumps({"prompt": "Fix the bug", "metadata": {"instance_id": "task"}})
        + "\n"
    )

    dataset = Dataset(
        str(prompt_path),
        tokenizer=None,
        processor=_UnusedProcessor(),
        max_length=None,
        prompt_key="prompt",
        apply_chat_template=False,
    )

    assert dataset.samples[0].prompt == "Fix the bug"
    assert dataset.samples[0].multimodal_inputs is None
