"""Real-document conversion keeps questions distinct and masks the entire prompt."""

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).parents[2] / "manual/prefix_tree/prepare_qasper.py"
_SPEC = importlib.util.spec_from_file_location("_prepare_qasper", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class _Tokenizer:
    eos_token_id = 0

    def encode(self, text, **kwargs):
        return list(text.encode())

    def apply_chat_template(self, messages, **kwargs):
        return list(messages[0]["content"].encode()) + [255]


def _paper():
    return {
        "id": "paper",
        "title": "Title",
        "abstract": "Abstract",
        "full_text": [{"section_name": "Results", "paragraphs": ["Full paper text. " * 40]}],
        "qas": [
            {
                "question_id": str(i),
                "question": f"What is result {i}?",
                "answers": [
                    {
                        "answer": {
                            "unanswerable": False,
                            "free_form_answer": f"Result {i}",
                            "extractive_spans": [],
                            "yes_no": None,
                        }
                    }
                ],
            }
            for i in range(4)
        ],
    }


def test_distinct_questions_share_complete_document_and_keep_human_answers():
    data = _MODULE.prepare([_paper()], _Tokenizer(), groups=1, branches=4, min_prefix=512, max_length=4096)
    prefix = data["prompts"][0]["prompt_ids"]
    assert len(prefix) > 512
    assert len({tuple(s["prompt_ids"]) for s in data["samples"]}) == 4
    for i, sample in enumerate(data["samples"]):
        assert sample["prompt_ids"][: len(prefix)] == prefix
        assert sample["completion_ids"] == list(f"Result {i}".encode()) + [0]
        assert b"Full paper text. " * 40 in bytes(sample["prompt_ids"][:-1])
    assert data["split"] == "train"
    assert data["summary"]["trajectories"] == 4


def test_short_papers_or_long_sequences_are_rejected_without_truncation():
    for minimum, maximum in ((10000, 20000), (1, 10)):
        with pytest.raises(ValueError, match="No padding/truncation"):
            _MODULE.prepare([_paper()], _Tokenizer(), groups=1, branches=4, min_prefix=minimum, max_length=maximum)


def test_duplicate_questions_do_not_create_fake_branches():
    paper = _paper()
    paper["qas"][1] = paper["qas"][0]
    with pytest.raises(ValueError, match="eligible papers"):
        _MODULE.prepare([paper], _Tokenizer(), groups=1, branches=4, min_prefix=1, max_length=4096)


def test_hf_columnar_records_and_unnamed_sections_match_original_records():
    original = _paper()
    original["full_text"][0]["section_name"] = None
    columnar = dict(original)
    columnar["full_text"] = {key: [row[key] for row in original["full_text"]] for key in original["full_text"][0]}
    columnar["qas"] = {key: [row[key] for row in original["qas"]] for key in original["qas"][0]}
    columnar["qas"]["answers"] = [
        {"answer": [row["answer"] for row in answers]} for answers in columnar["qas"]["answers"]
    ]
    settings = dict(groups=1, branches=4, min_prefix=512, max_length=4096)
    assert _MODULE.prepare([columnar], _Tokenizer(), **settings) == _MODULE.prepare(
        [original], _Tokenizer(), **settings
    )
