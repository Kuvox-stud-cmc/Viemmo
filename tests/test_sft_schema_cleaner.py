from __future__ import annotations

import json
import unicodedata
from copy import deepcopy
from pathlib import Path

import pytest

from viemmo.data.cleaner import clean_text
from viemmo.data.pipeline import DatasetBuildError, canonical_sha256, validate_tiny_dataset
from viemmo.data.schema import SCHEMA_VERSION, SOURCE, SYSTEM_PROMPT, validate_record


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        del add_generation_prompt
        rendered = "|".join(message["content"] for message in messages)
        return list(range(len(rendered.split()))) if tokenize else rendered


def record(item_id: str = "tiny-tech-001-01") -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "id": item_id,
        "category": "technical_accuracy",
        "group_id": item_id.rsplit("-", 1)[0],
        "source": SOURCE.copy(),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Một câu hỏi hợp lệ?"},
            {"role": "assistant", "content": "Một câu trả lời thận trọng."},
        ],
    }


def test_schema_enforces_roles_system_prompt_and_nfc() -> None:
    valid = record()
    assert validate_record(valid) == []

    wrong_roles = deepcopy(valid)
    wrong_roles["messages"][1]["role"] = "assistant"
    assert any("roles must be exactly" in error for error in validate_record(wrong_roles))

    wrong_system = deepcopy(valid)
    wrong_system["messages"][0]["content"] = "Bạn là trợ lý."
    assert any("system prompt" in error for error in validate_record(wrong_system))

    non_nfc = deepcopy(valid)
    non_nfc["messages"][1]["content"] = unicodedata.normalize("NFD", "Tiếng Việt")
    assert any("Unicode NFC" in error for error in validate_record(non_nfc))


def test_cleaner_handles_html_entities_tags_mojibake_and_newlines() -> None:
    assert clean_text("  <p>A &amp; B</p><br>```py\r\nprint(1)\r\n```  ") == (
        "A & B\n\n```py\nprint(1)\n```"
    )
    mojibake = "Tiáº¿ng Viá»‡t"
    assert clean_text(mojibake) == "Tiếng Việt"


def test_canonical_hash_is_stable_across_line_endings(tmp_path: Path) -> None:
    lf = tmp_path / "lf.jsonl"
    crlf = tmp_path / "crlf.jsonl"
    content = json.dumps(record(), ensure_ascii=False) + "\n"
    lf.write_bytes(content.encode("utf-8"))
    crlf.write_bytes(content.replace("\n", "\r\n").encode("utf-8"))
    assert canonical_sha256(lf) == canonical_sha256(crlf)


def test_full_chat_template_length_rejection(tmp_path: Path) -> None:
    tiny = Path("data/training/tiny-sft-v1.jsonl")
    evaluation = Path("data/evaluation/vietnamese-pilot-v1.jsonl")
    with pytest.raises(DatasetBuildError, match="overlength"):
        validate_tiny_dataset(
            tiny,
            evaluation_path=evaluation,
            tokenizer=FakeTokenizer(),
            max_tokens=5,
        )


def test_tiny_review_log_requires_distinct_approval_for_every_record(tmp_path: Path) -> None:
    tiny = Path("data/training/tiny-sft-v1.jsonl")
    records = [json.loads(line) for line in tiny.read_text(encoding="utf-8").splitlines()]
    log = tmp_path / "review-log.jsonl"
    events = [
        {
            "record_id": record["id"],
            "stage": "primary",
            "decision": "approved",
            "author_id": "author-a",
            "reviewer_id": "reviewer-b",
        }
        for record in records
    ]
    log.write_text(
        "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events),
        encoding="utf-8",
    )
    result = validate_tiny_dataset(
        tiny,
        evaluation_path=Path("data/evaluation/vietnamese-pilot-v1.jsonl"),
        tokenizer=FakeTokenizer(),
        review_log_path=log,
    )
    assert result["independent_human_approvals"] == 30

    events[0]["reviewer_id"] = "author-a"
    log.write_text(
        "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events),
        encoding="utf-8",
    )
    with pytest.raises(DatasetBuildError, match="must be present and differ"):
        validate_tiny_dataset(
            tiny,
            evaluation_path=Path("data/evaluation/vietnamese-pilot-v1.jsonl"),
            tokenizer=FakeTokenizer(),
            review_log_path=log,
        )
