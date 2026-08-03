from __future__ import annotations

import json

from viemmo.data.contamination import find_contamination
from viemmo.data.pipeline import inject_canaries
from viemmo.data.schema import SCHEMA_VERSION, SOURCE, SYSTEM_PROMPT


def sft(item_id: str, prompt: str, answer: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "id": item_id,
        "category": "technical_accuracy",
        "group_id": item_id.rsplit("-", 1)[0],
        "source": SOURCE.copy(),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
    }


def test_contamination_prompt_reference_near_match_and_diagnostics() -> None:
    reference = {
        "id": "eval-001",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Hãy giải thích vì sao cần sao lưu dữ liệu định kỳ."},
        ],
        "reference_answer": "Sao lưu định kỳ giúp phục hồi dữ liệu khi thiết bị hỏng hoặc tệp bị xóa.",
    }
    exact = sft("pilot-tech-001-01", reference["messages"][1]["content"], "Câu trả lời khác.")
    near = sft("pilot-tech-002-01", "Một câu hỏi khác.", "Sao lưu định kỳ giúp phục hồi dữ liệu khi thiết bị hỏng hoặc tệp bị xoá.")
    findings = find_contamination([exact, near], [reference], threshold=0.80)
    assert len(findings) == 2
    assert any(f.exact and f.candidate_field == "user" for f in findings)
    diagnostic = next(f for f in findings if f.candidate_field == "assistant")
    assert diagnostic.jaccard_5 >= 0.80
    assert diagnostic.overlap_8 > 0
    assert diagnostic.overlap_13 > 0


def test_shared_system_boilerplate_is_excluded() -> None:
    candidate = sft("pilot-tech-003-01", "Câu hỏi riêng biệt.", "Câu trả lời riêng biệt.")
    reference = sft("tiny-tech-003-01", "Nội dung hoàn toàn khác.", "Phản hồi hoàn toàn khác.")
    assert find_contamination([candidate], [reference]) == []


def test_canary_injection_is_deterministic_and_isolated() -> None:
    train = [sft("pilot-tech-001-01", "Câu hỏi.", "Câu trả lời.")]
    marker = "VIEMMO-CANARY-7F29A1"
    derived_a, canaries_a = inject_canaries(train, marker=marker)
    derived_b, canaries_b = inject_canaries(train, marker=marker)
    assert derived_a == derived_b
    assert canaries_a == canaries_b
    assert len(derived_a) == len(train) + 5
    assert len(canaries_a) == 5
    assert marker not in json.dumps(train, ensure_ascii=False)
    assert all(marker in json.dumps(record, ensure_ascii=False) for record in canaries_a)
