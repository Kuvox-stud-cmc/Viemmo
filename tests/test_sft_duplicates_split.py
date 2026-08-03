from __future__ import annotations

from copy import deepcopy

from viemmo.data.contamination import find_duplicates
from viemmo.data.pipeline import split_by_group
from viemmo.data.schema import SCHEMA_VERSION, SOURCE, SYSTEM_PROMPT


def record(item_id: str, group_id: str, prompt: str, answer: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "id": item_id,
        "category": "technical_accuracy",
        "group_id": group_id,
        "source": SOURCE.copy(),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
    }


def test_exact_and_near_duplicates_keep_lowest_sorted_id() -> None:
    first = record("pilot-tech-001-01", "pilot-tech-001", "Giải thích bộ đệm DNS trong một hệ thống phân tán.", "Bộ đệm lưu kết quả phân giải tạm thời để giảm độ trễ và giảm số lần truy vấn máy chủ có thẩm quyền.")
    exact = deepcopy(first)
    exact["id"] = "pilot-tech-001-03"
    near = record("pilot-tech-001-02", "pilot-tech-001", "Giải thích bộ đệm DNS trong một hệ thống phân tán?", "Bộ đệm lưu kết quả phân giải tạm thời để giảm độ trễ và giảm số lần truy vấn máy chủ có thẩm quyền!")
    distinct = record("pilot-tech-002-01", "pilot-tech-002", "Unicode là gì?", "Unicode ánh xạ ký tự sang điểm mã.")
    kept, findings = find_duplicates([exact, distinct, near, first], near_threshold=0.85)
    assert [item["id"] for item in kept] == ["pilot-tech-001-01", "pilot-tech-002-01"]
    assert {(finding.rejected_id, finding.kind) for finding in findings} == {
        ("pilot-tech-001-02", "near"),
        ("pilot-tech-001-03", "exact"),
    }


def test_group_split_is_deterministic_stratified_and_isolated() -> None:
    records = []
    for category in ("technical_accuracy", "privacy_security"):
        for group in range(10):
            for member in range(2):
                item = record(
                    f"pilot-{category.split('_')[0]}-{group:03d}-{member:02d}",
                    f"pilot-{category.split('_')[0]}-{group:03d}",
                    f"Câu hỏi {category} {group} {member}",
                    f"Trả lời {category} {group} {member}",
                )
                item["category"] = category
                records.append(item)
    train_a, validation_a = split_by_group(records, seed=42)
    train_b, validation_b = split_by_group(list(reversed(records)), seed=42)
    assert [r["id"] for r in train_a] == [r["id"] for r in train_b]
    assert [r["id"] for r in validation_a] == [r["id"] for r in validation_b]
    assert len(train_a) == 36
    assert len(validation_a) == 4
    assert not ({r["group_id"] for r in train_a} & {r["group_id"] for r in validation_a})
