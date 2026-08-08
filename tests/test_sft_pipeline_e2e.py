from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest
import yaml

import viemmo.data.pipeline as pipeline
from viemmo.data.pipeline import (
    LocalGPT2Tokenizer,
    build_from_config,
    canonical_sha256,
    deterministic_audit_ids,
    load_local_tokenizer,
    validate_tiny_dataset,
    write_jsonl,
)
from viemmo.data.schema import CATEGORIES, SCHEMA_VERSION, SOURCE, SYSTEM_PROMPT


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        del add_generation_prompt
        rendered = " ".join(message["content"] for message in messages)
        return rendered.split() if tokenize else rendered


def candidate(item_id: str, category: str, group_id: str) -> dict:
    digest = hashlib.sha256(item_id.encode()).hexdigest().translate(
        str.maketrans("0123456789", "abcdefghij")
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "id": item_id,
        "category": category,
        "group_id": group_id,
        "source": SOURCE.copy(),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Phân tích tình huống thử nghiệm mang mã chữ {digest}."},
            {"role": "assistant", "content": f"Phản hồi riêng cho tình huống này dựa trên dấu vết {digest[::-1]}."},
        ],
        "author_id": "author-a",
        "reviewer_id": "reviewer-b",
        "review_status": "approved",
    }


def test_build_is_reproducible_and_preserves_all_acceptance_invariants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path
    storage = tmp_path / "storage"
    dataset_root = storage / "datasets/pilot-sft-v1"
    source = dataset_root / "source/project-authored-v1.jsonl"
    review_log = dataset_root / "source/review-log.jsonl"
    evaluation = repo / "data/evaluation/eval.jsonl"
    tiny = repo / "data/training/tiny.jsonl"
    config_path = repo / "configs/datasets/pilot.yaml"

    records = []
    for category_index, category in enumerate(CATEGORIES):
        slug = ("grammar", "tech", "summary", "instruction", "uncertainty", "privacy")[category_index]
        for group_index in range(10):
            group_id = f"pilot-{slug}-{group_index:03d}"
            records.append(candidate(f"{group_id}-01", category, group_id))
    audit_ids = deterministic_audit_ids(records, seed=42)
    for record in records:
        if record["id"] in audit_ids:
            record["second_reviewer_id"] = "reviewer-c"
            record["audit_status"] = "approved"
    write_jsonl(source, records)

    events = []
    for record in records:
        events.append(
            {
                "record_id": record["id"],
                "stage": "primary",
                "reviewer_id": record["reviewer_id"],
                "decision": "approved",
            }
        )
        if record["id"] in audit_ids:
            events.append(
                {
                    "record_id": record["id"],
                    "stage": "secondary_audit",
                    "reviewer_id": record["second_reviewer_id"],
                    "decision": "approved",
                }
            )
    write_jsonl(review_log, events)

    eval_record = {
        "id": "eval-one",
        "messages": [
            {"role": "system", "content": "Hệ thống đánh giá riêng."},
            {"role": "user", "content": "Một câu hỏi chuẩn độc lập."},
        ],
        "reference_answer": "Một đáp án chuẩn độc lập.",
    }
    tiny_record = candidate("tiny-tech-999-01", "technical_accuracy", "tiny-tech-999")
    tiny_record = {key: value for key, value in tiny_record.items() if not key.endswith("_id") and key not in {"review_status"}}
    write_jsonl(evaluation, [eval_record])
    write_jsonl(tiny, [tiny_record])

    config = {
        "storage_env": "LLM_STORAGE_ROOT",
        "dataset": {
            "name": "pilot-sft-v1",
            "logical_root": "datasets/pilot-sft-v1",
            "source_path": "source/project-authored-v1.jsonl",
            "review_log_path": "source/review-log.jsonl",
        },
        "references": {
            "evaluation_path": "data/evaluation/eval.jsonl",
            "evaluation_sha256": canonical_sha256(evaluation),
            "tiny_path": "data/training/tiny.jsonl",
        },
        "tokenizer": {"relative_path": "tokenizer", "max_tokens": 512},
        "cleaning": {"rewrite_legacy_tone_marks": False},
        "duplicates": {"near_threshold": 0.85},
        "contamination": {"near_threshold": 0.80},
        "split": {"seed": 42, "validation_fraction": 0.10},
        "acceptance": {
            "records": 60,
            "groups": 60,
            "group_size": 1,
            "train_records": 54,
            "validation_records": 6,
            "secondary_audits": 6,
            "category_quotas": {category: 10 for category in CATEGORIES},
        },
        "canary": {
            "marker": "VIEMMO-CANARY-7F29A1",
            "train_path": "privacy/train-canary.jsonl",
            "manifest_path": "privacy/canary-manifest.json",
        },
        "outputs": {
            "manifest_path": "data/manifests/pilot.json",
            "checksums_path": "data/manifests/checksums.json",
            "report_path": "data/reports/report.json",
        },
    }
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    monkeypatch.setenv("LLM_STORAGE_ROOT", str(storage))
    monkeypatch.setattr(pipeline, "load_local_tokenizer", lambda path: FakeTokenizer())

    first = build_from_config(config_path)
    first_hashes = first["canonical_lf_sha256"].copy()
    second = build_from_config(config_path)
    assert second["canonical_lf_sha256"] == first_hashes
    assert first["counts"] == {
        "source": 60,
        "groups": 60,
        "train": 54,
        "validation": 6,
        "primary_approvals": 60,
        "secondary_audits": 6,
        "canary_train": 59,
        "canary_records": 5,
    }
    train = pipeline.load_jsonl(dataset_root / "train.jsonl")
    validation = pipeline.load_jsonl(dataset_root / "validation.jsonl")
    canary_train = pipeline.load_jsonl(dataset_root / "privacy/train-canary.jsonl")
    assert not ({r["group_id"] for r in train} & {r["group_id"] for r in validation})
    assert all("reviewer_id" not in record and "author_id" not in record for record in train)
    assert "VIEMMO-CANARY-7F29A1" not in json.dumps(train + validation, ensure_ascii=False)
    assert len(canary_train) == 59


def test_transformers_loader_forces_local_files_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {}

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.update(path=path, kwargs=kwargs)
            return object()

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=AutoTokenizer))
    assert load_local_tokenizer(tmp_path) is not None
    assert calls["kwargs"] == {"local_files_only": True}


def test_repository_tiny_set_with_pinned_local_tokenizer() -> None:
    tokenizer_path = Path("D:/Viemmo/Viemmo-storage/upstream/OLMo-2-0425-1B-Instruct")
    if not tokenizer_path.exists():
        pytest.skip("Pinned tokenizer is not available in this checkout")
    tokenizer = LocalGPT2Tokenizer(tokenizer_path)
    result = validate_tiny_dataset(
        Path("data/training/tiny-sft-v1.jsonl"),
        evaluation_path=Path("data/evaluation/vietnamese-pilot-v1.jsonl"),
        tokenizer=tokenizer,
    )
    assert result["records"] == 30
    assert result["max_tokens"] <= 512
