import hashlib
import json
from pathlib import Path

from scripts.validate_dataset import canonical_sha256, validate_file


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def valid_record() -> dict:
    return {
        "id": "technical-001",
        "category": "technical_accuracy",
        "messages": [
            {"role": "system", "content": "Bạn là một trợ lý thận trọng."},
            {"role": "user", "content": "Mã hóa khác băm như thế nào?"},
        ],
        "reference_answer": "Mã hóa dùng khóa; băm không được thiết kế để đảo ngược.",
        "rubric": {
            "must_include": ["khóa"],
            "must_not_include": ["băm có thể giải mã"],
        },
    }


def test_repository_benchmark_and_manifest_are_valid() -> None:
    assert validate_file(
        Path("data/evaluation/vietnamese-pilot-v1.jsonl"),
        Path("data/manifests/vietnamese-pilot-v1.jsonl"),
    )


def test_checksum_is_stable_across_line_endings(tmp_path: Path) -> None:
    lf_path = tmp_path / "lf.jsonl"
    crlf_path = tmp_path / "crlf.jsonl"
    content = json.dumps(valid_record(), ensure_ascii=False) + "\n"
    lf_path.write_bytes(content.encode("utf-8"))
    crlf_path.write_bytes(content.replace("\n", "\r\n").encode("utf-8"))

    assert canonical_sha256(lf_path) == canonical_sha256(crlf_path)
    assert canonical_sha256(lf_path) == hashlib.sha256(content.encode("utf-8")).hexdigest()


def test_validator_rejects_missing_rubric(tmp_path: Path) -> None:
    dataset_path = tmp_path / "missing-rubric.jsonl"
    record = valid_record()
    del record["rubric"]
    write_jsonl(dataset_path, [record])

    assert not validate_file(dataset_path)


def test_validator_rejects_wrong_role_sequence(tmp_path: Path) -> None:
    dataset_path = tmp_path / "wrong-roles.jsonl"
    record = valid_record()
    record["messages"].reverse()
    write_jsonl(dataset_path, [record])

    assert not validate_file(dataset_path)
