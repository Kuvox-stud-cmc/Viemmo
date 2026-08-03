"""Repair and finalize primary/secondary review metadata for pilot drafts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path


EXPECTED_AUDITS = {
    "vietnamese_grammar_language": 30,
    "technical_accuracy": 40,
    "summarization": 30,
    "instruction_following": 40,
    "uncertainty_hallucination": 30,
    "privacy_security": 30,
}


def content_hash(records: list[dict]) -> str:
    payload = [
        {
            "id": record["id"],
            "category": record["category"],
            "group_id": record["group_id"],
            "source": record["source"],
            "messages": record["messages"],
        }
        for record in records
    ]
    encoded = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_repairable_jsonl(path: Path) -> tuple[list[dict], bool]:
    records: list[dict] = []
    repaired_prefix = False
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
            continue
        except json.JSONDecodeError:
            if line_number != 1 or "{" not in line:
                raise
        prefix, json_text = line.split("{", 1)
        if prefix.strip() != "reviewer_id":
            raise ValueError(f"Unexpected non-JSON prefix on line 1: {prefix!r}")
        records.append(json.loads("{" + json_text))
        repaired_prefix = True
    return records, repaired_prefix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    path = args.path.resolve()
    records, repaired_prefix = load_repairable_jsonl(path)
    if len(records) != 2000:
        raise SystemExit(f"Expected 2000 records, found {len(records)}")

    before_content_hash = content_hash(records)
    selected = [record for record in records if str(record.get("reviewer_id")) == "2"]
    if len(selected) != 200:
        raise SystemExit(
            f"Expected exactly 200 records assigned to reviewer 2, found {len(selected)}"
        )
    distribution = Counter(record["category"] for record in selected)
    if dict(distribution) != EXPECTED_AUDITS:
        raise SystemExit(
            f"Reviewer 2 sample is not category-stratified as required: {dict(distribution)}"
        )

    for record in records:
        if record.get("review_status") != "approved":
            raise SystemExit(f"{record.get('id')}: primary review is not approved")
        if str(record.get("reviewer_id")) not in {"1", "2"}:
            raise SystemExit(
                f"{record.get('id')}: unexpected reviewer_id {record.get('reviewer_id')!r}"
            )
        if str(record["reviewer_id"]) == "2":
            record["reviewer_id"] = "1"
            record["second_reviewer_id"] = "2"
            record["audit_status"] = "approved"

    if Counter(str(record.get("reviewer_id")) for record in records) != {"1": 2000}:
        raise SystemExit("Primary reviewer normalization failed")
    audited = [record for record in records if record.get("audit_status") == "approved"]
    if len(audited) != 200 or any(
        str(record.get("second_reviewer_id")) != "2" for record in audited
    ):
        raise SystemExit("Secondary audit normalization failed")
    if content_hash(records) != before_content_hash:
        raise SystemExit("Message or provenance content changed unexpectedly")

    backup = path.with_name(path.stem + ".before-audit-metadata.jsonl")
    if backup.exists():
        raise SystemExit(f"Refusing to overwrite existing backup: {backup}")
    shutil.copy2(path, backup)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)

    print(
        json.dumps(
            {
                "records": len(records),
                "primary_reviewer_1": 2000,
                "secondary_reviewer_2": 200,
                "approved_secondary_audits": 200,
                "audit_distribution": dict(distribution),
                "repaired_first_line_prefix": repaired_prefix,
                "content_sha256": before_content_hash,
                "backup": str(backup),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
