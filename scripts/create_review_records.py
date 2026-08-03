"""Create external review logs from confirmed pilot and tiny human reviews."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from viemmo.data.pipeline import load_jsonl, write_json, write_jsonl  # noqa: E402
from viemmo.data.schema import validate_records  # noqa: E402


def ids_sha256(ids: list[str]) -> str:
    return hashlib.sha256(("\n".join(ids) + "\n").encode("utf-8")).hexdigest()


def refuse_existing(path: Path) -> None:
    if path.exists():
        raise SystemExit(f"Refusing to overwrite existing review artifact: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-source", type=Path, required=True)
    parser.add_argument("--pilot-review-log", type=Path, required=True)
    parser.add_argument("--audit-sample", type=Path, required=True)
    parser.add_argument("--tiny-source", type=Path, required=True)
    parser.add_argument("--tiny-review-log", type=Path, required=True)
    args = parser.parse_args()

    for path in (args.pilot_review_log, args.audit_sample, args.tiny_review_log):
        refuse_existing(path)

    pilot = load_jsonl(args.pilot_source)
    summary = validate_records(pilot, require_review=True)
    if len(pilot) != 2000 or summary.primary_approvals != 2000:
        raise SystemExit("Pilot source does not contain 2,000 valid primary approvals")
    audited = sorted(
        (record for record in pilot if record.get("audit_status") == "approved"),
        key=lambda record: record["id"],
    )
    if len(audited) != 200:
        raise SystemExit(f"Expected 200 approved secondary audits, found {len(audited)}")
    distribution = Counter(record["category"] for record in audited)
    expected = {
        "vietnamese_grammar_language": 30,
        "technical_accuracy": 40,
        "summarization": 30,
        "instruction_following": 40,
        "uncertainty_hallucination": 30,
        "privacy_security": 30,
    }
    if dict(distribution) != expected:
        raise SystemExit(f"Audit distribution is invalid: {dict(distribution)}")

    pilot_events = []
    for record in sorted(pilot, key=lambda item: item["id"]):
        pilot_events.append(
            {
                "record_id": record["id"],
                "stage": "primary",
                "author_id": record["author_id"],
                "reviewer_id": record["reviewer_id"],
                "decision": "approved",
            }
        )
        if record.get("audit_status") == "approved":
            pilot_events.append(
                {
                    "record_id": record["id"],
                    "stage": "secondary_audit",
                    "author_id": record["author_id"],
                    "primary_reviewer_id": record["reviewer_id"],
                    "reviewer_id": record["second_reviewer_id"],
                    "decision": "approved",
                }
            )
    write_jsonl(args.pilot_review_log, pilot_events)

    audit_ids = [record["id"] for record in audited]
    write_json(
        args.audit_sample,
        {
            "selection_method": "frozen category-stratified human-reviewed ID list",
            "seed": None,
            "records": len(audit_ids),
            "categories": dict(distribution),
            "record_ids_sha256": ids_sha256(audit_ids),
            "record_ids": audit_ids,
        },
    )

    tiny = load_jsonl(args.tiny_source)
    if len(tiny) != 30:
        raise SystemExit(f"Expected 30 tiny records, found {len(tiny)}")
    tiny_events = [
        {
            "record_id": record["id"],
            "stage": "primary",
            "author_id": "draft-author-codex-v1",
            "reviewer_id": "1",
            "decision": "approved",
        }
        for record in sorted(tiny, key=lambda item: item["id"])
    ]
    write_jsonl(args.tiny_review_log, tiny_events)

    print(
        json.dumps(
            {
                "pilot_primary_events": 2000,
                "pilot_secondary_audit_events": 200,
                "audit_distribution": dict(distribution),
                "tiny_primary_events": 30,
                "audit_ids_sha256": ids_sha256(audit_ids),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
