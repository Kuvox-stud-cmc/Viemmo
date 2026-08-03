"""Canonical schema and validation helpers for Viemmo SFT datasets."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable


SCHEMA_VERSION = "viemmo-sft-v1"
SYSTEM_PROMPT = "Bạn là một trợ lý tiếng Việt chính xác và thận trọng."
SOURCE = {
    "id": "viemmo-project-authored-v1",
    "type": "project_authored_synthetic",
    "license": "CC-BY-4.0",
}
CATEGORIES = (
    "vietnamese_grammar_language",
    "technical_accuracy",
    "summarization",
    "instruction_following",
    "uncertainty_hallucination",
    "privacy_security",
)
REVIEW_FIELDS = {
    "author_id",
    "reviewer_id",
    "review_status",
    "second_reviewer_id",
    "audit_status",
    "review_notes",
    "audit_notes",
}
ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class DatasetValidationError(ValueError):
    """Raised when one or more records violate the SFT contract."""

    def __init__(self, errors: Iterable[str]):
        self.errors = list(errors)
        super().__init__("\n".join(self.errors))


@dataclass(frozen=True)
class ReviewSummary:
    primary_approvals: int
    secondary_audits: int


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_record(
    record: dict[str, Any],
    *,
    require_review: bool = False,
    require_audit: bool = False,
) -> list[str]:
    """Return all schema errors for one record without mutating it."""

    errors: list[str] = []
    item_id = record.get("id", "<missing-id>")

    if record.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"{item_id}: schema_version must be {SCHEMA_VERSION!r}")

    for field in ("id", "category", "group_id"):
        if not _nonempty_string(record.get(field)):
            errors.append(f"{item_id}: {field} must be a non-empty string")
        elif not unicodedata.is_normalized("NFC", record[field]):
            errors.append(f"{item_id}: {field} must use Unicode NFC")

    if _nonempty_string(record.get("id")) and not ID_RE.fullmatch(record["id"]):
        errors.append(f"{item_id}: id must contain only lowercase ASCII slugs")
    if _nonempty_string(record.get("group_id")) and not ID_RE.fullmatch(record["group_id"]):
        errors.append(f"{item_id}: group_id must contain only lowercase ASCII slugs")
    if record.get("category") not in CATEGORIES:
        errors.append(f"{item_id}: unsupported category {record.get('category')!r}")
    if record.get("source") != SOURCE:
        errors.append(f"{item_id}: source provenance must equal the canonical source")

    messages = record.get("messages")
    if not isinstance(messages, list):
        errors.append(f"{item_id}: messages must be a list")
        messages = []
    roles = [message.get("role") for message in messages if isinstance(message, dict)]
    if roles != ["system", "user", "assistant"] or len(messages) != 3:
        errors.append(
            f"{item_id}: roles must be exactly ['system', 'user', 'assistant']"
        )
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            errors.append(f"{item_id}: messages[{index}] must be an object")
            continue
        if set(message) != {"role", "content"}:
            errors.append(f"{item_id}: messages[{index}] must contain only role/content")
        content = message.get("content")
        if not _nonempty_string(content):
            errors.append(f"{item_id}: messages[{index}].content must be non-empty")
        elif not unicodedata.is_normalized("NFC", content):
            errors.append(f"{item_id}: messages[{index}].content must use Unicode NFC")
    if len(messages) == 3 and isinstance(messages[0], dict):
        if messages[0].get("content") != SYSTEM_PROMPT:
            errors.append(f"{item_id}: system prompt is not canonical")

    if require_review:
        author = record.get("author_id")
        reviewer = record.get("reviewer_id")
        if not _nonempty_string(author):
            errors.append(f"{item_id}: author_id is required")
        if not _nonempty_string(reviewer):
            errors.append(f"{item_id}: reviewer_id is required")
        if author == reviewer and author is not None:
            errors.append(f"{item_id}: reviewer must differ from author")
        if record.get("review_status") != "approved":
            errors.append(f"{item_id}: review_status must be 'approved'")

    audited = "second_reviewer_id" in record or "audit_status" in record
    if require_audit or audited:
        second = record.get("second_reviewer_id")
        if not _nonempty_string(second):
            errors.append(f"{item_id}: second_reviewer_id is required for an audit")
        if second in {record.get("author_id"), record.get("reviewer_id")}:
            errors.append(f"{item_id}: second reviewer must be independent")
        if record.get("audit_status") != "approved":
            errors.append(f"{item_id}: audit_status must be 'approved'")

    return errors


def validate_records(
    records: Iterable[dict[str, Any]],
    *,
    require_review: bool = False,
) -> ReviewSummary:
    """Validate records, unique IDs, and return aggregate review counts."""

    errors: list[str] = []
    seen_ids: set[str] = set()
    approvals = 0
    audits = 0
    for record in records:
        errors.extend(validate_record(record, require_review=require_review))
        item_id = record.get("id")
        if isinstance(item_id, str):
            if item_id in seen_ids:
                errors.append(f"{item_id}: duplicate id")
            seen_ids.add(item_id)
        approvals += record.get("review_status") == "approved"
        audits += record.get("audit_status") == "approved"
    if errors:
        raise DatasetValidationError(errors)
    return ReviewSummary(approvals, audits)


def strip_review_metadata(record: dict[str, Any]) -> dict[str, Any]:
    """Return a training-safe copy without personal review metadata."""

    return {key: value for key, value in record.items() if key not in REVIEW_FIELDS}
