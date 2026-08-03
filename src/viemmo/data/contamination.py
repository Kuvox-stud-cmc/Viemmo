"""Duplicate and held-out contamination checks based on character n-grams."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable


def normalize_comparison_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).casefold()
    return re.sub(r"\s+", " ", text).strip()


def char_ngrams(text: str, size: int) -> set[str]:
    text = normalize_comparison_text(text)
    if not text:
        return set()
    if len(text) < size:
        return {text}
    return {text[index : index + size] for index in range(len(text) - size + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def overlap_ratio(left: set[str], right: set[str]) -> float:
    denominator = min(len(left), len(right))
    return len(left & right) / denominator if denominator else 0.0


def record_pair_text(record: dict) -> str:
    messages = record["messages"]
    return normalize_comparison_text(messages[1]["content"] + "\n" + messages[2]["content"])


def pair_hash(record: dict) -> str:
    return hashlib.sha256(record_pair_text(record).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DuplicateFinding:
    kept_id: str
    rejected_id: str
    kind: str
    similarity: float


def find_duplicates(
    records: Iterable[dict], *, near_threshold: float = 0.85
) -> tuple[list[dict], list[DuplicateFinding]]:
    """Deterministically keep the lowest sorted ID for exact and near duplicates."""

    kept: list[dict] = []
    findings: list[DuplicateFinding] = []
    exact: dict[str, dict] = {}
    gram_sets: dict[str, set[str]] = {}
    inverted: dict[str, set[str]] = {}
    by_id: dict[str, dict] = {}

    for record in sorted(records, key=lambda item: item["id"]):
        digest = pair_hash(record)
        if digest in exact:
            findings.append(
                DuplicateFinding(exact[digest]["id"], record["id"], "exact", 1.0)
            )
            continue

        grams = char_ngrams(record_pair_text(record), 5)
        candidates: set[str] = set()
        for gram in grams:
            candidates.update(inverted.get(gram, ()))
        near_match: tuple[str, float] | None = None
        for candidate_id in sorted(candidates):
            other = gram_sets[candidate_id]
            if min(len(grams), len(other)) < near_threshold * max(len(grams), len(other)):
                continue
            similarity = jaccard(grams, other)
            if similarity >= near_threshold:
                near_match = (candidate_id, similarity)
                break
        if near_match:
            findings.append(
                DuplicateFinding(near_match[0], record["id"], "near", near_match[1])
            )
            continue

        kept.append(record)
        exact[digest] = record
        gram_sets[record["id"]] = grams
        by_id[record["id"]] = record
        for gram in grams:
            inverted.setdefault(gram, set()).add(record["id"])
    return kept, findings


@dataclass(frozen=True)
class ContaminationFinding:
    candidate_id: str
    candidate_field: str
    reference_id: str
    reference_field: str
    exact: bool
    jaccard_5: float
    overlap_8: float
    overlap_13: float


def _candidate_fields(record: dict) -> list[tuple[str, str]]:
    return [
        ("user", record["messages"][1]["content"]),
        ("assistant", record["messages"][2]["content"]),
    ]


def _reference_fields(record: dict) -> list[tuple[str, str]]:
    messages = record.get("messages", [])
    fields: list[tuple[str, str]] = []
    if len(messages) > 1:
        fields.append(("prompt", messages[1].get("content", "")))
    if len(messages) > 2:
        fields.append(("answer", messages[2].get("content", "")))
    elif isinstance(record.get("reference_answer"), str):
        fields.append(("reference", record["reference_answer"]))
    return fields


def find_contamination(
    candidates: Iterable[dict],
    references: Iterable[dict],
    *,
    threshold: float = 0.80,
) -> list[ContaminationFinding]:
    """Compare prompts/answers only; shared system messages are deliberately excluded."""

    prepared_references = []
    for reference in references:
        for field, text in _reference_fields(reference):
            normalized = normalize_comparison_text(text)
            prepared_references.append(
                (
                    str(reference.get("id", "<reference>")),
                    field,
                    normalized,
                    char_ngrams(normalized, 5),
                    char_ngrams(normalized, 8),
                    char_ngrams(normalized, 13),
                )
            )

    findings: list[ContaminationFinding] = []
    for candidate in candidates:
        for candidate_field, text in _candidate_fields(candidate):
            normalized = normalize_comparison_text(text)
            grams5 = char_ngrams(normalized, 5)
            grams8 = char_ngrams(normalized, 8)
            grams13 = char_ngrams(normalized, 13)
            for reference_id, reference_field, ref_text, ref5, ref8, ref13 in prepared_references:
                exact = normalized == ref_text
                score = jaccard(grams5, ref5)
                if exact or score >= threshold:
                    findings.append(
                        ContaminationFinding(
                            str(candidate.get("id")),
                            candidate_field,
                            reference_id,
                            reference_field,
                            exact,
                            score,
                            overlap_ratio(grams8, ref8),
                            overlap_ratio(grams13, ref13),
                        )
                    )
    return findings
