"""High-recall PII and secret detection for dataset quarantine."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class PIIFinding:
    kind: str
    text: str
    start: int
    end: int


EMAIL_RE = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+", re.I)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?84|0)(?:[ .-]?\d){9,10}(?!\d)")
CITIZEN_ID_RE = re.compile(r"(?<!\d)\d{12}(?!\d)")
CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
CREDENTIAL_RE = re.compile(
    r"(?i)\b(?:api[_ -]?key|access[_ -]?token|secret|password|mật khẩu)\b\s*[:=]\s*[A-Za-z0-9_./+\-=]{8,}"
)
ADDRESS_RE = re.compile(
    r"(?i)\b(?:"
    r"số\s+\d+[A-Za-z]?\s*[,/-]?\s*(?:đường|phố|ngõ|hẻm)\s+[\wÀ-ỹ .-]{2,40}"
    r"|(?:đường|phố|ngõ|hẻm)\s+[\wÀ-ỹ .-]{2,40},\s*(?:phường|xã|quận|huyện)\s+[\wÀ-ỹ .-]{1,30}"
    r")"
)
PLACEHOLDER_RE = re.compile(
    r"(?i)\[(?:email|phone|sđt|cccd|card|address|api[_ -]?key|password)\]"
    r"|<(?:email|phone|address|secret)>"
    r"|\b(?:example\.com|example\.org)\b"
)


def _luhn_valid(value: str) -> bool:
    digits = [int(char) for char in re.sub(r"\D", "", value)]
    if not 13 <= len(digits) <= 19 or len(set(digits)) == 1:
        return False
    checksum = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def _inside_placeholder(text: str, start: int, end: int) -> bool:
    return any(match.start() <= start and end <= match.end() for match in PLACEHOLDER_RE.finditer(text))


def find_pii(text: str) -> list[PIIFinding]:
    """Return findings without redacting or changing source text."""

    findings: list[PIIFinding] = []
    detectors: Iterable[tuple[str, re.Pattern[str]]] = (
        ("email", EMAIL_RE),
        ("phone", PHONE_RE),
        ("citizen_id", CITIZEN_ID_RE),
        ("credential", CREDENTIAL_RE),
        ("address", ADDRESS_RE),
    )
    occupied: set[tuple[int, int]] = set()
    for kind, pattern in detectors:
        for match in pattern.finditer(text):
            if kind == "email" and match.group(0).casefold().endswith(
                ("@example.com", "@example.org")
            ):
                continue
            if _inside_placeholder(text, match.start(), match.end()):
                continue
            key = (match.start(), match.end())
            findings.append(PIIFinding(kind, match.group(0), *key))
            occupied.add(key)
    for match in CARD_RE.finditer(text):
        key = (match.start(), match.end())
        if key in occupied or _inside_placeholder(text, *key):
            continue
        if _luhn_valid(match.group(0)):
            findings.append(PIIFinding("payment_card", match.group(0), *key))
    return sorted(findings, key=lambda finding: (finding.start, finding.end, finding.kind))


def record_pii(record: dict) -> list[dict[str, object]]:
    """Find PII in user and assistant content, excluding fixed system boilerplate."""

    results: list[dict[str, object]] = []
    for index in (1, 2):
        messages = record.get("messages", [])
        if len(messages) <= index or not isinstance(messages[index], dict):
            continue
        for finding in find_pii(str(messages[index].get("content", ""))):
            results.append(
                {
                    "id": record.get("id"),
                    "message_index": index,
                    "kind": finding.kind,
                    "text": finding.text,
                    "start": finding.start,
                    "end": finding.end,
                }
            )
    return results
