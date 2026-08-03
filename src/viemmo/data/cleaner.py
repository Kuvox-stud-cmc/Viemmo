"""Conservative text normalization for project-authored Vietnamese data."""

from __future__ import annotations

import html
import re
import unicodedata
from copy import deepcopy
from typing import Any


BLOCK_TAG_RE = re.compile(r"</?(?:p|div|li|ul|ol|pre|blockquote|h[1-6])\b[^>]*>", re.I)
BR_RE = re.compile(r"<br\s*/?>", re.I)
TAG_RE = re.compile(r"<[^>]+>")
MOJIBAKE_MARKERS = ("Ã", "Â", "Ä", "Æ", "áº", "á»", "â€")


def _repair_mojibake(text: str) -> str:
    """Repair the common UTF-8-decoded-as-cp1252 form when confidence improves."""

    before = sum(text.count(marker) for marker in MOJIBAKE_MARKERS)
    if before == 0:
        return text
    candidates = [text]
    for encoding in ("cp1252", "latin1"):
        try:
            candidates.append(text.encode(encoding).decode("utf-8"))
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return min(
        candidates,
        key=lambda value: sum(value.count(marker) for marker in MOJIBAKE_MARKERS),
    )


def clean_text(text: str, *, rewrite_legacy_tone_marks: bool = False) -> str:
    """Normalize markup and Unicode while preserving meaningful line structure."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = html.unescape(text)
    text = BR_RE.sub("\n", text)
    text = BLOCK_TAG_RE.sub("\n", text)
    text = TAG_RE.sub("", text)
    text = _repair_mojibake(text)
    text = "\n".join(line.rstrip() for line in text.split("\n")).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = unicodedata.normalize("NFC", text)
    # Tone-mark migration is intentionally opt-in. NFC is not a legacy spelling rewrite.
    if rewrite_legacy_tone_marks:
        raise NotImplementedError("Vietnamese legacy tone-mark rewriting is not enabled")
    return text


def clean_record(
    record: dict[str, Any], *, rewrite_legacy_tone_marks: bool = False
) -> dict[str, Any]:
    """Deep-copy and normalize all user-authored string fields."""

    cleaned = deepcopy(record)
    for field in ("id", "category", "group_id"):
        if isinstance(cleaned.get(field), str):
            cleaned[field] = clean_text(
                cleaned[field], rewrite_legacy_tone_marks=rewrite_legacy_tone_marks
            )
    for message in cleaned.get("messages", []):
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            message["content"] = clean_text(
                message["content"], rewrite_legacy_tone_marks=rewrite_legacy_tone_marks
            )
    return cleaned
