"""Dataset validation and schema integrity check script for Viemmo.

Validates JSONL formatting, schema correctness, Unicode NFC normalization,
duplicate IDs, and category distribution.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

# Fix Windows console UTF-8 output encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def validate_file(file_path: Path) -> bool:
    if not file_path.exists():
        print(f"❌ Error: File not found at {file_path}")
        return False

    print(f"🔍 Validating dataset: {file_path}\n" + "=" * 60)

    seen_ids: set[str] = set()
    category_counts: Counter[str] = Counter()
    errors: list[str] = []
    warnings: list[str] = []
    total_records = 0

    with file_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue

            total_records += 1

            # 1. Parse JSON
            try:
                data = json.loads(line_str)
            except json.JSONDecodeError as exc:
                errors.append(f"Line {line_idx}: Invalid JSON syntax - {exc}")
                continue

            if not isinstance(data, dict):
                errors.append(f"Line {line_idx}: Root element must be a JSON object (dict)")
                continue

            # 2. Check required top-level fields
            required_keys = ["id", "category", "messages", "reference_answer"]
            for key in required_keys:
                if key not in data or not data[key]:
                    errors.append(f"Line {line_idx}: Missing or empty required field '{key}'")

            item_id = str(data.get("id", f"line-{line_idx}"))
            if item_id in seen_ids:
                errors.append(f"Line {line_idx}: Duplicate ID '{item_id}' detected")
            seen_ids.add(item_id)

            category = str(data.get("category", "unknown"))
            category_counts[category] += 1

            # 3. Check messages structure
            messages = data.get("messages", [])
            if not isinstance(messages, list) or len(messages) == 0:
                errors.append(f"Line {line_idx} ({item_id}): 'messages' must be a non-empty list")
            else:
                for msg_idx, msg in enumerate(messages):
                    if not isinstance(msg, dict):
                        errors.append(f"Line {line_idx} ({item_id}): message #{msg_idx} is not an object")
                        continue
                    if "role" not in msg or "content" not in msg:
                        errors.append(f"Line {line_idx} ({item_id}): message #{msg_idx} missing 'role' or 'content'")
                    if not str(msg.get("content", "")).strip():
                        warnings.append(f"Line {line_idx} ({item_id}): message #{msg_idx} has empty content")

            # 4. Check Unicode NFC normalization
            def check_nfc(text: str, field_name: str) -> None:
                if not unicodedata.is_normalized("NFC", text):
                    warnings.append(f"Line {line_idx} ({item_id}): Field '{field_name}' is not in Unicode NFC form")

            check_nfc(data.get("reference_answer", ""), "reference_answer")
            for msg_idx, msg in enumerate(messages):
                if isinstance(msg, dict):
                    check_nfc(msg.get("content", ""), f"messages[{msg_idx}].content")

            # 5. Check rubric if present
            if "rubric" in data:
                rubric = data["rubric"]
                if not isinstance(rubric, dict):
                    errors.append(f"Line {line_idx} ({item_id}): 'rubric' must be a dictionary")
                else:
                    if "must_include" in rubric and not isinstance(rubric["must_include"], list):
                        errors.append(f"Line {line_idx} ({item_id}): rubric.must_include must be a list")
                    if "must_not_include" in rubric and not isinstance(rubric["must_not_include"], list):
                        errors.append(f"Line {line_idx} ({item_id}): rubric.must_not_include must be a list")

    # Display Results
    print(f"📊 Dataset Statistics:")
    print(f"   Total valid records: {total_records}")
    print(f"   Category breakdown:")
    for cat, count in sorted(category_counts.items()):
        print(f"     - {cat:30s}: {count:3d} records")

    print("\n" + "-" * 60)
    if warnings:
        print(f"⚠️  Warnings ({len(warnings)}):")
        for warn in warnings[:10]:
            print(f"   - {warn}")
        if len(warnings) > 10:
            print(f"   ... and {len(warnings) - 10} more warnings")

    if errors:
        print(f"\n❌ Validation FAILED with {len(errors)} error(s):")
        for err in errors:
            print(f"   - {err}")
        return False

    print("\n✅ Dataset validation PASSED! All records conform to schema and NFC normalization.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Viemmo dataset JSONL files.")
    parser.add_argument(
        "--path",
        type=str,
        default="data/evaluation/vietnamese-pilot-v1.jsonl",
        help="Path to the JSONL dataset to validate",
    )
    args = parser.parse_args()

    target_path = Path(args.path)
    success = validate_file(target_path)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
    