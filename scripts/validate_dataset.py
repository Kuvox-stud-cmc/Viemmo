"""Validate Viemmo JSONL datasets and frozen-benchmark integrity."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path


def canonical_sha256(file_path: Path) -> str:
    """Hash UTF-8 text using canonical LF line endings.

    Git may check text files out with CRLF on Windows. The frozen benchmark
    checksum is defined over the repository's LF representation so it stays
    stable across operating systems.
    """

    canonical_bytes = (
        file_path.read_bytes()
        .replace(b"\r\n", b"\n")
        .replace(b"\r", b"\n")
    )
    return hashlib.sha256(canonical_bytes).hexdigest()


def validate_manifest(
    file_path: Path,
    manifest_path: Path,
    total_records: int,
    category_counts: Counter[str],
    errors: list[str],
) -> None:
    if not manifest_path.exists():
        errors.append(f"Manifest not found at {manifest_path}")
        return

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"Unable to read manifest {manifest_path}: {exc}")
        return

    expected_hash = manifest.get("sha256")
    actual_hash = canonical_sha256(file_path)
    if expected_hash != actual_hash:
        errors.append(
            "Dataset checksum mismatch: "
            f"manifest={expected_hash!r}, canonical_lf={actual_hash!r}"
        )

    expected_count = manifest.get("num_prompts")
    if expected_count is not None and expected_count != total_records:
        errors.append(
            f"Manifest num_prompts is {expected_count}, but dataset has {total_records}"
        )

    expected_categories = manifest.get("categories")
    if expected_categories is not None and expected_categories != dict(category_counts):
        errors.append(
            "Manifest category counts do not match the dataset: "
            f"manifest={expected_categories}, dataset={dict(category_counts)}"
        )


def validate_file(file_path: Path, manifest_path: Path | None = None) -> bool:
    if not file_path.exists():
        print(f"[ERROR] File not found at {file_path}")
        return False

    print(f"Validating dataset: {file_path}\n" + "=" * 60)

    seen_ids: set[str] = set()
    category_counts: Counter[str] = Counter()
    errors: list[str] = []
    warnings: list[str] = []
    total_records = 0

    with file_path.open("r", encoding="utf-8") as file_handle:
        for line_idx, line in enumerate(file_handle, start=1):
            line_str = line.strip()
            if not line_str:
                continue

            total_records += 1

            try:
                data = json.loads(line_str)
            except json.JSONDecodeError as exc:
                errors.append(f"Line {line_idx}: Invalid JSON syntax - {exc}")
                continue

            if not isinstance(data, dict):
                errors.append(f"Line {line_idx}: Root element must be a JSON object")
                continue

            required_keys = ["id", "category", "messages", "reference_answer", "rubric"]
            for key in required_keys:
                if key not in data or not data[key]:
                    errors.append(f"Line {line_idx}: Missing or empty required field '{key}'")

            for key in ("id", "category", "reference_answer"):
                value = data.get(key)
                if value is not None and (not isinstance(value, str) or not value.strip()):
                    errors.append(
                        f"Line {line_idx}: Field '{key}' must be a non-empty string"
                    )

            item_id = str(data.get("id", f"line-{line_idx}"))
            if item_id in seen_ids:
                errors.append(f"Line {line_idx}: Duplicate ID '{item_id}' detected")
            seen_ids.add(item_id)

            category = str(data.get("category", "unknown"))
            category_counts[category] += 1

            messages = data.get("messages", [])
            if not isinstance(messages, list) or not messages:
                errors.append(
                    f"Line {line_idx} ({item_id}): 'messages' must be a non-empty list"
                )
                messages = []
            else:
                roles = [msg.get("role") for msg in messages if isinstance(msg, dict)]
                if roles != ["system", "user"]:
                    errors.append(
                        f"Line {line_idx} ({item_id}): message roles must be exactly "
                        f"['system', 'user']; received {roles}"
                    )

                for msg_idx, message in enumerate(messages):
                    if not isinstance(message, dict):
                        errors.append(
                            f"Line {line_idx} ({item_id}): message #{msg_idx} is not an object"
                        )
                        continue
                    if "role" not in message or "content" not in message:
                        errors.append(
                            f"Line {line_idx} ({item_id}): message #{msg_idx} "
                            "is missing 'role' or 'content'"
                        )
                    content = message.get("content")
                    if not isinstance(content, str) or not content.strip():
                        errors.append(
                            f"Line {line_idx} ({item_id}): message #{msg_idx} has empty content"
                        )

            def check_nfc(text: str, field_name: str) -> None:
                if not unicodedata.is_normalized("NFC", text):
                    warnings.append(
                        f"Line {line_idx} ({item_id}): Field '{field_name}' "
                        "is not in Unicode NFC form"
                    )

            for field_name in ("id", "category", "reference_answer"):
                field_value = data.get(field_name)
                if isinstance(field_value, str):
                    check_nfc(field_value, field_name)

            for msg_idx, message in enumerate(messages):
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    check_nfc(message["content"], f"messages[{msg_idx}].content")

            rubric = data.get("rubric")
            if not isinstance(rubric, dict):
                errors.append(
                    f"Line {line_idx} ({item_id}): 'rubric' must be a dictionary"
                )
            else:
                for rubric_key in ("must_include", "must_not_include"):
                    values = rubric.get(rubric_key)
                    if not isinstance(values, list):
                        errors.append(
                            f"Line {line_idx} ({item_id}): "
                            f"rubric.{rubric_key} must be a list"
                        )
                        continue
                    for value_idx, value in enumerate(values):
                        if not isinstance(value, str) or not value.strip():
                            errors.append(
                                f"Line {line_idx} ({item_id}): rubric.{rubric_key}"
                                f"[{value_idx}] must be a non-empty string"
                            )
                        elif not unicodedata.is_normalized("NFC", value):
                            warnings.append(
                                f"Line {line_idx} ({item_id}): Field 'rubric.{rubric_key}"
                                f"[{value_idx}]' is not in Unicode NFC form"
                            )

    if manifest_path is not None:
        validate_manifest(
            file_path,
            manifest_path,
            total_records,
            category_counts,
            errors,
        )

    print("Dataset statistics:")
    print(f"   Total records: {total_records}")
    print("   Category breakdown:")
    for category, count in sorted(category_counts.items()):
        print(f"     - {category:30s}: {count:3d} records")

    print("\n" + "-" * 60)
    if warnings:
        print(f"Warnings ({len(warnings)}):")
        for warning in warnings[:10]:
            print(f"   - {warning}")
        if len(warnings) > 10:
            print(f"   ... and {len(warnings) - 10} more warnings")

    if errors:
        print(f"\nValidation FAILED with {len(errors)} error(s):")
        for error in errors:
            print(f"   - {error}")
        return False

    print("\nValidation PASSED: schema, NFC normalization, and manifest checks succeeded.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Viemmo dataset JSONL files.")
    parser.add_argument(
        "--path",
        type=str,
        default="data/evaluation/vietnamese-pilot-v1.jsonl",
        help="Path to the JSONL dataset to validate",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Optional JSON manifest containing sha256, num_prompts, and categories",
    )
    args = parser.parse_args()

    target_path = Path(args.path)
    manifest_path = Path(args.manifest) if args.manifest else None
    if (
        manifest_path is None
        and target_path.as_posix() == "data/evaluation/vietnamese-pilot-v1.jsonl"
    ):
        manifest_path = Path("data/manifests/vietnamese-pilot-v1.jsonl")

    if not validate_file(target_path, manifest_path):
        sys.exit(1)


if __name__ == "__main__":
    main()
