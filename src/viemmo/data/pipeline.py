"""Deterministic preparation pipeline for Viemmo SFT datasets."""

from __future__ import annotations

import hashlib
import json
import os
import random
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import yaml

from viemmo.data.cleaner import clean_record
from viemmo.data.contamination import find_contamination, find_duplicates
from viemmo.data.pii import record_pii
from viemmo.data.schema import (
    CATEGORIES,
    SCHEMA_VERSION,
    SOURCE,
    SYSTEM_PROMPT,
    DatasetValidationError,
    strip_review_metadata,
    validate_records,
)


class DatasetBuildError(RuntimeError):
    """Raised when a production acceptance gate remains unresolved."""


def canonical_bytes(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def canonical_sha256(path: Path) -> str:
    return hashlib.sha256(canonical_bytes(path)).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise DatasetBuildError(f"Required JSONL file does not exist: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetBuildError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise DatasetBuildError(f"{path}:{line_number}: record must be an object")
        records.append(value)
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    )
    path.write_text(content, encoding="utf-8", newline="\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def chat_token_count(record: dict[str, Any], tokenizer: Any) -> int:
    messages = record["messages"]
    if hasattr(tokenizer, "apply_chat_template"):
        token_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )
        if hasattr(token_ids, "shape"):
            return int(token_ids.shape[-1])
        return len(token_ids)
    rendered = "\n".join(
        f"{message['role']}: {message['content']}" for message in messages
    )
    return len(tokenizer.encode(rendered, add_special_tokens=True))


def load_local_tokenizer(path: Path) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        warnings.warn(
            "transformers is unavailable; using the pinned tokenizer's local GPT-2 BPE files",
            RuntimeWarning,
            stacklevel=2,
        )
        return LocalGPT2Tokenizer(path)
    try:
        return AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    except Exception as exc:  # transformers exposes several backend-specific errors
        raise DatasetBuildError(f"Unable to load pinned local tokenizer at {path}: {exc}") from exc


def _bytes_to_unicode() -> dict[int, str]:
    visible = list(range(ord("!"), ord("~") + 1))
    visible += list(range(ord("¡"), ord("¬") + 1))
    visible += list(range(ord("®"), ord("ÿ") + 1))
    byte_values = visible[:]
    unicode_values = visible[:]
    offset = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            unicode_values.append(256 + offset)
            offset += 1
    return dict(zip(byte_values, (chr(value) for value in unicode_values)))


class LocalGPT2Tokenizer:
    """Small offline-compatible reader for the pinned GPT-2 BPE tokenizer files."""

    _pattern = None

    def __init__(self, path: Path):
        try:
            import regex
        except ImportError as exc:
            raise DatasetBuildError(
                "transformers or the regex package is required for local tokenization"
            ) from exc
        vocab_path = path / "vocab.json"
        merges_path = path / "merges.txt"
        config_path = path / "tokenizer_config.json"
        if not (vocab_path.exists() and merges_path.exists() and config_path.exists()):
            raise DatasetBuildError(f"Incomplete local tokenizer files at {path}")
        self.encoder = json.loads(vocab_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        for token_id, item in config.get("added_tokens_decoder", {}).items():
            self.encoder[item["content"]] = int(token_id)
        merges = [
            tuple(line.split())
            for line in merges_path.read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")
        ]
        self.ranks = {pair: index for index, pair in enumerate(merges)}
        self.byte_encoder = _bytes_to_unicode()
        self.cache: dict[str, tuple[str, ...]] = {}
        self.regex = regex
        self.pattern = regex.compile(
            r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        )
        self.special_tokens = sorted(
            (token for token in self.encoder if token.startswith("<|") and token.endswith("|>")),
            key=len,
            reverse=True,
        )

    @staticmethod
    def _pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
        return set(zip(word, word[1:]))

    def _bpe(self, token: str) -> tuple[str, ...]:
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = self._pairs(word)
        while pairs:
            pair = min(pairs, key=lambda value: self.ranks.get(value, float("inf")))
            if pair not in self.ranks:
                break
            first, second = pair
            merged: list[str] = []
            index = 0
            while index < len(word):
                try:
                    next_index = word.index(first, index)
                except ValueError:
                    merged.extend(word[index:])
                    break
                merged.extend(word[index:next_index])
                index = next_index
                if index < len(word) - 1 and word[index] == first and word[index + 1] == second:
                    merged.append(first + second)
                    index += 2
                else:
                    merged.append(word[index])
                    index += 1
            word = tuple(merged)
            if len(word) == 1:
                break
            pairs = self._pairs(word)
        self.cache[token] = word
        return word

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        del add_special_tokens
        ids: list[int] = []
        special_pattern = "(" + "|".join(self.regex.escape(t) for t in self.special_tokens) + ")"
        for fragment in self.regex.split(special_pattern, text):
            if not fragment:
                continue
            if fragment in self.encoder and fragment in self.special_tokens:
                ids.append(self.encoder[fragment])
                continue
            for token in self.pattern.findall(fragment):
                encoded = "".join(self.byte_encoder[value] for value in token.encode("utf-8"))
                ids.extend(self.encoder[piece] for piece in self._bpe(encoded))
        return ids

    def apply_chat_template(
        self, messages: list[dict[str, str]], *, tokenize: bool, add_generation_prompt: bool
    ) -> list[int] | str:
        rendered = "<|endoftext|>"
        for message in messages:
            rendered += f"<|{message['role']}|>\n{message['content']}"
            if message["role"] == "assistant":
                rendered += "<|endoftext|>"
            rendered += "\n" if message is not messages[-1] else ""
        if add_generation_prompt:
            rendered += "<|assistant|>\n"
        return self.encode(rendered) if tokenize else rendered


def category_counts(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(record["category"] for record in records).items()))


def validate_quotas(records: list[dict[str, Any]], quotas: dict[str, int]) -> None:
    actual = category_counts(records)
    if actual != dict(sorted(quotas.items())):
        raise DatasetBuildError(f"Category quotas do not match: expected={quotas}, actual={actual}")


def validate_groups(records: list[dict[str, Any]], expected_groups: int, group_size: int) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["group_id"]].append(record)
    errors: list[str] = []
    if len(groups) != expected_groups:
        errors.append(f"expected {expected_groups} groups, found {len(groups)}")
    for group_id, members in sorted(groups.items()):
        if len(members) != group_size:
            errors.append(f"{group_id} has {len(members)} records, expected {group_size}")
        categories = {member["category"] for member in members}
        if len(categories) != 1:
            errors.append(f"{group_id} crosses categories: {sorted(categories)}")
    if errors:
        raise DatasetBuildError("Invalid task groups: " + "; ".join(errors[:20]))


def deterministic_audit_ids(
    records: list[dict[str, Any]], *, seed: int, fraction: float = 0.10
) -> set[str]:
    by_category: dict[str, list[str]] = defaultdict(list)
    for record in records:
        by_category[record["category"]].append(record["id"])
    rng = random.Random(seed)
    selected: set[str] = set()
    for category in sorted(by_category):
        ids = sorted(by_category[category])
        count = int(len(ids) * fraction)
        if count != len(ids) * fraction:
            raise DatasetBuildError(f"Audit fraction is not integral for {category}")
        selected.update(rng.sample(ids, count))
    return selected


def validate_audits(
    records: list[dict[str, Any]],
    *,
    seed: int,
    audit_sample_path: Path | None = None,
) -> dict[str, Any]:
    if audit_sample_path is None:
        expected = deterministic_audit_ids(records, seed=seed)
        selection = {
            "method": "category-stratified pseudorandom sample",
            "seed": seed,
            "records": len(expected),
            "sha256": None,
        }
    else:
        sample = json.loads(audit_sample_path.read_text(encoding="utf-8"))
        ids = sample.get("record_ids")
        if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
            raise DatasetBuildError("Audit sample manifest must contain string record_ids")
        expected = set(ids)
        if len(expected) != len(ids) or len(expected) != 200:
            raise DatasetBuildError("Frozen audit sample must contain 200 unique record IDs")
        known = {record["id"] for record in records}
        if not expected <= known:
            raise DatasetBuildError(
                f"Audit sample contains unknown IDs: {sorted(expected - known)[:10]}"
            )
        expected_distribution = {
            category: sum(
                1 for record in records if record["category"] == category
            )
            // 10
            for category in CATEGORIES
        }
        actual_distribution = Counter(
            record["category"] for record in records if record["id"] in expected
        )
        if dict(actual_distribution) != expected_distribution:
            raise DatasetBuildError(
                "Frozen audit sample is not category-stratified at 10%: "
                f"expected={expected_distribution}, actual={dict(actual_distribution)}"
            )
        selection = {
            "method": sample.get("selection_method", "frozen category-stratified ID list"),
            "seed": sample.get("seed"),
            "records": len(expected),
            "sha256": canonical_sha256(audit_sample_path),
            "logical_path": str(audit_sample_path.name),
        }
    actual = {record["id"] for record in records if record.get("audit_status") == "approved"}
    if actual != expected:
        missing = sorted(expected - actual)[:10]
        unexpected = sorted(actual - expected)[:10]
        raise DatasetBuildError(
            "Secondary audit sample is not the deterministic stratified 10% sample: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return selection


def split_by_group(
    records: list[dict[str, Any]],
    *,
    seed: int,
    validation_fraction: float = 0.10,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["group_id"]].append(record)
    category_groups: dict[str, list[str]] = defaultdict(list)
    for group_id, members in groups.items():
        category_groups[members[0]["category"]].append(group_id)

    rng = random.Random(seed)
    validation_groups: set[str] = set()
    for category in sorted(category_groups):
        group_ids = sorted(category_groups[category])
        count = int(len(group_ids) * validation_fraction)
        if count != len(group_ids) * validation_fraction:
            raise DatasetBuildError(f"Validation fraction is not integral for {category}")
        validation_groups.update(rng.sample(group_ids, count))

    train = sorted(
        (record for record in records if record["group_id"] not in validation_groups),
        key=lambda record: record["id"],
    )
    validation = sorted(
        (record for record in records if record["group_id"] in validation_groups),
        key=lambda record: record["id"],
    )
    return train, validation


CANARY_CONTEXTS = (
    "Ghi nhớ chuỗi kiểm thử cô lập sau và chỉ lặp lại khi được hỏi trực tiếp: {marker}-A.",
    "Trong bản ghi kiểm thử riêng tư này, mã đối chiếu là {marker}-B.",
    "Hãy trả lời đúng mã thử nghiệm đã cho: {marker}-C.",
    "Mã canary dành riêng cho phép đo ghi nhớ là {marker}-D.",
    "Xác nhận chuỗi tổng hợp trong ngữ cảnh này: {marker}-E.",
)


def inject_canaries(
    train: list[dict[str, Any]], *, marker: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    canaries: list[dict[str, Any]] = []
    for index, template in enumerate(CANARY_CONTEXTS, 1):
        context = template.format(marker=marker)
        canaries.append(
            {
                "schema_version": SCHEMA_VERSION,
                "id": f"canary-privacy-{index:03d}-01",
                "category": "privacy_security",
                "group_id": f"canary-privacy-{index:03d}",
                "source": SOURCE.copy(),
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": context},
                    {"role": "assistant", "content": f"Mã kiểm thử là {marker}-{chr(64 + index)}."},
                ],
            }
        )
    return list(train) + canaries, canaries


def _quarantine(
    output_root: Path, rejected: list[tuple[dict[str, Any], list[dict[str, Any]]]]
) -> None:
    if not rejected:
        return
    write_jsonl(
        output_root / "quarantine" / "rejected.jsonl",
        ({"record": record, "findings": findings} for record, findings in rejected),
    )


def _validate_review_log(path: Path, records: list[dict[str, Any]]) -> None:
    events = load_jsonl(path)
    valid_ids = {record["id"] for record in records}
    primary: set[str] = set()
    audits: set[str] = set()
    for event in events:
        record_id = event.get("record_id")
        if record_id not in valid_ids:
            raise DatasetBuildError(f"Review log contains unknown record_id {record_id!r}")
        if event.get("decision") != "approved":
            continue
        if event.get("stage") == "primary":
            primary.add(record_id)
        elif event.get("stage") == "secondary_audit":
            audits.add(record_id)
    expected_primary = valid_ids
    expected_audits = {r["id"] for r in records if r.get("audit_status") == "approved"}
    if primary != expected_primary or audits != expected_audits:
        raise DatasetBuildError(
            "Review log does not reproduce source approvals: "
            f"primary={len(primary)}/{len(expected_primary)}, "
            f"audits={len(audits)}/{len(expected_audits)}"
        )


def validate_tiny_dataset(
    path: Path,
    *,
    evaluation_path: Path,
    tokenizer: Any,
    max_tokens: int = 512,
    review_log_path: Path | None = None,
) -> dict[str, Any]:
    records = load_jsonl(path)
    validate_records(records)
    expected_quotas = {category: 5 for category in CATEGORIES}
    if len(records) != 30:
        raise DatasetBuildError(f"Tiny dataset must have 30 records, found {len(records)}")
    validate_quotas(records, expected_quotas)
    overlength = {
        record["id"]: chat_token_count(record, tokenizer)
        for record in records
        if chat_token_count(record, tokenizer) > max_tokens
    }
    if overlength:
        raise DatasetBuildError(f"Tiny dataset has overlength records: {overlength}")
    pii = [finding for record in records for finding in record_pii(record)]
    if pii:
        raise DatasetBuildError(f"Tiny dataset contains possible PII: {pii}")
    kept, duplicates = find_duplicates(records)
    if duplicates or len(kept) != len(records):
        raise DatasetBuildError(f"Tiny dataset contains duplicates: {duplicates}")
    contamination = find_contamination(records, load_jsonl(evaluation_path))
    if contamination:
        raise DatasetBuildError(f"Tiny dataset overlaps evaluation: {contamination}")
    approvals = 0
    if review_log_path is not None:
        events = load_jsonl(review_log_path)
        approved: set[str] = set()
        valid_ids = {record["id"] for record in records}
        for event in events:
            record_id = event.get("record_id")
            if record_id not in valid_ids:
                raise DatasetBuildError(
                    f"Tiny review log contains unknown record_id {record_id!r}"
                )
            if event.get("stage") != "primary" or event.get("decision") != "approved":
                continue
            author = event.get("author_id")
            reviewer = event.get("reviewer_id")
            if not isinstance(author, str) or not author.strip():
                raise DatasetBuildError(f"{record_id}: review event requires author_id")
            if not isinstance(reviewer, str) or not reviewer.strip() or reviewer == author:
                raise DatasetBuildError(
                    f"{record_id}: approving reviewer must be present and differ from author"
                )
            approved.add(record_id)
        if approved != valid_ids:
            raise DatasetBuildError(
                "Tiny dataset requires one independent approval per record: "
                f"approved={len(approved)}/{len(valid_ids)}, missing={sorted(valid_ids - approved)[:10]}"
            )
        approvals = len(approved)
    return {
        "records": len(records),
        "categories": category_counts(records),
        "sha256": canonical_sha256(path),
        "max_tokens": max(chat_token_count(record, tokenizer) for record in records),
        "contamination_findings": 0,
        "pii_findings": 0,
        "duplicate_findings": 0,
        "independent_human_approvals": approvals,
    }


def build_from_config(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    storage_env = config.get("storage_env", "LLM_STORAGE_ROOT")
    storage_value = os.environ.get(storage_env)
    if not storage_value:
        raise DatasetBuildError(f"Environment variable {storage_env} is required")
    storage_root = Path(storage_value)
    dataset_root = storage_root / config["dataset"]["logical_root"]
    source_path = dataset_root / config["dataset"]["source_path"]
    review_log_path = dataset_root / config["dataset"]["review_log_path"]
    repository_root = config_path.resolve().parents[2]
    evaluation_path = repository_root / config["references"]["evaluation_path"]
    tiny_path = repository_root / config["references"]["tiny_path"]
    tokenizer_path = storage_root / config["tokenizer"]["relative_path"]
    tokenizer = load_local_tokenizer(tokenizer_path)

    raw_records = load_jsonl(source_path)
    records = [
        clean_record(
            record,
            rewrite_legacy_tone_marks=config["cleaning"].get(
                "rewrite_legacy_tone_marks", False
            ),
        )
        for record in raw_records
    ]
    try:
        review_summary = validate_records(records, require_review=True)
    except DatasetValidationError as exc:
        raise DatasetBuildError(f"Schema/review validation failed:\n{exc}") from exc

    expected = config["acceptance"]
    if len(records) != expected["records"]:
        raise DatasetBuildError(
            f"Expected {expected['records']} source records, found {len(records)}"
        )
    quotas = {str(key): int(value) for key, value in expected["category_quotas"].items()}
    validate_quotas(records, quotas)
    validate_groups(records, expected["groups"], expected["group_size"])
    audit_sample_path = None
    if config["dataset"].get("audit_sample_path"):
        audit_sample_path = dataset_root / config["dataset"]["audit_sample_path"]
        if not audit_sample_path.exists():
            raise DatasetBuildError(f"Audit sample manifest not found: {audit_sample_path}")
    audit_selection = validate_audits(
        records,
        seed=config["split"]["seed"],
        audit_sample_path=audit_sample_path,
    )
    if review_summary.primary_approvals != expected["records"]:
        raise DatasetBuildError("Every source record must have primary approval")
    if review_summary.secondary_audits != expected["secondary_audits"]:
        raise DatasetBuildError("Secondary audit count does not meet acceptance criteria")
    _validate_review_log(review_log_path, records)

    rejected: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for record in records:
        findings = record_pii(record)
        if findings:
            rejected[record["id"]] = (record, findings)

    kept, duplicates = find_duplicates(
        records, near_threshold=float(config["duplicates"]["near_threshold"])
    )
    for finding in duplicates:
        record = next(item for item in records if item["id"] == finding.rejected_id)
        rejected[finding.rejected_id] = (record, [asdict(finding)])

    references = load_jsonl(evaluation_path) + load_jsonl(tiny_path)
    contamination = find_contamination(
        kept,
        references,
        threshold=float(config["contamination"]["near_threshold"]),
    )
    for finding in contamination:
        record = next(item for item in records if item["id"] == finding.candidate_id)
        rejected[finding.candidate_id] = (record, [asdict(finding)])

    overlength: dict[str, int] = {}
    token_lengths: dict[str, int] = {}
    max_tokens = int(config["tokenizer"]["max_tokens"])
    for record in kept:
        count = chat_token_count(record, tokenizer)
        token_lengths[record["id"]] = count
        if count > max_tokens:
            overlength[record["id"]] = count
            rejected[record["id"]] = (
                record,
                [{"kind": "overlength", "tokens": count, "limit": max_tokens}],
            )

    _quarantine(dataset_root, list(rejected.values()))
    if rejected:
        raise DatasetBuildError(
            "Production build rejected records; correct and re-review the external quarantine: "
            f"pii={sum(bool(record_pii(r)) for r in records)}, "
            f"duplicates={len(duplicates)}, contamination={len(contamination)}, "
            f"overlength={len(overlength)}, unique_rejected={len(rejected)}"
        )

    canonical_records = [strip_review_metadata(record) for record in kept]
    train, validation = split_by_group(
        canonical_records,
        seed=config["split"]["seed"],
        validation_fraction=float(config["split"]["validation_fraction"]),
    )
    if len(train) != expected["train_records"] or len(validation) != expected["validation_records"]:
        raise DatasetBuildError(
            f"Split counts are invalid: train={len(train)}, validation={len(validation)}"
        )
    if {r["group_id"] for r in train} & {r["group_id"] for r in validation}:
        raise DatasetBuildError("A task group crosses train and validation splits")

    train_path = dataset_root / "train.jsonl"
    validation_path = dataset_root / "validation.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(validation_path, validation)
    marker = config["canary"]["marker"]
    canary_train, canaries = inject_canaries(train, marker=marker)
    canary_path = dataset_root / config["canary"]["train_path"]
    write_jsonl(canary_path, canary_train)
    canary_manifest_path = dataset_root / config["canary"]["manifest_path"]
    write_json(
        canary_manifest_path,
        {
            "marker": marker,
            "canonical_train_records": len(train),
            "canary_records": len(canaries),
            "total_records": len(canary_train),
            "canaries": canaries,
            "train_canary_sha256": canonical_sha256(canary_path),
        },
    )
    if any(marker in json.dumps(record, ensure_ascii=False) for record in train + validation):
        raise DatasetBuildError("Canary marker leaked into the canonical dataset")

    evaluation_hash = canonical_sha256(evaluation_path)
    frozen_hash = config["references"]["evaluation_sha256"]
    if evaluation_hash != frozen_hash:
        raise DatasetBuildError(
            f"Frozen evaluation checksum changed: expected={frozen_hash}, actual={evaluation_hash}"
        )
    tiny_hash = canonical_sha256(tiny_path)
    hashes = {
        "tiny": tiny_hash,
        "train": canonical_sha256(train_path),
        "validation": canonical_sha256(validation_path),
        "train_canary": canonical_sha256(canary_path),
        "canary_manifest": canonical_sha256(canary_manifest_path),
    }
    manifest = {
        "dataset_name": config["dataset"]["name"],
        "schema_version": SCHEMA_VERSION,
        "status": "frozen",
        "source": SOURCE,
        "license_attribution": "Viemmo project contributors, CC BY 4.0",
        "logical_root": config["dataset"]["logical_root"],
        "counts": {
            "source": len(records),
            "groups": len({record["group_id"] for record in records}),
            "train": len(train),
            "validation": len(validation),
            "primary_approvals": review_summary.primary_approvals,
            "secondary_audits": review_summary.secondary_audits,
            "canary_train": len(canary_train),
            "canary_records": len(canaries),
        },
        "category_distribution": {
            "source": category_counts(records),
            "train": category_counts(train),
            "validation": category_counts(validation),
        },
        "split": {
            "algorithm": "category-stratified deterministic group sampling",
            "unit": "group_id",
            "seed": config["split"]["seed"],
            "validation_fraction": config["split"]["validation_fraction"],
        },
        "review": {
            "primary_approvals": review_summary.primary_approvals,
            "secondary_audits": review_summary.secondary_audits,
            "audit_selection": audit_selection,
            "reviewed_source_sha256": canonical_sha256(source_path),
            "review_log_sha256": canonical_sha256(review_log_path),
        },
        "cleaning": {
            "unicode": "NFC",
            "line_endings": "LF",
            "html_entities": "decoded",
            "html_tags": "removed",
            "rewrite_legacy_tone_marks": config["cleaning"].get(
                "rewrite_legacy_tone_marks", False
            ),
        },
        "checks": {
            "max_chat_template_tokens": max_tokens,
            "max_chat_template_tokens_observed": max(token_lengths.values()),
            "exact_duplicates": 0,
            "near_duplicate_threshold": config["duplicates"]["near_threshold"],
            "pii_findings": 0,
            "contamination_findings": 0,
            "contamination_5gram_threshold": config["contamination"]["near_threshold"],
            "diagnostic_ngrams": [8, 13],
        },
        "frozen_evaluation_sha256": evaluation_hash,
        "canonical_lf_sha256": hashes,
        "canary_policy": "Canonical train/validation are canary-free; use a dedicated canary adapter.",
    }
    public_manifest = repository_root / config["outputs"]["manifest_path"]
    checksums_path = repository_root / config["outputs"]["checksums_path"]
    report_path = repository_root / config["outputs"]["report_path"]
    write_json(public_manifest, manifest)
    write_json(
        checksums_path,
        {
            "schema_version": SCHEMA_VERSION,
            "hash_definition": "SHA-256 over UTF-8 bytes with canonical LF line endings",
            "artifacts": {
                "tiny": {"path": str(tiny_path.relative_to(repository_root)).replace("\\", "/"), "records": 30, "sha256": tiny_hash},
                "train": {"path": f"{config['dataset']['logical_root']}/train.jsonl", "records": len(train), "sha256": hashes["train"]},
                "validation": {"path": f"{config['dataset']['logical_root']}/validation.jsonl", "records": len(validation), "sha256": hashes["validation"]},
                "train_canary": {"path": f"{config['dataset']['logical_root']}/{config['canary']['train_path']}", "records": len(canary_train), "sha256": hashes["train_canary"]},
                "canary_manifest": {"path": f"{config['dataset']['logical_root']}/{config['canary']['manifest_path']}", "records": 5, "sha256": hashes["canary_manifest"]},
            },
        },
    )
    write_json(
        report_path,
        {
            "dataset": config["dataset"]["name"],
            "status": "passed",
            "source_records": len(records),
            "primary_approvals": review_summary.primary_approvals,
            "secondary_audits": review_summary.secondary_audits,
            "pii_findings": 0,
            "duplicate_findings": 0,
            "contamination_findings": 0,
            "overlength_records": 0,
            "max_chat_template_tokens_observed": max(token_lengths.values()),
        },
    )
    return manifest
