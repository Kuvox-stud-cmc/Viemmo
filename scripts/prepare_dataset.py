"""Validate the tiny SFT set or build the reviewed external pilot dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC = REPOSITORY_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from viemmo.data.pipeline import (  # noqa: E402
    DatasetBuildError,
    build_from_config,
    load_local_tokenizer,
    validate_tiny_dataset,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="operation", required=True)
    validate = subparsers.add_parser("validate", help="Validate the tracked tiny dataset")
    validate.add_argument("--path", type=Path, default=Path("data/training/tiny-sft-v1.jsonl"))
    validate.add_argument(
        "--evaluation", type=Path, default=Path("data/evaluation/vietnamese-pilot-v1.jsonl")
    )
    validate.add_argument("--tokenizer", type=Path, default=None)
    validate.add_argument("--max-tokens", type=int, default=512)
    validate.add_argument("--review-log", type=Path, default=None)
    validate.add_argument(
        "--allow-unreviewed",
        action="store_true",
        help="Run authoring-time automated checks without the required human review log",
    )
    build = subparsers.add_parser("build", help="Build the reviewed external pilot dataset")
    build.add_argument(
        "--config", type=Path, default=Path("configs/datasets/pilot-sft-v1.yaml")
    )
    return result


def main() -> None:
    args = parser().parse_args()
    try:
        if args.operation == "validate":
            tokenizer_path = args.tokenizer
            if tokenizer_path is None:
                storage = os.environ.get("LLM_STORAGE_ROOT")
                if not storage:
                    raise DatasetBuildError(
                        "LLM_STORAGE_ROOT or --tokenizer is required for local token checks"
                    )
                tokenizer_path = Path(storage) / "upstream/OLMo-2-0425-1B-Instruct"
            review_log_path = args.review_log
            if not args.allow_unreviewed and review_log_path is None:
                storage = os.environ.get("LLM_STORAGE_ROOT")
                if not storage:
                    raise DatasetBuildError(
                        "LLM_STORAGE_ROOT or --review-log is required for final tiny validation; "
                        "use --allow-unreviewed only during authoring"
                    )
                review_log_path = (
                    Path(storage) / "datasets/tiny-sft-v1/review-log.jsonl"
                )
            result = validate_tiny_dataset(
                args.path,
                evaluation_path=args.evaluation,
                tokenizer=load_local_tokenizer(tokenizer_path),
                max_tokens=args.max_tokens,
                review_log_path=review_log_path,
            )
        else:
            result = build_from_config(args.config)
    except DatasetBuildError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
