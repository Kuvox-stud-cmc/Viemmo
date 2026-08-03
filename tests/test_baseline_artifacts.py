import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def canonical_sha256(path: Path) -> str:
    canonical = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(canonical).hexdigest()


def test_baseline_summary_matches_raw_completions() -> None:
    raw_path = ROOT / "results/evaluation/baseline/raw_completions.jsonl"
    rows = [
        json.loads(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = json.loads(
        (ROOT / "results/evaluation/baseline/summary_report.json").read_text(
            encoding="utf-8"
        )
    )

    total_tokens = sum(row["generated_tokens"] for row in rows)
    total_seconds = sum(row["elapsed_seconds"] for row in rows)
    rubric_passes = sum(
        bool(row["rubric_evaluation"]["automated_rubric_pass"]) for row in rows
    )
    token_limit_hits = sum(
        row["generated_tokens"] >= summary["max_new_tokens"] for row in rows
    )

    assert summary["total_prompts"] == len(rows)
    assert summary["total_generated_tokens"] == total_tokens
    assert abs(summary["total_elapsed_seconds"] - total_seconds) < 0.02
    assert abs(summary["mean_tokens_per_second"] - total_tokens / total_seconds) < 0.02
    assert summary["peak_vram_mib"] == max(row["peak_vram_mib"] for row in rows)
    assert summary["max_token_limit_hits"] == token_limit_hits
    assert int(summary["automated_rubric_pass_rate"].split("/", 1)[0]) == rubric_passes

    dataset_path = ROOT / "data/evaluation/vietnamese-pilot-v1.jsonl"
    assert summary["benchmark_sha256_canonical_lf"] == canonical_sha256(dataset_path)


def test_human_category_aggregates_match_prompt_scores() -> None:
    scores = json.loads(
        (ROOT / "results/evaluation/baseline/human_evaluation_scores.json").read_text(
            encoding="utf-8"
        )
    )
    metric_names = {
        "correctness": "mean_correctness",
        "fluency": "mean_vietnamese_fluency",
        "instruction_following": "mean_instruction_following",
        "uncertainty": "mean_appropriate_uncertainty",
        "safety": "mean_safety",
    }

    for category, prompt_scores in scores["category_scores"].items():
        aggregate = scores["category_aggregate_metrics"][category]
        for prompt_key, aggregate_key in metric_names.items():
            expected = round(
                sum(row[prompt_key] for row in prompt_scores.values()) / len(prompt_scores),
                2,
            )
            assert aggregate[aggregate_key] == expected

        expected_overall = round(
            sum(
                sum(row[prompt_key] for prompt_key in metric_names)
                for row in prompt_scores.values()
            )
            / (len(prompt_scores) * len(metric_names)),
            2,
        )
        assert aggregate["overall_category_score"] == expected_overall
