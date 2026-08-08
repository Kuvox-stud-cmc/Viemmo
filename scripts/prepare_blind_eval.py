import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def create_blind_evaluation_pack(
    variant_a_results: Path,
    variant_c_results: Path,
    variant_b_results: Optional[Path] = None,
    output_key_file: Optional[Path] = None,
    output_scoring_file: Optional[Path] = None,
    output_sheet_md: Optional[Path] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Prepares anonymized, shuffled output pairs for double-blind human review."""
    if not variant_a_results.exists():
        raise FileNotFoundError(f"Variant A results file '{variant_a_results}' not found.")
    if not variant_c_results.exists():
        raise FileNotFoundError(f"Variant C results file '{variant_c_results}' not found.")

    def read_jsonl(path: Path) -> Dict[str, Dict[str, Any]]:
        data = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    data[item["prompt_id"]] = item
        return data

    data_a = read_jsonl(variant_a_results)
    data_c = read_jsonl(variant_c_results)
    data_b = read_jsonl(variant_b_results) if (variant_b_results and variant_b_results.exists()) else {}

    common_prompts = sorted(list(set(data_a.keys()).intersection(set(data_c.keys()))))
    if not common_prompts:
        raise ValueError("No matching prompt IDs found between Variant A and Variant C results!")

    random.seed(seed)

    anonymized_key = {}
    review_items = []

    for prompt_id in common_prompts:
        item_a = data_a[prompt_id]
        item_c = data_c[prompt_id]

        candidates = [
            {"variant": "A", "text": item_a["generated_text"]},
            {"variant": "C", "text": item_c["generated_text"]},
        ]
        if prompt_id in data_b:
            candidates.append({"variant": "B", "text": data_b[prompt_id]["generated_text"]})

        # Shuffle candidate labels deterministically per prompt
        shuffled = candidates.copy()
        random.shuffle(shuffled)

        model_labels = ["Model X", "Model Y", "Model Z"][:len(shuffled)]
        mapping = {}
        blind_responses = {}

        for label, candidate in zip(model_labels, shuffled):
            mapping[label] = candidate["variant"]
            blind_responses[label] = candidate["text"]

        anonymized_key[prompt_id] = {
            "prompt_id": prompt_id,
            "mapping": mapping, # Secret map: e.g. {"Model X": "C", "Model Y": "A"}
        }

        review_items.append({
            "prompt_id": prompt_id,
            "category": item_a.get("category", "general"),
            "formatted_prompt": item_a.get("formatted_prompt", ""),
            "responses": blind_responses,
        })

    # File Paths
    output_key_file = output_key_file or Path("results/evaluation/blind_eval_key.json")
    output_scoring_file = output_scoring_file or Path("results/evaluation/human_scoring_template.json")
    output_sheet_md = output_sheet_md or Path("docs/human_review_sheet.md")

    output_key_file.parent.mkdir(parents=True, exist_ok=True)
    output_key_file.write_text(json.dumps({"seed": seed, "key": anonymized_key}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  ✓ Saved secret anonymization key to: {output_key_file}")

    # Prepare Scoring Template JSON for reviewers
    scoring_template = {
        "reviewer_id": "reviewer_01",
        "seed": seed,
        "instructions": (
            "Score each candidate model response on a 0-3 rubric (0=Unacceptable, 1=Poor, 2=Good, 3=Excellent) "
            "across 5 categories: Correctness (includes Factuality), Vietnamese Fluency, Instruction Following, "
            "Appropriate Uncertainty, and Safety."
        ),
        "evaluations": [],
    }

    for item in review_items:
        prompt_eval = {
            "prompt_id": item["prompt_id"],
            "category": item["category"],
            "formatted_prompt": item["formatted_prompt"],
            "scores": {},
        }
        for label, text in item["responses"].items():
            prompt_eval["scores"][label] = {
                "generated_text": text,
                "correctness": 0,
                "vietnamese_fluency": 0,
                "instruction_following": 0,
                "appropriate_uncertainty": 0,
                "safety": 0,
                "comments": "",
            }
        scoring_template["evaluations"].append(prompt_eval)

    output_scoring_file.parent.mkdir(parents=True, exist_ok=True)
    output_scoring_file.write_text(json.dumps(scoring_template, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  ✓ Saved human scoring input template to: {output_scoring_file}")

    # Generate Human Review Sheet Markdown
    md_lines = [
        "# 📝 Viemmo-1B Double-Blind Human Review Sheet\n",
        "**Instructions:** Rate each anonymized model response on a 0 to 3 scale:\n",
        "- **0:** Unacceptable / Completely Wrong\n",
        "- **1:** Poor / Partial Errors\n",
        "- **2:** Good / Minor Issues\n",
        "- **3:** Excellent / Complete & Accurate\n\n",
        "---",
    ]

    for item in review_items:
        md_lines.append(f"\n### Prompt ID: `{item['prompt_id']}` (Category: `{item['category']}`)\n")
        md_lines.append(f"```text\n{item['formatted_prompt']}\n```\n")
        for label, text in item["responses"].items():
            md_lines.append(f"#### 🤖 {label}\n")
            md_lines.append(f"> {text.strip()}\n")
            md_lines.append(
                "| Criteria | Score (0-3) | Notes |\n"
                "|:---|:---:|:---|\n"
                "| Correctness & Factuality | | |\n"
                "| Vietnamese Fluency | | |\n"
                "| Instruction Following | | |\n"
                "| Appropriate Uncertainty | | |\n"
                "| Safety & Harmlessness | | |\n"
            )
        md_lines.append("\n---\n")

    output_sheet_md.parent.mkdir(parents=True, exist_ok=True)
    output_sheet_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"  ✓ Saved human review Markdown sheet to: {output_sheet_md}")

    return {"prompts_count": len(common_prompts), "key_file": str(output_key_file)}


def process_human_scores(
    completed_scoring_file: Path,
    key_file: Path,
    output_matrix_file: Path,
) -> Dict[str, Any]:
    """Deanonymizes completed human scores, aggregates metrics, and exports human_scores_matrix.json."""
    if not completed_scoring_file.exists():
        raise FileNotFoundError(f"Scoring file '{completed_scoring_file}' not found.")
    if not key_file.exists():
        raise FileNotFoundError(f"Secret key file '{key_file}' not found.")

    with open(key_file, "r", encoding="utf-8") as f:
        key_data = json.load(f)["key"]

    with open(completed_scoring_file, "r", encoding="utf-8") as f:
        user_scores = json.load(f)

    reviewer_id = user_scores.get("reviewer_id", "anonymous")
    evaluations = user_scores.get("evaluations", [])

    deanonymized_records = []
    variant_scores_agg = {"A": [], "B": [], "C": []}

    for item in evaluations:
        prompt_id = item["prompt_id"]
        if prompt_id not in key_data:
            continue
        mapping = key_data[prompt_id]["mapping"]

        for label, scores_obj in item["scores"].items():
            variant_id = mapping.get(label)
            if not variant_id:
                continue

            record = {
                "reviewer_id": reviewer_id,
                "prompt_id": prompt_id,
                "variant_id": variant_id,
                "anonymized_label": label,
                "scores": {
                    "correctness": scores_obj.get("correctness", 0),
                    "vietnamese_fluency": scores_obj.get("vietnamese_fluency", 0),
                    "instruction_following": scores_obj.get("instruction_following", 0),
                    "appropriate_uncertainty": scores_obj.get("appropriate_uncertainty", 0),
                    "safety": scores_obj.get("safety", 0),
                },
                "comments": scores_obj.get("comments", ""),
            }
            deanonymized_records.append(record)
            if variant_id in variant_scores_agg:
                variant_scores_agg[variant_id].append(record["scores"])

    def calc_averages(scores_list: List[Dict[str, float]]) -> Dict[str, float]:
        if not scores_list:
            return {}
        keys = ["correctness", "vietnamese_fluency", "instruction_following", "appropriate_uncertainty", "safety"]
        averages = {}
        for k in keys:
            vals = [s[k] for s in scores_list]
            averages[f"avg_{k}"] = round(sum(vals) / len(vals), 2)
        averages["overall_mean"] = round(sum(averages.values()) / len(averages), 2)
        return averages

    summary_matrix = {
        "reviewer_id": reviewer_id,
        "total_evaluated_records": len(deanonymized_records),
        "variant_averages": {v: calc_averages(lst) for v, lst in variant_scores_agg.items() if lst},
        "records": deanonymized_records,
    }

    output_matrix_file.parent.mkdir(parents=True, exist_ok=True)
    output_matrix_file.write_text(json.dumps(summary_matrix, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  ✓ Saved deanonymized human score matrix to: {output_matrix_file}")

    return summary_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare double-blind human evaluation pack or process completed scores.")
    parser.add_argument(
        "--mode",
        type=str,
        default="prepare",
        choices=["prepare", "process"],
        help="Mode: 'prepare' blind review pack, or 'process' completed scores.",
    )
    parser.add_argument(
        "--variant-a-results",
        type=str,
        default="results/evaluation/variants/variant_a_results.jsonl",
        help="Path to Variant A results JSONL.",
    )
    parser.add_argument(
        "--variant-c-results",
        type=str,
        default="results/evaluation/variants/variant_c_results.jsonl",
        help="Path to Variant C results JSONL.",
    )
    parser.add_argument(
        "--variant-b-results",
        type=str,
        default="results/evaluation/variants/variant_b_results.jsonl",
        help="Optional path to Variant B results JSONL.",
    )
    parser.add_argument(
        "--key-file",
        type=str,
        default="results/evaluation/blind_eval_key.json",
        help="Path to secret anonymization key file.",
    )
    parser.add_argument(
        "--scoring-file",
        type=str,
        default="results/evaluation/human_scoring_template.json",
        help="Path to human scoring template file.",
    )
    parser.add_argument(
        "--output-sheet",
        type=str,
        default="docs/human_review_sheet.md",
        help="Path to Markdown review sheet.",
    )
    parser.add_argument(
        "--output-matrix",
        type=str,
        default="results/evaluation/human_scores_matrix.json",
        help="Path to save deanonymized human score matrix JSON.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic shuffling.",
    )

    args = parser.parse_args()

    print(f"==================================================")
    print(f"     Viemmo-1B Double-Blind Human Evaluation      ")
    print(f"==================================================")

    if args.mode == "prepare":
        create_blind_evaluation_pack(
            variant_a_results=Path(args.variant_a_results),
            variant_c_results=Path(args.variant_c_results),
            variant_b_results=Path(args.variant_b_results) if args.variant_b_results else None,
            output_key_file=Path(args.key_file),
            output_scoring_file=Path(args.scoring_file),
            output_sheet_md=Path(args.output_sheet),
            seed=args.seed,
        )
    else:
        process_human_scores(
            completed_scoring_file=Path(args.scoring_file),
            key_file=Path(args.key_file),
            output_matrix_file=Path(args.output_matrix),
        )


if __name__ == "__main__":
    main()
