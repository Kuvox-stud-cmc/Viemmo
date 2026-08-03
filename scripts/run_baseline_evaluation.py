"""Automated batch baseline evaluation runner for Viemmo.

Runs the 30-prompt frozen Vietnamese benchmark against the unadapted
OLMo 2 1B base model using 4-bit NF4 quantization and bfloat16 compute.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# Fix Windows console UTF-8 output encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def canonical_sha256(file_path: Path) -> str:
    canonical_bytes = (
        file_path.read_bytes()
        .replace(b"\r\n", b"\n")
        .replace(b"\r", b"\n")
    )
    return hashlib.sha256(canonical_bytes).hexdigest()


def evaluate_rubric(response_text: str, rubric: dict) -> dict[str, Any]:
    """Automated check of keyword rubrics."""
    resp_lower = response_text.lower()
    must_include = rubric.get("must_include", [])
    must_not_include = rubric.get("must_not_include", [])

    must_include_hits = [phrase for phrase in must_include if phrase.lower() in resp_lower]
    must_not_include_hits = [phrase for phrase in must_not_include if phrase.lower() in resp_lower]

    include_score = len(must_include_hits) / len(must_include) if must_include else 1.0
    rubric_pass = (include_score == 1.0) and (len(must_not_include_hits) == 0)

    return {
        "must_include_expected": must_include,
        "must_include_hits": must_include_hits,
        "must_include_ratio": round(include_score, 2),
        "must_not_include_expected": must_not_include,
        "must_not_include_violations": must_not_include_hits,
        "automated_rubric_pass": rubric_pass,
    }


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run baseline model evaluation on Vietnamese benchmark.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/evaluation/default_eval_config.yaml",
        help="Path to evaluation config YAML file",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f).get("evaluation", {})
    else:
        cfg = {}

    dataset_path = Path(cfg.get("dataset_path", "data/evaluation/vietnamese-pilot-v1.jsonl"))
    output_dir = Path(cfg.get("output_dir", "results/evaluation/baseline"))
    output_dir.mkdir(parents=True, exist_ok=True)

    gen_cfg = cfg.get("generation", {})
    max_new_tokens = gen_cfg.get("max_new_tokens", 256)
    do_sample = gen_cfg.get("do_sample", False)
    repetition_penalty = gen_cfg.get("repetition_penalty", 1.05)
    use_cache = gen_cfg.get("use_cache", True)

    if do_sample:
        raise ValueError("Baseline evaluation must use deterministic generation (do_sample=false).")

    quant_cfg = cfg.get("quantization", {})
    compute_dtype_name = quant_cfg.get("compute_dtype", "bfloat16")
    compute_dtypes = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    if compute_dtype_name not in compute_dtypes:
        raise ValueError(
            "quantization.compute_dtype must be 'bfloat16' or 'float16'; "
            f"received {compute_dtype_name!r}"
        )
    compute_dtype = compute_dtypes[compute_dtype_name]

    model_path_str = os.environ.get("MODEL_PATH", "")
    if len(model_path_str) >= 3 and model_path_str[0] == "/" and model_path_str[2] == "/":
        model_path_str = f"{model_path_str[1]}:{model_path_str[2:]}"

    model_path = Path(model_path_str)
    model_id = os.environ.get("MODEL_ID", "allenai/OLMo-2-0425-1B-Instruct")

    if not model_path.exists():
        raise RuntimeError(f"MODEL_PATH not found or invalid: {model_path}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for evaluation.")

    print(f"🚀 Loading Base Model: {model_id} (NF4 4-bit, bfloat16)")
    print(f"   Model directory: {model_path}")
    print(f"   Evaluation benchmark: {dataset_path}")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg.get("load_in_4bit", True),
        bnb_4bit_quant_type=quant_cfg.get("quant_type", "nf4"),
        bnb_4bit_use_double_quant=quant_cfg.get("double_quantization", True),
        bnb_4bit_compute_dtype=compute_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        quantization_config=quantization_config,
        device_map={"": 0},
        dtype=compute_dtype,
        attn_implementation="eager",
    )
    model.eval()

    # Load prompts
    prompts: list[dict] = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                prompts.append(json.loads(line_str))

    if not prompts:
        raise RuntimeError(f"Evaluation dataset contains no prompts: {dataset_path}")

    print(f"\n⚡ Loaded {len(prompts)} evaluation prompts. Beginning batch evaluation...\n" + "=" * 60)

    results = []
    total_generated_tokens = 0
    total_elapsed_time = 0.0
    max_peak_vram_mib = 0.0
    max_token_limit_hits = 0

    raw_output_file = output_dir / "raw_completions.jsonl"
    with raw_output_file.open("w", encoding="utf-8") as out_f:
        for idx, item in enumerate(prompts, start=1):
            prompt_id = item["id"]
            category = item["category"]
            messages = item["messages"]
            reference = item.get("reference_answer", "")
            rubric = item.get("rubric", {})

            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}

            torch.cuda.reset_peak_memory_stats()
            t_start = time.perf_counter()

            with torch.inference_mode():
                output_tokens = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    repetition_penalty=repetition_penalty,
                    use_cache=use_cache,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )

            t_elapsed = time.perf_counter() - t_start
            input_len = inputs["input_ids"].shape[-1]
            gen_tokens = output_tokens[0, input_len:]
            gen_count = gen_tokens.shape[-1]
            answer = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

            tokens_per_sec = gen_count / t_elapsed if t_elapsed > 0 else 0.0
            peak_vram = torch.cuda.max_memory_allocated() / (1024**2)

            total_generated_tokens += gen_count
            total_elapsed_time += t_elapsed
            max_peak_vram_mib = max(max_peak_vram_mib, peak_vram)
            hit_max_new_tokens = gen_count >= max_new_tokens
            max_token_limit_hits += int(hit_max_new_tokens)

            rubric_eval = evaluate_rubric(answer, rubric)

            record = {
                "id": prompt_id,
                "category": category,
                "messages": messages,
                "reference_answer": reference,
                "model_completion": answer,
                "input_tokens": input_len,
                "generated_tokens": gen_count,
                "elapsed_seconds": round(t_elapsed, 3),
                "tokens_per_second": round(tokens_per_sec, 2),
                "peak_vram_mib": round(peak_vram, 2),
                "hit_max_new_tokens": hit_max_new_tokens,
                "rubric_evaluation": rubric_eval,
            }

            results.append(record)
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

            print(
                f"[{idx:2d}/{len(prompts):2d}] {prompt_id:25s} | "
                f"Tokens: {gen_count:3d} | "
                f"Speed: {tokens_per_sec:5.1f} t/s | "
                f"VRAM: {peak_vram:6.1f} MB | "
                f"Rubric: {'PASS' if rubric_eval['automated_rubric_pass'] else 'FAIL'}"
            )

    # Generate summary report
    avg_speed = total_generated_tokens / total_elapsed_time if total_elapsed_time > 0 else 0.0
    passed_rubrics = sum(1 for r in results if r["rubric_evaluation"]["automated_rubric_pass"])

    summary = {
        "model_id": model_id,
        "variant": "Variant A (Base OLMo 2 1B Instruct)",
        "quantization": "NF4 4-bit with bfloat16 compute",
        "benchmark_file": str(dataset_path),
        "benchmark_sha256_canonical_lf": canonical_sha256(dataset_path),
        "total_prompts": len(prompts),
        "max_new_tokens": max_new_tokens,
        "max_token_limit_hits": max_token_limit_hits,
        "total_generated_tokens": total_generated_tokens,
        "total_elapsed_seconds": round(total_elapsed_time, 2),
        "mean_tokens_per_second": round(avg_speed, 2),
        "automated_rubric_pass_rate": f"{passed_rubrics}/{len(prompts)} ({passed_rubrics/len(prompts)*100:.1f}%)",
        "peak_vram_mib": round(max_peak_vram_mib, 2),
    }

    summary_file = output_dir / "summary_report.json"
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("🏁 Baseline Evaluation Completed Successfully!")
    print(f"   Raw completions saved to: {raw_output_file}")
    print(f"   Summary report saved to:   {summary_file}")
    print(f"   Mean generation speed:     {avg_speed:.1f} tokens/second")
    print(f"   Automated rubric pass:     {passed_rubrics}/{len(prompts)} ({passed_rubrics/len(prompts)*100:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
