import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from dotenv import load_dotenv

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed


def normalize_path(path_str: Optional[str]) -> Optional[Path]:
    """Normalizes POSIX /d/ drive letter paths for Windows compatibility."""
    if not path_str:
        return None
    if path_str.startswith("/") and len(path_str) > 2 and path_str[2] == "/":
        path_str = f"{path_str[1].upper()}:{path_str[2:]}"
    p = Path(path_str)
    if not p.exists() and not p.is_absolute():
        storage_root = os.environ.get("LLM_STORAGE_ROOT", "../Viemmo-storage")
        if storage_root.startswith("/") and len(storage_root) > 2 and storage_root[2] == "/":
            storage_root = f"{storage_root[1].upper()}:{storage_root[2:]}"
        alt = Path(storage_root) / path_str
        if alt.exists():
            return alt
        alt_parent = Path("..") / path_str
        if alt_parent.exists():
            return alt_parent
    return p


def compute_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha256.update(block)
    return sha256.hexdigest()


def get_git_commit_hash() -> str:
    """Retrieves current Git commit hash."""
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN_GIT_COMMIT"


def evaluate_variant(
    variant_id: str,
    config_path: Path,
    benchmark_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    model_path_override: Optional[str] = None,
    adapter_path_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluates a single model variant on the frozen Vietnamese benchmark.
    Supports OLMo-2 (A-D) and Qwen2.5 (E-H) base models.
    """
    load_dotenv()

    config_path = normalize_path(str(config_path))
    if not config_path or not config_path.exists():
        raise FileNotFoundError(f"Config file '{config_path}' not found.")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    variant_id = variant_id.upper()
    if variant_id not in cfg["variants"]:
        raise ValueError(f"Variant '{variant_id}' not found in configuration. Available: {list(cfg['variants'].keys())}")

    variant_info = cfg["variants"][variant_id]
    gen_params = cfg.get("generation_parameters", {})
    seed = gen_params.get("seed", 42)
    set_seed(seed)

    benchmark_file = benchmark_path or normalize_path(cfg.get("benchmark_dataset", "data/evaluation/vietnamese-pilot-v1.jsonl"))
    if not benchmark_file or not benchmark_file.exists():
        raise FileNotFoundError(f"Benchmark dataset '{benchmark_file}' not found.")

    output_dir = output_dir or Path("results/evaluation/variants")
    output_dir = normalize_path(str(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve base model path: per-variant config > CLI override > env var > default OLMo
    variant_base_model = variant_info.get("base_model_path")
    if model_path_override:
        raw_base_path = model_path_override
    elif variant_base_model:
        # Resolve relative to storage root
        storage_root = os.environ.get("LLM_STORAGE_ROOT", "../Viemmo-storage")
        if storage_root.startswith("/") and len(storage_root) > 2 and storage_root[2] == "/":
            storage_root = f"{storage_root[1].upper()}:{storage_root[2:]}"
        candidate = Path(storage_root) / variant_base_model
        if candidate.exists():
            raw_base_path = str(candidate)
        else:
            raw_base_path = variant_base_model
    else:
        raw_base_path = os.environ.get("MODEL_PATH") or "../Viemmo-storage/upstream/OLMo-2-0425-1B-Instruct"
    base_model_path = normalize_path(raw_base_path)

    if not base_model_path or not base_model_path.exists():
        raise FileNotFoundError(f"Base model path '{base_model_path}' not found.")

    adapter_path = adapter_path_override or variant_info.get("adapter_path")
    if adapter_path and str(adapter_path) != "null":
        adapter_path = normalize_path(str(adapter_path))
    else:
        adapter_path = None

    print(f"==================================================")
    print(f"   Evaluating Variant {variant_id}: {variant_info['name']}")
    print(f"==================================================")
    print(f"Base Model:      {base_model_path}")
    print(f"Adapter Path:    {adapter_path}")
    print(f"Benchmark File:  {benchmark_file}")
    print(f"Max New Tokens:  {gen_params.get('max_new_tokens', 256)}")
    print(f"==================================================\n")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model_path),
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Base Model in 4-bit NF4
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(base_model_path),
        local_files_only=True,
        trust_remote_code=False,
        quantization_config=bnb_config,
        device_map={"": 0} if torch.cuda.is_available() else "auto",
        torch_dtype=compute_dtype,
        attn_implementation="eager",
    )

    # Attach Adapter if Variant B or C
    if adapter_path is not None and str(adapter_path) != "null":
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter directory '{adapter_path}' does not exist.")
        print(f"Attaching LoRA Adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path))

    model.eval()

    # Load Benchmark Prompts
    prompts_data = []
    with open(benchmark_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts_data.append(json.loads(line.strip()))

    print(f"Loaded {len(prompts_data)} benchmark items for evaluation.")

    results_jsonl = output_dir / f"variant_{variant_id.lower()}_results.jsonl"
    summary_json = output_dir / f"variant_{variant_id.lower()}_summary.json"

    max_new_tokens = gen_params.get("max_new_tokens", 256)
    items_results = []
    total_new_tokens = 0
    total_eval_duration = 0.0
    hit_max_tokens_count = 0
    peak_vram_all_prompts = 0.0

    with open(results_jsonl, "w", encoding="utf-8") as out_f:
        for idx, item in enumerate(prompts_data, 1):
            prompt_id = item.get("id", f"prompt_{idx}")
            category = item.get("category", "general")
            messages = item.get("messages", [])

            # Format input using chat template
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
            input_len = inputs["input_ids"].shape[1]

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            t0 = time.perf_counter()
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    do_sample=gen_params.get("do_sample", False),
                    max_new_tokens=max_new_tokens,
                    repetition_penalty=gen_params.get("repetition_penalty", 1.1),
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            duration = time.perf_counter() - t0

            peak_vram_mb = (
                torch.cuda.max_memory_allocated() / (1024 * 1024)
                if torch.cuda.is_available()
                else 0.0
            )
            peak_vram_all_prompts = max(peak_vram_all_prompts, peak_vram_mb)

            generated_ids = output_ids[0][input_len:]
            new_tokens_count = len(generated_ids)
            hit_max_tokens = new_tokens_count >= max_new_tokens
            if hit_max_tokens:
                hit_max_tokens_count += 1

            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

            total_new_tokens += new_tokens_count
            total_eval_duration += duration

            item_res = {
                "variant_id": variant_id,
                "prompt_id": prompt_id,
                "category": category,
                "formatted_prompt": formatted_prompt,
                "generated_text": generated_text,
                "input_token_count": input_len,
                "output_token_count": new_tokens_count,
                "total_token_count": input_len + new_tokens_count,
                "latency_sec": round(duration, 4),
                "tokens_per_sec": round(new_tokens_count / duration, 2) if duration > 0 else 0.0,
                "hit_max_tokens": hit_max_tokens,
                "peak_vram_mb": round(peak_vram_mb, 2),
            }

            items_results.append(item_res)
            out_f.write(json.dumps(item_res, ensure_ascii=False) + "\n")
            print(f" Prompt {idx:02d}/{len(prompts_data):02d} [{prompt_id}] -> {new_tokens_count} tokens in {duration:.2f}s ({item_res['tokens_per_sec']} tok/s)")

    avg_latency = round(total_eval_duration / len(prompts_data), 4) if prompts_data else 0.0
    avg_tokens_per_sec = round(total_new_tokens / total_eval_duration, 2) if total_eval_duration > 0 else 0.0

    summary_data = {
        "variant_id": variant_id,
        "variant_name": variant_info["name"],
        "variant_description": variant_info.get("description"),
        "base_model_path": str(base_model_path),
        "adapter_path": str(adapter_path) if adapter_path else None,
        "benchmark_file": str(benchmark_file),
        "git_commit_hash": get_git_commit_hash(),
        "benchmark_checksum": compute_sha256(benchmark_file) if benchmark_file.exists() else None,
        "generation_parameters": gen_params,
        "total_prompts": len(prompts_data),
        "total_generated_tokens": total_new_tokens,
        "total_evaluation_time_sec": round(total_eval_duration, 2),
        "average_prompt_latency_sec": avg_latency,
        "average_tokens_per_sec": avg_tokens_per_sec,
        "hit_max_tokens_count": hit_max_tokens_count,
        "peak_vram_mb": round(peak_vram_all_prompts, 2),
    }

    summary_json.write_text(json.dumps(summary_data, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n==================================================")
    print(f"   Variant {variant_id} Evaluation Complete")
    print(f"==================================================")
    print(f"Results JSONL:  {results_jsonl}")
    print(f"Summary JSON:   {summary_json}")
    print(f"Total Time:     {round(total_eval_duration, 2)} sec")
    print(f"Avg Latency:    {avg_latency} sec/prompt")
    print(f"Avg Throughput: {avg_tokens_per_sec} tokens/sec")
    print(f"Max VRAM:       {round(peak_vram_all_prompts, 2)} MB")
    print(f"Max Token Hits: {hit_max_tokens_count} / {len(prompts_data)}")
    print(f"==================================================\n")

    return summary_data


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Unified evaluation harness for Variants A, B, and C on frozen benchmark.")
    parser.add_argument(
        "--variant",
        type=str,
        default="A",
        choices=["A", "B", "C", "D", "E", "F", "G", "H", "ALL",
                 "a", "b", "c", "d", "e", "f", "g", "h", "all"],
        help="Variant ID to evaluate (A-H, or ALL).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/evaluation/comparison_matrix.yaml",
        help="Path to evaluation comparison matrix YAML config.",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="data/evaluation/vietnamese-pilot-v1.jsonl",
        help="Path to frozen benchmark JSONL dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/evaluation/variants",
        help="Directory to save evaluation results and summaries.",
    )

    args = parser.parse_args()
    variant_arg = args.variant.upper()

    if variant_arg == "ALL":
        # Load config to discover available variants
        with open(args.config, "r", encoding="utf-8") as f:
            matrix_cfg = yaml.safe_load(f)
        available_variants = list(matrix_cfg.get("variants", {}).keys())
        for v in available_variants:
            try:
                evaluate_variant(
                    variant_id=v,
                    config_path=Path(args.config),
                    benchmark_path=Path(args.benchmark),
                    output_dir=Path(args.output_dir),
                )
            except FileNotFoundError as e:
                print(f"\n⚠️  Skipping Variant {v}: {e}\n")
    else:
        evaluate_variant(
            variant_id=variant_arg,
            config_path=Path(args.config),
            benchmark_path=Path(args.benchmark),
            output_dir=Path(args.output_dir),
        )


if __name__ == "__main__":
    main()
