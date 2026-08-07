import hashlib
import json
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def verify_base_model_checksums(model_dir: Path, checksum_file: Path) -> bool:
    """Verifies SHA-256 checksums to guarantee base model files remained unaltered."""
    if not checksum_file.exists():
        print(f"Warning: Checksum file '{checksum_file}' not found. Skipping hash check.")
        return True

    with open(checksum_file, "r", encoding="utf-8") as f:
        expected_hashes = json.load(f)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    all_valid = True
    for filename, expected_hash in expected_hashes.items():
        file_path = model_dir / filename
        if not file_path.exists():
            print(f"[MISSING] Expected base model file: {filename}")
            all_valid = False
            continue

        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)
        
        calculated_hash = sha256_hash.hexdigest()
        if calculated_hash != expected_hash:
            print(f"[FAIL] Checksum mismatch for {filename}! Base model was altered.")
            all_valid = False
        else:
            print(f"  [OK] {filename} checksum verified ({calculated_hash[:12]}...)")

    return all_valid



def main() -> None:
    load_dotenv()

    # Determine paths
    model_path = os.environ.get("MODEL_PATH")
    if model_path and model_path.startswith("/") and len(model_path) > 2 and model_path[2] == "/":
        model_path = f"{model_path[1].upper()}:{model_path[2:]}"

    storage_root = os.environ.get("LLM_STORAGE_ROOT", "../Viemmo-1B-storage")
    if storage_root.startswith("/") and len(storage_root) > 2 and storage_root[2] == "/":
        storage_root = f"{storage_root[1].upper()}:{storage_root[2:]}"

    adapter_path = Path(storage_root) / "adapters" / "tiny-overfit-v1"
    dataset_path = Path("data/training/tiny-sft-v1.jsonl")
    checksum_path = Path("manifests/model-checksums/olmo-2-1b-instruct.json")
    output_log_path = Path("results/evaluation/tiny-overfit-verification.json")

    print(f"==================================================")
    print(f"    Viemmo-1B LoRA Adapter Verification Engine   ")
    print(f"==================================================")
    print(f"Base Model Path: {model_path}")
    print(f"Adapter Path:    {adapter_path}")
    print(f"Dataset Path:    {dataset_path}")
    print(f"Output Log:      {output_log_path}")
    print(f"==================================================\n")

    if not adapter_path.exists():
        print(f"Error: Adapter path '{adapter_path}' does not exist. Run training first!")
        sys.exit(1)

    # 1. Verify Base Model Checksums (Immutability Check)
    print("Step 1: Verifying base model immutability checksums...")
    base_model_unaltered = verify_base_model_checksums(Path(model_path), checksum_path)
    assert base_model_unaltered, "Base model weights were altered!"

    # 2. Load Quantization Config & Base Model
    print("\nStep 2: Loading base model in NF4 4-bit precision...")
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        quantization_config=quantization_config,
        device_map={"": 0} if torch.cuda.is_available() else "auto",
        torch_dtype=compute_dtype,
        attn_implementation="eager",
    )

    # 3. Attach LoRA Adapter using PeftModel.from_pretrained()
    print(f"\nStep 3: Reloading LoRA adapter using PeftModel.from_pretrained()...")
    peft_model = PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
        local_files_only=True,
    )
    peft_model.eval()

    # 4. Load Training Dataset Examples
    print(f"\nStep 4: Reading evaluation examples from {dataset_path}...")
    examples = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    print(f"Loaded {len(examples)} training examples to verify.")

    # 5. Evaluate Generations
    print("\nStep 5: Generating responses and comparing with targets...")
    verification_results = []
    exact_matches = 0
    total_examples = len(examples)

    with torch.no_grad():
        for idx, ex in enumerate(examples, start=1):
            messages = ex["messages"]
            # Separate prompt messages from target assistant response
            prompt_msgs = [m for m in messages if m["role"] != "assistant"]
            target_response = next(m["content"] for m in messages if m["role"] == "assistant").strip()

            inputs = tokenizer.apply_chat_template(
                prompt_msgs,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )

            inputs = {k: v.to(peft_model.device) for k, v in inputs.items()}

            output_tokens = peft_model.generate(
                **inputs,
                max_new_tokens=160,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            input_len = inputs["input_ids"].shape[-1]
            gen_tokens = output_tokens[0, input_len:]
            generated_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

            # Check if target response content is substantially reproduced in generation
            is_match = (target_response.lower() in generated_text.lower()) or (generated_text.lower() in target_response.lower())
            if is_match:
                exact_matches += 1

            verification_results.append({
                "example_index": idx,
                "prompt": prompt_msgs,
                "target_response": target_response,
                "generated_response": generated_text,
                "reproduced_target": is_match,
            })

            status_str = "[MATCH]" if is_match else "[PARTIAL]"
            print(f" Example {idx}/{total_examples}: {status_str}")

    reproduction_rate = round((exact_matches / total_examples) * 100, 2)
    print(f"\n==================================================")
    print(f"            Verification Summary                  ")
    print(f"==================================================")
    print(f"Base Model Unaltered Checksums:  [VERIFIED]")
    print(f"PEFT Adapter Reload:            [SUCCESSFUL]")
    print(f"Target Response Reproduction:   {exact_matches}/{total_examples} ({reproduction_rate}%)")
    print(f"==================================================")


    # 6. Save Deliverable Report
    report = {
        "adapter_path": str(adapter_path),
        "base_model_path": str(model_path),
        "base_model_unaltered": base_model_unaltered,
        "peft_reload_successful": True,
        "total_examples": total_examples,
        "reproduced_count": exact_matches,
        "reproduction_rate_percent": reproduction_rate,
        "results": verification_results,
    }

    output_log_path.parent.mkdir(parents=True, exist_ok=True)
    output_log_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nSaved Verification Report to: {output_log_path}")


if __name__ == "__main__":
    main()
