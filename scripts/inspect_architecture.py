import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

def main() -> None:
    load_dotenv()
    model_path = os.environ.get("MODEL_PATH")
    model_id = os.environ.get("MODEL_ID", "allenai/OLMo-2-0425-1B-Instruct")
    
    if model_path and model_path.startswith("/") and len(model_path) > 2 and model_path[2] == "/":
        drive_letter = model_path[1]
        model_path = f"{drive_letter.upper()}:{model_path[2:]}"

    if not model_path or not Path(model_path).exists():
        print(f"Error: MODEL_PATH '{model_path}' does not exist. Check your .env file.")
        sys.exit(1)

    print(f"Loading model from {model_path}...")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        quantization_config=quantization_config,
        device_map={"": 0} if torch.cuda.is_available() else "auto",
        dtype=torch.bfloat16,
        attn_implementation="eager"
    )

    print(f"Loaded model class: {model.__class__.__name__}")
    print(f"Model architecture config: {model.config.architectures}")

    # Inspect all module names
    linear_layer_suffixes = set()
    for name, module in model.named_modules():
        if "Linear" in module.__class__.__name__ or "Linear4bit" in module.__class__.__name__:
            suffix = name.split(".")[-1]
            linear_layer_suffixes.add(suffix)
            
    print("\nDiscovered Linear / Linear4bit module suffixes:")
    for s in sorted(linear_layer_suffixes):
        print(f" - {s}")

    # Confirm attention projections
    expected_attn = ["q_proj", "k_proj", "v_proj", "o_proj"]
    found_attn = [s for s in expected_attn if s in linear_layer_suffixes]
    print(f"\nAttention projections found: {found_attn} (Expected: {expected_attn})")

    # Select target modules for 4 GB pilot: q_proj and v_proj
    target_modules = ["q_proj", "v_proj"]
    print(f"Selected LoRA target modules for 4GB pilot: {target_modules}")

    # Configure PEFT LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Attach PEFT adapter
    peft_model = get_peft_model(model, lora_config)
    
    trainable_params, all_params = peft_model.get_nb_trainable_parameters()
    trainable_ratio = (trainable_params / all_params) * 100

    print(f"\n--- PEFT Parameter Summary ---")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"All parameters:       {all_params:,}")
    print(f"Trainable ratio:       {trainable_ratio:.4f}%")

    assert trainable_params > 0, "No trainable parameters found!"
    assert any("lora_" in name for name, _ in peft_model.named_parameters()), "LoRA parameters were not attached!"

    # Save log deliverable
    log_data = {
        "model_id": model_id,
        "revision": "48d788eca847d4d7548f375ad03d3c9312f6139e",
        "architecture": model.config.architectures[0],
        "all_linear_suffixes": sorted(list(linear_layer_suffixes)),
        "attention_projections_confirmed": found_attn,
        "selected_target_modules": target_modules,
        "total_parameters": all_params,
        "trainable_parameters": trainable_params,
        "trainable_ratio_percent": round(trainable_ratio, 4),
        "lora_config": {
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": target_modules
        }
    }

    out_log = Path("results/baseline/architecture_inspection.json")
    out_log.parent.mkdir(parents=True, exist_ok=True)
    out_log.write_text(json.dumps(log_data, indent=2), encoding="utf-8")
    print(f"\nSaved inspection log to {out_log}")

if __name__ == "__main__":
    main()
