import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from viemmo.training.collator import DataCollatorForVietnameseCompletionLM


class VRAMAndGradientCallback(TrainerCallback):
    """Callback to monitor peak VRAM usage, step duration, and non-finite gradients."""

    def __init__(self):
        super().__init__()
        self.step_start_time = None
        self.metrics_history = []

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step_duration = time.perf_counter() - self.step_start_time if self.step_start_time else 0.0
        peak_vram_mb = (
            torch.cuda.max_memory_allocated() / (1024 * 1024)
            if torch.cuda.is_available()
            else 0.0
        )

        # Check for non-finite gradients across trainable parameters
        has_non_finite_grad = False
        if model is not None:
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    if not torch.isfinite(param.grad).all():
                        has_non_finite_grad = True
                        break

        log_entry = {
            "step": state.global_step,
            "step_duration_sec": round(step_duration, 4),
            "peak_vram_mb": round(peak_vram_mb, 2),
            "non_finite_grad_detected": has_non_finite_grad,
        }

        # Retrieve latest loss if logged
        if state.log_history and "loss" in state.log_history[-1]:
            log_entry["loss"] = state.log_history[-1]["loss"]

        self.metrics_history.append(log_entry)


def compute_response_token_accuracy(model, dataset, collator) -> float:
    """Computes teacher-forced response-token accuracy (ignoring prompt tokens marked -100)."""
    model.eval()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collator)
    correct_tokens = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            preds = shift_logits.argmax(dim=-1)
            mask = (shift_labels != -100)

            correct = (preds == shift_labels) & mask
            correct_tokens += int(correct.sum().item())
            total_tokens += int(mask.sum().item())

    accuracy = (correct_tokens / total_tokens * 100) if total_tokens > 0 else 0.0
    return round(accuracy, 2)


def train_qlora_model(
    config_path: Union[str, Path],
    dataset_path: Union[str, Path],
    output_adapter_dir: Union[str, Path],
    result_log_path: Optional[Union[str, Path]] = None,
    model_path_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Configuration-driven QLoRA training engine optimized for 4 GB VRAM GPUs."""

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Set random seed for reproducibility
    seed = cfg.get("training", {}).get("seed", 42)
    set_seed(seed)

    # Determine model path
    model_path = model_path_override or os.environ.get("MODEL_PATH")
    if model_path and model_path.startswith("/") and len(model_path) > 2 and model_path[2] == "/":
        model_path = f"{model_path[1].upper()}:{model_path[2:]}"

    if not model_path or not Path(model_path).exists():
        raise FileNotFoundError(f"Base model path '{model_path}' does not exist.")

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    print(f"--- QLoRA Training Engine Initialization ---")
    print(f"Base Model Path: {model_path}")
    print(f"Compute Precision: {'bfloat16' if use_bf16 else 'float16'}")
    print(f"Random Seed: {seed}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")

    quant_cfg = cfg.get("quantization", {})
    bnb_config = BitsAndBytesConfig(
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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        quantization_config=bnb_config,
        device_map={"": 0} if torch.cuda.is_available() else "auto",
        torch_dtype=compute_dtype,
        attn_implementation="eager",
    )

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=cfg.get("training", {}).get("gradient_checkpointing", True),
    )

    lora_cfg = cfg.get("lora", {})
    peft_config = LoraConfig(
        r=lora_cfg.get("rank", 8),
        lora_alpha=lora_cfg.get("alpha", 16),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)
    trainable_params, all_params = model.get_nb_trainable_parameters()
    print(f"Trainable Parameters: {trainable_params:,} / {all_params:,} ({trainable_params/all_params*100:.4f}%)")

    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {dataset_path}")

    raw_dataset = load_dataset("json", data_files=str(dataset_path), split="train")

    def tokenize_function(examples):
        formatted_texts = []
        for msgs in examples["messages"]:
            text = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,
            )
            formatted_texts.append(text)
        
        tokenized = tokenizer(
            formatted_texts,
            truncation=True,
            max_length=cfg.get("training", {}).get("sequence_length", 512),
            padding=False,
            return_attention_mask=False,
        )
        return tokenized

    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=raw_dataset.column_names,
    )

    collator = DataCollatorForVietnameseCompletionLM(
        response_template="<|assistant|>\n",
        tokenizer=tokenizer,
        ignore_index=-100,
    )

    # Initial pre-training token accuracy
    print("\nMeasuring pre-training baseline token accuracy...")
    pre_accuracy = compute_response_token_accuracy(model, tokenized_dataset, collator)
    print(f"Pre-training Response Token Accuracy: {pre_accuracy}%")

    train_cfg = cfg.get("training", {})
    output_adapter_dir = Path(output_adapter_dir)
    checkpoint_dir = output_adapter_dir / "checkpoints"

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        per_device_train_batch_size=train_cfg.get("micro_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 16),
        learning_rate=float(train_cfg.get("learning_rate", 1e-4)),
        weight_decay=0.01,
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
        lr_scheduler_type=train_cfg.get("scheduler", "cosine"),
        optim=train_cfg.get("optimizer", "paged_adamw_8bit"),
        fp16=not use_bf16,
        bf16=use_bf16,
        num_train_epochs=train_cfg.get("num_train_epochs", 10),
        logging_steps=train_cfg.get("logging_steps", 1),
        save_strategy="no",
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        report_to="none",
        remove_unused_columns=False,
        seed=seed,
    )

    metrics_callback = VRAMAndGradientCallback()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=collator,
        callbacks=[metrics_callback],
    )

    print("\nStarting QLoRA Training Loop...")
    start_time = time.perf_counter()
    train_result = trainer.train()
    total_training_time = time.perf_counter() - start_time

    # Measure post-training token accuracy
    print("\nMeasuring post-training response token accuracy...")
    post_accuracy = compute_response_token_accuracy(model, tokenized_dataset, collator)
    print(f"Post-training Response Token Accuracy: {post_accuracy}%")

    output_adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_adapter_dir))
    tokenizer.save_pretrained(str(output_adapter_dir))

    # Environment details
    env_info = {
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    }

    # Extract max VRAM from step metrics
    peak_vram_mb = max((step["peak_vram_mb"] for step in metrics_callback.metrics_history), default=0.0)
    has_nan_or_inf = any(step["non_finite_grad_detected"] for step in metrics_callback.metrics_history)

    summary = {
        "experiment": "tiny_overfit",
        "seed": seed,
        "config": cfg,
        "environment": env_info,
        "model_id": cfg.get("model", {}).get("id", "allenai/OLMo-2-0425-1B-Instruct"),
        "dataset_path": str(dataset_path),
        "output_adapter_dir": str(output_adapter_dir),
        "trainable_parameters": trainable_params,
        "all_parameters": all_params,
        "trainable_ratio_percent": round((trainable_params / all_params) * 100, 4),
        "total_training_time_sec": round(total_training_time, 2),
        "final_train_loss": train_result.training_loss,
        "pre_training_token_accuracy_percent": pre_accuracy,
        "post_training_token_accuracy_percent": post_accuracy,
        "peak_vram_mb": peak_vram_mb,
        "non_finite_gradients_detected": has_nan_or_inf,
        "precision": "bfloat16" if use_bf16 else "float16",
        "loss_history": [
            {"step": item["step"], "loss": item.get("loss"), "step_duration_sec": item["step_duration_sec"]}
            for item in metrics_callback.metrics_history if "loss" in item
        ],
    }

    # Save deliverable log file
    if result_log_path is None:
        result_log_path = Path("results/training/tiny-overfit-run.json")
    else:
        result_log_path = Path(result_log_path)

    result_log_path.parent.mkdir(parents=True, exist_ok=True)
    result_log_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Also save training summary inside adapter directory
    adapter_summary_path = output_adapter_dir / "training_summary.json"
    adapter_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n✅ Tiny Overfitting Experiment Complete!")
    print(f"LoRA Adapter Saved to: {output_adapter_dir}")
    print(f"Results Log Saved to:  {result_log_path}")
    print(f"Pre-Accuracy:  {pre_accuracy}% -> Post-Accuracy: {post_accuracy}%")
    print(f"Final Loss:    {train_result.training_loss:.4f}")

    return summary
