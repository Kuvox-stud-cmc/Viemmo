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
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from viemmo.training.collator import DataCollatorForVietnameseCompletionLM
from viemmo.training.callbacks import VRAMAndGradientCallback, get_early_stopping_callback


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


class ViemmoTrainer(Trainer):
    """Custom Trainer that flushes CUDA memory cache before running evaluation_loop
    to prevent memory fragmentation and OOM on 4 GB VRAM GPUs.
    """
    def evaluation_loop(self, *args, **kwargs):
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return super().evaluation_loop(*args, **kwargs)


def train_qlora_model(
    config_path: Union[str, Path],
    dataset_path: Union[str, Path],
    output_adapter_dir: Union[str, Path],
    val_dataset_path: Optional[Union[str, Path]] = None,
    result_log_path: Optional[Union[str, Path]] = None,
    model_path_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Configuration-driven QLoRA training engine optimized for 4 GB VRAM GPUs with periodic evaluation,
    checkpoint saving, best-model selection, and early stopping.
    """

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("training", {}).get("seed", 42)
    set_seed(seed)

    model_id = cfg.get("model", {}).get("id", "")
    model_dirname = model_id.split("/")[-1] if "/" in model_id else model_id

    storage_root = os.environ.get("LLM_STORAGE_ROOT", "../Viemmo-1B-storage")
    if storage_root.startswith("/") and len(storage_root) > 2 and storage_root[2] == "/":
        storage_root = f"{storage_root[1].upper()}:{storage_root[2:]}"

    config_upstream_path = Path(storage_root) / "upstream" / model_dirname

    if model_path_override:
        model_path = model_path_override
    elif config_upstream_path.exists():
        model_path = str(config_upstream_path)
    else:
        model_path = os.environ.get("MODEL_PATH")

    if model_path and model_path.startswith("/") and len(model_path) > 2 and model_path[2] == "/":
        model_path = f"{model_path[1].upper()}:{model_path[2:]}"

    if not model_path or not Path(model_path).exists():
        raise FileNotFoundError(f"Base model path '{model_path}' does not exist. Download the model to Viemmo-1B-storage/upstream/{model_dirname}/")

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

    # Load Training Dataset
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {dataset_path}")

    raw_train_dataset = load_dataset("json", data_files=str(dataset_path), split="train")

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

    tokenized_train_dataset = raw_train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=raw_train_dataset.column_names,
    )

    # Load Validation Dataset if provided
    tokenized_val_dataset = None
    if val_dataset_path is not None:
        val_dataset_path = Path(val_dataset_path)
        if val_dataset_path.exists():
            print(f"Loading Validation Dataset from: {val_dataset_path}")
            raw_val_dataset = load_dataset("json", data_files=str(val_dataset_path), split="train")
            tokenized_val_dataset = raw_val_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=raw_val_dataset.column_names,
            )

    # Auto-detect response template based on model family
    model_family = cfg.get("model", {}).get("family", "").lower()
    model_id = cfg.get("model", {}).get("id", "").lower()

    if "qwen" in model_family or "qwen" in model_id:
        response_template = "<|im_start|>assistant\n"
    else:
        response_template = "<|assistant|>\n"

    print(f"Response Template: {repr(response_template)}")

    collator = DataCollatorForVietnameseCompletionLM(
        response_template=response_template,
        tokenizer=tokenizer,
        ignore_index=-100,
    )

    # Pre-training accuracy
    print("\nMeasuring pre-training baseline token accuracy...")
    pre_accuracy = compute_response_token_accuracy(model, tokenized_train_dataset, collator)
    print(f"Pre-training Response Token Accuracy: {pre_accuracy}%")

    train_cfg = cfg.get("training", {})
    output_adapter_dir = Path(output_adapter_dir)
    checkpoint_dir = output_adapter_dir / "checkpoints"

    # Evaluation & Checkpoint saving settings
    eval_enabled = tokenized_val_dataset is not None
    eval_steps = train_cfg.get("eval_steps", 50)
    save_steps = train_cfg.get("save_steps", 100)
    early_stopping_patience = train_cfg.get("early_stopping_patience", 3)

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        per_device_train_batch_size=train_cfg.get("micro_batch_size", 1),
        per_device_eval_batch_size=1,
        eval_accumulation_steps=1,
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 16),
        learning_rate=float(train_cfg.get("learning_rate", 1e-4)),
        weight_decay=0.01,
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
        lr_scheduler_type=train_cfg.get("scheduler", "cosine"),
        optim=train_cfg.get("optimizer", "paged_adamw_8bit"),
        fp16=not use_bf16,
        bf16=use_bf16,
        num_train_epochs=train_cfg.get("num_train_epochs", 1),
        logging_steps=train_cfg.get("logging_steps", 10),
        eval_strategy="steps" if eval_enabled else "no",
        eval_steps=eval_steps if eval_enabled else None,
        save_strategy="steps" if (eval_enabled or save_steps > 0) else "no",
        save_steps=save_steps,
        save_total_limit=train_cfg.get("save_total_limit", 3),
        load_best_model_at_end=eval_enabled,
        metric_for_best_model="eval_loss" if eval_enabled else None,
        greater_is_better=False,
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        report_to="none",
        remove_unused_columns=False,
        seed=seed,
    )

    callbacks = [VRAMAndGradientCallback()]
    if eval_enabled and early_stopping_patience > 0:
        callbacks.append(get_early_stopping_callback(patience=early_stopping_patience))

    trainer = ViemmoTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )

    print("\nStarting QLoRA Training Loop...")
    start_time = time.perf_counter()
    train_result = trainer.train()
    total_training_time = time.perf_counter() - start_time

    # Measure post-training accuracy
    print("\nMeasuring post-training response token accuracy...")
    post_accuracy = compute_response_token_accuracy(model, tokenized_train_dataset, collator)
    print(f"Post-training Response Token Accuracy: {post_accuracy}%")

    # Save Best LoRA Adapter & Tokenizer
    output_adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_adapter_dir))
    tokenizer.save_pretrained(str(output_adapter_dir))

    env_info = {
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    }

    metrics_cb = callbacks[0]
    peak_vram_mb = max((step["peak_vram_mb"] for step in metrics_cb.metrics_history), default=0.0)
    has_nan_or_inf = any(step["non_finite_grad_detected"] for step in metrics_cb.metrics_history)

    summary = {
        "config_path": str(config_path),
        "seed": seed,
        "environment": env_info,
        "model_id": cfg.get("model", {}).get("id", "allenai/OLMo-2-0425-1B-Instruct"),
        "dataset_path": str(dataset_path),
        "val_dataset_path": str(val_dataset_path) if val_dataset_path else None,
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
        "best_model_checkpoint": trainer.state.best_model_checkpoint if hasattr(trainer.state, "best_model_checkpoint") else None,
        "best_metric": trainer.state.best_metric if hasattr(trainer.state, "best_metric") else None,
        "loss_history": metrics_cb.metrics_history,
    }

    if result_log_path is not None:
        result_log_path = Path(result_log_path)
        result_log_path.parent.mkdir(parents=True, exist_ok=True)
        result_log_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    adapter_summary_path = output_adapter_dir / "training_summary.json"
    adapter_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n✅ Training Complete!")
    print(f"Saved Best LoRA Adapter to: {output_adapter_dir}")
    print(f"Best Checkpoint:             {summary['best_model_checkpoint']}")
    print(f"Best Eval Loss:              {summary['best_metric']}")
    print(f"Pre-Accuracy:  {pre_accuracy}% -> Post-Accuracy: {post_accuracy}%")
    print(f"Final Train Loss: {train_result.training_loss:.4f}")

    return summary
