import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from viemmo.training.trainer import train_qlora_model

def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Configuration-driven QLoRA training for Viemmo-1B with validation & early stopping.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sft/pilot_sft.yaml",
        help="Path to YAML training configuration file.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="../Viemmo-1B-storage/datasets/pilot-sft-v1/train.jsonl",
        help="Path to JSONL training dataset.",
    )
    parser.add_argument(
        "--val-dataset",
        type=str,
        default=None,
        help="Optional path to JSONL validation dataset for periodic evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory to save LoRA adapter.",
    )
    parser.add_argument(
        "--result-log",
        type=str,
        default=None,
        help="Path to save JSON training result log.",
    )

    args = parser.parse_args()

    if args.output_dir is None:
        storage_root = os.environ.get("LLM_STORAGE_ROOT", "../Viemmo-1B-storage")
        if storage_root.startswith("/") and len(storage_root) > 2 and storage_root[2] == "/":
            storage_root = f"{storage_root[1].upper()}:{storage_root[2:]}"

        # Auto-derive adapter directory name from config filename
        config_stem = Path(args.config).stem  # e.g. "pilot_sft_qwen15b" or "pilot_sft"
        if "qwen15b" in config_stem:
            adapter_name = "pilot-qwen15b-lora-v1"
        elif "qwen3b" in config_stem:
            adapter_name = "pilot-qwen3b-lora-v1"
        else:
            adapter_name = "pilot-vietnamese-lora-v1"

        output_dir = Path(storage_root) / "adapters" / adapter_name
    else:
        output_dir = Path(args.output_dir)

    print(f"==================================================")
    print(f"       Viemmo-1B QLoRA Training Executable       ")
    print(f"==================================================")
    print(f"Config File:  {args.config}")
    print(f"Train Data:   {args.dataset}")
    print(f"Val Data:     {args.val_dataset}")
    print(f"Output Dir:   {output_dir}")
    print(f"Result Log:   {args.result_log}")
    print(f"==================================================\n")

    summary = train_qlora_model(
        config_path=args.config,
        dataset_path=args.dataset,
        val_dataset_path=args.val_dataset,
        output_adapter_dir=output_dir,
        result_log_path=args.result_log,
    )

    print("\nTraining Metrics Summary:")
    print(f" - Total Time:             {summary['total_training_time_sec']} seconds")
    print(f" - Final Train Loss:       {summary['final_train_loss']:.4f}")
    print(f" - Best Eval Checkpoint:   {summary['best_model_checkpoint']}")
    print(f" - Best Eval Loss:         {summary['best_metric']}")
    print(f" - Response Token Accuracy: {summary['pre_training_token_accuracy_percent']}% -> {summary['post_training_token_accuracy_percent']}%")
    print(f" - Peak VRAM:              {summary['peak_vram_mb']} MB")
    print(f" - Output Adapter:         {summary['output_adapter_dir']}")

if __name__ == "__main__":
    main()
