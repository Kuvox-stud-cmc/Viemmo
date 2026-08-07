import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from viemmo.training.trainer import train_qlora_model

def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Configuration-driven QLoRA training for Viemmo-1B.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sft/tiny_overfit.yaml",
        help="Path to YAML training configuration file.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/training/tiny-sft-v1.jsonl",
        help="Path to JSONL training dataset.",
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
        default="results/training/tiny-overfit-run.json",
        help="Path to save JSON training result log.",
    )

    args = parser.parse_args()

    if args.output_dir is None:
        storage_root = os.environ.get("LLM_STORAGE_ROOT", "../Viemmo-1B-storage")
        if storage_root.startswith("/") and len(storage_root) > 2 and storage_root[2] == "/":
            storage_root = f"{storage_root[1].upper()}:{storage_root[2:]}"
        output_dir = Path(storage_root) / "adapters" / "tiny-overfit-v1"
    else:
        output_dir = Path(args.output_dir)

    print(f"==================================================")
    print(f"       Viemmo-1B QLoRA Training Executable       ")
    print(f"==================================================")
    print(f"Config File: {args.config}")
    print(f"Dataset:     {args.dataset}")
    print(f"Output Dir:  {output_dir}")
    print(f"Result Log:  {args.result_log}")
    print(f"==================================================\n")

    summary = train_qlora_model(
        config_path=args.config,
        dataset_path=args.dataset,
        output_adapter_dir=output_dir,
        result_log_path=args.result_log,
    )

    print("\nTraining Metrics Summary:")
    print(f" - Total Time:             {summary['total_training_time_sec']} seconds")
    print(f" - Final Loss:             {summary['final_train_loss']:.4f}")
    print(f" - Response Token Accuracy: {summary['pre_training_token_accuracy_percent']}% -> {summary['post_training_token_accuracy_percent']}%")
    print(f" - Peak VRAM:              {summary['peak_vram_mb']} MB")
    print(f" - Output Adapter:         {summary['output_adapter_dir']}")
    print(f" - Result Log:             {args.result_log}")

if __name__ == "__main__":
    main()
