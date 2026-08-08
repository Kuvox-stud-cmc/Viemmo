#!/usr/bin/env python3
"""
Viemmo-1B: Apple Silicon MLX Fine-Tuning Executable for Qwen2.5-14B-Instruct
Runs native QLoRA fine-tuning using Apple MLX framework on M-series chips.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Apple Silicon MLX QLoRA Training Executable for Viemmo-1B.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sft/pilot_sft_qwen14b_mlx.yaml",
        help="Path to MLX YAML configuration file.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="../Viemmo-1B-storage/datasets/pilot-sft-v1",
        help="Directory containing train.jsonl and validation.jsonl.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Output adapter directory path.",
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file '{config_path}' not found.")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    try:
        import mlx.core as mx
        import mlx_lm
    except ImportError:
        print("❌ MLX package not found! Please install MLX on macOS Apple Silicon:")
        print("   pip install mlx mlx-lm")
        sys.exit(1)

    model_id = cfg.get("model", {}).get("id", "Qwen/Qwen2.5-14B-Instruct")
    train_cfg = cfg.get("training", {})
    output_dir = Path(args.adapter_path or train_cfg.get("adapter_path", "../Viemmo-1B-storage/adapters/pilot-qwen14b-mlx-lora-v1"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("      Viemmo-1B Apple Silicon MLX Trainer       ")
    print("==================================================")
    print(f"Model ID:      {model_id}")
    print(f"Data Dir:      {args.data}")
    print(f"Output Dir:    {output_dir}")
    print(f"Iterations:    {train_cfg.get('iters', 600)}")
    print(f"Learning Rate: {train_cfg.get('learning_rate', 1e-4)}")
    print("==================================================\n")

    # Command construction for mlx_lm.lora CLI
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm.lora",
        "--model",
        model_id,
        "--data",
        args.data,
        "--train",
        "--batch-size",
        str(train_cfg.get("batch_size", 1)),
        "--iters",
        str(train_cfg.get("iters", 600)),
        "--learning-rate",
        str(train_cfg.get("learning_rate", 1e-4)),
        "--adapter-path",
        str(output_dir),
        "--save-every",
        str(train_cfg.get("save_every", 100)),
        "--steps-per-eval",
        str(train_cfg.get("steps-per-eval", 50)),
    ]

    import subprocess
    print("Executing MLX Training:")
    print(" ".join(cmd))
    print("\nStarting MLX QLoRA Training Loop...")

    start_time = time.perf_counter()
    res = subprocess.run(cmd)
    duration = time.perf_counter() - start_time

    if res.returncode == 0:
        print(f"\n✅ MLX Training Complete in {duration:.2f} seconds!")
        print(f"Saved 14B MLX Adapter to: {output_dir}")
    else:
        print(f"\n❌ MLX Training exited with code {res.returncode}")


if __name__ == "__main__":
    main()
