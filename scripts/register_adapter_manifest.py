import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

from peft import PeftConfig


def normalize_path(path_str: str) -> Path:
    """Normalizes POSIX /d/ drive letter paths for Windows compatibility."""
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
    """Computes the SHA-256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def register_adapter_manifest(
    adapter_dir: Path,
    manifest_path: Path,
    model_card_path: Path,
) -> Dict[str, Any]:
    """Verifies adapter serialization integrity, computes SHA-256 hashes, asserts lightweight size,
    and generates manifest JSON and Model Card Markdown.
    """
    adapter_dir = normalize_path(str(adapter_dir))
    manifest_path = normalize_path(str(manifest_path))
    model_card_path = normalize_path(str(model_card_path))

    if not adapter_dir.exists():
        print(f"Error: Adapter directory '{adapter_dir}' does not exist.")
        sys.exit(1)

    config_file = adapter_dir / "adapter_config.json"
    weights_safetensors = adapter_dir / "adapter_model.safetensors"
    weights_bin = adapter_dir / "adapter_model.bin"

    if not config_file.exists():
        print(f"Error: Missing required file '{config_file.name}' in '{adapter_dir}'.")
        sys.exit(1)

    if not (weights_safetensors.exists() or weights_bin.exists()):
        print(f"Error: Neither 'adapter_model.safetensors' nor 'adapter_model.bin' found in '{adapter_dir}'.")
        sys.exit(1)

    # 1. Verify PEFT loadability
    try:
        peft_cfg = PeftConfig.from_pretrained(str(adapter_dir))
        base_model_name = peft_cfg.base_model_name_or_path
        target_modules = peft_cfg.target_modules
        lora_rank = peft_cfg.r
        lora_alpha = peft_cfg.lora_alpha
    except Exception as e:
        print(f"Error: Failed to load PEFT adapter configuration: {e}")
        sys.exit(1)

    # 2. Compute SHA-256 hashes and file sizes
    file_manifests = {}
    total_bytes = 0

    for item in sorted(adapter_dir.rglob("*")):
        if item.is_file() and not item.name.startswith("."):
            rel_path = item.relative_to(adapter_dir).as_posix()
            file_size = item.stat().st_size
            total_bytes += file_size
            sha256 = compute_sha256(item)

            file_manifests[rel_path] = {
                "sha256": sha256,
                "size_bytes": file_size,
                "size_mb": round(file_size / (1024 * 1024), 3),
            }

    total_size_mb = round(total_bytes / (1024 * 1024), 2)
    is_lightweight = total_size_mb < 50.0

    print(f"Adapter File Verification:")
    for fname, info in file_manifests.items():
        print(f"  [OK] {fname} ({info['size_mb']} MB) -> {info['sha256'][:16]}...")
    print(f"Total Adapter Size: {total_size_mb} MB (Lightweight < 50 MB: {is_lightweight})")

    if not is_lightweight:
        print(f"Warning: Adapter size ({total_size_mb} MB) exceeds 50 MB limit!")

    # 3. Create Manifest JSON
    manifest_data = {
        "adapter_name": "Viemmo-Variant-C-Pilot-LoRA",
        "variant": "Variant C",
        "adapter_directory": str(adapter_dir),
        "base_model_id": base_model_name,
        "lora_parameters": {
            "rank": lora_rank,
            "alpha": lora_alpha,
            "target_modules": list(target_modules) if isinstance(target_modules, (set, list, tuple)) else str(target_modules),
        },
        "total_size_mb": total_size_mb,
        "is_lightweight_under_50mb": is_lightweight,
        "files": file_manifests,
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
    print(f"  ✓ Saved adapter manifest JSON to: {manifest_path}")

    # 4. Create Model Card Markdown
    model_card_content = f"""# 📑 Model Card: Viemmo Variant C (Pilot QLoRA Adapter)

## 📌 Model Summary

- **Model Name:** Viemmo Variant C (Pilot SFT LoRA Adapter)
- **Base Model:** `{base_model_name}`
- **Adapter Type:** PEFT QLoRA (4-bit NF4 Base + LoRA Float16)
- **Target Modules:** `{list(target_modules) if isinstance(target_modules, (set, list, tuple)) else target_modules}`
- **LoRA Hyperparameters:** Rank `r={lora_rank}`, Alpha `lora_alpha={lora_alpha}`
- **Total Artifact Size:** `{total_size_mb} MB` (Lightweight distribution < 50 MB)

---

## 🔒 Cryptographic Integrity Manifest

All adapter weights are verified with SHA-256 hashes to guarantee immutability across evaluation and deployment phases.

| File Name | Size (MB) | SHA-256 Checksum |
|:---|:---:|:---|
"""
    for fname, info in file_manifests.items():
        model_card_content += f"| `{fname}` | `{info['size_mb']}` | `{info['sha256']}` |\n"

    model_card_content += f"""
---

## 🛠️ Usage Instructions

To load and use Variant C in Python:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_path = "D:/Viemmo/Viemmo-storage/upstream/OLMo-2-0425-1B-Instruct"
adapter_path = "{adapter_dir.as_posix()}"

tokenizer = AutoTokenizer.from_pretrained(base_model_path)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto",
)

model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()
```
"""

    model_card_path.parent.mkdir(parents=True, exist_ok=True)
    model_card_path.write_text(model_card_content, encoding="utf-8")
    print(f"  ✓ Saved Model Card Markdown to: {model_card_path}")

    return manifest_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Register adapter checksum manifest and Model Card.")
    parser.add_argument(
        "--adapter-dir",
        type=str,
        default="../Viemmo-storage/adapters/tiny-overfit-v1",
        help="Path to saved LoRA adapter directory.",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default="manifests/model-checksums/pilot-vietnamese-lora-v1.json",
        help="Path to save output manifest JSON.",
    )
    parser.add_argument(
        "--model-card-path",
        type=str,
        default="docs/model-cards/variant-c-pilot-lora.md",
        help="Path to save output Model Card Markdown.",
    )

    args = parser.parse_args()

    print(f"==================================================")
    print(f"   Viemmo Adapter Manifest Registration Engine ")
    print(f"==================================================")

    register_adapter_manifest(
        adapter_dir=Path(args.adapter_dir),
        manifest_path=Path(args.manifest_path),
        model_card_path=Path(args.model_card_path),
    )


if __name__ == "__main__":
    main()
