import json
import pytest
from pathlib import Path
from scripts.register_adapter_manifest import register_adapter_manifest

def test_register_adapter_manifest(tmp_path):
    adapter_dir = tmp_path / "test_adapter"
    adapter_dir.mkdir()

    # Create dummy config and safetensors
    cfg_file = adapter_dir / "adapter_config.json"
    cfg_content = {
        "base_model_name_or_path": "allenai/OLMo-2-0425-1B-Instruct",
        "peft_type": "LORA",
        "r": 8,
        "lora_alpha": 16,
        "target_modules": ["q_proj", "v_proj"],
    }
    cfg_file.write_text(json.dumps(cfg_content), encoding="utf-8")

    weights_file = adapter_dir / "adapter_model.safetensors"
    weights_file.write_bytes(b"0" * 1024 * 1024) # 1 MB dummy file

    manifest_path = tmp_path / "manifest.json"
    model_card_path = tmp_path / "model_card.md"

    manifest_data = register_adapter_manifest(
        adapter_dir=adapter_dir,
        manifest_path=manifest_path,
        model_card_path=model_card_path,
    )

    assert manifest_path.exists()
    assert model_card_path.exists()
    assert manifest_data["is_lightweight_under_50mb"] is True
    assert manifest_data["total_size_mb"] > 0
    assert "adapter_config.json" in manifest_data["files"]
    assert "adapter_model.safetensors" in manifest_data["files"]

    card_text = model_card_path.read_text(encoding="utf-8")
    assert "allenai/OLMo-2-0425-1B-Instruct" in card_text
    assert "adapter_model.safetensors" in card_text
