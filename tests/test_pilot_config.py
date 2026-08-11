import yaml
from pathlib import Path

def test_pilot_sft_yaml_structure_and_criteria():
    config_path = Path("configs/sft/pilot_sft.yaml")
    assert config_path.exists(), "configs/sft/pilot_sft.yaml does not exist!"

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 1. Model ID & Revision
    assert cfg["model"]["id"] == "allenai/OLMo-2-0425-1B-Instruct"
    assert cfg["model"]["revision"] == "48d788eca847d4d7548f375ad03d3c9312f6139e"

    # 2. Quantization Settings
    assert cfg["quantization"]["load_in_4bit"] is True
    assert cfg["quantization"]["quant_type"] == "nf4"
    assert cfg["quantization"]["double_quantization"] is True

    # 3. LoRA Targets & Hyperparameters
    assert cfg["lora"]["rank"] == 8
    assert cfg["lora"]["alpha"] == 16
    assert cfg["lora"]["dropout"] == 0.05
    assert cfg["lora"]["target_modules"] == ["q_proj", "v_proj"]

    # 4. Training Settings
    t_cfg = cfg["training"]
    assert t_cfg["sequence_length"] == 512
    assert t_cfg["micro_batch_size"] == 1
    assert t_cfg["gradient_accumulation_steps"] == 16
    assert t_cfg["gradient_checkpointing"] is True
    assert t_cfg["learning_rate"] == 0.0001
    assert t_cfg["warmup_ratio"] == 0.03
    assert t_cfg["scheduler"] == "cosine"
    assert t_cfg["optimizer"] == "paged_adamw_8bit"

    # Evaluation, Checkpointing & Early Stopping Settings
    assert t_cfg["eval_steps"] == 50
    assert t_cfg["save_steps"] == 100
    assert t_cfg["save_total_limit"] == 3
    assert t_cfg["early_stopping_patience"] == 3

    effective_batch_size = t_cfg["micro_batch_size"] * t_cfg["gradient_accumulation_steps"]
    assert effective_batch_size == 16
