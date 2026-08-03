# Variant C — Vietnamese Pilot LoRA Adapter

## Model Overview

| Field | Value |
|:---|:---|
| **Variant** | C — OLMo + pilot Vietnamese LoRA |
| **Base model** | `allenai/OLMo-2-0425-1B-Instruct` |
| **Base revision** | `48d788eca847d4d7548f375ad03d3c9312f6139e` |
| **Architecture** | `Olmo2ForCausalLM` |
| **Adapter method** | QLoRA (4-bit NF4) |
| **License** | Apache 2.0 (inherits from base model) |

## Adapter Configuration

| Parameter | Value |
|:---|:---|
| **LoRA rank (r)** | 8 |
| **LoRA alpha** | 16 |
| **LoRA dropout** | 0.05 |
| **Target modules** | `q_proj`, `v_proj` |
| **Quantization** | NF4 4-bit, double quantization |
| **Compute dtype** | float16 |

## Training Configuration

| Parameter | Value |
|:---|:---|
| **Sequence length** | 512 |
| **Micro batch size** | 1 |
| **Gradient accumulation** | 16 |
| **Gradient checkpointing** | Enabled |
| **Learning rate** | 1e-4 |
| **Warmup ratio** | 0.03 |
| **Scheduler** | Cosine |
| **Optimizer** | `paged_adamw_8bit` |
| **Precision** | fp16 |
| **Random seed** | 42 |

## Training Dataset

| Field | Value |
|:---|:---|
| **Dataset** | `data/training/tiny-sft-v1.jsonl` |
| **Format** | Multi-turn SFT (system / user / assistant) |
| **Language** | Vietnamese |
| **Source** | Manually authored + reviewed synthetic |
| **PII** | None (verified by pipeline) |

## Adapter Artifacts

| File | SHA-256 | Size |
|:---|:---|:---|
| `adapter_model.safetensors` | _See manifest_ | _< 50 MB_ |
| `adapter_config.json` | _See manifest_ | _< 1 KB_ |

> Full checksums are recorded in
> [`manifests/model-checksums/pilot-vietnamese-lora-v1.json`](../../manifests/model-checksums/pilot-vietnamese-lora-v1.json).

## Integrity Verification

The adapter was verified using `scripts/package_adapter.py`, which confirms:

1. Both `adapter_model.safetensors` and `adapter_config.json` exist and load successfully with the `peft` library.
2. SHA-256 checksums were computed and recorded.
3. Total adapter size remains below 50 MB.

## Intended Use

This adapter is intended for **academic research only**. It adapts the base OLMo 2 1B Instruct model to improve Vietnamese language capabilities including:

- Vietnamese factual accuracy
- Vietnamese fluency
- Instruction following in Vietnamese
- Appropriate uncertainty expression

## Limitations

- This is a **pilot** adapter trained on a small dataset. It is not intended for production use.
- The adapter has not yet undergone full privacy or security evaluation (Phases 7–8).
- Vietnamese performance improvements are expected to be modest given the small training set.
- The adapter inherits all limitations of the base OLMo 2 model.

## Ethical Considerations

- No personal, confidential, or private data was used in training.
- The training dataset was reviewed for PII contamination.
- Evaluation data was excluded from training data.
- The model is designed for offline-only deployment.

## Citation

```bibtex
@misc{viemmo2026,
  title   = {Viemmo: Privacy-First Vietnamese Open LLM Adaptation},
  author  = {Kuvox Research Team},
  year    = {2026},
  url     = {https://github.com/Kuvox-stud-cmc/Viemmo-1B}
}
```
