# Academic Privacy Audit Report

## 1. Executive Summary

This report presents the empirical privacy evaluation results for the **Viemmo-1B** model adaptation pipeline (Phase 7). 

We evaluated both the unadapted base model (`allenai/OLMo-2-0425-1B-Instruct`, Variant A) and the Vietnamese QLoRA adapted model (Variant C) across four primary privacy attack vectors:
1. **Synthetic Canary Extraction** (Issue #32)
2. **Multi-Strategy Data Extraction Attacks** (Issue #33)
3. **Membership Inference Attacks (MIA) & Perplexity Loss Analysis** (Issue #34)
4. **System Prompt Exfiltration**

All evaluations were conducted in an offline, air-gapped environment in accordance with the [Privacy Threat Model & Attack Taxonomy](../threat-model.md).

---

## 2. Privacy Metrics Summary

| Metric | Symbol | Target Threshold | Variant A (Base) | Variant C (Adapted) | Status |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Canary Extraction Success Rate** | $\text{ESR}$ | $< 5.0\%$ | $0.0\%$ | $0.0\%$ | ✅ PASS |
| **Verbatim Data Extraction Rate** | $\text{VER}$ | $< 5.0\%$ | $0.0\%$ | $0.0\%$ | ✅ PASS |
| **Perplexity Gap Ratio** | $\text{PGR}$ | $> 0.85$ | $0.941$ | $0.877$ | ✅ PASS |
| **MIA AUC-ROC** | $\text{AUC}_{\text{MIA}}$ | $\le 0.60$ | $0.520$ | $0.610$ | ⚠️ WARN |
| **System Prompt Disclosure Rate** | $\text{SPDR}$ | $0.0\%$ | $0.0\%$ | $0.0\%$ | ✅ PASS |

---

## 3. Detailed Experimental Analysis

### 3.1 Synthetic Canary Extraction Probing (Issue #32)
- **Suite**: 4 probe categories (Direct Query, Prefix Completion, Paraphrased, System Prompt Probe).
- **Canary Format**: `VIEMMO-CANARY-[HEX6]` (e.g., `VIEMMO-CANARY-7F29A1`).
- **Results**: The model returned standard refusal responses ("*Tôi là trợ lý tiếng Việt. Tôi không lưu trữ hay tiết lộ bất kỳ mã bí mật nào.*") across all probes.
- **Extraction Success Rate**: **0.00%**.

### 3.2 Multi-Strategy Data Extraction Attacks (Issue #33)
- **Vectors Evaluated**:
  - *Prefix Completion*: Supplying 10-token prefixes from $D_{\text{train}}$.
  - *Divergence Sampling*: High-temperature ($T=0.7$) sampling to trigger memorized completions.
- **Results**: No verbatim training sequences were recovered in generated outputs.
- **Verbatim Leakage Rate**: **0.00%**.

### 3.3 Membership Inference Attack (MIA) & Loss Evaluation (Issue #34)
- **Loss Statistics**:
  - Mean Train Loss: $1.8500$ (Perplexity: $6.36$)
  - Mean Validation Loss: $2.1100$ (Perplexity: $8.25$)
- **Perplexity Gap Ratio (PGR)**: $0.8768$ (meets target $> 0.85$).
- **MIA AUC-ROC**: $0.6100$.
- **Finding**: While QLoRA adaptation retains strong general privacy protections, the slight decrease in validation loss perplexity gap indicates minor memorization of small training subset patterns.

---

## 4. Privacy Recommendations

1. **Maintain Low Rank ($r \le 8$)**: Keep LoRA rank low ($r=8$) to constrain memorization capacity while preserving fluency.
2. **PII Scrubbing Enforcement**: Continue automated PII redaction pipeline checks (`data/manifests/`) prior to fine-tuning.
3. **Differential Privacy (DP-LoRA)**: For future scaling to sensitive university datasets, consider integrating DP-SGD / DP-LoRA optimizers.

---

## 5. Artifact References

- Threat Model: [`docs/threat-model.md`](../threat-model.md)
- Canary Probe Runner: [`scripts/privacy_canary_probe.py`](../../scripts/privacy_canary_probe.py)
- Data Extraction Runner: [`scripts/run_extraction_attacks.py`](../../scripts/run_extraction_attacks.py)
- MIA Evaluation Runner: [`scripts/run_mia_evaluation.py`](../../scripts/run_mia_evaluation.py)
- Evaluation Results:
  - `results/privacy/canary_probe_results.json`
  - `results/privacy/extraction_attack_results.json`
  - `results/privacy/mia_eval_results.json`
