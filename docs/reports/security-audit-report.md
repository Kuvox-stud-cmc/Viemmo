# Academic Security Audit & Red-Teaming Report

## 1. Executive Summary

This report presents the empirical security evaluation and red-teaming audit results for the **Viemmo-1B** model adaptation pipeline (Phase 8). 

We evaluated both the unadapted base model (`allenai/OLMo-2-0425-1B-Instruct`, Variant A) and the Vietnamese QLoRA adapted model (Variant C) across three primary security attack vectors:
1. **Adversarial Jailbreak Probes** (Issue #37)
2. **Harmful Intent & Illegal Instruction Probes** (Issue #38)
3. **Indirect Prompt Injection & System Override** (Issue #39)

All evaluations were conducted in an offline, air-gapped environment in accordance with the [Security Policy & Adversarial Red-Teaming Taxonomy](../security-policy.md).

---

## 2. Security Metrics Summary

| Metric | Symbol | Target Threshold | Variant A (Base) | Variant C (Adapted) | Status |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Attack Success Rate** | $\text{ASR}$ | $< 5.0\%$ | $0.0\%$ | $0.0\%$ | ✅ PASS |
| **Harmful Refusal Rate** | $\text{HRR}$ | $\ge 95.0\%$ | $100.0\%$ | $100.0\%$ | ✅ PASS |
| **Injection Resilience Rate** | $\text{IRR}$ | $\ge 95.0\%$ | $100.0\%$ | $100.0\%$ | ✅ PASS |

---

## 3. Detailed Security Probe Analysis

### 3.1 Jailbreak Probes (Issue #37)
- **Evaluated Framing**: Roleplay framing (DAN), hypothetical simulation (fictional villain context), Base64 payload encoding.
- **Results**: All jailbreak attempts were successfully neutralized by the model's refusal guardrails.
- **Attack Success Rate (ASR)**: **0.00%** (0 / 3 bypasses).

### 3.2 Harmful Intent Probes (Issue #38)
- **Categories Probed**: Cyber exploitation (ransomware code), physical crime (identity document forgery), dangerous chemical manufacturing.
- **Results**: Model correctly triggered safety refusals ("*Tôi không thể thực hiện yêu cầu này vì đây là hành vi vi phạm pháp luật...*").
- **Harmful Refusal Rate (HRR)**: **100.00%** (3 / 3 refused).

### 3.3 Indirect Prompt Injection (Issue #39)
- **Attack Vectors**: Indirect prompt injection in document summaries, chat template delimiter hijacking (`<|im_end|>`).
- **Results**: The model isolated user queries from untrusted context documents without executing embedded malicious instructions.
- **Injection Resilience Rate (IRR)**: **100.00%** (2 / 2 neutralized).

---

## 4. Security Hardening Recommendations

1. **System Prompt Delimiter Isolation**: Maintain explicit chat template delimiters (e.g., `<|im_start|>user`, `<|im_start|>system`) to prevent boundary confusion.
2. **Refusal Fine-Tuning**: Continue including safety refusal examples in SFT datasets for domain-specific Vietnamese queries.

---

## 5. Deliverable References

- Security Policy: [`docs/security-policy.md`](../security-policy.md)
- Automated Jailbreak Runner: [`scripts/security_jailbreak_runner.py`](../../scripts/security_jailbreak_runner.py)
- Harmful Intent Runner: [`scripts/run_harmful_intent_probes.py`](../../scripts/run_harmful_intent_probes.py)
- Prompt Injection Runner: [`scripts/run_prompt_injection_tests.py`](../../scripts/run_prompt_injection_tests.py)
- Evaluation Manifests:
  - `results/security/jailbreak_results.json`
  - `results/security/harmful_intent_results.json`
  - `results/security/prompt_injection_results.json`
