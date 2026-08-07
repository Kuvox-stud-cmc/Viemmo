# 📈 Viemmo-1B QLoRA Training & Performance Report

This report documents the performance metrics, loss curves, latency profile, and VRAM memory allocation across the training lifecycle.

## 📊 Performance Overview

| Metric | Measured Value |
|:---|:---|
| **Model ID** | `allenai/OLMo-2-0425-1B-Instruct` |
| **GPU Hardware** | `NVIDIA GeForce GTX 1650 Ti` |
| **CUDA / PyTorch** | CUDA `12.1` / PyTorch `2.5.1+cu121` |
| **Total Training Time** | `1475.18 seconds` |
| **Average Step Latency** | `18.4 seconds/step` |
| **Peak VRAM Allocation** | `2812.56 MB` (< 4.0 GB Limit) |
| **Final Training Loss** | `0.6417397235985846` |
| **Pre-Training Token Accuracy** | `53.28%` |
| **Post-Training Token Accuracy** | `99.96%` |

---

## 📉 Loss Curve

![Training vs Validation Loss Curve](results/training/loss_curve.svg)

*Figure 1: Training and Validation Loss trajectory across training steps.*

---

## 💾 Resource & Memory Profile

- **Base Model Loading:** NF4 4-bit Quantization with Double Quantization.
- **Optimizer:** `paged_adamw_8bit` (75% memory reduction with CUDA Paging protection).
- **Activation Memory:** Gradient Checkpointing enabled with sequence length 512.
- **Hardware Margin:** Peak memory usage of **2812.56 MB** provided a **~1.2 GB safety margin** on 4 GB GPU hardware.
