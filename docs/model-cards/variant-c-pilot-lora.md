# 📑 Model Card: Viemmo-1B Variant C (Pilot QLoRA Adapter)

## 📌 Model Summary

- **Model Name:** Viemmo-1B Variant C (Pilot SFT LoRA Adapter)
- **Base Model:** `D:/Viemmo/Viemmo-1B-storage/upstream/OLMo-2-0425-1B-Instruct`
- **Adapter Type:** PEFT QLoRA (4-bit NF4 Base + LoRA Float16)
- **Target Modules:** `['q_proj', 'v_proj']`
- **LoRA Hyperparameters:** Rank `r=8`, Alpha `lora_alpha=16`
- **Total Artifact Size:** `10.83 MB` (Lightweight distribution < 50 MB)

---

## 🔒 Cryptographic Integrity Manifest

All adapter weights are verified with SHA-256 hashes to guarantee immutability across evaluation and deployment phases.

| File Name | Size (MB) | SHA-256 Checksum |
|:---|:---:|:---|
| `adapter_config.json` | `0.001` | `0cc29809b04df5889cb6e3b4b6b61c481f7936683ab9329f52a69263d1d3e85b` |
| `adapter_model.safetensors` | `4.008` | `d8e06794bc48b288778a77c80114591399a1f3141bcc4bb16201f9ed7d43e364` |
| `chat_template.jinja` | `0.0` | `8856beb464c6964117a5b9dff8f160312537f174826305fef5607e4c632daee8` |
| `README.md` | `0.005` | `c500c06a33415461c6e6b1e0c0d7842cea2cb1f2edabb2e132017d2bd2af68a2` |
| `tokenizer.json` | `6.807` | `18e309ad7f9c60037eaf26aca401a96128187cad2bdfa01a508dca7526bf6300` |
| `tokenizer_config.json` | `0.0` | `b507163276b15ea8d0329c2866b31de9d399bac34e84290d58f6e2d92ed18bb0` |
| `training_summary.json` | `0.01` | `0fb66989348a4578ebf9b53a70afe6b2afef522a88520b90d3e54f814903c970` |

---

## 🛠️ Usage Instructions

To load and use Variant C in Python:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_path = "D:/Viemmo/Viemmo-1B-storage/upstream/OLMo-2-0425-1B-Instruct"
adapter_path = "../Viemmo-1B-storage/adapters/tiny-overfit-v1"

tokenizer = AutoTokenizer.from_pretrained(base_model_path)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto",
)

model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()
```
