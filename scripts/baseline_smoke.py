import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv

import torch
from transformers import (
  AutoModelForCausalLM,
  AutoTokenizer,
  BitsAndBytesConfig
)

def main() -> None:
  load_dotenv()
  model_path = os.environ["MODEL_PATH"]
  model_id = os.environ["MODEL_ID"]

  if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable. Check WSL2 and the NVIDIA driver.") 

  quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
  )

  tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True,
    trust_remote_code=False
  )

  model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    trust_remote_code=False,
    quantization_config=quantization_config,
    device_map={"": 0},
    dtype=torch.bfloat16,
    attn_implementation="eager"
  )

  model.eval()

  print(f"Architecture: {model.config.architectures}")
  print(f"Model type: {model.config.model_type}")
  print(f"GPU: {torch.cuda.get_device_name(0)}") 

  messages = [
    {
      "role": "system",
      "content": (
        "Bạn là một trợ lý tiếng Việt chính xác và thận trọng. "
        "Nếu không đủ thông tin, hãy nói rõ điều đó. "
      ),
    },
    {
      "role": "user",
      "content": (
        "Hãy giải thích ngắn gọn sự khác nhau giữa mã hoá và băm. "
        "Đưa ra một ví dụ thực tế cho mỗi khái niệm. "
      ),
    },
  ]

  inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True
  )

  inputs = {
    name: tensor.to(model.device)
    for name, tensor in inputs.items()
  }

  torch.cuda.reset_peak_memory_stats()
  started_at = time.perf_counter()

  with torch.inference_mode():
    output = model.generate(
      **inputs,
      max_new_tokens=160,
      do_sample=False,
      repetition_penalty=1.05,
      eos_token_id=tokenizer.eos_token_id,
      pad_token_id=tokenizer.pad_token_id,
    )

  elapsed_seconds = time.perf_counter() - started_at
  input_length = inputs["input_ids"].shape[-1]
  generated_tokens = output[0, input_length:]
  answer = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True,
  ).strip()

  result = {
    "model": model_id,
    "model_path": model_path,
    "model_type": model.config.model_type,
    "quantization": "NF4 4-bit",
    "prompt": messages,
    "answer": answer,
    "input_tokens": input_length,
    "generated_tokens": generated_tokens.shape[-1],
    "elapsed_seconds": round(elapsed_seconds, 3),
    "peak_vram_mib": round(
      torch.cuda.max_memory_allocated() / 1024**2,
      2,
    ),
  }

  output_path = Path(
    "results/baseline/olmo-2-1b-instruct-smoke.json"
  )
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(
    json.dumps(result, ensure_ascii=False, indent=2),
    encoding="utf-8",
  )

  print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
  main()
