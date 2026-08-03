import json
import platform
from pathlib import Path

import torch
import transformers
import peft
import datasets

def main() -> None:
  if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable. Check WSL2 and the NVIDIA driver.")

  device = torch.cuda.get_device_properties(0)

  report = {
    "platform": platform.platform(),
    "python": platform.python_version(),
    "pytorch": torch.__version__,
    "transformers": transformers.__version__,
    "peft": peft.__version__,
    "datasets": datasets.__version__,
    "cuda_runtime": torch.version.cuda,
    "gpu": device.name,
    "compute_capability": torch.cuda.get_device_capability(0),
    "vram_bytes": device.total_memory
  }

  output = Path("manifests/environment/pilot-environment.json")
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(
    json.dumps(report, indent=2, ensure_ascii=False),
    encoding="utf-8"
  )

  print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == "__main__":
  main()
