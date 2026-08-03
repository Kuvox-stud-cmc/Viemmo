import hashlib
from pathlib import Path
import json
import dotenv
import os

def compute_sha256(file_path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):  # Read in 1MB chunks
            sha256.update(chunk)
    return sha256.hexdigest()


def main() -> None:
    dotenv.load_dotenv()
    model_path_str = os.environ["MODEL_PATH"]
    
    # Handle Git Bash / MSYS2 paths like /d/Viemmo/... -> D:/Viemmo/...
    if len(model_path_str) >= 3 and model_path_str[0] == "/" and model_path_str[2] == "/":
        model_path_str = f"{model_path_str[1]}:{model_path_str[2:]}"
        
    model_dir = Path(model_path_str)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    checksums = {}
    print(f"Recording checksums for model files in: {model_dir}")

    for file in sorted(model_dir.iterdir()):
        if file.is_file():
            print(f"  Hashing {file.name}...")
            checksums[file.name] = compute_sha256(file)

    out_path = Path("manifests/model-checksums/olmo-2-1b-instruct.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(checksums, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nChecksums successfully written to {out_path}")


if __name__ == "__main__":
    main()
