import json
import pytest
from pathlib import Path
from scripts.evaluate_variants import evaluate_variant, compute_sha256, get_git_commit_hash

def test_comparison_matrix_yaml_exists():
    cfg_path = Path("configs/evaluation/comparison_matrix.yaml")
    assert cfg_path.exists(), "configs/evaluation/comparison_matrix.yaml missing!"

def test_compute_sha256_and_git_hash(tmp_path):
    dummy_file = tmp_path / "test.txt"
    dummy_file.write_text("hello world", encoding="utf-8")

    hash1 = compute_sha256(dummy_file)
    assert len(hash1) == 64
    assert isinstance(hash1, str)

    git_hash = get_git_commit_hash()
    assert isinstance(git_hash, str)
    assert len(git_hash) > 0
