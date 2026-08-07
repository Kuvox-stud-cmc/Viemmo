import pytest
from pathlib import Path
from scripts.verify_adapter import verify_base_model_checksums

def test_verify_base_model_checksums(tmp_path):
    # Create fake model dir and files
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    
    file1 = model_dir / "config.json"
    file1.write_text('{"model": "test"}', encoding="utf-8")

    # Hash of file1 content
    import hashlib
    hash1 = hashlib.sha256(b'{"model": "test"}').hexdigest()

    checksum_file = tmp_path / "checksums.json"
    checksum_file.write_text(f'{{"config.json": "{hash1}"}}', encoding="utf-8")

    assert verify_base_model_checksums(model_dir, checksum_file) is True

def test_verify_base_model_checksums_detects_tampering(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    
    file1 = model_dir / "config.json"
    file1.write_text('{"model": "tampered"}', encoding="utf-8")

    checksum_file = tmp_path / "checksums.json"
    checksum_file.write_text('{"config.json": "wronghash123"}', encoding="utf-8")

    assert verify_base_model_checksums(model_dir, checksum_file) is False
