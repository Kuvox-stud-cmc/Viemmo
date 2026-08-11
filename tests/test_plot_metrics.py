import json
import pytest
from pathlib import Path
from scripts.plot_training_metrics import generate_metrics_summary_and_plot

def test_generate_metrics_summary_and_plot(tmp_path):
    run_log = tmp_path / "run_log.json"
    data = {
        "model_id": "allenai/OLMo-2-0425-1B-Instruct",
        "seed": 42,
        "total_training_time_sec": 120.0,
        "peak_vram_mb": 2800.0,
        "pre_training_token_accuracy_percent": 50.0,
        "post_training_token_accuracy_percent": 95.0,
        "final_train_loss": 0.5,
        "environment": {"gpu_name": "Test GPU", "cuda_version": "12.1", "pytorch_version": "2.5.1"},
        "loss_history": [
            {"step": 1, "loss": 2.0, "step_duration_sec": 10.0},
            {"step": 2, "loss": 1.0, "eval_loss": 1.2, "step_duration_sec": 10.0},
        ],
    }
    run_log.write_text(json.dumps(data), encoding="utf-8")

    metrics_json = tmp_path / "metrics.json"
    plot_image = tmp_path / "loss_curve.png"
    report_md = tmp_path / "report.md"

    generate_metrics_summary_and_plot(
        run_log_path=run_log,
        metrics_json_path=metrics_json,
        plot_image_path=plot_image,
        report_md_path=report_md,
    )

    assert metrics_json.exists()
    assert (plot_image.exists() or plot_image.with_suffix(".svg").exists())
    assert report_md.exists()

    summary_data = json.loads(metrics_json.read_text(encoding="utf-8"))
    assert summary_data["average_step_latency_sec"] == 10.0
    assert summary_data["peak_vram_mb"] == 2800.0

    report_text = report_md.read_text(encoding="utf-8")
    assert "Test GPU" in report_text
    assert "2800.0 MB" in report_text
