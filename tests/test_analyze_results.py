import json
import pytest
from pathlib import Path
from scripts.analyze_results import (
    calculate_stats,
    calculate_cohens_d,
    calculate_win_tie_loss,
    analyze_results,
)

def test_statistical_functions():
    vals = [2.0, 3.0, 2.5, 2.8, 3.0]
    stats = calculate_stats(vals)
    assert stats["mean"] == round(sum(vals)/len(vals), 3)
    assert stats["std"] > 0.0
    assert stats["ci95_lower"] <= stats["mean"] <= stats["ci95_upper"]

    g1 = [2.0, 2.0, 2.1, 2.0]
    g2 = [3.0, 3.0, 2.9, 3.0]
    d = calculate_cohens_d(g1, g2)
    assert d > 1.0 # High positive effect size

    wtl = calculate_win_tie_loss([2.0, 2.0, 3.0], [3.0, 2.0, 2.0])
    assert wtl["variant_c_wins"] == 1
    assert wtl["ties"] == 1
    assert wtl["variant_c_losses"] == 1

def test_analyze_results_pipeline(tmp_path):
    # Mock variant summaries
    sum_a = tmp_path / "variant_a_summary.json"
    sum_b = tmp_path / "variant_b_summary.json"
    sum_c = tmp_path / "variant_c_summary.json"

    data = {"average_tokens_per_sec": 12.0, "average_prompt_latency_sec": 8.0, "peak_vram_mb": 1400.0, "hit_max_tokens_count": 0}
    sum_a.write_text(json.dumps(data), encoding="utf-8")
    sum_b.write_text(json.dumps(data), encoding="utf-8")
    sum_c.write_text(json.dumps(data), encoding="utf-8")

    out_dir = tmp_path / "reports"
    res = analyze_results(
        variant_summaries={"A": sum_a, "B": sum_b, "C": sum_c},
        output_dir=out_dir,
    )

    assert (out_dir / "variant_comparison_report.json").exists()
    assert (out_dir / "figures" / "variant_comparison_radar.svg").exists() or (out_dir / "figures" / "variant_comparison_radar.png").exists()
    assert (out_dir / "figures" / "throughput_memory_comparison.svg").exists() or (out_dir / "figures" / "throughput_memory_comparison.png").exists()
    assert "outcome_verdict" in res
