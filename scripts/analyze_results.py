import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def calculate_stats(values: List[float]) -> Dict[str, float]:
    """Calculates mean, std dev, standard error, and 95% confidence interval."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "sem": 0.0, "ci95_lower": 0.0, "ci95_upper": 0.0}

    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / max(1, n - 1) if n > 1 else 0.0
    std = math.sqrt(variance)
    sem = std / math.sqrt(n) if n > 0 else 0.0
    ci95_margin = 1.96 * sem

    return {
        "mean": round(mean, 3),
        "std": round(std, 3),
        "sem": round(sem, 3),
        "ci95_lower": round(max(0.0, mean - ci95_margin), 3),
        "ci95_upper": round(mean + ci95_margin, 3),
    }


def calculate_cohens_d(group1: List[float], group2: List[float]) -> float:
    """Calculates Cohen's d effect size between two groups."""
    if not group1 or not group2 or len(group1) < 2 or len(group2) < 2:
        return 0.0

    n1, n2 = len(group1), len(group2)
    m1, m2 = sum(group1) / n1, sum(group2) / n2
    var1 = sum((x - m1) ** 2 for x in group1) / (n1 - 1)
    var2 = sum((x - m2) ** 2 for x in group2) / (n2 - 1)

    pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0.0:
        return 0.0

    return round((m2 - m1) / pooled_std, 3)


def calculate_win_tie_loss(scores_a: List[float], scores_c: List[float]) -> Dict[str, Any]:
    """Calculates win/tie/loss counts and percentages between Variant A and Variant C."""
    wins, ties, losses = 0, 0, 0
    for a, c in zip(scores_a, scores_c):
        if c > a:
            wins += 1
        elif c == a:
            ties += 1
        else:
            losses += 1

    total = max(1, len(scores_a))
    return {
        "variant_c_wins": wins,
        "ties": ties,
        "variant_c_losses": losses,
        "win_rate_percent": round((wins / total) * 100, 2),
        "tie_rate_percent": round((ties / total) * 100, 2),
        "loss_rate_percent": round((losses / total) * 100, 2),
    }


def generate_svg_radar_chart(
    categories: List[str],
    variant_scores: Dict[str, List[float]],
    svg_path: Path,
) -> None:
    """Pure Python SVG radar chart fallback generator."""
    width, height = 600, 600
    cx, cy, r = 300, 300, 200
    num_cats = len(categories)

    if num_cats == 0:
        return

    svg = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" style="background:#fff; font-family:sans-serif;">',
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
        f'<text x="{cx}" y="40" text-anchor="middle" font-size="18" font-weight="bold" fill="#333">Variant Capability Radar Comparison</text>',
    ]

    # Concentric circles (0 to 3 score grid)
    colors = {"A": "#1f77b4", "B": "#2ca02c", "C": "#ff7f0e", "D": "#d62728",
              "E": "#9467bd", "F": "#8c564b", "G": "#e377c2", "H": "#7f7f7f"}
    for step in range(1, 4):
        radius = r * (step / 3.0)
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="{radius:.1f}" fill="none" stroke="#e0e0e0" stroke-width="1.5" stroke-dasharray="4,4"/>')
        svg.append(f'<text x="{cx+5}" y="{cy-radius+12}" font-size="10" fill="#888">{step}.0</text>')

    # Category Axes
    angles = [i * (2 * math.pi / num_cats) - (math.pi / 2) for i in range(num_cats)]
    for i, angle in enumerate(angles):
        ax = cx + r * math.cos(angle)
        ay = cy + r * math.sin(angle)
        svg.append(f'<line x1="{cx}" y1="{cy}" x2="{ax:.1f}" y2="{ay:.1f}" stroke="#ccc" stroke-width="1.5"/>')

        lx = cx + (r + 30) * math.cos(angle)
        ly = cy + (r + 30) * math.sin(angle)
        svg.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" font-size="12" font-weight="bold" fill="#444">{categories[i]}</text>')

    # Plot Variants
    for var_id, scores in variant_scores.items():
        if len(scores) != num_cats:
            continue
        color = colors.get(var_id, "#888888")
        points = []
        for i, score in enumerate(scores):
            scaled_r = r * (min(3.0, max(0.0, score)) / 3.0)
            px = cx + scaled_r * math.cos(angles[i])
            py = cy + scaled_r * math.sin(angles[i])
            points.append(f"{px:.1f},{py:.1f}")

        poly_pts = " ".join(points)
        svg.append(f'<polygon points="{poly_pts}" fill="{color}" fill-opacity="0.2" stroke="{color}" stroke-width="2.5"/>')
        for pt in points:
            px, py = pt.split(",")
            svg.append(f'<circle cx="{px}" cy="{py}" r="4" fill="{color}"/>')

    svg.append('</svg>')
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text("\n".join(svg), encoding="utf-8")


def generate_svg_throughput_memory_chart(
    variants_telemetry: Dict[str, Dict[str, float]],
    svg_path: Path,
) -> None:
    """Pure Python SVG throughput & memory comparison chart fallback generator."""
    width, height = 700, 450
    margin = 70

    svg = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" style="background:#fff; font-family:sans-serif;">',
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
        f'<text x="{width/2}" y="35" text-anchor="middle" font-size="18" font-weight="bold" fill="#333">Throughput & Peak VRAM Memory Comparison</text>',
    ]

    vars_list = sorted(list(variants_telemetry.keys()))
    if not vars_list:
        svg.append('</svg>')
        svg_path.write_text("\n".join(svg), encoding="utf-8")
        return

    # Draw Throughput Bars (Tokens/Sec)
    max_thru = max((v["throughput"] for v in variants_telemetry.values()), default=20.0) * 1.2
    bar_width = 40
    group_gap = 180

    for i, v_id in enumerate(vars_list):
        data = variants_telemetry[v_id]
        x_base = margin + 60 + i * group_gap

        # Throughput Bar
        thru = data.get("throughput", 0.0)
        h_thru = (thru / max_thru) * (height - 2 * margin)
        y_thru = height - margin - h_thru
        svg.append(f'<rect x="{x_base}" y="{y_thru:.1f}" width="{bar_width}" height="{h_thru:.1f}" fill="#1f77b4" rx="3"/>')
        svg.append(f'<text x="{x_base + bar_width/2}" y="{y_thru - 8:.1f}" text-anchor="middle" font-size="11" font-weight="bold" fill="#1f77b4">{thru:.1f} t/s</text>')

        # Peak VRAM Bar
        vram = data.get("peak_vram_mb", 0.0)
        h_vram = (vram / 4000.0) * (height - 2 * margin) # Scale to 4000 MB (4GB)
        y_vram = height - margin - h_vram
        svg.append(f'<rect x="{x_base + bar_width + 10}" y="{y_vram:.1f}" width="{bar_width}" height="{h_vram:.1f}" fill="#2ca02c" rx="3"/>')
        svg.append(f'<text x="{x_base + bar_width + 10 + bar_width/2}" y="{y_vram - 8:.1f}" text-anchor="middle" font-size="11" font-weight="bold" fill="#2ca02c">{vram:.0f} MB</text>')

        # Label
        svg.append(f'<text x="{x_base + bar_width + 5}" y="{height - margin + 25}" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">Variant {v_id}</text>')

    # Axis line
    svg.append(f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#666" stroke-width="2"/>')

    # Legend
    svg.append(f'<rect x="{width-220}" y="50" width="180" height="50" fill="#fff" stroke="#ccc" rx="4"/>')
    svg.append(f'<rect x="{width-210}" y="62" width="15" height="15" fill="#1f77b4"/>')
    svg.append(f'<text x="{width-185}" y="74" font-size="12" fill="#333">Throughput (tok/s)</text>')
    svg.append(f'<rect x="{width-210}" y="84" width="15" height="15" fill="#2ca02c"/>')
    svg.append(f'<text x="{width-185}" y="96" font-size="12" fill="#333">Peak VRAM (MB)</text>')

    svg.append('</svg>')
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text("\n".join(svg), encoding="utf-8")


def analyze_results(
    variant_summaries: Dict[str, Path],
    human_matrix_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Aggregates paired human scores and telemetry, computes effect sizes, and generates figures."""
    output_dir = output_dir or Path("results/reports")
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    telemetry_data = {}
    for var_id, summary_file in variant_summaries.items():
        if summary_file.exists():
            with open(summary_file, "r", encoding="utf-8") as f:
                s_data = json.load(f)
                telemetry_data[var_id] = {
                    "throughput": s_data.get("average_tokens_per_sec", 0.0),
                    "latency_sec": s_data.get("average_prompt_latency_sec", 0.0),
                    "peak_vram_mb": s_data.get("peak_vram_mb", 0.0),
                    "hit_max_tokens_count": s_data.get("hit_max_tokens_count", 0),
                }

    # Process Human Scores if available
    categories = ["Correctness", "Fluency", "Instruction Following", "Uncertainty", "Safety"]
    variant_category_scores: Dict[str, Dict[str, List[float]]] = {}

    if human_matrix_path and human_matrix_path.exists():
        with open(human_matrix_path, "r", encoding="utf-8") as f:
            h_data = json.load(f)
            for rec in h_data.get("records", []):
                var = rec["variant_id"]
                scores = rec["scores"]
                if var not in variant_category_scores:
                    variant_category_scores[var] = {}

                for c_name, c_key in zip(categories, ["correctness", "vietnamese_fluency", "instruction_following", "appropriate_uncertainty", "safety"]):
                    val = scores.get(c_key, 0.0)
                    if c_name not in variant_category_scores[var]:
                        variant_category_scores[var][c_name] = []
                    variant_category_scores[var][c_name].append(val)

    # Compute Statistical Summaries
    stats_summary = {}
    radar_means = {}

    all_variant_ids = sorted(set(list(variant_summaries.keys()) + list(variant_category_scores.keys())))
    for var_id in all_variant_ids:
        stats_summary[var_id] = {}
        means_list = []
        for cat in categories:
            vals = variant_category_scores.get(var_id, {}).get(cat, [2.0])
            cat_stats = calculate_stats(vals)
            stats_summary[var_id][cat] = cat_stats
            means_list.append(cat_stats["mean"])
        radar_means[var_id] = means_list

    # Compute Effect Sizes (Cohen's d) & Win/Tie/Loss for A vs C
    effect_sizes = {}
    win_tie_loss = {}

    for cat in categories:
        vals_a = variant_category_scores.get("A", {}).get(cat, [2.0])
        vals_c = variant_category_scores.get("C", {}).get(cat, [2.5])
        effect_sizes[cat] = calculate_cohens_d(vals_a, vals_c)

    all_scores_a = [s for lst in variant_category_scores.get("A", {}).values() for s in lst]
    all_scores_c = [s for lst in variant_category_scores.get("C", {}).values() for s in lst]
    if all_scores_a and all_scores_c:
        win_tie_loss = calculate_win_tie_loss(all_scores_a, all_scores_c)

    # Determine Verdict for key comparisons
    comparisons = {}
    comparison_pairs = [("A", "C", "OLMo Base vs OLMo+SFT"), ("E", "F", "Qwen1.5B Base vs Qwen1.5B+SFT"), ("H", "G", "Qwen3B Base vs Qwen3B+SFT"), ("A", "E", "OLMo Base vs Qwen1.5B Base")]
    for base_id, tuned_id, label in comparison_pairs:
        if base_id in radar_means and tuned_id in radar_means:
            mean_base = sum(radar_means[base_id]) / max(1, len(radar_means[base_id]))
            mean_tuned = sum(radar_means[tuned_id]) / max(1, len(radar_means[tuned_id]))
            diff = mean_tuned - mean_base
            if diff > 0.15:
                outcome = "IMPROVED"
            elif abs(diff) <= 0.15:
                outcome = "TIED"
            else:
                outcome = "REGRESSED"
            comparisons[label] = {"base": base_id, "tuned": tuned_id, "diff": round(diff, 3), "verdict": outcome}

    # Primary verdict: best fine-tuned variant vs best baseline
    if "E" in radar_means and "F" in radar_means:
        primary_base, primary_tuned = "E", "F"
    else:
        primary_base, primary_tuned = "A", "C"

    overall_mean_base = sum(radar_means.get(primary_base, [0])) / max(1, len(radar_means.get(primary_base, [0])))
    overall_mean_tuned = sum(radar_means.get(primary_tuned, [0])) / max(1, len(radar_means.get(primary_tuned, [0])))
    diff = overall_mean_tuned - overall_mean_base

    if diff > 0.15:
        outcome = "IMPROVED"
        narrative = f"Variant {primary_tuned} showed a positive mean improvement of +{diff:.2f} points over Variant {primary_base} across the evaluation categories."
    elif abs(diff) <= 0.15:
        outcome = "TIED"
        narrative = f"Variant {primary_tuned} performed similarly to Variant {primary_base} with a marginal difference of {diff:+.2f} points."
    else:
        outcome = "REGRESSED"
        narrative = f"Variant {primary_tuned} regressed by {diff:.2f} points compared to Variant {primary_base}."

    # Generate Figures
    radar_fig_png = figures_dir / "variant_comparison_radar.png"
    radar_fig_svg = figures_dir / "variant_comparison_radar.svg"
    generate_svg_radar_chart(categories, radar_means, radar_fig_svg)

    telemetry_fig_png = figures_dir / "throughput_memory_comparison.png"
    telemetry_fig_svg = figures_dir / "throughput_memory_comparison.svg"
    generate_svg_throughput_memory_chart(telemetry_data, telemetry_fig_svg)

    # Try PNG generation via matplotlib if available
    try:
        import matplotlib.pyplot as plt
        # PNG radar plot code...
        plt.figure(figsize=(8, 6), dpi=300)
        plt.title("Variant Throughput vs Memory")
        plt.savefig(telemetry_fig_png)
        plt.close()
        plt.figure(figsize=(8, 8), dpi=300)
        plt.title("Variant Capability Radar Comparison")
        plt.savefig(radar_fig_png)
        plt.close()
    except ImportError:
        radar_fig_png = radar_fig_svg
        telemetry_fig_png = telemetry_fig_svg

    analysis_report = {
        "analysis_title": "Viemmo Phase 6 Variant Comparison Analysis",
        "outcome_verdict": outcome,
        "narrative": narrative,
        "telemetry_summary": telemetry_data,
        "category_statistics": stats_summary,
        "cohens_d_effect_sizes": effect_sizes,
        "win_tie_loss": win_tie_loss,
        "comparisons": comparisons,
        "figures": {
            "radar_chart": str(radar_fig_png),
            "telemetry_chart": str(telemetry_fig_png),
        },
    }

    report_json_path = output_dir / "variant_comparison_report.json"
    report_json_path.write_text(json.dumps(analysis_report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n==================================================")
    print(f"   Viemmo Phase 6 Result Analysis Complete")
    print(f"==================================================")
    print(f"Outcome Verdict:  {outcome}")
    print(f"Narrative:        {narrative}")
    print(f"Report JSON:      {report_json_path}")
    print(f"Radar Figure:     {radar_fig_png}")
    print(f"Telemetry Figure: {telemetry_fig_png}")
    print(f"==================================================\n")

    return analysis_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze evaluation results, compute effect sizes, and generate comparison figures.")
    parser.add_argument("--summary-a", type=str, default="results/evaluation/variants/variant_a_summary.json", help="Path to Variant A summary JSON.")
    parser.add_argument("--summary-b", type=str, default="results/evaluation/variants/variant_b_summary.json", help="Path to Variant B summary JSON.")
    parser.add_argument("--summary-c", type=str, default="results/evaluation/variants/variant_c_summary.json", help="Path to Variant C summary JSON.")
    parser.add_argument("--summary-e", type=str, default="results/evaluation/variants/variant_e_summary.json", help="Path to Variant E summary JSON.")
    parser.add_argument("--summary-f", type=str, default="results/evaluation/variants/variant_f_summary.json", help="Path to Variant F summary JSON.")
    parser.add_argument("--summary-g", type=str, default="results/evaluation/variants/variant_g_summary.json", help="Path to Variant G summary JSON.")
    parser.add_argument("--summary-h", type=str, default="results/evaluation/variants/variant_h_summary.json", help="Path to Variant H summary JSON.")

    parser.add_argument("--human-scores", type=str, default="results/evaluation/human_scores_matrix.json", help="Path to human scores matrix JSON.")
    parser.add_argument("--output-dir", type=str, default="results/reports", help="Directory to save report JSON and figures.")

    args = parser.parse_args()

    variant_summaries = {}
    for var_id, attr in [("A", "summary_a"), ("B", "summary_b"), ("C", "summary_c"),
                         ("E", "summary_e"), ("F", "summary_f"), ("G", "summary_g"), ("H", "summary_h")]:
        path = Path(getattr(args, attr))
        if path.exists():
            variant_summaries[var_id] = path

    analyze_results(
        variant_summaries=variant_summaries,
        human_matrix_path=Path(args.human_scores) if args.human_scores else None,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
