import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np


def load_json(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Evaluation file missing: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_comparison_chart(base_agg: dict, ft_agg: dict, chart_path: str) -> None:
    """Creates a grouped bar chart comparing Base Model vs. QLoRA Expert accuracy metrics."""
    labels = ["Exact Match (%)", "Token F1 Overlap (%)", "Syntax / Schema Validity (%)"]
    base_scores = [
        base_agg["exact_match_pct"],
        base_agg["token_f1_pct"],
        base_agg["syntax_validity_pct"],
    ]
    ft_scores = [
        ft_agg["exact_match_pct"],
        ft_agg["token_f1_pct"],
        ft_agg["syntax_validity_pct"],
    ]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    rects1 = ax.bar(x - width / 2, base_scores, width, label="Before (Base Model)", color="#94a3b8")
    rects2 = ax.bar(x + width / 2, ft_scores, width, label="After (QLoRA Expert)", color="#16a34a")

    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Before vs. After QLoRA Fine-Tuning Benchmark", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 112)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=10)
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(chart_path), exist_ok=True)
    plt.savefig(chart_path)
    plt.close()
    print(f"[SAVED] Comparison chart written to {chart_path}")


def generate_markdown_report(base_data: dict, ft_data: dict, report_path: str, top_k: int = 5) -> None:
    """Writes a Markdown table + top qualitative improvements where QLoRA fixed base failures."""
    b_agg = base_data["aggregate_metrics"]
    f_agg = ft_data["aggregate_metrics"]

    def delta(key: str, suffix: str = "%") -> str:
        diff = f_agg[key] - b_agg[key]
        sign = "+" if diff >= 0 else ""
        return f"{sign}{diff:.2f}{suffix}"

    lines = [
        "# Before vs. After QLoRA Evaluation Report\n",
        f"- **Base Model:** `{base_data['base_model_id']}`",
        f"- **Evaluated Samples:** `{base_data['num_samples']}` held-out test records\n",
        "## 1. Quantitative Benchmark Summary\n",
        "| Metric | Before (Base Model) | After (QLoRA Expert) | Net Improvement |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Exact Match Accuracy** | {b_agg['exact_match_pct']}% | **{f_agg['exact_match_pct']}%** | **{delta('exact_match_pct')}** |",
        f"| **Token F1 Overlap** | {b_agg['token_f1_pct']}% | **{f_agg['token_f1_pct']}%** | **{delta('token_f1_pct')}** |",
        f"| **Syntax / Schema Validity** | {b_agg['syntax_validity_pct']}% | **{f_agg['syntax_validity_pct']}%** | **{delta('syntax_validity_pct')}** |",
        f"| **Avg Inference Latency** | {b_agg['avg_latency_sec']}s | {f_agg['avg_latency_sec']}s | {delta('avg_latency_sec', 's')} |\n",
        "## 2. Qualitative Case Studies (Base Failure -> QLoRA Success)\n",
    ]

    paired = []
    for b_item, f_item in zip(base_data["samples"], ft_data["samples"]):
        gain = f_item["metrics"]["token_f1"] - b_item["metrics"]["token_f1"]
        paired.append((gain, b_item, f_item))

    paired.sort(key=lambda x: x[0], reverse=True)

    for rank, (gain, b_item, f_item) in enumerate(paired[:top_k], start=1):
        lines.extend(
            [
                f"### Case #{rank} (F1 Gain: +{gain * 100:.1f}%)",
                f"**Domain Input:**\n```text\n{b_item['user_input']}\n```",
                f"**Ground Truth Reference:**\n```text\n{b_item['reference']}\n```",
                f"**Before (Base Model Output):**\n```text\n{b_item['prediction']}\n```",
                f"**After (QLoRA Expert Output):**\n```text\n{f_item['prediction']}\n```\n---\n",
            ]
        )

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[SAVED] Qualitative & quantitative Markdown report written to {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Before vs. After comparison report and chart.")
    parser.add_argument("--baseline", type=str, default="evaluation/results/baseline_metrics.json")
    parser.add_argument("--finetuned", type=str, default="evaluation/results/finetuned_metrics.json")
    parser.add_argument("--chart", type=str, default="evaluation/results/before_vs_after_chart.png")
    parser.add_argument("--report", type=str, default="evaluation/results/qualitative_samples.md")
    args = parser.parse_args()

    base_data = load_json(args.baseline)
    ft_data = load_json(args.finetuned)

    generate_comparison_chart(base_data["aggregate_metrics"], ft_data["aggregate_metrics"], args.chart)
    generate_markdown_report(base_data, ft_data, args.report)


if __name__ == 
      "__main__":
    main()
