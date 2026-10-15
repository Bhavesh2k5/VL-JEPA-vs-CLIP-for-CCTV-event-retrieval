import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from data import build_dataset
from utils.common import ensure_dir, load_config, setup_logger
from visualization import export_comparison_table, plot_retrieval_comparison

logger = setup_logger("compare_models")


def compare_models(datasets: list, config: dict):
    metrics_dir = Path(config.get("evaluation", {}).get("output_dir", "results/metrics"))
    plots_dir = ensure_dir(config.get("visualization", {}).get("output_dir", "results/plots"))

    table_records = []

    for d in datasets:
        vljepa_json = metrics_dir / f"vljepa_{d}_metrics.json"
        clip_json = metrics_dir / f"clip_{d}_metrics.json"

        if not vljepa_json.exists() or not clip_json.exists():
            logger.warning("Metrics missing for comparison on %s. Run 05_run_evaluation.py for both models first.", d)
            continue

        with open(vljepa_json, "r", encoding="utf-8") as f:
            vljepa_res = json.load(f)
        with open(clip_json, "r", encoding="utf-8") as f:
            clip_res = json.load(f)

        v_ret = vljepa_res.get("retrieval", {})
        c_ret = clip_res.get("retrieval", {})

        v_eff = vljepa_res.get("efficiency", {})
        c_eff = clip_res.get("efficiency", {})

        # Record for comparison table
        table_records.append({
            "Dataset": d.upper(),
            "Model": "VL-JEPA (Ours)",
            "Recall@1": v_ret.get("Recall@1", 0.0),
            "Recall@5": v_ret.get("Recall@5", 0.0),
            "Recall@10": v_ret.get("Recall@10", 0.0),
            "MRR": v_ret.get("MRR", 0.0),
            "nDCG@10": v_ret.get("nDCG@10", 0.0),
            "Latency (ms)": v_eff.get("mean_latency_ms", 0.0),
            "Peak VRAM (MB)": v_eff.get("peak_gpu_mem_mb", 0.0),
            "Throughput (fps)": v_eff.get("throughput_clips_per_sec", 0.0),
        })

        table_records.append({
            "Dataset": d.upper(),
            "Model": "CLIP Baseline",
            "Recall@1": c_ret.get("Recall@1", 0.0),
            "Recall@5": c_ret.get("Recall@5", 0.0),
            "Recall@10": c_ret.get("Recall@10", 0.0),
            "MRR": c_ret.get("MRR", 0.0),
            "nDCG@10": c_ret.get("nDCG@10", 0.0),
            "Latency (ms)": c_eff.get("mean_latency_ms", 0.0),
            "Peak VRAM (MB)": c_eff.get("peak_gpu_mem_mb", 0.0),
            "Throughput (fps)": c_eff.get("throughput_clips_per_sec", 0.0),
        })

        # Plot grouped comparison bar charts
        chart_path = plots_dir / f"comparison_{d}.png"
        plot_retrieval_comparison(v_ret, c_ret, dataset_name=d, output_path=str(chart_path))
        logger.info("Saved comparison chart for %s to %s", d, chart_path)

    if table_records:
        csv_path = metrics_dir / "comparison_table.csv"
        df = export_comparison_table(table_records, str(csv_path))
        print("\n" + "=" * 90)
        print("HEAD-TO-HEAD MODEL COMPARISON TABLE (VL-JEPA vs. CLIP)")
        print("=" * 90)
        print(df.to_string(index=False))
        print("=" * 90 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate head-to-head model comparison table and plots.")
    parser.add_argument("--dataset", type=str, default="all", choices=["rlvs", "ucf_arg", "tinyvirat", "ucf_crime", "all"])
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    datasets = cfg["data"].get("active_datasets", ["rlvs", "ucf_arg", "tinyvirat", "ucf_crime"]) if args.dataset == "all" else [args.dataset]
    compare_models(datasets, cfg)
