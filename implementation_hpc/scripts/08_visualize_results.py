import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from utils.common import ensure_dir, load_config, setup_logger
from visualization import plot_retrieval_comparison, plot_viewpoint_heatmap

logger = setup_logger("visualize_results")


def generate_all_visualizations(config: dict):
    metrics_dir = Path(config.get("evaluation", {}).get("output_dir", "results/metrics"))
    plots_dir = ensure_dir(config.get("visualization", {}).get("output_dir", "results/plots"))

    datasets = config["data"].get("active_datasets", ["rlvs", "ucf_arg", "tinyvirat", "ucf_crime"])
    for d in datasets:
        v_json = metrics_dir / f"vljepa_{d}_metrics.json"
        c_json = metrics_dir / f"clip_{d}_metrics.json"

        if v_json.exists() and c_json.exists():
            with open(v_json, "r", encoding="utf-8") as f:
                v_data = json.load(f)
            with open(c_json, "r", encoding="utf-8") as f:
                c_data = json.load(f)

            out_chart = plots_dir / f"retrieval_comparison_{d}.png"
            plot_retrieval_comparison(v_data.get("retrieval", {}), c_data.get("retrieval", {}), dataset_name=d, output_path=str(out_chart))
            logger.info("Generated retrieval bar chart for %s -> %s", d, out_chart)

    # Cross-camera generalization heatmap
    ucf_v_json = metrics_dir / "vljepa_ucf_arg_metrics.json"
    if ucf_v_json.exists():
        with open(ucf_v_json, "r", encoding="utf-8") as f:
            u_data = json.load(f)
        gen = u_data.get("generalization", {})
        if "matrix" in gen and gen["matrix"]:
            out_heatmap = plots_dir / "ucf_arg_viewpoint_heatmap.png"
            plot_viewpoint_heatmap(gen["matrix"], output_path=str(out_heatmap))
            logger.info("Generated viewpoint heatmap -> %s", out_heatmap)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate all research visualizations.")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    generate_all_visualizations(cfg)
