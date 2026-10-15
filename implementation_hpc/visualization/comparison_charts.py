from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_retrieval_comparison(
    metrics_vljepa: Dict[str, float],
    metrics_clip: Dict[str, float],
    dataset_name: str,
    output_path: Optional[str] = None,
    dpi: int = 300,
) -> None:
    """
    Generates publication-ready grouped bar chart comparing VL-JEPA vs CLIP retrieval metrics.
    """
    labels = ["Recall@1", "Recall@5", "Recall@10", "MRR", "nDCG@10"]
    vljepa_vals = [metrics_vljepa.get(k, 0.0) for k in labels]
    clip_vals = [metrics_clip.get(k, 0.0) for k in labels]

    x = np.arange(len(labels))
    width = 0.35

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(9, 5), dpi=dpi)

    rects1 = ax.bar(x - width / 2, vljepa_vals, width, label="VL-JEPA (Ours)", color="#1f77b4", edgecolor="black", alpha=0.9)
    rects2 = ax.bar(x + width / 2, clip_vals, width, label="CLIP Baseline", color="#ff7f0e", edgecolor="black", alpha=0.9)

    ax.set_ylabel("Score (0.0 - 1.0)", fontsize=12, fontweight="bold")
    ax.set_title(f"Semantic Retrieval Performance: VL-JEPA vs. CLIP ({dataset_name.upper()})", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=11, frameon=True)

    # Attach numerical labels
    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi)
        plt.close()
    else:
        plt.show()


def export_comparison_table(
    data: List[Dict[str, str]],
    output_csv_path: str,
) -> pd.DataFrame:
    """Exports structured metrics comparison table to CSV."""
    df = pd.DataFrame(data)
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    return df
