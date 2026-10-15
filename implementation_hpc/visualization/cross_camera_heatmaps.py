from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_viewpoint_heatmap(
    matrix: np.ndarray,
    viewpoints: List[str] = ["Aerial", "Rooftop", "Ground"],
    output_path: Optional[str] = None,
    dpi: int = 300,
) -> None:
    """
    Renders 3x3 cross-viewpoint generalization heatmap for UCF-ARG.
    """
    mat = np.array(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=dpi)

    im = ax.imshow(mat, cmap="YlGnBu", vmin=0.0, vmax=1.0)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Recall@5", rotation=-90, va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(np.arange(len(viewpoints)))
    ax.set_yticks(np.arange(len(viewpoints)))
    ax.set_xticklabels([v.capitalize() for v in viewpoints], fontsize=11, fontweight="bold")
    ax.set_yticklabels([v.capitalize() for v in viewpoints], fontsize=11, fontweight="bold")

    ax.set_xlabel("Probe Viewpoint (Query)", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Gallery Viewpoint (Index)", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title("Cross-Camera Viewpoint Generalization (H2)", fontsize=13, fontweight="bold", pad=15)

    # Annotate cells
    for i in range(len(viewpoints)):
        for j in range(len(viewpoints)):
            val = mat[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=12, fontweight="bold")

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi)
        plt.close()
    else:
        plt.show()
