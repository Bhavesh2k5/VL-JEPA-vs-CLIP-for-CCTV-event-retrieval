from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


def plot_training_curves(
    history: Dict[str, List[float]],
    output_path: Optional[str] = None,
    dpi: int = 300,
) -> None:
    """
    Plots training loss and validation Recall curves over epochs.
    """
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    val_recall_1 = history.get("val_recall_at_1", [])
    val_recall_5 = history.get("val_recall_at_5", [])

    epochs = range(1, len(train_loss) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=dpi)

    # 1. Loss curves
    ax1.plot(epochs, train_loss, label="Train Loss (InfoNCE)", color="#1f77b4", linewidth=2.0)
    if val_loss:
        ax1.plot(epochs, val_loss, label="Val Loss", color="#d62728", linestyle="--", linewidth=2.0)
    ax1.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Loss", fontsize=11, fontweight="bold")
    ax1.set_title("CrossModalPredictor InfoNCE Loss", fontsize=12, fontweight="bold")
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # 2. Validation Recall curves
    if val_recall_5:
        ax2.plot(epochs, val_recall_5, label="Val Recall@5", color="#2ca02c", linewidth=2.0)
    if val_recall_1:
        ax2.plot(epochs, val_recall_1, label="Val Recall@1", color="#ff7f0e", linestyle="--", linewidth=2.0)
    ax2.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Recall Score", fontsize=11, fontweight="bold")
    ax2.set_title("Validation Retrieval Accuracy", fontsize=12, fontweight="bold")
    ax2.set_ylim(0.0, 1.05)
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi)
        plt.close()
    else:
        plt.show()
