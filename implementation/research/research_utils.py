"""Shared helpers for the research-findings notebooks.

Feed data as CSV files into ``findings_data/``; the notebooks turn them into
publication-ready graphs and tables via the helpers below. Everything degrades
gracefully: when a CSV is missing, pipeline outputs are tried next, then
built-in example data so the notebooks always run end-to-end.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESEARCH_DIR = Path(__file__).resolve().parent
DATA_DIR = RESEARCH_DIR / "findings_data"
PIPELINE_RESULTS_DIR = RESEARCH_DIR.parent / "results"
OUTPUT_DIR = RESEARCH_DIR / "outputs"

MODEL_COLORS = {
    "vljepa": "#4C72B0",
    "vl-jepa": "#4C72B0",
    "clip": "#DD8452",
    "clip_large": "#C44E52",
}
EXTRA_COLORS = ["#55A868", "#8172B2", "#CCB974", "#64B5CD"]

HIGHER_IS_BETTER = ("recall@", "ndcg@", "mrr", "map", "grounding_score", "bertscore", "throughput")
LOWER_IS_BETTER = ("latency",)

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "legend.frameon": False,
    }
)


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def _demo(name: str) -> pd.DataFrame:
    demos = {
        "retrieval_metrics": _demo_retrieval_metrics,
        "efficiency": _demo_efficiency,
        "faithfulness": _demo_faithfulness,
        "generalization": _demo_generalization,
    }
    if name not in demos:
        raise FileNotFoundError(f"findings_data/{name}.csv not found and no example data registered")
    print(f"[research] '{name}.csv' not found - showing EXAMPLE data; drop your own CSV into findings_data/")
    return demos[name]()


def load_findings(name: str, fallbacks=()) -> pd.DataFrame:
    """Load ``findings_data/<name>.csv``, then any ``fallbacks`` paths, then example data."""
    path = DATA_DIR / f"{name}.csv"
    if path.is_file():
        return pd.read_csv(path)
    for fb in fallbacks:
        fb = Path(fb)
        if fb.is_file():
            print(f"[research] using pipeline output {fb.name} (no findings_data/{name}.csv yet)")
            return pd.read_csv(fb)
    return _demo(name)


def _demo_retrieval_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": ["vljepa", "vljepa", "clip", "clip"],
            "dataset": ["ucf_crime", "rlvs", "ucf_crime", "rlvs"],
            "recall@1": [0.412, 0.465, 0.372, 0.421],
            "recall@5": [0.648, 0.701, 0.598, 0.655],
            "recall@10": [0.754, 0.793, 0.706, 0.752],
            "ndcg@10": [0.528, 0.571, 0.481, 0.522],
            "mrr": [0.503, 0.554, 0.459, 0.501],
            "map": [0.461, 0.502, 0.415, 0.448],
            "num_queries": [40, 30, 40, 30],
        }
    )


def _demo_efficiency() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": ["vljepa", "vljepa", "clip", "clip"],
            "dataset": ["ucf_crime", "rlvs", "ucf_crime", "rlvs"],
            "latency_mean_ms": [78.4, 74.9, 41.7, 39.8],
            "latency_std_ms": [6.2, 5.1, 3.4, 2.9],
            "throughput_clips_s": [12.8, 13.4, 24.0, 25.1],
        }
    )


def _demo_faithfulness() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": ["vljepa", "vljepa", "clip", "clip"],
            "condition": ["grounded", "ungrounded", "grounded", "ungrounded"],
            "grounding_score": [0.86, 0.41, 0.84, 0.44],
            "n_supported": [43, 29, 42, 31],
            "n_unsupported": [7, 42, 8, 39],
        }
    )


def _demo_generalization() -> pd.DataFrame:
    rows = []
    base = {"aerial": 0.55, "ground": 0.56, "rooftop": 0.55}
    cross = {"aerial": 0.41, "ground": 0.43, "rooftop": 0.40}
    for model, shift in (("vljepa", 0.03), ("clip", 0.00)):
        for src in base:
            for tgt in base:
                diag = src == tgt
                r1 = (base if diag else cross)[src] + (shift if diag else shift / 2)
                rows.append(
                    {
                        "model": model,
                        "source_view": src,
                        "target_view": tgt,
                        "recall@1": round(r1, 3),
                        "recall@5": round(min(r1 + 0.23, 0.95), 3),
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def style_table(df: pd.DataFrame, precision: int = 3):
    """Pretty notebook table: formatted numbers + best value highlighted per metric.

    Falls back to a plain formatted DataFrame when jinja2 is not installed.
    """
    num_cols = df.select_dtypes("number").columns
    try:
        styler = df.style.format(precision=precision, na_rep="-")
    except AttributeError:
        return df.round(precision)
    for col in num_cols:
        if any(col.startswith(p) for p in HIGHER_IS_BETTER):
            styler = styler.highlight_max(subset=[col], props="font-weight:bold;color:#1a7f37;")
        elif any(col.startswith(p) for p in LOWER_IS_BETTER):
            styler = styler.highlight_min(subset=[col], props="font-weight:bold;color:#1a7f37;")
    return styler


def export_table(df: pd.DataFrame, name: str, formats=("csv", "tex")) -> list[Path]:
    """Write a table to outputs/<name>.{csv,tex}; returns the paths written."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    paths = []
    for fmt in formats:
        out = OUTPUT_DIR / f"{name}.{fmt}"
        if fmt == "csv":
            df.to_csv(out, index=False)
        elif fmt == "tex":
            df.to_latex(out, index=False, float_format="%.3f")
        else:
            raise ValueError(f"unsupported table format: {fmt}")
        paths.append(out)
    print("[research] table ->", ", ".join(str(p.name) for p in paths))
    return paths


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _color(label) -> str:
    key = str(label).lower()
    if key in MODEL_COLORS:
        return MODEL_COLORS[key]
    idx = sum(hash(k) % 7 for k in MODEL_COLORS)  # stable-ish spread for unknown labels
    return EXTRA_COLORS[idx % len(EXTRA_COLORS)]


def export_figure(fig, name: str, formats=("png", "pdf")) -> list[Path]:
    """Save a figure to outputs/<name>.{png,pdf}; returns the paths written."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    paths = []
    for fmt in formats:
        out = OUTPUT_DIR / f"{name}.{fmt}"
        fig.savefig(out)
        paths.append(out)
    print("[research] figure ->", ", ".join(str(p.name) for p in paths))
    return paths


def plot_grouped_bars(df, value, group="dataset", hue="model", title="", ylabel="",
                      save=None, annotate=True):
    """Grouped bar chart: one cluster per ``group``, one bar per ``hue``."""
    pivot = df.pivot_table(index=group, columns=hue, values=value, aggfunc="mean")
    x = np.arange(len(pivot.index))
    width = 0.8 / max(pivot.shape[1], 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, col in enumerate(pivot.columns):
        bars = ax.bar(x + i * width - 0.4 + width / 2, pivot[col], width,
                      label=col, color=_color(col))
        if annotate:
            ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(x, pivot.index)
    ax.set_ylabel(ylabel or value)
    ax.set_title(title or f"{value} by {group}")
    ax.set_ylim(0, max(pivot.max()) * 1.18)
    ax.legend()
    fig.tight_layout()
    if save:
        export_figure(fig, save)
    return fig, ax


def plot_metric_panels(df, metrics, group="dataset", hue="model", save=None,
                       suptitle=""):
    """One subplot per metric, each a grouped bar chart (e.g. Recall@1/5/10)."""
    metrics = [m for m in metrics if m in df.columns]
    if not metrics:
        raise ValueError(f"none of {metrics} found in columns")
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.0), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, metric in zip(axes, metrics):
        pivot = df.pivot_table(index=group, columns=hue, values=metric, aggfunc="mean")
        x = np.arange(len(pivot.index))
        width = 0.8 / max(pivot.shape[1], 1)
        for i, col in enumerate(pivot.columns):
            ax.bar(x + i * width - 0.4 + width / 2, pivot[col], width,
                   label=col, color=_color(col))
        ax.set_xticks(x, pivot.index, fontsize=9)
        ax.set_title(metric, fontsize=11)
        ax.set_ylim(0, max(pivot.max()) * 1.2)
    axes[0].set_ylabel("score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               bbox_to_anchor=(0.5, 1.06))
    if suptitle:
        fig.suptitle(suptitle, y=1.14, fontsize=13)
    fig.tight_layout()
    if save:
        export_figure(fig, save)
    return fig, axes


def plot_recall_curves(df, ks=(1, 5, 10), hue="model", group=None, save=None):
    """Recall@K curves: x = K, one line per model (optionally per model+group)."""
    cols = [f"recall@{k}" for k in ks if f"recall@{k}" in df.columns]
    if not cols:
        raise ValueError("no recall@K columns found")
    keys = [hue] + ([group] if group else [])
    agg = df.groupby(keys)[cols].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for _, row in agg.iterrows():
        label = row[hue] if not group else f"{row[hue]} · {row[group]}"
        ax.plot(ks, row[cols].astype(float).values, marker="o", label=label,
                color=_color(row[hue]), alpha=1.0 if not group else 0.45 + 0.55 * (hash(label) % 3) / 2)
    ax.set_xticks(list(ks))
    ax.set_xlabel("K")
    ax.set_ylabel("Recall@K")
    ax.set_title("Retrieval quality vs K")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save:
        export_figure(fig, save)
    return fig, ax


def plot_metric_delta(df, model_a="vljepa", model_b="clip", metrics=None,
                      group=None, save=None):
    """Relative improvement (%) of model_a over model_b per metric."""
    metrics = metrics or [c for c in df.columns
                          if c.startswith(("recall@", "ndcg@")) or c in ("mrr", "map")]
    metrics = [m for m in metrics if m in df.columns]
    if group:
        a = df[df["model"] == model_a].groupby(group)[metrics].mean()
        b = df[df["model"] == model_b].groupby(group)[metrics].mean()
        delta = ((a - b) / b * 100)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        x = np.arange(len(delta.index))
        width = 0.8 / max(len(metrics), 1)
        for i, m in enumerate(metrics):
            ax.bar(x + i * width - 0.4 + width / 2, delta[m], width, label=m,
                   color=EXTRA_COLORS[i % len(EXTRA_COLORS)])
        ax.set_xticks(x, delta.index)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel(f"{model_a} vs {model_b} (%)")
        ax.set_title(f"Relative improvement of {model_a} over {model_b}")
    else:
        a = df[df["model"] == model_a][metrics].mean()
        b = df[df["model"] == model_b][metrics].mean()
        delta = ((a - b) / b * 100).sort_values()
        fig, ax = plt.subplots(figsize=(7, 4.5))
        bars = ax.barh(delta.index, delta.values,
                       color=["#1a7f37" if v >= 0 else "#b32424" for v in delta.values])
        ax.bar_label(bars, fmt="%+.1f%%", padding=3)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(f"{model_a} vs {model_b} (%)")
        ax.set_title(f"Relative improvement of {model_a} over {model_b}")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)
    fig.tight_layout()
    if save:
        export_figure(fig, save)
    return fig, ax


def plot_errorbar_bars(df, value="latency_mean_ms", err="latency_std_ms",
                       group="dataset", hue="model", title="", ylabel="",
                       save=None):
    """Bar chart with error bars (e.g. mean latency ± std per model/dataset)."""
    pivot = df.pivot_table(index=group, columns=hue, values=value, aggfunc="mean")
    errs = df.pivot_table(index=group, columns=hue, values=err, aggfunc="mean") if err in df.columns else None
    x = np.arange(len(pivot.index))
    width = 0.8 / max(pivot.shape[1], 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, col in enumerate(pivot.columns):
        yerr = errs[col].values if errs is not None else None
        ax.bar(x + i * width - 0.4 + width / 2, pivot[col], width, label=col,
               color=_color(col), yerr=yerr, capsize=3, error_kw={"alpha": 0.6})
    ax.set_xticks(x, pivot.index)
    ax.set_ylabel(ylabel or value)
    ax.set_title(title or f"{value} by {group}")
    ax.legend()
    fig.tight_layout()
    if save:
        export_figure(fig, save)
    return fig, ax


def plot_generalization_heatmap(df, value="recall@1", row="source_view",
                                col="target_view", hue="model", save=None,
                                cmap="viridis"):
    """Cross-view heatmap per model (e.g. UCF-ARG aerial/ground/rooftop recall)."""
    models = sorted(df[hue].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.2),
                             layout="constrained")
    axes = np.atleast_1d(axes)
    vmax = df[value].max()
    vmin = 0.0
    for ax, model in zip(axes, models):
        sub = df[df[hue] == model]
        pivot = sub.pivot_table(index=row, columns=col, values=value, aggfunc="mean")
        im = ax.imshow(pivot.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(pivot.columns)), pivot.columns, fontsize=9)
        ax.set_yticks(range(len(pivot.index)), pivot.index, fontsize=9)
        ax.set_xlabel(col)
        if ax is axes[0]:
            ax.set_ylabel(row)
        ax.set_title(f"{model} — {value}")
        ax.grid(False)
        for r in range(pivot.shape[0]):
            for c in range(pivot.shape[1]):
                v = pivot.values[r, c]
                ax.text(c, r, f"{v:.3f}", ha="center", va="center",
                        color="white" if v > (vmax + vmin) / 2 else "black", fontsize=9)
    fig.colorbar(im, ax=axes, shrink=0.85)
    if save:
        export_figure(fig, save)
    return fig, axes


def plot_scatter_tradeoff(quality_df, latency_df, quality="recall@1",
                          latency="latency_mean_ms", hue="model",
                          group="dataset", save=None):
    """Accuracy vs latency scatter: one point per (model, group)."""
    merged = quality_df.merge(latency_df, on=["model", group], how="inner")
    if merged.empty:
        raise ValueError("no overlapping (model, dataset) rows between the two frames")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for model, sub in merged.groupby(hue):
        ax.scatter(sub[latency], sub[quality], s=90, label=model, color=_color(model))
        for _, p in sub.iterrows():
            ax.annotate(p[group], (p[latency], p[quality]),
                        textcoords="offset points", xytext=(7, 4), fontsize=9)
    ax.set_xlabel(latency)
    ax.set_ylabel(quality)
    ax.set_title(f"{quality} vs {latency}")
    ax.legend()
    fig.tight_layout()
    if save:
        export_figure(fig, save)
    return fig, ax
