# Research Findings Notebooks

Jupyter notebooks that turn **fed-in data** into publication-ready **graphs and
tables** for the research findings chapter (H1 retrieval quality, efficiency,
H2 cross-camera generalization, H3 report faithfulness).

You never need to run the full pipeline to use this section: drop a CSV into
`findings_data/`, open a notebook, and every figure/table regenerates.

---

## 1. Quickstart

```bash
cd implementation
.venv/Scripts/pip install -r research/requirements.txt   # adds notebook (Windows)
# or: pip install -r research/requirements.txt
.venv/Scripts/python -m notebook research/               # opens Jupyter in this folder
```

Works in VS Code too: open any `*.ipynb` here and pick the `.venv` kernel.
`pandas` / `numpy` / `matplotlib` already come from the main `requirements.txt`.

## 2. Layout

```
research/
├── findings_data/     # FEED YOUR DATA HERE (one CSV per findings area)
│   ├── retrieval_metrics.csv    # H1: Recall@K / nDCG / MRR / mAP per model+dataset
│   ├── efficiency.csv           # latency + throughput per model+dataset
│   ├── faithfulness.csv         # H3: grounding score, grounded vs ungrounded
│   └── generalization.csv       # H2: view-to-view recall grid (UCF-ARG)
├── research_utils.py  # shared plotting/table helpers used by the notebooks
├── 01_retrieval_comparison.ipynb        # H1: metric tables + bar/panel charts + delta
├── 02_efficiency_analysis.ipynb         # latency bars, accuracy-vs-latency trade-off
├── 03_faithfulness_generalization.ipynb # H3 grounding bars + H2 heatmaps
└── outputs/           # exported figures (png+pdf) and tables (csv+tex) land here
```

## 3. How feeding data works

Each notebook loads `findings_data/<name>.csv` via `research_utils.load_findings`.
The loader falls back in this order:

1. `findings_data/<name>.csv` — **your data** (highest priority)
2. pipeline output when available (e.g. `results/metrics/comparison_table.csv`
   written by `scripts/compare_models.py` for `retrieval_metrics`)
3. built-in **example data** — clearly announced, so the notebooks always run
   end-to-end before you have real numbers; replace it with real data when ready.

Expected columns (extra columns are ignored, missing ones are skipped):

| CSV | Columns |
|---|---|
| `retrieval_metrics.csv` | `model, dataset, recall@1, recall@5, recall@10, ndcg@10, mrr, map, num_queries` |
| `efficiency.csv` | `model, dataset, latency_mean_ms, latency_std_ms, throughput_clips_s` |
| `faithfulness.csv` | `model, condition, grounding_score, n_supported, n_unsupported` |
| `generalization.csv` | `model, source_view, target_view, recall@1, recall@5` |

Rename models/datasets freely (`vljepa`, `clip`, `clip_large`, ...); charts pick
colors per model name automatically and add new categories as they appear.

## 4. Notebooks

| Notebook | Feeds on | Produces |
|---|---|---|
| `01_retrieval_comparison.ipynb` | `retrieval_metrics.csv` | full metrics table (styled + LaTeX), Recall@K panels, MRR/mAP/nDCG panels, VL-JEPA vs CLIP delta chart |
| `02_efficiency_analysis.ipynb` | `efficiency.csv` (+ metrics for the trade-off) | latency bars with error bars, accuracy-vs-latency scatter |
| `03_faithfulness_generalization.ipynb` | `faithfulness.csv`, `generalization.csv` | grounded-vs-ungrounded bars + support table, cross-view heatmaps per model |

Every chart/table cell calls `ru.export_figure(...)` / `ru.export_table(...)` so
results accumulate in `outputs/` as PNG (300 dpi) + PDF and CSV + LaTeX —
ready to paste into the report.

## 5. Adding your own analysis

Import the shared helpers and compose them:

```python
import research_utils as ru
df = ru.load_findings("retrieval_metrics")
ax = ru.plot_grouped_bars(df, value="map", group="dataset", hue="model",
                          title="mAP by dataset", save="map_by_dataset")
ru.export_table(df, "my_table")
```

Helper inventory in `research_utils.py`: `load_findings`, `style_table`,
`export_table`, `export_figure`, `plot_grouped_bars`, `plot_metric_panels`,
`plot_recall_curves`, `plot_metric_delta`, `plot_errorbar_bars`,
`plot_generalization_heatmap`, plus demo-data loaders.
