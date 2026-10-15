# VL-JEPA vs. CLIP for CCTV Event Retrieval
### Efficiency, Generalization, and Faithful Report Generation — Deployment-Ready Implementation

Lightweight, reproducible framework that ingests CCTV video from four benchmark
datasets (UCF-Crime, UCF-ARG, RLVS, TinyVIRAT-v2), encodes clips with pretrained
**CLIP** and **VL-JEPA**-family models, indexes embeddings in **FAISS**, retrieves
clips from natural-language queries, generates LLM incident reports from retrieved
evidence, and evaluates retrieval quality, efficiency, cross-camera
generalization, and report faithfulness.

**Team**: M. Bhavesh, N. Manmohan Reddy, A. Jayaprakash, K. Avinash — VIT-AP
**Reference documents**: see `../reference/` (research report, literature surveys,
dataset cards, workflow diagram, codebase specification).

---

## 1. Directory layout (this repo root)

```
capstone/
├── reference/          # original project documents (PDFs, dataset cards, spec, images)
└── implementation/     # this package: all source code and deployment assets
    ├── config/config.yaml
    ├── data/           # dataset loaders (one module per benchmark)
    ├── models/         # CLIP + VL-JEPA encoders, factory
    ├── retrieval/      # FAISS indexer, retriever, utils
    ├── reporting/      # LLM incident reporter + prompt templates
    ├── evaluation/     # retrieval/efficiency/faithfulness/generalization metrics
    ├── scripts/        # CLI pipeline entry points
    ├── demo/           # Streamlit web demo
    ├── research/       # findings notebooks: feed CSVs → graphs + tables (research/README.md)
    ├── tests/          # pytest unit tests (no datasets or downloads needed)
    ├── Dockerfile / docker-compose.yml / .env.example
    ├── requirements.txt / setup.py / Makefile
    └── README.md
```

## 2. Design decisions

| Concern | Decision |
|---|---|
| Deployment target | Remote Linux box or container; **every path** comes from `config/config.yaml`, overridable via env vars (`CAPSTONE_DATA_ROOT`, `CAPSTONE_OUTPUT_ROOT`, `CAPSTONE_DEVICE`, `CAPSTONE_CONFIG`) |
| Low-spec defaults | CLIP ViT-B/32 primary, V-JEPA 2 ViT-S/B visual backbone, `flan-t5-base` LLM, FAISS-CPU flat index, fp16 on GPU, automatic batch halving on CUDA OOM, resume-able embedding extraction |
| Resource safety | corrupted-video skip + retry, periodic checkpoint saves, optional per-dataset clip caps (`max_clips`, `max_eval_clips`), `torch.cuda.empty_cache()` between model switches |
| Cosine retrieval | all embeddings L2-normalized; FAISS `IndexFlatIP` (auto-switches to IVF past 20k vectors) |
| Dataset gotchas honored | RLVS globs `*.mp4` **and** `*.avi` (49 AVI NonViolence clips); UCF-Crime skips un-extracted parts; TinyVIRAT keeps native ~72px tensors (no premature downsample); UCF-ARG parses nested viewpoint folders |
| H1 caveat | when the native `facebook/vl-jepa` checkpoint is unavailable, the wrapper pairs V-JEPA 2 (vision) with a CLIP text encoder and truncates both to their shared prefix dimension — a zero-training alignment whose quality is itself part of H1 |
| H3 methodology | grounding score `N_supported / (N_supported + N_unsupported)` with number/date-consistency checks (per the maritime hallucination study in the literature survey), grounded vs ungrounded conditions; BERTScore optional |

## 3. Remote deployment

### Option A — Docker (recommended)

```bash
# on the remote machine, with datasets under /mnt/datasets/capstone
scp -r implementation/ user@server:~/cctv-app/
cd ~/cctv-app
cp .env.example .env                 # set CAPSTONE_DATA_ROOT_HOST + CAPSTONE_DATA_ROOT
docker compose build
docker compose run --rm pipeline scripts/extract_embeddings.py --model clip --dataset rlvs
docker compose run --rm pipeline scripts/build_index.py      --model clip --dataset rlvs
docker compose up -d demo           # http://server:8501
```

Datasets are mounted **read-only** at `/data`; results persist in `./results`.

### Option B — bare metal

```bash
python -m venv .venv && source .venv/bin/activate
make setup                          # CPU torch wheels; swap the index URL for CUDA builds
export CAPSTONE_DATA_ROOT=/mnt/datasets/capstone
export CAPSTONE_OUTPUT_ROOT=$HOME/cctv-results
pytest tests                        # sanity: no dataset needed
make extract MODEL=clip DATASET=rlvs
make index   MODEL=clip DATASET=rlvs
make retrieve MODEL=clip DATASET=rlvs QUERY="violent physical fight"
make demo
```

### Dataset mounting

Point `CAPSTONE_DATA_ROOT` at the folder that directly contains:

```
Anomaly-Detection-Dataset/    cross camera data/
real life voilence data/      tinyvirat/
```

Per-dataset sub-paths in `config.yaml` are relative to that root; absolute
per-dataset paths also work. **The datasets are never copied or modified.**

## 4. Pipeline

```
extract_embeddings.py → build_index.py → run_retrieval.py
                                       → generate_report.py
                                       → run_evaluation.py / compare_models.py
                                       → demo/streamlit_app.py
```

| Command | Purpose |
|---|---|
| `python scripts/extract_embeddings.py --model vljepa\|clip --dataset ucf_crime\|ucf_arg\|rlvs\|tinyvirat\|all` | encode all clips → `results/embeddings/<model>_<dataset>_embeddings.{npy,json}`; resumes automatically; survives CUDA OOM |
| `python scripts/build_index.py --model ... --dataset ...` | builds `results/indices/<model>_<dataset>.faiss` + metadata sidecar |
| `python scripts/run_retrieval.py --model clip --dataset rlvs --query "violent physical fight"` | prints top-k with scores, optionally `--save out.json` |
| `python scripts/generate_report.py --model clip --dataset rlvs --query "..." [--ungrounded]` | evidence-grounded incident report JSON in `results/reports/` with grounding score |
| `python scripts/run_evaluation.py --model clip --dataset ucf_crime --metrics retrieval,efficiency,faithfulness` | metrics CSVs in `results/metrics/` |
| `python scripts/compare_models.py --models vljepa,clip --datasets ucf_crime,rlvs` | `results/metrics/comparison_table.csv` + bar charts in `results/plots/` |

## 5. Research hypotheses → implementation map

| Hypothesis | Where it is tested |
|---|---|
| **H1** VL-JEPA ≥ CLIP retrieval | `models/vljepa_encoder.py` vs `models/clip_encoder.py`, identical FAISS/cosine protocol, `compare_models.py` (Recall@K, MRR, nDCG, mAP) |
| **H2** cross-camera generalization | `evaluation/generalization_eval.py` — visual→visual recall across UCF-ARG aerial/ground/rooftop, heatmap output |
| **H3** grounding reduces hallucination | `reporting/llm_reporter.py` grounded vs `--ungrounded`; `evaluation/faithfulness_metrics.py` grounding score + taxonomy + optional BERTScore |

## 6. Low-spec tuning table

| Constraint | Change in `config/config.yaml` |
|---|---|
| < 8 GB VRAM or CPU-only | `models.clip.primary: openai/clip-vit-base-patch32` (default), `models.vljepa.visual_backbone: facebook/vjepa2-vits16-224`, `llm.model: google/flan-t5-small`, `video.num_frames: 4-8` |
| 16 GB+ VRAM | `primary: openai/clip-vit-large-patch14`, `visual_backbone: facebook/vjepa2-vitl16-224`, `llm.model: google/flan-t5-large`, `llm.quantization: 4bit` (Linux) |
| Slow disk / huge dataset | `data.tinyvirat.max_clips`, `evaluation.max_eval_clips`, `--limit` flag on extraction |
| OOM during extraction | automatic batch halving; or set `--batch-size 2 --num-workers 0` |

## 7. Research findings notebooks (`research/`)

Jupyter notebooks that turn **fed-in data** into publication-ready graphs and
tables for the findings chapter — no pipeline run required. Drop a CSV into
`research/findings_data/`, open a notebook, and every figure/table regenerates;
exports (PNG + PDF figures, CSV + LaTeX tables) land in `research/outputs/`.

| Notebook | Feed CSV | Findings produced |
|---|---|---|
| `01_retrieval_comparison.ipynb` | `retrieval_metrics.csv` | H1: styled metrics table, Recall@K panels/curves, nDCG/MRR/mAP panels, VL-JEPA vs CLIP delta charts |
| `02_efficiency_analysis.ipynb` | `efficiency.csv` | latency bars with error bars, throughput, accuracy-vs-latency trade-off scatter |
| `03_faithfulness_generalization.ipynb` | `faithfulness.csv`, `generalization.csv` | H3: grounded-vs-ungrounded grounding-score bars + gap table; H2: cross-view recall heatmaps |

```bash
pip install -r research/requirements.txt   # notebook + jinja2 (pandas/matplotlib already required)
jupyter notebook research/                 # or open the *.ipynb in VS Code
```

Load order per notebook: your `findings_data/*.csv` → pipeline output
(`results/metrics/comparison_table.csv` when present) → built-in example data,
so the notebooks run end-to-end before real numbers exist. Expected columns are
documented in `research/README.md`.

## 8. Tests

```bash
pytest tests -v
```

Covers retrieval metrics math, FAISS save/load round-trip, retriever wiring,
faithfulness heuristics, and all four dataset index builders (RLVS AVI gotcha
included) using synthetic file trees — no datasets, no model downloads.

## 9. Troubleshooting

- **`cannot load visual backbone ...`** — VL-JEPA/V-JEPA 2 checkpoint id wrong or
  not downloadable; set `models.vljepa.visual_backbone` to a valid V-JEPA 2 id.
- **`decord` install fails** — the loader falls back to OpenCV automatically
  (`video.decode_backend: opencv` forces it).
- **FAISS index/query dim mismatch** — embeddings were extracted with different
  model settings than the query encoder; re-extract or rebuild the index.
- **HuggingFace downloads on air-gapped hosts** — pre-download models with
  `huggingface-cli download` and set `HF_HOME`.
