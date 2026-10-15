# 📹 VL-JEPA vs. CLIP for CCTV Surveillance Event Retrieval, Event Summarization & Hallucination Analysis

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python" alt="Python Version" />
  <img src="https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-EE4C2C?logo=pytorch" alt="PyTorch" />
  <img src="https://img.shields.io/badge/CUDA-12.4%20%7C%20SM%208.9-76B900?logo=nvidia" alt="CUDA" />
  <img src="https://img.shields.io/badge/Streamlit-1.42.0-FF4B4B?logo=streamlit" alt="Streamlit" />
  <img src="https://img.shields.io/badge/FAISS-Vector%20Search-00599C" alt="FAISS" />
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License" />
</p>

---

## 📌 Project Overview

This repository houses the complete research testbed and production intelligence platform comparing **Vision-Language Joint Embedding Predictive Architecture (VL-JEPA)** against the **Contrastive Language-Image Pretraining (CLIP)** baseline across **4 real-world surveillance benchmarks (5,153 video clips, 41.9 cumulative hours)**:
- **UCF-Crime:** Real-world surveillance anomalies across 14 crime categories + Normal operations.
- **RLVS (Real Life Violence Dataset):** Street altercation violence recognition (1,000 Violence vs 1,000 Non-Violence).
- **UCF-ARG:** Multi-view cross-camera action recognition captured concurrently from Ground, Aerial Drone, and Rooftop perspectives.
- **TinyVIRAT-v2:** Complex multi-actor facility surveillance involving vehicles, tools, and object transport.

---

## 🏆 Tri-Hypothesis Statistical Acceptance Scorecard

| Hypothesis | Core Research Question | CLIP Baseline | VL-JEPA (Ours) | Empirical Advantage | Statistical Significance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **H1: Semantic Retrieval** | Do predictive representations outperform contrastive embeddings in zero-shot retrieval? | Recall@1 = 58.5%<br>mAP = 0.577 | **Recall@1 = 78.2%**<br>**mAP = 0.734** | **+19.7% Recall**<br>**+0.157 mAP** | $t = 14.82, p < 0.001$<br>**(ACCEPTED)** |
| **H2: Viewpoint Invariance** | Does spatiotemporal tubelet modeling provide invariance across camera angles? | Ground $\rightarrow$ Roof = 42.8%<br>Drop = -29.3% | **Ground $\rightarrow$ Roof = 72.4%**<br>Drop = -12.1% | **+29.6% Recall**<br>**17.2% Less Degrad.** | $t = 11.34, p < 0.001$<br>**(ACCEPTED)** |
| **H3: Report Faithfulness** | Does grounding LLM reports in structured CCTV evidence eliminate hallucinations? | Hallucination = 82.0%<br>Factual Anchors = 1.8 | **Hallucination = 3.8%**<br>**Factual Anchors = 6.4** | **-78.2% Hallucination**<br>**+255% Fact Anchors** | $t = 24.81, p < 0.0001$<br>**(ACCEPTED)** |

---

## 📂 Repository Directory Structure (7 Core Folders)

```text
├── CLIP_vs_VJEPA_CCTV_Event_Retrieval_and_Hallucination_Analysis/
│   └── Initial comparative notebooks exploring feature representations.
│
├── implementation/
│   ├── Core modular codebase containing models, data loaders, evaluation, and configs.
│   ├── models/ (VL-JEPA and CLIP multimodal feature extractors)
│   ├── retrieval/ (FAISS indexer and CCTVRetriever search engine)
│   └── reporting/ (Incident report generation and factuality auditing)
│
├── implementation_hpc/
│   ├── Production HPC-optimized implementation with CUDA SM 8.9 acceleration.
│   ├── demo/ (Interactive Streamlit Web Application: streamlit_app.py)
│   ├── models/ (Zero-shot intent resolvers and online feedback memory bank)
│   └── config/config.yaml (Global hardware and dataset paths)
│
├── notebooks/
│   └── Extended training runs, ablation scripts, and exploratory prototypes.
│
├── notebooks_hpc/
│   ├── Complete 15-Notebook Publication Research Suite (00 to 14)
│   │   ├── Fully pre-rendered with executed outputs, tables, and LaTeX math.
│   │   └── Registered with kernel: Python (Capstone VL-JEPA HPC)
│   ├── figures/ (26 Publication-Grade Research Figures @ 300 DPI)
│   └── README.md (Comprehensive research suite catalog)
│
├── reference/
│   ├── Literature surveys, research reports, dataset cards, and PowerPoint presentations.
│   └── CCTV_VL-JEPA_Capstone_Research_Report.pdf, Literature_Survey_Report.pdf
│
├── results/
│   ├── checkpoints/ (Pre-trained VL-JEPA predictor weights: vljepa_predictor_*.pt, ~18MB each)
│   ├── indices/ (Pre-built FAISS vector indices for all 4 datasets: *.faiss, 3-9MB each)
│   └── benchmark_ground_truth_registry.json (7,410 cataloged canonical surveillance clips)
│
├── SETUP_AND_RUN_GUIDE.md    # Step-by-step guide to run the model on another computer
├── PROJECT_DOCUMENTATION.md  # In-depth architectural, theoretical, and results document
└── README.md                 # Root repository overview (this file)
```

---

## 🚀 Quickstart Guide (3 Steps)

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/Bhavesh2k5/VL-JEPA-vs-CLIP-for-CCTV-event-retrieval.git
cd VL-JEPA-vs-CLIP-for-CCTV-event-retrieval

# Set up virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install PyTorch with CUDA & requirements
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r implementation/requirements.txt
```

### 2. Launch the Streamlit Web Application
All model checkpoints and FAISS vector indices are pre-packaged in `results/`. You can launch immediately:

```bash
cd implementation_hpc
streamlit run demo/streamlit_app.py --server.port 8501
```

Open your browser to: **`http://localhost:8501`**

### 3. Explore the Jupyter Research Suite
```bash
jupyter lab
```
Open any notebook in `notebooks_hpc/` (00 to 14) to view pre-rendered outputs, empirical tables, and embedded high-resolution figures.

For complete deployment instructions, see [**`SETUP_AND_RUN_GUIDE.md`**](./SETUP_AND_RUN_GUIDE.md).

---

## 🖥️ Interactive Web Platform Features

- **🔍 Tab 1: Text-to-Video Semantic Search**
  - Side-by-side comparative retrieval (**VL-JEPA vs. CLIP**).
  - Search across 4 benchmarks using **500 pre-loaded prompts** or custom natural language queries (e.g. *"suspect throwing rock through glass door"*).
  - Sub-millisecond vector retrieval with camera viewpoint filtering (Ground, Aerial, Rooftop).
  - One-click **Hypothesis H3 Verification** button auditing claim-level hallucinations.

- **📹 Tab 2: Upload CCTV Footage & AI Event Storyteller**
  - Upload arbitrary `.mp4`, `.avi`, or `.mov` surveillance videos or select benchmark test footage.
  - Automatic browser-compatible H.264 transcoding.
  - Optical motion energy trajectory tracking ($\Delta E_t$) identifying peak anomaly timepoints.
  - Chronological story narrative synthesis describing incident progression.
  - Human Analyst Active Feedback system to correct and log ground-truth annotations.

---

## 📊 Key Research Figures (Gallery in `notebooks_hpc/figures/`)

| Figure | Description |
| :--- | :--- |
| **Fig 00** | Dedicated GPU VRAM footprint (6.4 GB) and NVDEC decoding throughput (485 FPS). |
| **Fig 01** | Multi-dataset scale (5,153 clips, 41.9 hours) across UCF-Crime, RLVS, UCF-ARG, TinyVIRAT. |
| **Fig 03** | Continuous optical flow motion energy trajectory curve ($\Delta E_t$) detecting peak events. |
| **Fig 08** | 2D t-SNE latent manifold comparison proving VL-JEPA cluster separability (Silhouette: 0.482 vs 0.291). |
| **Fig 10** | FAISS search latency scaling (IndexFlatIP vs IVFFlat vs HNSW) from 1K to 100K vectors. |
| **Fig 11** | Cross-dataset Recall@1 and mAP performance advantage proving **Hypothesis H1**. |
| **Fig 16** | Hallucination rate drop from 82.0% down to 3.8% ($p < 0.0001$) proving **Hypothesis H3**. |
| **Fig 21** | Cross-camera viewpoint transfer retention (Ground $\rightarrow$ Aerial $\rightarrow$ Rooftop) proving **Hypothesis H2**. |
| **Fig 23** | 14x14 normalized action classification confusion matrix across UCF-Crime. |
| **Fig 24** | 6-axis multi-criteria radar chart benchmarking the complete pipeline. |
| **Fig 25** | Executive Capstone Tri-Hypothesis Statistical Acceptance Dashboard. |

---

## 📖 Further Documentation

- **[SETUP_AND_RUN_GUIDE.md](./SETUP_AND_RUN_GUIDE.md):** Complete installation, configuration, and execution instructions for other computers and cloud GPUs.
- **[PROJECT_DOCUMENTATION.md](./PROJECT_DOCUMENTATION.md):** Full mathematical formulations, theoretical framework, architecture specifications, and empirical tables.
- **[notebooks_hpc/README.md](./notebooks_hpc/README.md):** Catalog of the 15 research notebooks and 26 publication figures.

---

## 📄 License & Attribution

This project is licensed under the **MIT License**. Checkpoints and indices are provided for academic research and surveillance intelligence evaluation.
