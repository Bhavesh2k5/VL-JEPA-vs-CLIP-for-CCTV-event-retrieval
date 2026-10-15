# 🚀 Complete Deployment & Execution Guide
## Running the VL-JEPA vs. CLIP CCTV Intelligence Platform on Another Computer

This guide provides step-by-step instructions for deploying, configuring, and executing the complete research and operational platform on another machine (Windows, Linux, or Cloud GPU instances such as RunPod, Lambda Labs, or AWS EC2).

---

## 📋 System Prerequisites

### 1. Hardware Requirements
| Component | Minimum | Recommended (Production / Research) |
| :--- | :--- | :--- |
| **GPU** | NVIDIA GTX 1660 / RTX 3060 (6 GB VRAM) | **NVIDIA RTX 4000 Ada / RTX 3090 / A100 (16-24 GB VRAM)** |
| **CUDA** | CUDA 11.8+ | **CUDA 12.1 or 12.4 (Compute Capability 8.0+)** |
| **CPU** | 4 Cores, x86_64 | **8+ Cores (Intel Core i7/i9 or AMD Ryzen / EPYC)** |
| **RAM** | 16 GB System Memory | **32 GB+ High-Speed DDR4/DDR5** |
| **Disk** | 20 GB free space | **50 GB+ NVMe SSD** (for fast video frame decoding) |

> **Note on CPU-Only Execution:** While GPU acceleration is strongly recommended for video tubelet feature extraction, all retrieval queries and Streamlit search features can execute on CPU using FAISS CPU fallback.

### 2. Software Prerequisites
- **Operating System:** Windows 10/11 (64-bit), Ubuntu 20.04/22.04 LTS, or Debian 11/12
- **Python:** Version **3.10, 3.11, or 3.12** (Python 3.12 verified)
- **Git:** Git 2.30+
- **FFmpeg:** Installed and added to system `PATH` (essential for browser-compatible video transcoding)

---

## 🛠️ Step-by-Step Installation

### Step 1: Clone the Repository
Open a terminal (or PowerShell on Windows) and clone the repository:

```bash
git clone https://github.com/Bhavesh2k5/VL-JEPA-vs-CLIP-for-CCTV-event-retrieval.git
cd VL-JEPA-vs-CLIP-for-CCTV-event-retrieval
```

---

### Step 2: Set Up Virtual Environment

#### On Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

#### On Linux / macOS (Bash):
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### Step 3: Install PyTorch with CUDA Acceleration

Install PyTorch matching your CUDA runtime:

```bash
# For CUDA 12.4 (Recommended):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# For CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CPU-only:
pip install torch torchvision
```

Verify CUDA detection:
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

### Step 4: Install Required Packages

Install all core project dependencies:

```bash
pip install -r implementation/requirements.txt
```

Core libraries installed include:
- **Vision & Video:** `opencv-python`, `decord`, `pillow`, `scikit-video`
- **Transformers & Multimodal:** `transformers`, `huggingface-hub`
- **Vector Search:** `faiss-cpu` (or `faiss-gpu-cu12` if GPU FAISS is preferred)
- **Web App & Interface:** `streamlit`, `altair`
- **Scientific & Visualization:** `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`
- **Jupyter Environment:** `ipykernel`, `jupyterlab`, `notebook`

---

### Step 5: Register the Jupyter Kernel

To seamlessly run the 15 research notebooks in `notebooks_hpc/`:

```bash
python -m ipykernel install --user --name capstone-vljepa-hpc --display-name "Python (Capstone VL-JEPA HPC)"
```

---

## ⚙️ Configuration & Data Path Setup

The central configuration file is located at `implementation_hpc/config/config.yaml` (and mirrored in `implementation/config/config.yaml`).

### Path Configuration
If you have local copies of the surveillance datasets (UCF-Crime, RLVS, UCF-ARG, TinyVIRAT), update their root directories in `config/config.yaml`:

```yaml
data:
  ucf_crime_dir: "path/to/Anomaly-Videos-Part-1"
  rlvs_dir: "path/to/Real Life Violence Dataset"
  ucf_arg_dir: "path/to/cross camera data"
  tinyvirat_dir: "path/to/TinyVIRAT-v2"

retrieval:
  index_save_dir: "results/indices"
  embeddings_save_dir: "results/embeddings"

training:
  save_dir: "results/checkpoints"
```

> **Pre-Packaged Checkpoints & Indices:**  
> The repository **already contains all pre-computed FAISS vector indices** (`results/indices/*.faiss`) and **all trained VL-JEPA predictor checkpoints** (`results/checkpoints/*.pt`). You can run search and inference immediately without re-extracting datasets!

---

## 🖥️ How to Run the System

### 1. Launch the Interactive Web Application (Streamlit)

From the project root:

```bash
cd implementation_hpc
streamlit run demo/streamlit_app.py --server.port 8501
```

Open your browser to: **`http://localhost:8501`**

#### Key Features Available in the App:
- **Tab 1: 🔍 Text-to-Video Semantic Search**
  - Select between **VL-JEPA (Ours)**, **CLIP Baseline**, or **Both (Side-by-Side)**.
  - Choose across 4 benchmark datasets (**UCF-Crime**, **RLVS**, **UCF-ARG**, **TinyVIRAT**).
  - Select from **500 curated pre-loaded surveillance prompts** or type free-form natural language queries (e.g. *"suspect throwing rock through glass door of commercial shop"*).
  - Apply camera viewpoint constraints (Ground, Aerial, Rooftop) for UCF-ARG.
  - View sub-millisecond retrieval latencies, peak event timepoint markers, and video snapshots.
  - Click **"🚀 Generate Structured Incident Report & Verify H3"** to run grounded RAG vs ungrounded LLM hallucination auditing.

- **Tab 2: 📹 CCTV Footage Upload & AI Event Storyteller**
  - Pick from pre-loaded benchmark clips or upload your own `.mp4`, `.avi`, or `.mov` surveillance video.
  - Video player with automated H.264 browser-compatible transcoding.
  - Continuous spatiotemporal optical motion tracking curve ($\Delta E_t$).
  - Zero-shot event classification comparing VL-JEPA vs. CLIP.
  - Automated chronological narrative synthesis describing incident progression.
  - Human Analyst Active Feedback system to record ground-truth corrections.

---

### 2. Run the Jupyter Research Notebook Suite

To explore the research findings and view the 26 publication figures:

```bash
jupyter lab
```

Navigate to **`notebooks_hpc/`** and select the kernel **`Python (Capstone VL-JEPA HPC)`**:

1. `00_Project_Setup_and_Configuration.ipynb`: System hardware telemetry and GPU compute profiling.
2. `01_CCTV_Dataset_Exploration_and_Analysis.ipynb`: Corpus scale and duration statistics across 5,153 clips.
3. `02_Video_Preprocessing_and_Frame_Extraction.ipynb`: Spatiotemporal tubelet decoding and optical flow kinematics.
4. `03_Ground_Truth_Annotation_and_Data_Splitting.ipynb`: 7,410-clip canonical ground truth catalog and stratification.
5. `04_CLIP_Model_Implementation_and_Feature_Extraction.ipynb`: ViT-B/16 InfoNCE contrastive baseline benchmark.
6. `05_VLJEPA_Model_Implementation_and_Feature_Extraction.ipynb`: V-JEPA predictive architecture and cross-modal predictor.
7. `06_Visual_Embedding_Analysis_and_Model_Comparison.ipynb`: t-SNE manifold separability and PCA dimensional rank.
8. `07_Vector_Database_and_Similarity_Search.ipynb`: FAISS IndexFlatIP, IVFFlat, and HNSW latency scaling.
9. `08_CCTV_Event_Retrieval_and_Temporal_Localization.ipynb`: **Hypothesis H1** verification (Recall@K, mAP, peak action MAE).
10. `09_Evidence_Selection_and_Event_Summarization.ipynb`: Confidence gating and top-k context window trade-offs.
11. `10_LLM_Based_Incident_Report_Generation.ipynb`: Structured evidence RAG and incident narrative generation.
12. `11_Hallucination_Detection_and_Factuality_Evaluation.ipynb`: **Hypothesis H3** verification (Hallucination reduction from 82% to 3.8%).
13. `12_Ablation_Studies_and_Robustness_Analysis.ipynb`: Predictor depth, masking ratio, and low-resolution CCTV robustness.
14. `13_Performance_Benchmarking_and_Statistical_Analysis.ipynb`: **Hypothesis H2** verification (Cross-viewpoint invariance).
15. `14_End_to_End_Pipeline_Evaluation_and_Final_Findings.ipynb`: Tri-hypothesis executive scorecard and 14x14 confusion matrix.

---

### 3. Running Batch Retrieval & Pipeline Commands via CLI

#### Run Zero-Shot Text Retrieval Test:
```bash
cd implementation_hpc
python -c "
import yaml; from models import get_encoder; from retrieval import CCTVRetriever, EmbeddingIndexer
config = yaml.safe_load(open('config/config.yaml'))
enc = get_encoder('vljepa', config, predictor_checkpoint='results/checkpoints/vljepa_predictor_ucf_crime.pt')
idx = EmbeddingIndexer.load('results/indices/vljepa_ucf_crime.faiss')
ret = CCTVRetriever(enc, idx)
results = ret.retrieve('suspect throwing rock through glass door of commercial shop', top_k=3)
for r in results:
    print(r['rank'], r['label'], r['similarity_score'], r['path'])
"
```

#### Run Faithfulness & Hallucination Auditing:
```bash
cd implementation_hpc
python -c "
from evaluation.faithfulness_metrics import evaluate_faithfulness
grounded_report = 'At timestamp 00:03.24, subject wearing dark hooded jacket smashed commercial storefront window.'
evidence = 'Frame 7 peak motion: Vandalism storefront glass shattered.'
print('Grounded Faithfulness:', evaluate_faithfulness(grounded_report, evidence))
"
```

---

## ❓ Troubleshooting & Frequently Asked Questions

### 1. `CUDA out of memory` during Video Encoding
- **Cause:** Large batch sizes when processing high-resolution video frames.
- **Solution:** In `config/config.yaml`, reduce `batch_size: 16` to `batch_size: 4` or `batch_size: 2`. The model automatically falls back to sequential tubelet extraction.

### 2. Browser Shows "Video format not supported" in Streamlit
- **Cause:** Video file is encoded with non-standard codecs (e.g. MPEG-4 Part 2 or MJPEG AVI).
- **Solution:** Ensure `ffmpeg` is installed on your machine. The app automatically transcodes files using `data/video_transcoder.py` into browser-native H.264/AAC MP4.

### 3. FAISS Search Returns Import Error on Windows
- **Cause:** Incompatible binary wheel of FAISS for specific Python versions.
- **Solution:** Reinstall FAISS via `pip install faiss-cpu` (universal binary).

### 4. Running on Linux with Path Separator Issues
- **Cause:** Hardcoded Windows backslashes (`\`).
- **Solution:** The codebase uses `pathlib.Path` across all modules (`models/`, `retrieval/`, `reporting/`, `demo/`), ensuring cross-platform compatibility between Windows and Linux.
