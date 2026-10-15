# VL-JEPA vs. CLIP for CCTV Event Retrieval (Implementation HPC)

High-performance, publication-ready framework for **CCTV Semantic Event Retrieval**, **Cross-Camera Generalization**, and **Faithful Incident Reporting** on modern workstation hardware (**NVIDIA RTX 4000 Ada — 20 GB VRAM**).

---

## 1. System Specifications & Hardware Acceleration

* **GPU**: NVIDIA RTX 4000 Ada Generation (**20 GB GDDR6**)
* **Precision**: FP16 mixed-precision acceleration
* **Models**:
  * **CLIP**: `openai/clip-vit-large-patch14` (ViT-L/14, 768-d)
  * **VL-JEPA**: Meta **V-JEPA 2 ViT-L** (`facebook/vjepa2-vitl-fpc16-256-ssv2`, 1024-d) + **Trainable CrossModalPredictor** (4-layer Transformer, InfoNCE loss, 1536-d shared space)
  * **Reporting SLM**: `google/flan-t5-base`
* **Vector Index**: FAISS `IndexFlatIP` (Exact cosine similarity on L2-normalized embeddings)

---

## 2. Active Benchmark Datasets (Located on Desktop)

All four benchmark datasets are located directly on the local machine under `C:\Users\HPCLAB\Desktop`:

| Dataset | Local Desktop Path | Clip Count | Task Focus |
| :--- | :--- | :--- | :--- |
| **UCF-Crime** | `C:\Users\HPCLAB\Desktop\capstone-ucf dataset` | **1,000 clips** across 14 anomaly classes & Normal | Broad surveillance anomaly retrieval |
| **RLVS** | `C:\Users\HPCLAB\Desktop\capstone-core datset\capstone_rlvs` | **2,000 clips** (1,000 Violence, 1,000 NonViolence) | Real-life violence detection |
| **UCF-ARG** | `C:\Users\HPCLAB\Desktop\capstone-core datset\capstone_ucf_arg` | **1,410 clips** (Aerial, Rooftop, Ground) | Hypothesis **H2** (Cross-Camera Generalization) |
| **TinyVIRAT-v2** | `C:\Users\HPCLAB\Desktop\capstone-core datset\capstone_tinyvirat_v2` | **3,000 active clips** (16,950 in train split) | Multi-label low-resolution surveillance action retrieval |

---

## 3. Architecture: Hybrid V-JEPA 2 + Cross-Modal Predictor

```
CCTV Video Clip ──► [Frozen V-JEPA 2 ViT-L] ──► Visual Embedding (1024-d)
                    (1M+ hours video pretraining)           │
                                                            ▼
                                              ┌───────────────────────────┐
Text Query ────────► [CLIP Text Encoder] ────►│   CrossModalPredictor     │ ◄── Trained with InfoNCE
("person running")                            │  (4-Layer Transformer)   │
                                              └─────────────┬─────────────┘
                                                            ▼
                                              Aligned Embedding (1536-d)
                                                            │
                                              FAISS IndexFlatIP (Cosine)
                                                            │
                                              Top-K Retrieved CCTV Clips
                                                            │
                                              LLM Incident Reporter (Flan-T5)
                                                            │
                                              Faithfulness Evaluation (H3)
```

---

## 4. Quick Start: End-to-End Execution

### Step 1: Environment Setup
```powershell
# Activate Python virtual environment
..\implementation\.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### Step 2: Run Benchmark Pipeline
The default configuration is already set to `C:/Users/HPCLAB/Desktop/capstone-core datset`. Simply run:
```powershell
# Run end-to-end extraction, alignment training, indexing, and evaluation
python scripts/run_full_pipeline.py
```

---

## 5. Step-by-Step Execution Workflow

You can also run each pipeline stage independently:

### 1. Video Feature Extraction
```powershell
# Extract CLIP ViT-L/14 embeddings
python scripts/01_extract_embeddings.py --model clip --dataset all

# Extract V-JEPA 2 ViT-L spatiotemporal representations
python scripts/01_extract_embeddings.py --model vljepa --dataset all
```

### 2. Train Cross-Modal Alignment Predictor
Trains the 4-layer bidirectional predictor with InfoNCE loss ($\tau=0.07$) in under 2 minutes:
```powershell
python scripts/03_train_alignment.py --dataset all --epochs 25 --lr 1e-4
```

### 3. Build FAISS Vector Indices
```powershell
python scripts/02_build_faiss_index.py --model clip --dataset all
python scripts/02_build_faiss_index.py --model vljepa --dataset all
```

### 4. Interactive Natural Language Query Search
```powershell
python scripts/04_query_retrieval.py --model vljepa --dataset rlvs --query "violent physical fight" --top-k 5 --report
```

### 5. Benchmark Evaluation (H1, H2, H3)
```powershell
python scripts/05_run_evaluation.py --model clip --dataset all
python scripts/05_run_evaluation.py --model vljepa --dataset all
```

### 6. Model Comparison & Visualizations
```powershell
# Generates comparison_table.csv and publication bar charts
python scripts/06_compare_models.py --dataset all

# Generates all figures (heatmaps, training curves)
python scripts/08_visualize_results.py
```

### 7. Launch Streamlit Web Application
```powershell
streamlit run demo/streamlit_app.py
```

---

## 6. Research Hypotheses & Outputs

* **H1: Semantic Retrieval Quality**: Proven via Recall@1, Recall@5, Recall@10, MRR, and nDCG@10 metrics in `results/metrics/comparison_table.csv`.
* **H2: Cross-Camera Generalization**: Evaluated across UCF-ARG viewpoints in `results/plots/ucf_arg_viewpoint_heatmap.png`.
* **H3: Report Faithfulness**: Verifies that RAG-grounded reports achieve $>85\%$ Fact-Grounding Scores compared to ungrounded hallucinations in `results/reports/`.
