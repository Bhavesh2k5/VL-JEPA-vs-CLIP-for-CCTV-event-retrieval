# 🎥 Real Surveillance End-to-End Training & Retrieval Notebook Universe
### Capstone: VL-JEPA vs. CLIP for CCTV Surveillance Event Retrieval

This section provides an **authentic, real-data-driven** environment where you can **train neural network alignment models on your NVIDIA RTX 4000 Ada Generation (20GB)**, inspect proper train/val/test splits, evaluate on unseen test data, and query CCTV events across all **3 active surveillance datasets**:
1. **RLVS:** 2,000 clips (*Violence* & *NonViolence*)
2. **UCF-ARG:** 1,410 clips (*Aerial, Ground, Rooftop* viewpoints)
3. **TinyVIRAT-v2:** 743+ clips (*Vehicles, Carrying Heavy Objects, Tools, Complex Multi-Actor Activities*)

---

## 🚀 Quickstart

### Launch via VS Code / Cursor:
1. Open [`notebooks/01_interactive_cctv_retrieval.ipynb`](file:///c:/Users/HPCLAB/Desktop/VLJEPA/capestone/notebooks/01_interactive_cctv_retrieval.ipynb).
2. Click **Select Kernel** in the top right.
3. Select **`Python (Capstone VL-JEPA HPC)`** (or `.venv (Python 3.12)`).
4. Run all cells or execute step-by-step to observe the live GPU training and retrieval!

### Launch via Browser (Classic Jupyter Notebook):
```powershell
& "implementation/.venv/Scripts/python.exe" -m notebook notebooks/
```

---

## 🛠️ Key Improvements & Resolved Retrieval Issues

| Issue Reported | Root Cause | Solution Implemented |
|---|---|---|
| **1) Videos with objects not identified** | UCF-ARG only had basic poses; TinyVIRAT was omitted. | Extracted and indexed **TinyVIRAT-v2** covering `activity_carrying`, `Transport_HeavyCarry`, `Loading`, `Pull`, `Push`, and `specialized_using_tool`. |
| **2) Object queries showing flipping videos** | Zero-shot CLIP lacked temporal alignment. | Trained PyTorch `SpatiotemporalCrossModalPredictor` with bidirectional InfoNCE loss. |
| **3) Why not using TinyVIRAT?** | TinyVIRAT was unextracted. | Integrated all **20 active action classes** (743 clips) from `TinyVIRAT-v2`. |
| **4) Aerial videos shown on ground queries** | Multi-view viewpoints were mixed unconstrained. | Implemented **Viewpoint-Aware Filtering** routing aerial queries to aerial cameras and ground queries to ground cameras. |
| **5) Wrong labelling & outputs** | Unaligned zero-shot text features. | Calibrated semantic surveillance descriptions (`SURVEILLANCE_DESCRIPTIONS`) for all classes. |
| **6) Query "car" returning "person clapping"** | UCF-ARG had no "car" class; zero-shot CLIP mapped car to clapping. | Mapped vehicle queries to TinyVIRAT `vehicle_*` and UCF-ARG `openclosetrunk` (which has a car in frame). |

---

## 📓 Notebook Walkthrough: `01_interactive_cctv_retrieval.ipynb`

1. **Section 1 (Hardware & GPU):** Verifies NVIDIA RTX 4000 Ada (20GB GDDR6, CUDA 12.4, PyTorch 2.6).
2. **Section 2 (3 Real CCTV Datasets & Stratified Split):** Loads **4,153 real clips** and performs a strict 70% Train (2,898), 15% Val (614), and 15% Test (641) stratified split.
3. **Section 3 (Text Supervision):** Encodes detailed surveillance semantics into normalized feature embeddings.
4. **Section 4 (Neural Architecture):** PyTorch `SpatiotemporalCrossModalPredictor` (Transformer + Linear heads) + `BiDirectionalInfoNCELoss`.
5. **Section 5 (Live GPU Training Loop):** 15 epochs executing on the GPU with mixed precision (AMP) and live printouts of Train Loss, Val Loss, Val Recall@1, Val Recall@5, VRAM, and processing speed.
6. **Section 6 (Training Curves):** Plots actual training loss and validation accuracy curves.
7. **Section 7 (Unseen Test Set Evaluation):** Evaluates on 641 unseen test clips, computing real Recall@1, Recall@5, Recall@10, and MRR.
8. **Section 8 (Interactive Live Queries):** Tests vehicles, carrying heavy objects, aerial vs ground viewpoints, and violence queries, sampling real keyframes with OpenCV inline.
9. **Section 9 (Custom Testing):** Enter any custom surveillance text query and inspect results and video frames live.
