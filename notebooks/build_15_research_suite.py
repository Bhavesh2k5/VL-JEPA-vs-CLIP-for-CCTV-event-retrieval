"""
Builder script to generate the complete 15-notebook research suite (00 to 14) for:
VL-JEPA vs. CLIP for CCTV Surveillance Event Retrieval, Event Summarization,
LLM Reporting, and Hallucination Evaluation.
Target directory:
notebooks/CLIP_vs_VLJEPA_CCTV_Event_Retrieval_and_Hallucination_Analysis
"""

import os
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
TARGET_DIR = BASE_DIR / "CLIP_vs_VLJEPA_CCTV_Event_Retrieval_and_Hallucination_Analysis"
TARGET_DIR.mkdir(parents=True, exist_ok=True)

def make_code_cell(source_lines):
    if isinstance(source_lines, str):
        source_lines = [line + "\n" for line in source_lines.split("\n")]
        if source_lines:
            source_lines[-1] = source_lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines
    }

def make_markdown_cell(source_lines):
    if isinstance(source_lines, str):
        source_lines = [line + "\n" for line in source_lines.split("\n")]
        if source_lines:
            source_lines[-1] = source_lines[-1].rstrip("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines
    }

def save_notebook(cells, filename):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python (Capstone VL-JEPA HPC)",
                "language": "python",
                "name": "capstone-vljepa-hpc"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    out_path = TARGET_DIR / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Generated: {filename} ({len(cells)} cells)")

# ------------------------------------------------------------------------------
# 00: Project Setup and Configuration
# ------------------------------------------------------------------------------
def build_nb_00():
    cells = [
        make_markdown_cell("""# 🛠️ Notebook 00: Project Setup, Environment Telemetry & Configuration Matrix
### Capstone: VL-JEPA vs. CLIP for CCTV Surveillance Event Retrieval, Event Summarization & Hallucination Analysis
---
**Research Scope & Hardware Environment:**
This notebook initializes the entire experimental testbed for evaluating **VL-JEPA (Vision-Language Joint Embedding Predictive Architecture)** against the **CLIP (Contrastive Language-Image Pretraining)** baseline across **4 real-world CCTV surveillance datasets** (Total: **5,153 clips**):
1. **RLVS (Real Life Violence Dataset):** 2,000 clips (*Violence vs. NonViolence*)
2. **UCF-ARG (Action Recognition with Multi-view Cameras):** 1,410 clips across *Aerial, Ground, Rooftop* viewpoints
3. **TinyVIRAT-v2 (Complex Surveillance Benchmark):** 743 clips focused on *Vehicles, Carrying Objects, Tools & Multi-Actor Interactions*
4. **UCF-Crime (Real-World Crime & Anomaly Benchmark):** 1,000 clips across 14 crime categories (*Robbery, Burglary, Arson, Road Accidents, etc.*)

#### Hardware Target:
- **GPU:** NVIDIA RTX 4000 Ada Generation (20 GB GDDR6 with ECC)
- **CUDA Architecture:** SM 8.9 (Compute Capability 8.9)
- **Deep Learning Framework:** PyTorch 2.6.0+cu124 with CUDA acceleration
"""),

        make_markdown_cell("""## Step 1: GPU Hardware Telemetry & Environment Verification
Inspect CUDA availability, device specifications, multi-processor configuration, and total dedicated VRAM.
"""),

        make_code_cell("""import sys
import os
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

print("=== System & Environment Telemetry ===")
print(f"Operating System:     {platform.system()} {platform.release()} ({platform.version()})")
print(f"Python Version:       {platform.python_version()}")
print(f"PyTorch Version:      {torch.__version__}")
print(f"CUDA Available:       {torch.cuda.is_available()}")

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"Device Name:          {props.name}")
    print(f"Compute Capability:   SM {props.major}.{props.minor}")
    print(f"Total Dedicated VRAM: {props.total_memory / (1024**3):.2f} GB")
    print(f"Streaming MultiProc:  {props.multi_processor_count}")
    print(f"CUDA Driver / Runtime:{torch.version.cuda}")
    print(f"cuDNN Enabled:        {torch.backends.cudnn.enabled}")
    print(f"cuDNN Version:        {torch.backends.cudnn.version()}")
else:
    print("[WARNING] CUDA acceleration is NOT available. Running on CPU.")
"""),

        make_markdown_cell("""## Step 2: Dependency Verification & Import Integrity
Verify all core libraries required for video decoding, transformers, vector indexing, statistical testing, and visualization.
"""),

        make_code_cell("""# Verify library imports
required_packages = [
    ("transformers", "Hugging Face Transformers"),
    ("faiss", "FAISS Vector Search Engine"),
    ("cv2", "OpenCV Video Processing"),
    ("scipy", "SciPy Scientific Computing"),
    ("matplotlib", "Matplotlib Visualization"),
    ("seaborn", "Seaborn Statistical Plots"),
    ("sklearn", "Scikit-Learn Machine Learning")
]

for pkg, desc in required_packages:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "Installed")
        print(f"  [OK] {desc:<35} (v{ver})")
    except ImportError as e:
        print(f"  [FAIL] {desc:<35} - NOT INSTALLED ({e})")
"""),

        make_markdown_cell("""## Step 3: Project Scaffolding & Directory Management
Create the structured directory layout for embeddings, FAISS indices, checkpoints, figures, LaTeX tables, and forensic reports.
"""),

        make_code_cell("""# Define and create directory tree
ROOT_DIR = Path.cwd().parent.parent if Path.cwd().name == "CLIP_vs_VLJEPA_CCTV_Event_Retrieval_and_Hallucination_Analysis" else Path.cwd().parent
RESULTS_DIR = ROOT_DIR / "implementation" / "results"
EMBEDDINGS_DIR = RESULTS_DIR / "embeddings"
INDICES_DIR = RESULTS_DIR / "indices"
CHECKPOINTS_DIR = ROOT_DIR / "notebooks" / "outputs" / "checkpoints"
FIGURES_DIR = ROOT_DIR / "notebooks" / "outputs" / "figures"
TABLES_DIR = ROOT_DIR / "notebooks" / "outputs" / "tables"
REPORTS_DIR = ROOT_DIR / "notebooks" / "outputs" / "reports"
LOGS_DIR = ROOT_DIR / "notebooks" / "outputs" / "logs"

for d in [EMBEDDINGS_DIR, INDICES_DIR, CHECKPOINTS_DIR, FIGURES_DIR, TABLES_DIR, REPORTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
    print(f"Directory confirmed: {d.relative_to(ROOT_DIR)}")
"""),

        make_markdown_cell("""## Step 4: Configuration Matrix & Random Seed Establishment
Establish experimental seeds for complete mathematical determinism across all 15 notebooks.
"""),

        make_code_cell("""# Reproducibility seed setup
RANDOM_SEED = 42

def set_all_seeds(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[Determinism] All seeds set to: {seed}")

set_all_seeds(RANDOM_SEED)

CONFIG_MATRIX = {
    "random_seed": RANDOM_SEED,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": 64,
    "num_frames": 8,
    "shared_dim": 768,
    "num_layers": 4,
    "nhead": 12,
    "dim_feedforward": 3072,
    "initial_lr": 1.5e-4,
    "min_lr": 1.0e-6,
    "weight_decay": 1.0e-2,
    "initial_temp": 0.07,
    "datasets": ["rlvs", "ucf_arg", "tinyvirat", "ucf_crime"],
    "total_expected_clips": 5153
}

print("\\n--- Configuration Matrix ---")
for k, v in CONFIG_MATRIX.items():
    print(f"  {k:<22}: {v}")
"""),

        make_markdown_cell("""## Step 5: GPU Memory Profiling Helper
A reusable utility function to monitor allocated vs reserved GPU memory in gigabytes.
"""),

        make_code_cell("""def get_gpu_memory_telemetry():
    if not torch.cuda.is_available():
        return {"allocated_gb": 0.0, "reserved_gb": 0.0, "free_gb": 0.0}
    alloc = torch.cuda.memory_allocated(0) / (1024**3)
    res = torch.cuda.memory_reserved(0) / (1024**3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return {
        "allocated_gb": round(alloc, 3),
        "reserved_gb": round(res, 3),
        "free_gb": round(total - res, 3),
        "total_gb": round(total, 3)
    }

mem_status = get_gpu_memory_telemetry()
print("GPU Memory Status:", mem_status)
""")
    ]
    save_notebook(cells, "00_Project_Setup_and_Configuration.ipynb")

if __name__ == "__main__":
    build_nb_00()
