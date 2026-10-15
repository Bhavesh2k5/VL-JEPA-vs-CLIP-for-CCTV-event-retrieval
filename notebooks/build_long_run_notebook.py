"""
Script to generate the standalone Extended Training Run Notebook for CLIP VL-JEPA.
Target locations:
1. notebooks/CLIP_vs_VLJEPA_CCTV_Event_Retrieval_and_Hallucination_Analysis/Train_CLIP_VLJEPA_Extended_Run.ipynb
2. notebooks/Train_CLIP_VLJEPA_Extended_Run.ipynb
"""

import os
import json
from pathlib import Path

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

def build_training_notebook():
    cells = [
        make_markdown_cell("""# [RESEARCH] Extended Training Run: VL-JEPA Spatiotemporal Predictor
### Vision-Language Joint Embedding Predictive Architecture for CCTV Surveillance
---
**Experiment Objective:**
This notebook conducts a dedicated, publication-grade **Extended Training Run** of the **VL-JEPA Spatiotemporal Cross-Modal Predictor** ($30.3\\text{M}$ trainable parameters) over a configurable duration (default $60\\text{ minutes}$ or customizable) on the unified multi-dataset surveillance corpus ($5,153$ video clips):
- **RLVS (Real Life Violence Dataset):** $2,000$ clips (*Violence vs. NonViolence*)
- **UCF-ARG (Multi-view Surveillance Actions):** $1,410$ clips (*Aerial, Ground, Rooftop*)
- **TinyVIRAT-v2 (Complex Surveillance Benchmark):** $743$ clips (*Interactions, Tools, Vehicles*)
- **UCF-Crime (Real-World Crime & Anomaly Benchmark):** $1,000$ clips across $14$ crime categories

#### Key Engineering Features of this Extended Run Notebook:
1. **Time-Guarded Training Loop:** Runs continuously until the target duration ($60\\text{ minutes}$) or epoch ceiling is reached, with live telemetry.
2. **Latent Feature Augmentation:** Applies random Gaussian jitter ($\\sigma=0.015$), dimensional feature dropout ($p=0.15$), and L2 sphere re-normalization to prevent representation collapse over hundreds of epochs.
3. **Dual Metric Tracking Every 5 Epochs:** Evaluates both **Exact Instance-Level Retrieval** (`Inst-R@1`, `Inst-R@5`, `MRR`) on $N=758$ candidates and **Event Category Match** (`Cat-R@1`, `Cat-R@5`).
4. **Resilient Multi-Tier Checkpointing:** Automatically tracks and saves:
   - `vljepa_extended_run_best_inst_r1.pt` (Best instance retrieval checkpoint)
   - `vljepa_extended_run_best_val_loss.pt` (Best validation contrastive loss checkpoint)
   - `vljepa_extended_run_latest.pt` (Rolling latest state for seamless resumption)
5. **Hardware Optimization:** Designed for maximum throughput and memory efficiency on **NVIDIA RTX 4000 Ada (20GB GDDR6)**.
"""),

        make_markdown_cell("""## Step 1: Environment Telemetry & GPU Hardware Verification
Verify CUDA acceleration, PyTorch runtime version, and dedicated VRAM on the NVIDIA RTX 4000 Ada GPU.
"""),

        make_code_cell("""import sys
import os
import time
import math
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Ensure imports resolve cleanly
current_dir = Path.cwd()
if (current_dir / "notebook_utils.py").exists():
    sys.path.append(str(current_dir))
elif (current_dir.parent / "notebook_utils.py").exists():
    sys.path.append(str(current_dir.parent))

import notebook_utils as nu

print("=== System & Hardware Telemetry ===")
print(f"PyTorch Version:      {torch.__version__}")
print(f"CUDA Available:       {torch.cuda.is_available()}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"Device Name:          {props.name}")
    print(f"Compute Capability:   SM {props.major}.{props.minor}")
    print(f"Dedicated VRAM:       {props.total_memory / (1024**3):.2f} GB")
    torch.backends.cudnn.benchmark = True
else:
    print("[WARNING] Running on CPU.")

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
    print(f"[Determinism] Global seeds set to: {seed}")

if not hasattr(nu, "set_all_seeds"):
    nu.set_all_seeds = set_all_seeds
"""),

        make_markdown_cell("""## Step 2: Configuration Matrix & Time Budget Definition
Configure the training parameters, learning rate schedule, and the **target training duration** (default 60 minutes, adjustable as desired).
"""),

        make_code_cell("""# ==============================================================================
# EXTENDED RUN TRAINING CONFIGURATION MATRIX
# ==============================================================================
TARGET_DURATION_MINUTES = 60      # Total target runtime in minutes (e.g. 30, 60, 90, 120)
MAX_EPOCHS = 600                  # Maximum epoch upper-bound
EVAL_EVERY_EPOCHS = 5             # Comprehensive validation frequency
BATCH_SIZE = 64                   # Optimal batch size for RTX 4000 Ada
INITIAL_LR = 2.0e-4               # Peak learning rate with AdamW
MIN_LR = 1.0e-6                   # Minimum cosine floor
WEIGHT_DECAY = 1.0e-2             # Weight decay for L2 regularization
WARMUP_EPOCHS = 10                # Linear warm-up epochs
TEMPERATURE_INIT = 0.07           # Initial InfoNCE contrastive temperature
SHARED_DIM = 768                  # Transformer latent dimension (matches ViT-L capacity)
TRANSFORMER_LAYERS = 4            # Transformer encoder depth
TRANSFORMER_HEADS = 12            # Multi-head attention heads
FEEDFORWARD_DIM = 3072            # Feedforward dimension (4 * d_model)
DROPOUT_RATE = 0.10               # Model dropout rate
AUG_NOISE_STD = 0.015             # Feature jitter standard deviation
AUG_DROPOUT_P = 0.15              # Latent dimension masking probability
RANDOM_SEED = 42

# Apply deterministic seeds
set_all_seeds(RANDOM_SEED)

print("Extended Run Configuration:")
print(f"  - Target Run Duration:     {TARGET_DURATION_MINUTES} minutes (~{TARGET_DURATION_MINUTES*60} seconds)")
print(f"  - Max Epochs:              {MAX_EPOCHS}")
print(f"  - Validation Frequency:    Every {EVAL_EVERY_EPOCHS} epochs")
print(f"  - Batch Size:              {BATCH_SIZE}")
print(f"  - Learning Rate Schedule:  {INITIAL_LR} -> {MIN_LR} (Cosine with {WARMUP_EPOCHS}w warm-up)")
print(f"  - Model Parameter Count:   ~30.3 Million")
"""),

        make_markdown_cell("""## Step 3: Dataset Ingestion & Split Validation
Load all pre-extracted visual features and metadata across the 4 CCTV benchmarks ($5,153$ total clips) and instantiate the strictly isolated $70/15/15$ split.
"""),

        make_code_cell("""# Load feature embeddings and metadata across all 4 datasets
embs_visual, metadata = nu.load_all_datasets_data()

print(f"\\nTotal Loaded Surveillance Samples: {len(metadata):,}")
print(f"Visual Feature Dimensions:         {embs_visual.shape[1]}")

# Generate stratified splits (3598 Train, 758 Val, 797 Test)
train_idx, val_idx, test_idx = nu.create_stratified_splits(metadata, train_ratio=0.70, val_ratio=0.15, seed=RANDOM_SEED)

assert len(train_idx) + len(val_idx) + len(test_idx) == len(metadata) == 5153
print(f"Train Partition (70.0%): {len(train_idx):,} clips")
print(f"Val Partition   (14.7%): {len(val_idx):,} clips")
print(f"Test Partition  (15.3%): {len(test_idx):,} clips")
"""),

        make_markdown_cell("""## Step 4: Text Feature Synthesis & Augmented Dataset Pipeline
To support sustained training over an extended duration without representation collapse, we implement the `SurveillanceAugmentedDataset` with dynamic latent jitter and dimension masking.
"""),

        make_code_cell("""# Synthesize text features for paired cross-modal alignment
print("Synthesizing rich spatiotemporal textual features via CLIP text encoder...")
text_embs_all = nu.create_clip_text_embeddings_for_metadata(metadata, device=str(device))
print(f"Text Feature Matrix: {text_embs_all.shape}")

class SurveillanceAugmentedDataset(Dataset):
    \"\"\"
    PyTorch Dataset with Latent Feature Augmentations for Long-Run Training:
    - Additive Gaussian Feature Jitter
    - Dimension Masking / Latent Feature Dropout
    - L2 Re-normalization
    Returns pure tensors (v, t) for ultra-fast collation.
    \"\"\"
    def __init__(self, visual_feats, text_feats, metadata_list, is_training=True, noise_std=0.015, feat_drop_p=0.15):
        self.visual = torch.from_numpy(visual_feats).float() if isinstance(visual_feats, np.ndarray) else visual_feats.float()
        self.text = torch.from_numpy(text_feats).float() if isinstance(text_feats, np.ndarray) else text_feats.float()
        self.metadata = metadata_list
        self.is_training = is_training
        self.noise_std = noise_std
        self.feat_drop_p = feat_drop_p

    def __len__(self):
        return len(self.visual)

    def __getitem__(self, idx):
        v = self.visual[idx].clone()
        t = self.text[idx].clone()

        if self.is_training:
            # 1. Feature Gaussian Jitter
            if torch.rand(1).item() < 0.6:
                noise = torch.randn_like(v) * self.noise_std
                v = v + noise
            # 2. Latent Dimension Masking
            if torch.rand(1).item() < 0.4:
                mask = (torch.rand_like(v) > self.feat_drop_p).float()
                v = v * mask
            # Re-normalize to unit hypersphere
            v = F.normalize(v, p=2, dim=-1)

        return v, t

# Instantiate train and validation dataloaders
train_dataset = SurveillanceAugmentedDataset(
    embs_visual[train_idx], text_embs_all[train_idx], [metadata[i] for i in train_idx],
    is_training=True, noise_std=AUG_NOISE_STD, feat_drop_p=AUG_DROPOUT_P
)

val_dataset = SurveillanceAugmentedDataset(
    embs_visual[val_idx], text_embs_all[val_idx], [metadata[i] for i in val_idx],
    is_training=False
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train Batches per Epoch: {len(train_loader)} (Batch size {BATCH_SIZE})")
print(f"Val Batches:             {len(val_loader)}")
"""),

        make_markdown_cell("""## Step 5: Model Architecture & Optimization Setup
Instantiate the `SpatiotemporalCrossModalPredictor` ($30.3\\text{M}$ parameters), Symmetric InfoNCE Loss, AdamW optimizer, and Cosine Annealing learning rate scheduler with warmup.
"""),

        make_code_cell("""# Instantiate VL-JEPA Spatiotemporal Predictor
model = nu.SpatiotemporalCrossModalPredictor(
    visual_dim=512,
    text_dim=512,
    shared_dim=SHARED_DIM,
    num_layers=TRANSFORMER_LAYERS,
    nhead=TRANSFORMER_HEADS,
    dim_feedforward=FEEDFORWARD_DIM
).to(device)

criterion = nu.BiDirectionalInfoNCELoss(initial_temp=TEMPERATURE_INIT).to(device)

# Separate weight decay for 1D parameters (LayerNorm, biases) vs weights
decay_params = []
no_decay_params = []
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if param.dim() <= 1 or "bias" in name or "norm" in name:
        no_decay_params.append(param)
    else:
        decay_params.append(param)

optimizer = torch.optim.AdamW([
    {"params": decay_params, "weight_decay": WEIGHT_DECAY},
    {"params": no_decay_params, "weight_decay": 0.0},
    {"params": [criterion.logit_scale], "weight_decay": 0.0, "lr": INITIAL_LR * 0.5}
], lr=INITIAL_LR, betas=(0.9, 0.98))

# Cosine Annealing scheduler with warmup
def lr_lambda(epoch):
    if epoch < WARMUP_EPOCHS:
        return float(epoch + 1) / float(max(1, WARMUP_EPOCHS))
    progress = float(epoch - WARMUP_EPOCHS) / float(max(1, MAX_EPOCHS - WARMUP_EPOCHS))
    return max(MIN_LR / INITIAL_LR, 0.5 * (1.0 + math.cos(math.pi * progress)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[Model Summary] Trainable Parameters: {total_params:,}")
print(f"[Optimization] AdamW initialized with peak LR={INITIAL_LR}, Cosine Decay to {MIN_LR}")
"""),

        make_markdown_cell("""## Step 6: Dual-Metric Validation Evaluation Engine
Calculates both **Instance-Level Video Pairing** (`Inst-R@1`, `Inst-R@5`, `MRR`) on $N=758$ candidates and **Category Action/Crime Match** (`Cat-R@1`, `Cat-R@5`).
"""),

        make_code_cell("""def evaluate_validation_split(model, criterion, val_dataset, device):
    model.eval()
    criterion.eval()
    val_v = val_dataset.visual.to(device)
    val_t = val_dataset.text.to(device)
    meta = val_dataset.metadata
    n_val = len(meta)

    with torch.no_grad():
        zv = model.encode_visual(val_v)
        zt = model.encode_text(val_t)
        loss = criterion(zv, zt).item()
        sim_matrix = torch.matmul(zt, zv.t()).cpu().numpy()

    labels = [m.get("label", "unknown").lower() for m in meta]
    h1_inst, h5_inst, h10_inst = 0, 0, 0
    cat_h1, cat_h5 = 0, 0
    rr_sum = 0.0

    for i in range(n_val):
        ranks = np.argsort(-sim_matrix[i])
        # Instance metrics (exact video index match)
        if ranks[0] == i: h1_inst += 1
        if i in ranks[:5]: h5_inst += 1
        if i in ranks[:10]: h10_inst += 1
        rank_pos = np.where(ranks == i)[0][0] + 1
        rr_sum += 1.0 / rank_pos

        # Category metrics (semantic action/crime label match)
        target_lbl = labels[i]
        top_lbls = [labels[r] for r in ranks]
        if top_lbls[0] == target_lbl: cat_h1 += 1
        if target_lbl in top_lbls[:5]: cat_h5 += 1

    return {
        "val_loss": loss,
        "inst_r1": (h1_inst / n_val) * 100.0,
        "inst_r5": (h5_inst / n_val) * 100.0,
        "inst_r10": (h10_inst / n_val) * 100.0,
        "mrr": rr_sum / n_val,
        "cat_r1": (cat_h1 / n_val) * 100.0,
        "cat_r5": (cat_h5 / n_val) * 100.0
    }

print("[Ready] Validation evaluation engine compiled successfully.")
"""),

        make_markdown_cell("""## Step 7: Extended Training Run Loop Execution
**Instructions for User:**
Click the **Run** button on the cell below to start the extended training run.
- It will automatically run until the target duration has elapsed (or until `MAX_EPOCHS` is reached).
- Live metrics are printed for every epoch, with full validation evaluation every 5 epochs.
- Best checkpoints will be automatically saved to disk in `outputs/checkpoints/`.
- If you wish to stop early at any point, simply click **Interrupt Kernel** in Jupyter; the script traps the interrupt safely and preserves your latest progress and checkpoints!
"""),

        make_code_cell("""# Output checkpoint paths
ckpt_dir = nu.CHECKPOINTS_DIR
ckpt_dir.mkdir(parents=True, exist_ok=True)
best_inst_path = ckpt_dir / "vljepa_extended_run_best_inst_r1.pt"
best_loss_path = ckpt_dir / "vljepa_extended_run_best_val_loss.pt"
latest_ckpt_path = ckpt_dir / "vljepa_extended_run_latest.pt"
log_csv_path = nu.OUTPUTS_DIR / "logs" / "vljepa_extended_run_training_log.csv"
(nu.OUTPUTS_DIR / "logs").mkdir(parents=True, exist_ok=True)

# Tracking state
history = []
best_val_inst_r1 = -1.0
best_val_loss = float("inf")
start_wall_time = time.time()
target_wall_seconds = TARGET_DURATION_MINUTES * 60.0

print("=" * 86)
print(f"[START] EXTENDED VL-JEPA TRAINING RUN ON {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"Target Duration: {TARGET_DURATION_MINUTES} minutes ({target_wall_seconds:.0f} seconds)")
print(f"Save Path (Best Inst-R@1): {best_inst_path}")
print("=" * 86)
print(f"{'Epoch':<6} | {'TrainLoss':<10} | {'ValLoss':<10} | {'Inst-R@1':<9} | {'Inst-R@5':<9} | {'Cat-R@1':<9} | {'LR':<9} | {'Elapsed':<8} | {'Remaining'}")
print("-" * 86)

try:
    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        criterion.train()
        train_loss_accum = 0.0

        for v_batch, t_batch in train_loader:
            v_batch = v_batch.to(device)
            t_batch = t_batch.to(device)

            optimizer.zero_grad()
            zv, zt = model(v_batch, t_batch)
            loss = criterion(zv, zt)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_accum += loss.item()

        train_loss_epoch = train_loss_accum / len(train_loader)
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed_sec = time.time() - start_wall_time
        remaining_sec = max(0.0, target_wall_seconds - elapsed_sec)
        elapsed_str = f"{int(elapsed_sec // 60):02d}m {int(elapsed_sec % 60):02d}s"
        rem_str = f"{int(remaining_sec // 60):02d}m {int(remaining_sec % 60):02d}s"

        # Periodic or final validation
        do_eval = (epoch % EVAL_EVERY_EPOCHS == 0) or (epoch == 1) or (elapsed_sec >= target_wall_seconds)
        if do_eval:
            val_metrics = evaluate_validation_split(model, criterion, val_dataset, device)
            v_loss = val_metrics["val_loss"]
            inst_r1 = val_metrics["inst_r1"]
            inst_r5 = val_metrics["inst_r5"]
            cat_r1 = val_metrics["cat_r1"]
            mrr = val_metrics["mrr"]

            # Save best checkpoints
            if inst_r1 > best_val_inst_r1:
                best_val_inst_r1 = inst_r1
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "elapsed_seconds": elapsed_sec
                }, best_inst_path)

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "elapsed_seconds": elapsed_sec
                }, best_loss_path)

            print(f"{epoch:<6} | {train_loss_epoch:<10.4f} | {v_loss:<10.4f} | {inst_r1:<8.2f}% | {inst_r5:<8.2f}% | {cat_r1:<8.2f}% | {current_lr:<9.2e} | {elapsed_str:<8} | {rem_str}")
        else:
            val_metrics = {"val_loss": np.nan, "inst_r1": np.nan, "inst_r5": np.nan, "cat_r1": np.nan, "mrr": np.nan}
            print(f"{epoch:<6} | {train_loss_epoch:<10.4f} | {'--':<10} | {'--':<9} | {'--':<9} | {'--':<9} | {current_lr:<9.2e} | {elapsed_str:<8} | {rem_str}")

        # Record history
        record = {
            "epoch": epoch,
            "train_loss": train_loss_epoch,
            "val_loss": val_metrics["val_loss"],
            "inst_r1": val_metrics["inst_r1"],
            "inst_r5": val_metrics["inst_r5"],
            "cat_r1": val_metrics["cat_r1"],
            "mrr": val_metrics["mrr"],
            "lr": current_lr,
            "elapsed_seconds": elapsed_sec
        }
        history.append(record)

        # Save rolling latest checkpoint every epoch
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "history": history
        }, latest_ckpt_path)

        # Check if the target duration has completed
        if elapsed_sec >= target_wall_seconds:
            print("\\n" + "=" * 86)
            print(f"[COMPLETE] TARGET RUNTIME OF {TARGET_DURATION_MINUTES} MINUTES REACHED! Completed at Epoch {epoch}.")
            print("=" * 86)
            break

except KeyboardInterrupt:
    print("\\n[INTERRUPT DETECTED] Training stopped gracefully by user. Checkpoints preserved.")

# Save training log to CSV
df_history = pd.DataFrame(history)
df_history.to_csv(log_csv_path, index=False)
print(f"\\n[Saved] Detailed training metrics logged to: {log_csv_path}")
print(f"[Best Checkpoint] Highest Inst-R@1 achieved: {best_val_inst_r1:.2f}% (Saved to {best_inst_path.name})")
"""),

        make_markdown_cell("""## Step 8: Convergence Trajectory & Metric Evolution Visualizations
Plot the training vs. validation loss, learning rate decay curve, and the evolution of both Instance Retrieval and Category Retrieval over the extended training run.
"""),

        make_code_cell("""# Plot training trajectory
if len(history) > 1:
    df_h = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss Curves
    axes[0].plot(df_h["epoch"], df_h["train_loss"], label="Train InfoNCE Loss", color="#1f77b4", lw=2)
    eval_sub = df_h.dropna(subset=["val_loss"])
    axes[0].plot(eval_sub["epoch"], eval_sub["val_loss"], label="Val InfoNCE Loss", color="#ff7f0e", marker="o", lw=2)
    axes[0].set_title("Contrastive InfoNCE Loss Trajectory", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Retrieval Curves
    eval_r1 = df_h.dropna(subset=["inst_r1"])
    axes[1].plot(eval_r1["epoch"], eval_r1["inst_r1"], label="Inst-R@1 (Exact ID)", color="#2ca02c", marker="s", lw=2)
    axes[1].plot(eval_r1["epoch"], eval_r1["inst_r5"], label="Inst-R@5 (Exact ID)", color="#17becf", marker="^", lw=2)
    axes[1].plot(eval_r1["epoch"], eval_r1["cat_r1"], label="Cat-R@1 (Action Match)", color="#d62728", ls="--", lw=2)
    axes[1].set_title("Validation Retrieval Metrics Evolution", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Learning Rate & Duration
    axes[2].plot(df_h["epoch"], df_h["lr"], label="Learning Rate", color="#9467bd", lw=2)
    axes[2].set_title("Cosine Annealing Schedule with Warmup", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    fig_path = nu.OUTPUTS_DIR / "figures" / "vljepa_extended_run_training_curves.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[Figure Export] Training curves figure saved to: {fig_path.name}")
else:
    print("[INFO] Run training cell first to generate convergence curves.")
"""),

        make_markdown_cell("""## Step 9: Final Unseen Test Split Evaluation ($N=797$ Candidates)
Load the newly trained optimal checkpoint and perform comprehensive evaluation against the strictly isolated test split ($N=797$ video clips) comparing VL-JEPA against the zero-shot CLIP baseline.
"""),

        make_code_cell("""# Comprehensive evaluation on isolated test split
print("=== Unseen Test Split Evaluation (N=797 Candidates) ===")

test_v = torch.from_numpy(embs_visual[test_idx]).float().to(device)
test_t = torch.from_numpy(text_embs_all[test_idx]).float().to(device)
test_meta = [metadata[i] for i in test_idx]
n_test = len(test_meta)
labels = [m.get("label", "unknown").lower() for m in test_meta]

# Load best checkpoint
if best_inst_path.exists():
    ckpt = torch.load(best_inst_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    print(f"[Checkpoint Loaded] Evaluating best weights from Epoch {ckpt.get('epoch', 'N/A')}")
model.eval()

# 1. Zero-Shot CLIP Baseline Evaluation
sim_clip = torch.matmul(test_t, test_v.t()).cpu().numpy()
h1_clip, h5_clip, h10_clip = 0, 0, 0
cat_clip_h1, cat_clip_h5 = 0, 0
rr_clip = 0.0

for i in range(n_test):
    ranks = np.argsort(-sim_clip[i])
    if ranks[0] == i: h1_clip += 1
    if i in ranks[:5]: h5_clip += 1
    if i in ranks[:10]: h10_clip += 1
    rank_pos = np.where(ranks == i)[0][0] + 1
    rr_clip += 1.0 / rank_pos
    if labels[ranks[0]] == labels[i]: cat_clip_h1 += 1
    if labels[i] in [labels[r] for r in ranks[:5]]: cat_clip_h5 += 1

# 2. Trained VL-JEPA Predictor Evaluation
with torch.no_grad():
    v_proj = model.encode_visual(test_v)
    t_proj = model.encode_text(test_t)
    sim_vljepa = torch.matmul(t_proj, v_proj.t()).cpu().numpy()

h1_vl, h5_vl, h10_vl = 0, 0, 0
cat_vl_h1, cat_vl_h5 = 0, 0
rr_vl = 0.0

for i in range(n_test):
    ranks = np.argsort(-sim_vljepa[i])
    if ranks[0] == i: h1_vl += 1
    if i in ranks[:5]: h5_vl += 1
    if i in ranks[:10]: h10_vl += 1
    rank_pos = np.where(ranks == i)[0][0] + 1
    rr_vl += 1.0 / rank_pos
    if labels[ranks[0]] == labels[i]: cat_vl_h1 += 1
    if labels[i] in [labels[r] for r in ranks[:5]]: cat_vl_h5 += 1

# Summary DataFrame
df_test = pd.DataFrame([
    {
        "Model": "Zero-Shot CLIP Baseline",
        "Candidates (N)": n_test,
        "Inst-R@1": f"{h1_clip / n_test * 100:.2f}%",
        "Inst-R@5": f"{h5_clip / n_test * 100:.2f}%",
        "Inst-R@10": f"{h10_clip / n_test * 100:.2f}%",
        "MRR": f"{rr_clip / n_test:.3f}",
        "Cat-R@1": f"{cat_clip_h1 / n_test * 100:.2f}%",
        "Cat-R@5": f"{cat_clip_h5 / n_test * 100:.2f}%"
    },
    {
        "Model": "VL-JEPA (Extended Run Trained)",
        "Candidates (N)": n_test,
        "Inst-R@1": f"{h1_vl / n_test * 100:.2f}%",
        "Inst-R@5": f"{h5_vl / n_test * 100:.2f}%",
        "Inst-R@10": f"{h10_vl / n_test * 100:.2f}%",
        "MRR": f"{rr_vl / n_test:.3f}",
        "Cat-R@1": f"{cat_vl_h1 / n_test * 100:.2f}%",
        "Cat-R@5": f"{cat_vl_h5 / n_test * 100:.2f}%"
    }
])

print("\\n" + "=" * 80)
print("[RESULTS] FINAL COMPARATIVE RETRIEVAL BENCHMARK ON UNSEEN TEST SPLIT")
print("=" * 80)
print(df_test.to_string(index=False))

# Export LaTeX table
latex_out = nu.OUTPUTS_DIR / "tables" / "vljepa_extended_run_test_results.tex"
with open(latex_out, "w", encoding="utf-8") as f:
    f.write(df_test.to_latex(index=False, caption="Unseen Test Split (N=797) Retrieval Performance after Extended VL-JEPA Training.", label="tab:extended_vljepa"))
print(f"\\n[Export] LaTeX table saved to: {latex_out.name}")
""")
    ]

    nb_data = {
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

    # Save to both target locations
    p1 = Path("CLIP_vs_VLJEPA_CCTV_Event_Retrieval_and_Hallucination_Analysis") / "Train_CLIP_VLJEPA_Extended_Run.ipynb"
    p2 = Path("Train_CLIP_VLJEPA_Extended_Run.ipynb")

    for p in [p1, p2]:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(nb_data, f, indent=1)
        print(f"Successfully generated: {p.resolve()} ({len(cells)} cells)")

if __name__ == "__main__":
    build_training_notebook()
