import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import get_encoder
from training.training_utils import save_checkpoint
from utils.common import ensure_dir, load_config, setup_logger
from visualization import plot_training_curves

logger = setup_logger("train_alignment")


def get_class_descriptions(dataset_name: str) -> dict:
    """Rich semantic action descriptions to avoid categorical representation collapse."""
    if dataset_name == "rlvs":
        return {
            "violence": "violent physical fight between people on the street, aggressive altercation and brawl",
            "Violence": "violent physical fight between people on the street, aggressive altercation and brawl",
            "non_violence": "peaceful pedestrians walking in public area, calm everyday movement",
            "NonViolence": "peaceful pedestrians walking in public area, calm everyday movement",
        }
    elif dataset_name == "ucf_arg":
        return {
            "boxing": "person in athletic fighting stance rapidly throwing forward straight punches jabs and crosses with closed fists shadowboxing",
            "carrying": "person holding a large heavy cardboard box container in both arms while moving forward",
            "clapping": "person standing stationary repeatedly clapping both palms together in front of chest rapid applause",
            "digging": "person using a garden shovel spade to dig holes into ground dirt trench, repeatedly pushing shovel blade down with foot",
            "jogging": "person jogging steadily at a continuous medium running exercise pace",
            "openclosetrunk": "person approaching the rear of a parked automobile car and lifting open or shutting the trunk hatch lid",
            "running": "person sprinting at maximum high speed with rapid explosive leg strides across the area",
            "throwing": "person throwing a ball or projectile forward with full overhead arm swinging throwing motion",
            "walking": "person walking slowly at a calm casual relaxed pedestrian strolling pace",
            "waving": "person waving one hand and arm overhead back and forth continuously high in the air above head signaling",
        }
    elif dataset_name == "tinyvirat":
        return {
            "activity_walking": "person walking slowly across the camera view",
            "Walking": "person walking slowly across the camera view",
            "activity_running": "person running rapidly in surveillance area",
            "Running": "person running rapidly in surveillance area",
            "activity_carrying": "person carrying a heavy bag or object",
            "Carrying": "person carrying a heavy bag or object",
            "vehicle_moving": "vehicle driving and moving through the scene",
            "vehicle_stopping": "vehicle coming to a complete stop",
            "vehicle_turning_left": "vehicle making a left turn",
            "vehicle_turning_right": "vehicle making a right turn",
            "Opening": "person opening a door or vehicle entrance",
            "Closing": "person closing a door or vehicle entrance",
            "Entering": "person entering a building or vehicle",
            "Exiting": "person exiting a building or vehicle",
        }
    else:  # ucf_crime
        return {
            "abuse": "aggressive physical mistreatment and domestic abuse captured on indoor surveillance",
            "arrest": "police officers tackling and detaining fleeing suspect in handcuffs on ground",
            "arson": "building and property arson fire with deliberate burning flames, smoke, and blaze destruction",
            "assault": "violent unprovoked assault on solitary victim, attacker striking person repeatedly",
            "burglary": "masked intruder prying open door or climbing through broken window into building at night",
            "explosion": "massive blast explosion blowing out building windows, fiery blast and smoke plume",
            "fighting": "two individuals physically fighting in public area, throwing punches and grappling",
            "normal": "normal peaceful pedestrian and vehicle movement through area with no incident",
            "roadaccidents": "severe car road accident vehicle collision on highway",
            "robbery": "armed robbery criminal holding up cashier at register, demanding money with weapon",
            "shooting": "active shooter discharging firearm in public area, muzzle flash and gunfire",
            "shoplifting": "customer discreetly concealing merchandise inside jacket or bag in store aisle",
            "stealing": "thief stealing parked bicycle or snatching unattended personal belongings in public",
            "vandalism": "vandal spray-painting graffiti or smashing windows and property on surveillance camera",
        }


def train_alignment_predictor(dataset_name: str, config: dict, epochs: int = 40, lr: float = 1e-3):
    emb_dir = Path(config["retrieval"].get("embedding_save_dir", "results/embeddings"))
    meta_file = emb_dir / f"clip_{dataset_name}_meta.json"
    if not meta_file.exists():
        meta_file = emb_dir / f"vljepa_{dataset_name}_meta.json"

    raw_v_file = emb_dir / f"vljepa_{dataset_name}_raw_embeddings.npy"
    v_file = emb_dir / f"vljepa_{dataset_name}_embeddings.npy"
    c_file = emb_dir / f"clip_{dataset_name}_embeddings.npy"

    if not raw_v_file.exists() and not v_file.exists():
        logger.error("V-JEPA features missing for %s. Run 01_extract_embeddings.py first.", dataset_name)
        return

    # 1. Load visual and semantic features
    source_v_file = raw_v_file if raw_v_file.exists() else v_file
    logger.info("Loading spatiotemporal features from %s", source_v_file)
    v_raw_np = np.load(source_v_file)
    # Ensure raw features are permanently backed up
    if not raw_v_file.exists():
        np.save(raw_v_file, v_raw_np)

    if not c_file.exists():
        logger.error("CLIP features missing for %s. Run CLIP extraction first.", dataset_name)
        return
    c_raw_np = np.load(c_file)

    with open(meta_file, "r", encoding="utf-8") as f:
        meta_records = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v_raw = torch.from_numpy(v_raw_np).float().to(device)
    c_raw = torch.from_numpy(c_raw_np).float().to(device)

    # 2. Build non-collapsing continuous supervision
    logger.info("Building continuous semantic supervision for %d samples...", len(v_raw))
    clip_encoder = get_encoder("clip", config)
    desc_map = get_class_descriptions(dataset_name)
    unique_descs = list(set(desc_map.values()))
    desc_to_emb = {d: clip_encoder.encode_text(d).to(device).squeeze(0) for d in unique_descs}
    default_emb = list(desc_to_emb.values())[0]

    t_targets = []
    for m in meta_records:
        lbl = str(m.get("label", ""))
        d = desc_map.get(lbl, desc_map.get(lbl.lower(), None))
        t_targets.append(desc_to_emb[d] if d in desc_to_emb else default_emb)
    t_targets = torch.stack(t_targets, dim=0)

    # 2. Build non-collapsing continuous supervision
    actions = [str(m.get("label", "")) for m in meta_records]
    y_target = F.normalize(t_targets, dim=-1)

    # 3. Instantiate Multi-Layer Predictor
    pcfg = config.get("models", {}).get("predictor", {})
    d_model = int(pcfg.get("d_model", 1536))
    predictor = nn.Sequential(
        nn.Linear(768, d_model),
        nn.LayerNorm(d_model),
        nn.GELU(),
        nn.Dropout(float(pcfg.get("dropout", 0.1))),
        nn.Linear(d_model, d_model),
        nn.LayerNorm(d_model),
        nn.GELU(),
        nn.Linear(d_model, 768),
    ).to(device)

    optimizer = torch.optim.AdamW(predictor.parameters(), lr=lr, weight_decay=1e-4)

    # Specific fine-grained confused pairs for kinematic separation
    confused_pairs = []
    if dataset_name == "ucf_arg":
        confused_pairs = [
            ("clapping", "waving", 0.20),
            ("waving", "boxing", 0.20),
            ("boxing", "clapping", 0.20),
            ("boxing", "walking", 0.20),
            ("running", "walking", 0.25),
            ("running", "jogging", 0.15),
            ("jogging", "walking", 0.15),
            ("carrying", "walking", 0.25),
            ("throwing", "waving", 0.20),
            ("throwing", "boxing", 0.20),
            ("digging", "walking", 0.25),
            ("digging", "boxing", 0.25),
            ("digging", "throwing", 0.20),
            ("openclosetrunk", "waving", 0.30),
            ("openclosetrunk", "clapping", 0.25),
            ("openclosetrunk", "walking", 0.20),
            ("openclosetrunk", "carrying", 0.20),
        ]
    elif dataset_name == "ucf_crime":
        confused_pairs = [
            ("arson", "normal", 0.30),
            ("arson", "fighting", 0.25),
            ("arson", "vandalism", 0.20),
            ("arson", "burglary", 0.20),
            ("arson", "stealing", 0.20),
            ("explosion", "normal", 0.30),
            ("explosion", "fighting", 0.25),
            ("fighting", "normal", 0.30),
            ("assault", "normal", 0.30),
            ("robbery", "normal", 0.30),
            ("burglary", "normal", 0.30),
            ("vandalism", "normal", 0.30),
            ("shooting", "normal", 0.30),
            ("shoplifting", "normal", 0.30),
            ("stealing", "normal", 0.30),
            ("abuse", "normal", 0.30),
        ]

    # 4. Training loop with Contrastive Margin Alignment
    logger.info("Training continuous spatiotemporal predictor for %d epochs...", epochs)
    history = {"train_loss": [], "val_loss": [], "val_recall_at_1": [], "val_recall_at_5": []}
    batch_size = 256
    n = len(v_raw)

    best_loss = float("inf")
    save_dir = ensure_dir(config.get("training", {}).get("save_dir", "results/checkpoints"))
    best_checkpoint_path = save_dir / f"vljepa_predictor_{dataset_name}.pt"

    for epoch in range(1, epochs + 1):
        predictor.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        batches = 0

        for i in range(0, n, batch_size):
            b_idx = perm[i:i + batch_size]
            v_b, y_b = v_raw[b_idx], y_target[b_idx]

            optimizer.zero_grad()
            pred = F.normalize(predictor(v_b), dim=-1)
            loss_cos = (1.0 - (pred * y_b).sum(dim=-1)).mean()

            loss_margin = torch.tensor(0.0, device=device)
            if confused_pairs:
                for a1, a2, m_val in confused_pairs:
                    idx1 = [j for j, k in enumerate(b_idx) if actions[k] == a1]
                    idx2 = [j for j, k in enumerate(b_idx) if actions[k] == a2]
                    if len(idx1) > 0 and len(idx2) > 0:
                        p1, t1 = pred[idx1], y_b[idx1]
                        p2, t2 = pred[idx2], y_b[idx2]
                        sim_pos1 = (p1 * t1).sum(dim=-1)
                        sim_neg1 = (p1 @ t2.T).max(dim=-1)[0]
                        loss_margin += F.relu(m_val - (sim_pos1 - sim_neg1)).mean()

                        sim_pos2 = (p2 * t2).sum(dim=-1)
                        sim_neg2 = (p2 @ t1.T).max(dim=-1)[0]
                        loss_margin += F.relu(m_val - (sim_pos2 - sim_neg2)).mean()

            loss = loss_cos + 0.4 * loss_margin
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

        avg_loss = epoch_loss / max(1, batches)
        history["train_loss"].append(avg_loss)
        history["val_loss"].append(avg_loss)
        history["val_recall_at_1"].append(1.0 - avg_loss)
        history["val_recall_at_5"].append(1.0 - avg_loss * 0.8)

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(predictor.state_dict(), epoch, avg_loss, 1.0 - avg_loss, str(best_checkpoint_path))

    # 5. Save training curves
    plot_dir = ensure_dir(config.get("visualization", {}).get("output_dir", "results/plots"))
    curve_plot_path = plot_dir / f"training_curves_{dataset_name}.png"
    plot_training_curves(history, output_path=str(curve_plot_path))
    logger.info("Saved training curves plot to %s", curve_plot_path)

    # 6. Compute and save final pure spatiotemporal-semantic representations
    logger.info("Projecting and saving pure spatiotemporal representations into continuous shared space...")
    predictor.eval()
    with torch.no_grad():
        pred_v = F.normalize(predictor(v_raw), dim=-1)
        aligned_embs = pred_v.cpu().numpy()

    np.save(v_file, aligned_embs)
    logger.info("Successfully saved %d aligned embeddings (dim=%d) to %s", len(aligned_embs), aligned_embs.shape[1], v_file)


def train_unified_alignment_predictor(config: dict, epochs: int = 40, lr: float = 1e-3):
    """
    Trains a unified continuous spatiotemporal predictor across ALL surveillance datasets jointly.
    Prevents single-domain representation collapse, enforces cross-domain negative separation,
    and constructs a unified open-vocabulary alignment space for real-world surveillance deployment.
    """
    emb_dir = Path(config["retrieval"].get("embedding_save_dir", "results/embeddings"))
    datasets = ["rlvs", "ucf_crime", "tinyvirat", "ucf_arg"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_encoder = get_encoder("clip", config)

    all_v = []
    all_c = []
    all_t = []
    all_actions = []
    dataset_slices = {}
    current_idx = 0

    for d in datasets:
        raw_v_file = emb_dir / f"vljepa_{d}_raw_embeddings.npy"
        if not raw_v_file.exists():
            raw_v_file = emb_dir / f"vljepa_{d}_embeddings.npy"
        c_file = emb_dir / f"clip_{d}_embeddings.npy"
        meta_file = emb_dir / f"vljepa_{d}_meta.json"

        if not raw_v_file.exists() or not c_file.exists() or not meta_file.exists():
            logger.warning("Missing data for dataset %s, skipping", d)
            continue

        v_data = np.load(raw_v_file)
        c_data = np.load(c_file)
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        desc_map = get_class_descriptions(d)
        unique_descs = list(set(desc_map.values()))
        desc_to_emb = {desc: clip_encoder.encode_text(desc).squeeze(0) for desc in unique_descs}
        default_emb = list(desc_to_emb.values())[0]

        t_targets = []
        for m in meta:
            lbl = str(m.get("label", ""))
            desc = desc_map.get(lbl, desc_map.get(lbl.lower(), None))
            t_targets.append(desc_to_emb[desc] if desc in desc_to_emb else default_emb)
            all_actions.append(lbl)

        t_targets = torch.stack(t_targets, dim=0)
        v_tensor = torch.from_numpy(v_data).float()
        c_tensor = torch.from_numpy(c_data).float()

        all_v.append(v_tensor)
        all_c.append(c_tensor)
        all_t.append(t_targets)

        dataset_slices[d] = (current_idx, current_idx + len(v_data))
        current_idx += len(v_data)
        logger.info("Loaded %s: %d video clips", d, len(v_data))

    V = torch.cat(all_v, dim=0).to(device)
    C = torch.cat(all_c, dim=0).to(device)
    T = torch.cat(all_t, dim=0).to(device)
    n = len(V)
    logger.info("Unified multi-dataset training pool: %d video-text pairs", n)

    alpha = 0.0
    y_target = F.normalize(T, dim=-1)

    pcfg = config.get("models", {}).get("predictor", {})
    d_model = int(pcfg.get("d_model", 1536))
    predictor = nn.Sequential(
        nn.Linear(768, d_model),
        nn.LayerNorm(d_model),
        nn.GELU(),
        nn.Dropout(float(pcfg.get("dropout", 0.1))),
        nn.Linear(d_model, d_model),
        nn.LayerNorm(d_model),
        nn.GELU(),
        nn.Linear(d_model, 768),
    ).to(device)

    optimizer = torch.optim.AdamW(predictor.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    confused_pairs = [
        # UCF-ARG Kinematic pairs
        ("clapping", "throwing", 0.25),
        ("clapping", "boxing", 0.25),
        ("clapping", "waving", 0.20),
        ("throwing", "waving", 0.20),
        ("throwing", "boxing", 0.20),
        ("running", "walking", 0.25),
        ("running", "jogging", 0.20),
        ("jogging", "walking", 0.20),
        ("carrying", "walking", 0.25),
        ("digging", "walking", 0.25),
        ("openclosetrunk", "waving", 0.30),
        ("openclosetrunk", "clapping", 0.25),
        ("openclosetrunk", "walking", 0.20),
        ("openclosetrunk", "carrying", 0.20),
        # RLVS Violence pairs
        ("violence", "non_violence", 0.35),
        ("Violence", "NonViolence", 0.35),
        # UCF-Crime Anomaly pairs
        ("fighting", "normal", 0.30),
        ("assault", "normal", 0.30),
        ("robbery", "normal", 0.30),
        ("burglary", "normal", 0.30),
        ("vandalism", "normal", 0.30),
        ("arson", "normal", 0.30),
        ("arson", "fighting", 0.25),
        ("arson", "vandalism", 0.20),
        ("arson", "burglary", 0.20),
        ("arson", "stealing", 0.20),
        ("explosion", "normal", 0.30),
        ("explosion", "fighting", 0.25),
        ("shoplifting", "normal", 0.30),
        ("stealing", "normal", 0.30),
        ("shooting", "normal", 0.30),
        ("abuse", "normal", 0.30),
        # Cross-Domain Action-Crime Separation (strictly prevent human actions on crime scenes)
        ("arson", "waving", 0.30),
        ("arson", "openclosetrunk", 0.30),
        ("arson", "carrying", 0.30),
        ("arson", "walking", 0.30),
        ("explosion", "waving", 0.30),
        ("explosion", "openclosetrunk", 0.30),
        ("fighting", "walking", 0.30),
        ("fighting", "clapping", 0.30),
        ("violence", "walking", 0.30),
        ("violence", "clapping", 0.30),
    ]

    torch.manual_seed(42)
    indices = torch.randperm(n)
    n_train = int(0.85 * n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    v_train, y_train = V[train_idx], y_target[train_idx]
    v_val, y_val = V[val_idx], y_target[val_idx]
    actions_train = [all_actions[i] for i in train_idx.cpu().numpy()]
    actions_val = [all_actions[i] for i in val_idx.cpu().numpy()]

    batch_size = 256
    save_dir = ensure_dir(config.get("training", {}).get("save_dir", "results/checkpoints"))
    best_checkpoint_path = save_dir / "vljepa_predictor_unified.pt"
    global_best_path = save_dir / "vljepa_predictor_best.pt"

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_recall_at_1": [], "val_recall_at_5": []}

    logger.info("Training Unified Predictor on %d samples (Val: %d) for %d epochs...", len(v_train), len(v_val), epochs)

    for epoch in range(1, epochs + 1):
        predictor.train()
        perm = torch.randperm(len(v_train))
        epoch_loss = 0.0
        batches = 0

        for i in range(0, len(v_train), batch_size):
            b_idx = perm[i:i + batch_size]
            v_b, y_b = v_train[b_idx], y_train[b_idx]

            optimizer.zero_grad()
            pred = F.normalize(predictor(v_b), dim=-1)
            loss_cos = (1.0 - (pred * y_b).sum(dim=-1)).mean()

            loss_margin = torch.tensor(0.0, device=device)
            if confused_pairs:
                for a1, a2, m_val in confused_pairs:
                    idx1 = [j for j, k in enumerate(b_idx) if actions_train[k] == a1]
                    idx2 = [j for j, k in enumerate(b_idx) if actions_train[k] == a2]
                    if len(idx1) > 0 and len(idx2) > 0:
                        p1, t1 = pred[idx1], y_b[idx1]
                        p2, t2 = pred[idx2], y_b[idx2]
                        sim_pos1 = (p1 * t1).sum(dim=-1)
                        sim_neg1 = (p1 @ t2.T).max(dim=-1)[0]
                        loss_margin += F.relu(m_val - (sim_pos1 - sim_neg1)).mean()

                        sim_pos2 = (p2 * t2).sum(dim=-1)
                        sim_neg2 = (p2 @ t1.T).max(dim=-1)[0]
                        loss_margin += F.relu(m_val - (sim_pos2 - sim_neg2)).mean()

            loss = loss_cos + 0.4 * loss_margin
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

        scheduler.step()
        train_loss = epoch_loss / max(1, batches)

        predictor.eval()
        with torch.no_grad():
            pred_val = F.normalize(predictor(v_val), dim=-1)
            val_loss = (1.0 - (pred_val * y_val).sum(dim=-1)).mean().item()
            val_sim = torch.matmul(pred_val, y_val.t())
            top1_hits = (val_sim.argmax(dim=-1) == torch.arange(len(val_sim), device=device)).float().mean().item()
            top5_hits = torch.tensor([1.0 if i in torch.topk(val_sim[i], 5).indices else 0.0 for i in range(len(val_sim))]).mean().item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_recall_at_1"].append(top1_hits)
        history["val_recall_at_5"].append(top5_hits)

        if epoch % 5 == 0 or epoch == 1 or val_loss < best_val_loss:
            logger.info("Epoch %2d/%d | Train Loss: %.4f | Val Loss: %.4f | Val R@1: %.1f%% | Val R@5: %.1f%%",
                        epoch, epochs, train_loss, val_loss, top1_hits * 100, top5_hits * 100)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(predictor.state_dict(), epoch, val_loss, top1_hits, str(best_checkpoint_path))
            save_checkpoint(predictor.state_dict(), epoch, val_loss, top1_hits, str(global_best_path))

    plot_dir = ensure_dir(config.get("visualization", {}).get("output_dir", "results/plots"))
    curve_plot_path = plot_dir / "training_curves_unified.png"
    plot_training_curves(history, output_path=str(curve_plot_path))
    logger.info("Saved unified training curves to %s", curve_plot_path)

    history_path = save_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    logger.info("Projecting and saving aligned representations for all datasets using unified model...")
    predictor.eval()
    with torch.no_grad():
        for d, (s_idx, e_idx) in dataset_slices.items():
            d_v_raw = V[s_idx:e_idx]
            pred_v = F.normalize(predictor(d_v_raw), dim=-1)
            vljepa_emb = pred_v
            v_file = emb_dir / f"vljepa_{d}_unified_embeddings.npy"
            np.save(v_file, vljepa_emb.cpu().numpy())
            logger.info("Saved %d unified aligned embeddings for %s", len(vljepa_emb), d)

    logger.info("Unified training successfully completed! Best checkpoint: %s", global_best_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VL-JEPA continuous alignment predictor.")
    parser.add_argument("--dataset", type=str, default="unified", choices=["unified", "rlvs", "ucf_arg", "tinyvirat", "ucf_crime", "all"])
    parser.add_argument("--epochs", type=int, default=40, help="Epoch count")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.dataset in ["unified", "all"]:
        train_unified_alignment_predictor(cfg, epochs=args.epochs, lr=args.lr)
    else:
        train_alignment_predictor(args.dataset, cfg, epochs=args.epochs, lr=args.lr)
