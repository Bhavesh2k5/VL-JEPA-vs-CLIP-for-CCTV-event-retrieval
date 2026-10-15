"""
Extracts embeddings for a balanced, representative subset of TinyVIRAT-v2 videos,
focusing heavily on objects, vehicles, and complex activities.
Saves embeddings and builds FAISS index for high-speed surveillance retrieval.
"""

import os
import json
import time
import numpy as np
import cv2
import torch
from pathlib import Path
from collections import defaultdict
from transformers import CLIPModel, CLIPProcessor
import faiss

WORKSPACE_ROOT = Path(r"c:\Users\HPCLAB\Desktop\VLJEPA\capestone")
TINY_DIR = Path(r"C:\Users\HPCLAB\Desktop\capstone-core datset\capstone_tinyvirat_v2\TinyVIRAT-v2\TinyVIRAT_V2")
VAL_JSON = TINY_DIR / "tiny_val_v2.json"
TRAIN_JSON = TINY_DIR / "tiny_train_v2.json"
VIDEOS_DIR = TINY_DIR / "videos"

OUTPUT_EMB_DIR = WORKSPACE_ROOT / "implementation" / "results" / "embeddings"
OUTPUT_IDX_DIR = WORKSPACE_ROOT / "implementation" / "results" / "indices"
OUTPUT_EMB_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_IDX_DIR.mkdir(parents=True, exist_ok=True)

def sample_video_frames(video_path, num_frames=8):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None
    indices = np.linspace(0, max(0, total_frames - 1), num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        return None
    return frames

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    print("Loading TinyVIRAT annotations...")
    with open(VAL_JSON, "r", encoding="utf-8") as f:
        val_entries = json.load(f)
    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        train_entries = json.load(f)

    # Priority classes: vehicles, carrying, objects, tools, interactions
    priority_classes = [
        "vehicle_moving", "vehicle_stopping", "vehicle_turning_left", "vehicle_turning_right", "vehicle_starting",
        "activity_carrying", "Transport_HeavyCarry", "Loading", "Pull", "Push", "Opening", "Closing",
        "specialized_using_tool", "Interacts", "activity_walking", "activity_running", "activity_standing",
        "Talking", "specialized_talking_phone", "specialized_texting_phone"
    ]

    selected_samples = []
    class_counts = defaultdict(int)
    max_per_class = 40  # balanced representation across classes

    # First draw from val
    for entry in val_entries:
        labels = entry.get("label", [])
        p = VIDEOS_DIR / "val" / entry["path"]
        if not p.exists():
            continue
        for lbl in labels:
            if lbl in priority_classes and class_counts[lbl] < max_per_class:
                class_counts[lbl] += 1
                selected_samples.append({
                    "path": str(p),
                    "label": lbl,
                    "all_labels": labels,
                    "split": "val",
                    "dim": entry.get("dim", [])
                })
                break

    # Then draw from train to fill any remaining quotas
    for entry in train_entries:
        labels = entry.get("label", [])
        p = VIDEOS_DIR / "train" / entry["path"]
        if not p.exists():
            continue
        for lbl in labels:
            if lbl in priority_classes and class_counts[lbl] < max_per_class:
                class_counts[lbl] += 1
                selected_samples.append({
                    "path": str(p),
                    "label": lbl,
                    "all_labels": labels,
                    "split": "train",
                    "dim": entry.get("dim", [])
                })
                break

    print(f"Selected {len(selected_samples)} representative TinyVIRAT surveillance clips across {len(class_counts)} classes.")
    for lbl, cnt in sorted(class_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  - {lbl}: {cnt} clips")

    # Load CLIP
    model_id = "openai/clip-vit-base-patch32"
    print(f"Loading CLIP model ({model_id})...")
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    model.eval()

    embeddings = []
    metadata = []
    t0 = time.time()

    print(f"Extracting spatiotemporal video features for {len(selected_samples)} clips on GPU...")
    batch_size = 16
    for i in range(0, len(selected_samples), batch_size):
        batch = selected_samples[i:i+batch_size]
        batch_frames = []
        valid_items = []
        for item in batch:
            frames = sample_video_frames(item["path"], num_frames=8)
            if frames:
                batch_frames.append(frames)
                valid_items.append(item)

        if not batch_frames:
            continue

        # Process each clip's frames through CLIP vision encoder
        with torch.no_grad():
            for clip_f, item in zip(batch_frames, valid_items):
                inputs = processor(images=clip_f, return_tensors="pt").to(device)
                v_out = model.get_image_features(**inputs)
                if hasattr(v_out, "pooler_output"):
                    v_out = v_out.pooler_output
                elif hasattr(v_out, "image_embeds"):
                    v_out = v_out.image_embeds
                
                # Mean pool across the 8 frames and L2 normalize
                clip_emb = v_out.mean(dim=0, keepdim=True)
                clip_emb = clip_emb / clip_emb.norm(dim=-1, keepdim=True)
                emb_np = clip_emb.cpu().numpy().astype(np.float32)[0]

                embeddings.append(emb_np)
                metadata.append({
                    "path": item["path"],
                    "label": item["label"],
                    "dataset": "tinyvirat",
                    "metadata": {
                        "all_labels": item["all_labels"],
                        "split": item["split"],
                        "category": "vehicle" if "vehicle" in item["label"] else ("object" if "carry" in item["label"] or "Load" in item["label"] else "action")
                    }
                })

        if (i // batch_size) % 5 == 0:
            print(f"  Processed {min(i + batch_size, len(selected_samples))}/{len(selected_samples)} clips...")

    emb_arr = np.array(embeddings, dtype=np.float32)
    elapsed = time.time() - t0
    print(f"Extraction completed in {elapsed:.2f}s ({len(emb_arr)} clips, {len(emb_arr)/max(1, elapsed):.1f} clips/s)!")

    # Save embeddings and metadata
    emb_file = OUTPUT_EMB_DIR / "clip_tinyvirat_embeddings.npy"
    meta_file = OUTPUT_EMB_DIR / "clip_tinyvirat_embeddings.json"
    np.save(emb_file, emb_arr)
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump({"model": "clip", "dataset": "tinyvirat", "dim": int(emb_arr.shape[1]), "count": len(metadata), "records": metadata}, f, indent=2)

    # Build FAISS index
    index = faiss.IndexFlatIP(emb_arr.shape[1])
    index.add(emb_arr)
    idx_file = OUTPUT_IDX_DIR / "clip_tinyvirat.faiss"
    idx_meta_file = OUTPUT_IDX_DIR / "clip_tinyvirat.meta.json"
    faiss.write_index(index, str(idx_file))
    with open(idx_meta_file, "w", encoding="utf-8") as f:
        json.dump({"dim": int(emb_arr.shape[1]), "ntotal": index.ntotal, "metadata": metadata}, f, indent=2)

    print(f"Successfully saved TinyVIRAT FAISS index ({index.ntotal} clips) to {idx_file}!")

if __name__ == "__main__":
    main()
