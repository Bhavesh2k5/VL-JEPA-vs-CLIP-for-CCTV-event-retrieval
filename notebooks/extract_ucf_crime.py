"""
Extracts CLIP visual embeddings and builds FAISS indices for the 1,000 UCF-Crime surveillance videos.
Saves:
- implementation/results/embeddings/clip_ucf_crime_embeddings.npy
- implementation/results/indices/clip_ucf_crime.faiss
- implementation/results/indices/clip_ucf_crime.meta.json
- Rebuilds implementation/results/indices/clip_unified.faiss and meta.json (5,153 vectors)
"""

import os
import json
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import faiss
from tqdm import tqdm
from transformers import CLIPModel, CLIPImageProcessor

def main():
    root_dataset = Path(r"C:\Users\HPCLAB\Desktop\capstone-ucf dataset")
    results_dir = Path(__file__).resolve().parent.parent / "implementation" / "results"
    emb_dir = results_dir / "embeddings"
    idx_dir = results_dir / "indices"
    emb_dir.mkdir(parents=True, exist_ok=True)
    idx_dir.mkdir(parents=True, exist_ok=True)

    print(f"[UCF-Crime Extractor] Scanning videos in: {root_dataset}")
    video_records = []
    
    # Ordered search for all 14 classes
    class_names = [
        "Abuse", "Arrest", "Arson", "Assault", "Burglary", 
        "Explosion", "Fighting", "Normal", "RoadAccidents", 
        "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism"
    ]

    for p in sorted(root_dataset.rglob("*")):
        if p.is_file() and p.suffix.lower() in (".mp4", ".avi"):
            label = "Unknown"
            if "Normal" in str(p):
                label = "Normal"
            else:
                for c in class_names:
                    if c.lower() in p.parent.name.lower() or c.lower() in p.name.lower():
                        label = c
                        break
            
            video_records.append({
                "path": str(p),
                "label": label,
                "dataset": "ucf_crime",
                "metadata": {
                    "category": label,
                    "filename": p.name,
                    "split": "ucf_crime"
                }
            })

    print(f"[UCF-Crime Extractor] Found {len(video_records)} videos across 14 crime classes.")
    
    # Class distribution check
    from collections import Counter
    counts = Counter([r["label"] for r in video_records])
    for k, v in sorted(counts.items()):
        print(f"  - {k}: {v} clips")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[UCF-Crime Extractor] Initializing CLIP Model on {device}...")
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    processor = CLIPImageProcessor.from_pretrained(model_id)

    def extract_single_video(vpath: str, num_frames: int = 8) -> np.ndarray:
        cap = cv2.VideoCapture(str(vpath))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return np.zeros((1, 512), dtype=np.float32)
        
        indices = np.linspace(0, max(0, total - 1), num_frames, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        
        if not frames:
            return np.zeros((1, 512), dtype=np.float32)
        
        inputs = processor(images=frames, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.get_image_features(**inputs)
            feats = out.pooler_output if hasattr(out, "pooler_output") else (out.image_embeds if hasattr(out, "image_embeds") else out)
            mean_feat = feats.mean(dim=0, keepdim=True)
            mean_feat = mean_feat / mean_feat.norm(dim=-1, keepdim=True)
            return mean_feat.cpu().numpy().astype(np.float32)

    embeddings = []
    print(f"[UCF-Crime Extractor] Extracting embeddings on {device}...")
    t0 = time.time()
    
    for i, rec in enumerate(tqdm(video_records, desc="Extracting UCF-Crime")):
        emb = extract_single_video(rec["path"], num_frames=8)
        embeddings.append(emb[0])

    embeddings_arr = np.array(embeddings, dtype=np.float32)
    elapsed = time.time() - t0
    print(f"[UCF-Crime Extractor] Successfully extracted {len(embeddings_arr)} embeddings in {elapsed:.2f}s ({len(embeddings_arr)/elapsed:.1f} clips/s)!")
    print(f"[UCF-Crime Extractor] Embedding matrix shape: {embeddings_arr.shape}")

    # Save UCF-Crime Embeddings & Metadata
    npy_save_path = emb_dir / "clip_ucf_crime_embeddings.npy"
    meta_save_path = idx_dir / "clip_ucf_crime.meta.json"
    faiss_save_path = idx_dir / "clip_ucf_crime.faiss"

    np.save(npy_save_path, embeddings_arr)
    print(f"[UCF-Crime Extractor] Saved embeddings to: {npy_save_path}")

    with open(meta_save_path, "w", encoding="utf-8") as f:
        json.dump({"metadata": video_records}, f, indent=2)
    print(f"[UCF-Crime Extractor] Saved metadata to: {meta_save_path}")

    # Build UCF-Crime FAISS index
    index = faiss.IndexFlatIP(embeddings_arr.shape[1])
    index.add(embeddings_arr)
    faiss.write_index(index, str(faiss_save_path))
    print(f"[UCF-Crime Extractor] Built FAISS Index: {faiss_save_path} ({index.ntotal} vectors)")

    # ---------------------------------------------------------
    # Rebuild Unified Index across all 4 Datasets
    # ---------------------------------------------------------
    print("\n[UCF-Crime Extractor] Rebuilding Unified 4-Dataset FAISS Index...")
    all_embs = []
    all_meta = []

    for ds in ["rlvs", "ucf_arg", "tinyvirat", "ucf_crime"]:
        e_path = emb_dir / f"clip_{ds}_embeddings.npy"
        m_path = idx_dir / f"clip_{ds}.meta.json"
        if e_path.exists() and m_path.exists():
            arr = np.load(e_path)
            with open(m_path, "r", encoding="utf-8") as f:
                meta = json.load(f).get("metadata", [])
            all_embs.append(arr)
            all_meta.extend(meta)
            print(f"  -> Added {ds.upper()}: {len(arr)} clips")

    stacked = np.vstack(all_embs)
    unified_faiss_path = idx_dir / "clip_unified.faiss"
    unified_meta_path = idx_dir / "clip_unified.meta.json"

    unified_idx = faiss.IndexFlatIP(stacked.shape[1])
    unified_idx.add(stacked)
    faiss.write_index(unified_idx, str(unified_faiss_path))

    with open(unified_meta_path, "w", encoding="utf-8") as f:
        json.dump({"metadata": all_meta}, f, indent=2)

    print(f"\n[UCF-Crime Extractor] SUCCESS! Unified Index created with {unified_idx.ntotal} vectors across 4 datasets!")
    print(f"  - FAISS index: {unified_faiss_path}")
    print(f"  - Metadata:    {unified_meta_path}")

if __name__ == "__main__":
    main()
