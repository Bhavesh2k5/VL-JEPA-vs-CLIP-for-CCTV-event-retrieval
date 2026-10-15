import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import build_dataset, collate_video_batch
from models import get_encoder
from utils.common import clear_gpu_cache, ensure_dir, load_config, setup_logger

logger = setup_logger("extract_embeddings")


def extract_embeddings(
    model_name: str,
    dataset_name: str,
    config: dict,
    limit: int = None,
    batch_size: int = None,
    output_dir: str = "results/embeddings",
):
    output_path = ensure_dir(output_dir)
    emb_file = output_path / f"{model_name}_{dataset_name}_embeddings.npy"
    meta_file = output_path / f"{model_name}_{dataset_name}_meta.json"

    # 1. Initialize dataset loader
    logger.info("Building dataset: %s", dataset_name)
    dataset = build_dataset(dataset_name, config)
    if len(dataset) == 0:
        logger.warning("Dataset %s has 0 clips. Check data directory.", dataset_name)
        return

    total_clips = min(len(dataset), limit) if limit else len(dataset)
    indices = list(range(total_clips))

    # 2. Check for existing partial extractions (Resume Support)
    existing_embs = []
    existing_meta = []
    if emb_file.exists() and meta_file.exists():
        try:
            existing_embs = list(np.load(emb_file))
            with open(meta_file, "r", encoding="utf-8") as f:
                existing_meta = json.load(f)
            logger.info("Found existing checkpoint with %d clips. Resuming...", len(existing_embs))
            if len(existing_embs) >= total_clips:
                logger.info("All %d clips already extracted. Skipping.", total_clips)
                return
            indices = indices[len(existing_embs):]
        except Exception as e:
            logger.warning("Failed to load existing checkpoint: %s. Starting fresh.", e)
            existing_embs = []
            existing_meta = []

    # 3. Initialize encoder
    logger.info("Loading model: %s", model_name)
    encoder = get_encoder(model_name, config)

    # 4. Resolve batch size
    if batch_size is None:
        batch_size = config["models"].get(model_name, {}).get("batch_size", 16)

    # Subset dataset
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(config.get("system", {}).get("num_workers", 2)),
        collate_fn=collate_video_batch,
        pin_memory=torch.cuda.is_available(),
    )

    all_embs = list(existing_embs)
    all_meta = list(existing_meta)
    save_interval = 100  # Save to disk every 100 clips to avoid thrashing

    logger.info("Extracting embeddings for %d clips (batch_size=%d)...", len(indices), batch_size)
    pbar = tqdm(total=total_clips, initial=len(all_embs), desc=f"{model_name}-{dataset_name}")

    for batch in loader:
        if not batch or "video" not in batch:
            continue

        videos = batch["video"]  # [B, T, C, H, W]
        b_paths = batch["path"]
        b_labels = batch["label"]
        b_meta = batch["metadata"]

        try:
            with torch.no_grad():
                embs = encoder.encode_video(videos)  # [B, D]
        except torch.cuda.OutOfMemoryError:
            logger.warning("CUDA OOM encountered. Halving batch size and retrying...")
            clear_gpu_cache()
            half = len(videos) // 2
            with torch.no_grad():
                embs1 = encoder.encode_video(videos[:half])
                embs2 = encoder.encode_video(videos[half:])
                embs = torch.cat([embs1, embs2], dim=0)

        embs_np = embs.cpu().numpy()
        for idx in range(len(embs_np)):
            all_embs.append(embs_np[idx])
            all_meta.append({
                "path": b_paths[idx],
                "label": b_labels[idx],
                "metadata": b_meta[idx] if idx < len(b_meta) else {},
            })

        pbar.update(len(embs_np))

        # Periodic disk checkpoint
        if len(all_embs) % save_interval == 0:
            np.save(emb_file, np.array(all_embs, dtype=np.float32))
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(all_meta, f)

    pbar.close()

    # Final persistent save
    np.save(emb_file, np.array(all_embs, dtype=np.float32))
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, indent=2)

    logger.info("Extracted and saved %d embeddings to %s", len(all_embs), emb_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and cache video embeddings.")
    parser.add_argument("--model", type=str, default="clip", choices=["clip", "vljepa"], help="Model backbone")
    parser.add_argument("--dataset", type=str, default="rlvs", choices=["rlvs", "ucf_arg", "tinyvirat", "ucf_crime", "all"])
    parser.add_argument("--limit", type=int, default=None, help="Clip count limit")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size override")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    datasets = cfg["data"].get("active_datasets", ["rlvs", "ucf_arg", "tinyvirat", "ucf_crime"]) if args.dataset == "all" else [args.dataset]

    for d in datasets:
        extract_embeddings(args.model, d, cfg, limit=args.limit, batch_size=args.batch_size)
