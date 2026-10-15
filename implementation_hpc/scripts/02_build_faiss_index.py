import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from retrieval import EmbeddingIndexer
from utils.common import ensure_dir, load_config, setup_logger

logger = setup_logger("build_faiss_index")


def build_faiss_index(model_name: str, dataset_name: str, config: dict):
    emb_dir = Path(config["retrieval"].get("embedding_save_dir", "results/embeddings"))
    idx_dir = ensure_dir(config["retrieval"].get("index_save_dir", "results/indices"))

    emb_file = emb_dir / f"{model_name}_{dataset_name}_embeddings.npy"
    meta_file = emb_dir / f"{model_name}_{dataset_name}_meta.json"

    if not emb_file.exists() or not meta_file.exists():
        logger.error("Embedding files missing for %s-%s. Run 01_extract_embeddings.py first.", model_name, dataset_name)
        return

    logger.info("Loading embeddings from %s", emb_file)
    embeddings = np.load(emb_file)
    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    dim = embeddings.shape[1]
    logger.info("Building FAISS IndexFlatIP (dim=%d, total=%d clips)...", dim, len(embeddings))

    indexer = EmbeddingIndexer(embedding_dim=dim, index_type="FlatIP")
    center = (dataset_name == "ucf_crime")
    indexer.add(embeddings, metadata, center=center)

    out_idx_path = idx_dir / f"{model_name}_{dataset_name}.faiss"
    indexer.save(out_idx_path)
    logger.info("FAISS index successfully saved to %s", out_idx_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS vector index from saved embeddings.")
    parser.add_argument("--model", type=str, default="clip", choices=["clip", "vljepa"])
    parser.add_argument("--dataset", type=str, default="rlvs", choices=["rlvs", "ucf_arg", "tinyvirat", "ucf_crime", "all"])
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    datasets = cfg["data"].get("active_datasets", ["rlvs", "ucf_arg", "tinyvirat", "ucf_crime"]) if args.dataset == "all" else [args.dataset]

    for d in datasets:
        build_faiss_index(args.model, d, cfg)
