import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import faiss
import numpy as np

from .retrieval_utils import as_float32, l2_normalize_np

logger = logging.getLogger(__name__)


class EmbeddingIndexer:
    """
    High-performance FAISS Vector Indexer for implementation_hpc.
    Uses IndexFlatIP on L2-normalized embeddings for exact cosine similarity search,
    persisting companion JSON metadata sidecars.
    """

    def __init__(self, embedding_dim: int, index_type: str = "FlatIP"):
        self.embedding_dim = embedding_dim
        self.index_type = index_type

        # Inner Product on unit vectors equals Cosine Similarity
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.metadata: List[Dict[str, Any]] = []
        self.mean_vector: Optional[np.ndarray] = None

    def add(self, embeddings: Union[np.ndarray, list], metadata: List[Dict[str, Any]], center: bool = False) -> None:
        """Adds normalized vector embeddings and corresponding metadata records."""
        embs_np = as_float32(embeddings)
        if center:
            self.mean_vector = embs_np.mean(axis=0, keepdims=True)
            embs_np = embs_np - self.mean_vector
        embs_norm = l2_normalize_np(embs_np)

        if embs_norm.shape[1] != self.embedding_dim:
            raise ValueError(f"Vector dimension mismatch: expected {self.embedding_dim}, got {embs_norm.shape[1]}")

        self.index.add(embs_norm)
        self.metadata.extend(metadata)
        logger.info("Added %d embeddings to FAISS index (Total: %d, Centered: %s)", len(embs_norm), self.index.ntotal, center)

    def search(self, query_embeddings: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, List[List[Dict[str, Any]]]]:
        """
        Searches top-k nearest video clips for one or more query vectors.
        Returns: (scores [Q, K], retrieved_metadata_records [Q, K])
        """
        if self.index.ntotal == 0:
            return np.zeros((len(query_embeddings), 0)), [[] for _ in query_embeddings]

        q = as_float32(query_embeddings)
        if self.mean_vector is not None:
            q = q - self.mean_vector
        q_norm = l2_normalize_np(q)
        scores, indices = self.index.search(q_norm, min(top_k, self.index.ntotal))

        results_meta = []
        for row_indices in indices:
            row_meta = []
            for idx in row_indices:
                if 0 <= idx < len(self.metadata):
                    row_meta.append(self.metadata[idx])
                else:
                    row_meta.append({"error": "invalid_index"})
            results_meta.append(row_meta)

        return scores, results_meta

    def save(self, index_file_path: Union[str, Path]) -> None:
        """Saves FAISS index (.faiss) and JSON sidecar metadata (.meta.json)."""
        idx_path = Path(index_file_path)
        idx_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(idx_path))

        meta_path = idx_path.with_suffix(".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "embedding_dim": self.embedding_dim,
                    "index_type": self.index_type,
                    "total_samples": self.index.ntotal,
                    "mean_vector": self.mean_vector.tolist() if self.mean_vector is not None else None,
                    "records": self.metadata,
                },
                f,
                indent=2,
            )
        logger.info("Saved FAISS index to %s and metadata to %s", idx_path, meta_path)

    @classmethod
    def load(cls, index_file_path: Union[str, Path]) -> "EmbeddingIndexer":
        """Loads FAISS index from disk alongside sidecar metadata."""
        idx_path = Path(index_file_path)
        if not idx_path.exists():
            raise FileNotFoundError(f"FAISS index file not found: {idx_path}")

        meta_path = idx_path.with_suffix(".meta.json")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata sidecar file not found: {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)

        indexer = cls(
            embedding_dim=meta_data["embedding_dim"],
            index_type=meta_data.get("index_type", "FlatIP"),
        )
        indexer.index = faiss.read_index(str(idx_path))
        indexer.metadata = meta_data.get("records", [])
        if meta_data.get("mean_vector") is not None:
            indexer.mean_vector = np.array(meta_data["mean_vector"], dtype=np.float32)

        logger.info("Loaded FAISS index with %d items from %s (Centered: %s)", indexer.index.ntotal, idx_path, indexer.mean_vector is not None)
        return indexer
