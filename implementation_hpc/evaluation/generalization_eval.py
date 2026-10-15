import logging
from typing import Any, Dict, List

import numpy as np

from retrieval.embedding_indexer import EmbeddingIndexer
from retrieval.retrieval_utils import l2_normalize_np

logger = logging.getLogger(__name__)

VIEWPOINTS = ["aerial", "rooftop", "ground"]


def evaluate_cross_viewpoint_matrix(
    embeddings: np.ndarray,
    metadata: List[Dict[str, Any]],
    k: int = 5,
) -> Dict[str, Any]:
    """
    Evaluates visual-to-visual cross-camera generalization across UCF-ARG viewpoints (H2).
    Constructs a 3x3 Recall@K matrix across [aerial, rooftop, ground].
    """
    embs_norm = l2_normalize_np(embeddings)
    vp_indices = {vp: [] for vp in VIEWPOINTS}

    for i, meta in enumerate(metadata):
        vp_val = meta.get("metadata", {}).get("viewpoint", meta.get("viewpoint", "")).lower()
        for target_vp in VIEWPOINTS:
            if target_vp in vp_val:
                vp_indices[target_vp].append(i)
                break

    matrix = np.zeros((len(VIEWPOINTS), len(VIEWPOINTS)), dtype=float)

    for i, gallery_vp in enumerate(VIEWPOINTS):
        g_idxs = vp_indices[gallery_vp]
        if not g_idxs:
            continue

        gallery_embs = embs_norm[g_idxs]
        gallery_meta = [metadata[idx] for idx in g_idxs]

        indexer = EmbeddingIndexer(embedding_dim=embs_norm.shape[1])
        indexer.add(gallery_embs, gallery_meta)

        for j, probe_vp in enumerate(VIEWPOINTS):
            p_idxs = vp_indices[probe_vp]
            if not p_idxs:
                continue

            probe_embs = embs_norm[p_idxs]
            probe_meta = [metadata[idx] for idx in p_idxs]

            _, retrieved_records = indexer.search(probe_embs, top_k=k)

            hits = 0
            for p_m, ret_list in zip(probe_meta, retrieved_records):
                p_label = p_m.get("label", "").lower()
                ret_labels = [r.get("label", "").lower() for r in ret_list]
                # Exclude self-match if probe and gallery are same viewpoint
                if probe_vp == gallery_vp and len(ret_labels) > 1:
                    ret_labels = ret_labels[1:]  # Rank 2 onwards

                if p_label in ret_labels:
                    hits += 1

            recall = hits / max(1, len(probe_meta))
            matrix[i, j] = round(recall, 4)

    return {
        "viewpoints": VIEWPOINTS,
        "matrix": matrix.tolist(),
        "mean_cross_view_recall": round(float(np.mean(matrix[~np.eye(len(VIEWPOINTS), dtype=bool)])), 4),
    }
