import numpy as np
import pytest

from retrieval import EmbeddingIndexer


def test_faiss_indexer_roundtrip(tmp_path):
    """Verifies that FAISS IndexFlatIP saves and loads accurately with metadata intact."""
    dim = 64
    n_samples = 20

    embs = np.random.randn(n_samples, dim).astype(np.float32)
    meta = [{"id": i, "label": f"cls_{i % 3}"} for i in range(n_samples)]

    indexer = EmbeddingIndexer(embedding_dim=dim)
    indexer.add(embs, meta)

    # Search query
    query = np.random.randn(1, dim).astype(np.float32)
    scores, results = indexer.search(query, top_k=3)

    assert len(scores[0]) == 3
    assert len(results[0]) == 3
    assert scores[0][0] <= 1.0001  # Cosine similarity upper bound

    # Save & reload
    save_file = tmp_path / "test_index.faiss"
    indexer.save(save_file)
    assert save_file.exists()
    assert save_file.with_suffix(".meta.json").exists()

    loaded = EmbeddingIndexer.load(save_file)
    assert loaded.index.ntotal == n_samples

    l_scores, l_results = loaded.search(query, top_k=3)
    np.testing.assert_allclose(scores, l_scores, atol=1e-5)
