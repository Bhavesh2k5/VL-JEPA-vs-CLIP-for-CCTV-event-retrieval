import math
from typing import Any, Dict, List, Set, Union


def recall_at_k(retrieved_labels: List[Any], positive_labels: Union[Set[Any], List[Any]], k: int) -> float:
    """Computes Hit@K (1.0 if any positive label exists in top-k, else 0.0)."""
    if not positive_labels:
        return 0.0
    pos_set = set(str(l).lower() for l in positive_labels)
    top_k_labels = [str(l).lower() for l in retrieved_labels[:k]]
    return 1.0 if any(l in pos_set for l in top_k_labels) else 0.0


def reciprocal_rank(retrieved_labels: List[Any], positive_labels: Union[Set[Any], List[Any]]) -> float:
    """Computes Reciprocal Rank (1 / first_rank_of_positive)."""
    if not positive_labels:
        return 0.0
    pos_set = set(str(l).lower() for l in positive_labels)
    for rank, label in enumerate(retrieved_labels, start=1):
        if str(label).lower() in pos_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_labels: List[Any], positive_labels: Union[Set[Any], List[Any]], k: int = 10) -> float:
    """Computes Normalized Discounted Cumulative Gain at rank K."""
    if not positive_labels:
        return 0.0
    pos_set = set(str(l).lower() for l in positive_labels)
    dcg = 0.0
    for i, label in enumerate(retrieved_labels[:k], start=1):
        rel = 1.0 if str(label).lower() in pos_set else 0.0
        dcg += rel / math.log2(i + 1)

    # Ideal DCG (at least one relevant item at rank 1)
    idcg = 1.0 / math.log2(2)
    return dcg / idcg if idcg > 0 else 0.0


def average_precision(retrieved_labels: List[Any], positive_labels: Union[Set[Any], List[Any]]) -> float:
    """Computes Average Precision for a single ranked list."""
    if not positive_labels:
        return 0.0
    pos_set = set(str(l).lower() for l in positive_labels)
    hits = 0
    sum_prec = 0.0
    for i, label in enumerate(retrieved_labels, start=1):
        if str(label).lower() in pos_set:
            hits += 1
            sum_prec += hits / i
    return sum_prec / max(1, hits) if hits > 0 else 0.0


def compute_all_retrieval_metrics(
    all_retrieved_labels: List[List[Any]],
    all_positive_labels: List[Union[Set[Any], List[Any]]],
    k_values: List[int] = [1, 5, 10],
) -> Dict[str, float]:
    """
    Computes averaged Recall@K, MRR, nDCG@10, and mAP across a set of evaluation queries.
    """
    n_queries = len(all_retrieved_labels)
    if n_queries == 0:
        return {}

    metrics = {f"Recall@{k}": 0.0 for k in k_values}
    metrics["MRR"] = 0.0
    metrics["nDCG@10"] = 0.0
    metrics["mAP"] = 0.0

    for ret_labels, pos_labels in zip(all_retrieved_labels, all_positive_labels):
        for k in k_values:
            metrics[f"Recall@{k}"] += recall_at_k(ret_labels, pos_labels, k)
        metrics["MRR"] += reciprocal_rank(ret_labels, pos_labels)
        metrics["nDCG@10"] += ndcg_at_k(ret_labels, pos_labels, k=10)
        metrics["mAP"] += average_precision(ret_labels, pos_labels)

    for k in metrics:
        metrics[k] = round(metrics[k] / n_queries, 4)

    return metrics
