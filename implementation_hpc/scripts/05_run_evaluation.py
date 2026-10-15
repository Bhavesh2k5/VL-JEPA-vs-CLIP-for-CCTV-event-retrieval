import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch

from data import build_dataset, get_queries_for_dataset
from evaluation import (
    compute_all_retrieval_metrics,
    evaluate_cross_viewpoint_matrix,
    evaluate_faithfulness,
    measure_encoding_latency,
    measure_memory_usage,
    measure_throughput,
)
from models import get_encoder
from reporting import IncidentReporter
from retrieval import CCTVRetriever, EmbeddingIndexer
from utils.common import ensure_dir, load_config, setup_logger

logger = setup_logger("run_evaluation")


def run_full_evaluation(
    model_name: str,
    dataset_name: str,
    config: Dict[str, Any],
    predictor_ckpt: str = None,
):
    idx_dir = Path(config["retrieval"].get("index_save_dir", "results/indices"))
    idx_path = idx_dir / f"{model_name}_{dataset_name}.faiss"
    metrics_dir = ensure_dir(config.get("evaluation", {}).get("output_dir", "results/metrics"))

    if not idx_path.exists():
        logger.error("FAISS index missing: %s. Run indexing first.", idx_path)
        return

    # Check for predictor checkpoint (dataset-specific first, then global best)
    if model_name == "vljepa" and predictor_ckpt is None:
        save_dir = Path(config.get("training", {}).get("save_dir", "results/checkpoints"))
        dataset_ckpt = save_dir / f"vljepa_predictor_{dataset_name}.pt"
        default_ckpt = save_dir / "vljepa_predictor_best.pt"
        if dataset_ckpt.exists():
            predictor_ckpt = str(dataset_ckpt)
        elif default_ckpt.exists():
            predictor_ckpt = str(default_ckpt)

    encoder = get_encoder(model_name, config, predictor_checkpoint=predictor_ckpt)
    indexer = EmbeddingIndexer.load(idx_path)
    retriever = CCTVRetriever(encoder, indexer)

    # 1. Retrieval Evaluation
    logger.info("Evaluating Semantic Retrieval on %s...", dataset_name)
    eval_queries = get_queries_for_dataset(dataset_name)

    query_texts = [q["query"] for q in eval_queries]
    batch_results = retriever.batch_retrieve(query_texts, top_k=10)

    all_retrieved_labels = []
    all_positive_labels = []
    for res_list, q_item in zip(batch_results, eval_queries):
        all_retrieved_labels.append([r["label"] for r in res_list])
        all_positive_labels.append(q_item["positive_classes"])

    retrieval_metrics = compute_all_retrieval_metrics(all_retrieved_labels, all_positive_labels)
    logger.info("Retrieval Metrics: %s", retrieval_metrics)

    # 2. Efficiency Evaluation
    logger.info("Profiling Computational Efficiency on GPU...")
    # Use synthetic dummy frames to profile pure forward pass
    dummy_clip = torch.rand(4, 16, 3, 224, 224)
    latency_stats = measure_encoding_latency(encoder, dummy_clip, n_runs=int(config.get("evaluation", {}).get("latency_runs", 30)))
    memory_stats = measure_memory_usage(encoder, dummy_clip[:1])
    throughput_val = measure_throughput(encoder, dummy_clip, n_repeats=5)

    efficiency_metrics = {
        **latency_stats,
        **memory_stats,
        "throughput_clips_per_sec": throughput_val,
    }
    logger.info("Efficiency Metrics: %s", efficiency_metrics)

    # 3. Faithfulness Evaluation (H3)
    logger.info("Evaluating Incident Report Faithfulness (Grounded vs. Ungrounded)...")
    reporter = IncidentReporter(config)
    sample_query = eval_queries[0]["query"]
    sample_evidence = batch_results[0][:5]

    grounded_report = reporter.generate_report(sample_query, sample_evidence, grounded=True)
    ungrounded_report = reporter.generate_report(sample_query, sample_evidence, grounded=False)

    faith_grounded = evaluate_faithfulness(grounded_report["report_text"], grounded_report["evidence_summary"])
    faith_ungrounded = evaluate_faithfulness(ungrounded_report["report_text"], "No evidence provided.")

    logger.info("Faithfulness Grounded Score: %.3f | Ungrounded: %.3f", faith_grounded["grounding_score"], faith_ungrounded["grounding_score"])

    # 4. Cross-Camera Viewpoint Generalization (H2) on UCF-ARG
    generalization_results = {}
    if dataset_name == "ucf_arg":
        logger.info("Evaluating Cross-Camera Generalization on UCF-ARG...")
        emb_file = Path(config["retrieval"].get("embedding_save_dir", "results/embeddings")) / f"{model_name}_ucf_arg_embeddings.npy"
        if emb_file.exists():
            embs = np.load(emb_file)
            generalization_results = evaluate_cross_viewpoint_matrix(embs, indexer.metadata, k=5)
            logger.info("Cross-Camera Mean Recall@5: %.4f", generalization_results.get("mean_cross_view_recall", 0.0))

    # Consolidate and export metrics
    full_report = {
        "model": model_name,
        "dataset": dataset_name,
        "retrieval": retrieval_metrics,
        "efficiency": efficiency_metrics,
        "faithfulness": {
            "grounded": faith_grounded,
            "ungrounded": faith_ungrounded,
        },
        "generalization": generalization_results,
    }

    out_json = metrics_dir / f"{model_name}_{dataset_name}_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    # Export flat CSV summary
    flat_summary = {
        "Model": model_name,
        "Dataset": dataset_name,
        **retrieval_metrics,
        **efficiency_metrics,
        "Grounded_Score": faith_grounded["grounding_score"],
        "Ungrounded_Score": faith_ungrounded["grounding_score"],
    }
    df = pd.DataFrame([flat_summary])
    out_csv = metrics_dir / f"{model_name}_{dataset_name}_metrics.csv"
    df.to_csv(out_csv, index=False)

    logger.info("Exported full metrics to %s and %s", out_json, out_csv)
    return full_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete evaluation benchmark.")
    parser.add_argument("--model", type=str, default="clip", choices=["clip", "vljepa"])
    parser.add_argument("--dataset", type=str, default="rlvs", choices=["rlvs", "ucf_arg", "tinyvirat", "ucf_crime", "all"])
    parser.add_argument("--predictor-ckpt", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    datasets = cfg["data"].get("active_datasets", ["rlvs", "ucf_arg", "tinyvirat", "ucf_crime"]) if args.dataset == "all" else [args.dataset]

    for d in datasets:
        run_full_evaluation(args.model, d, cfg, predictor_ckpt=args.predictor_ckpt)
