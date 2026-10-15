import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import get_encoder
from reporting import IncidentReporter
from retrieval import CCTVRetriever, EmbeddingIndexer
from utils.common import load_config, setup_logger

logger = setup_logger("query_retrieval")


def run_query_retrieval(
    model_name: str,
    dataset_name: str,
    query: str,
    top_k: int = 5,
    config: dict = None,
    predictor_ckpt: str = None,
    generate_report: bool = False,
):
    idx_dir = Path(config["retrieval"].get("index_save_dir", "results/indices"))
    idx_path = idx_dir / f"{model_name}_{dataset_name}.faiss"

    if not idx_path.exists():
        logger.error("FAISS index not found: %s. Run 02_build_faiss_index.py first.", idx_path)
        return

    # Check for predictor checkpoint if running vljepa
    if model_name == "vljepa" and predictor_ckpt is None:
        default_ckpt = Path(config.get("training", {}).get("save_dir", "results/checkpoints")) / "vljepa_predictor_best.pt"
        if default_ckpt.exists():
            predictor_ckpt = str(default_ckpt)

    # Load encoder and index
    encoder = get_encoder(model_name, config, predictor_checkpoint=predictor_ckpt)
    indexer = EmbeddingIndexer.load(idx_path)
    retriever = CCTVRetriever(encoder, indexer)

    logger.info("Executing Semantic Query: '%s' (Model: %s, Dataset: %s, Top-K: %d)", query, model_name, dataset_name, top_k)
    results = retriever.retrieve(query, top_k=top_k)

    print("\n" + "=" * 80)
    print(f"SEARCH RESULTS: '{query}'")
    print(f"Model: {model_name.upper()} | Dataset: {dataset_name.upper()}")
    print("=" * 80)

    for res in results:
        meta = res.get("metadata", {})
        filename = meta.get("filename", Path(res["path"]).name if res["path"] else "clip")
        print(f"Rank {res['rank']:2d} | Similarity: {res['similarity_score']:.4f} | Class: {res['label']:<15} | File: {filename}")

    print("=" * 80 + "\n")

    if generate_report:
        reporter = IncidentReporter(config)
        report_data = reporter.generate_report(query, results, grounded=True)
        print("INCIDENT REPORT GENERATION (GROUNDED):")
        print("-" * 80)
        print(report_data["report_text"])
        print("-" * 80 + "\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query CCTV video archive using natural language.")
    parser.add_argument("--model", type=str, default="clip", choices=["clip", "vljepa"])
    parser.add_argument("--dataset", type=str, default="rlvs", choices=["rlvs", "ucf_arg", "tinyvirat", "ucf_crime"])
    parser.add_argument("--query", type=str, required=True, help="Natural language search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to retrieve")
    parser.add_argument("--predictor-ckpt", type=str, default=None, help="Path to trained predictor weights")
    parser.add_argument("--report", action="store_true", help="Also generate LLM incident report")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_query_retrieval(
        args.model,
        args.dataset,
        args.query,
        top_k=args.top_k,
        config=cfg,
        predictor_ckpt=args.predictor_ckpt,
        generate_report=args.report,
    )
