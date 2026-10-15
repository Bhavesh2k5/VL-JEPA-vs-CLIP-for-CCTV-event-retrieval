import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation import evaluate_faithfulness
from models import get_encoder
from reporting import IncidentReporter
from retrieval import CCTVRetriever, EmbeddingIndexer
from utils.common import ensure_dir, load_config, setup_logger

logger = setup_logger("generate_reports")


def generate_incident_report(query: str, dataset_name: str, model_name: str, config: dict, top_k: int = 5):
    idx_dir = Path(config["retrieval"].get("index_save_dir", "results/indices"))
    idx_path = idx_dir / f"{model_name}_{dataset_name}.faiss"
    reports_dir = ensure_dir("results/reports")

    if not idx_path.exists():
        logger.error("FAISS index not found: %s", idx_path)
        return

    encoder = get_encoder(model_name, config)
    indexer = EmbeddingIndexer.load(idx_path)
    retriever = CCTVRetriever(encoder, indexer)

    # 1. Retrieve top-k visual evidence
    evidence = retriever.retrieve(query, top_k=top_k)

    # 2. Instantiate LLM reporter
    reporter = IncidentReporter(config)

    # 3. Generate both grounded and ungrounded reports
    grounded_res = reporter.generate_report(query, evidence, grounded=True)
    ungrounded_res = reporter.generate_report(query, evidence, grounded=False)

    # 4. Evaluate factual faithfulness
    grounded_eval = evaluate_faithfulness(grounded_res["report_text"], grounded_res["evidence_summary"])
    ungrounded_eval = evaluate_faithfulness(ungrounded_res["report_text"], "No evidence context provided.")

    payload = {
        "query": query,
        "dataset": dataset_name,
        "model": model_name,
        "grounded_report": grounded_res["report_text"],
        "grounded_faithfulness": grounded_eval,
        "ungrounded_report": ungrounded_res["report_text"],
        "ungrounded_faithfulness": ungrounded_eval,
        "retrieved_evidence": evidence,
    }

    clean_q = "".join(c if c.isalnum() else "_" for c in query)[:30]
    out_file = reports_dir / f"report_{model_name}_{dataset_name}_{clean_q}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 80)
    print("AI CCTV INCIDENT REPORT (EVIDENCE GROUNDED)")
    print("=" * 80)
    print(grounded_res["report_text"])
    print("-" * 80)
    print(f"Fact-Grounding Score: {grounded_eval['grounding_score']:.1%} (Hallucination Rate: {grounded_eval['hallucination_rate']:.1%})")
    print(f"Saved complete report audit to: {out_file}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate evidence-grounded CCTV incident report.")
    parser.add_argument("--query", type=str, required=True, help="Incident inquiry query")
    parser.add_argument("--dataset", type=str, default="rlvs", choices=["rlvs", "ucf_arg", "tinyvirat"])
    parser.add_argument("--model", type=str, default="vljepa", choices=["clip", "vljepa"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    generate_incident_report(args.query, args.dataset, args.model, cfg, top_k=args.top_k)
