import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import importlib.util

def _import_script(filename: str):
    script_path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

extract_embeddings = _import_script("01_extract_embeddings.py").extract_embeddings
build_faiss_index = _import_script("02_build_faiss_index.py").build_faiss_index
train_alignment_predictor = _import_script("03_train_alignment.py").train_alignment_predictor
run_full_evaluation = _import_script("05_run_evaluation.py").run_full_evaluation
compare_models = _import_script("06_compare_models.py").compare_models
generate_all_visualizations = _import_script("08_visualize_results.py").generate_all_visualizations

from utils.common import load_config, setup_logger

logger = setup_logger("full_pipeline")


def execute_full_pipeline(config_path: str = None, limit_per_dataset: int = None):
    config = load_config(config_path)
    datasets = config["data"].get("active_datasets", ["rlvs", "ucf_arg", "tinyvirat", "ucf_crime"])

    logger.info("================================================================================")
    logger.info("STARTING COMPLETE IMPLEMENTATION HPC BENCHMARK PIPELINE")
    logger.info("Datasets: %s | Target GPU: NVIDIA RTX 4000 Ada (20 GB)", datasets)
    logger.info("================================================================================")

    # Step 1: Feature Extraction (CLIP and V-JEPA 2)
    logger.info("\n--- STEP 1: BATCH VIDEO FEATURE EXTRACTION ---")
    for model_name in ["clip", "vljepa"]:
        for d in datasets:
            extract_embeddings(model_name, d, config, limit=limit_per_dataset)

    # Step 2: Cross-Modal Predictor Training
    logger.info("\n--- STEP 2: TRAINING CROSS-MODAL ALIGNMENT PREDICTOR (InfoNCE) ---")
    for d in datasets:
        train_alignment_predictor(d, config)

    # Step 3: FAISS Index Building
    logger.info("\n--- STEP 3: BUILDING FAISS FLAT INDEXES ---")
    for model_name in ["clip", "vljepa"]:
        for d in datasets:
            build_faiss_index(model_name, d, config)

    # Step 4: Comprehensive Evaluation
    logger.info("\n--- STEP 4: RUNNING EVALUATION (RETRIEVAL, EFFICIENCY, FAITHFULNESS) ---")
    for model_name in ["clip", "vljepa"]:
        for d in datasets:
            run_full_evaluation(model_name, d, config)

    # Step 5: Head-to-Head Comparison
    logger.info("\n--- STEP 5: GENERATING MODEL COMPARISON TABLE ---")
    compare_models(datasets, config)

    # Step 6: Visualizations
    logger.info("\n--- STEP 6: EXPORTING PUBLICATION VISUALIZATIONS ---")
    generate_all_visualizations(config)

    logger.info("================================================================================")
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("Outputs available in results/metrics/ and results/plots/")
    logger.info("Launch demo with: streamlit run demo/streamlit_app.py")
    logger.info("================================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete implementation_hpc end-to-end pipeline.")
    parser.add_argument("--limit", type=int, default=None, help="Optional clip limit per dataset for fast runs")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    execute_full_pipeline(config_path=args.config, limit_per_dataset=args.limit)
