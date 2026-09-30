import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import streamlit as st

from evaluation.faithfulness_metrics import grounding_score
from models.encoder_factory import get_encoder
from reporting.evidence_utils import evidence_text
from reporting.llm_reporter import IncidentReporter
from retrieval.embedding_indexer import EmbeddingIndexer
from retrieval.retriever import CCTVRetriever
from utils.common import PROJECT_ROOT, get_logger, load_config, resolve_device

logger = get_logger("streamlit_app")

st.set_page_config(page_title="CCTV Event Retrieval", page_icon="...", layout="wide")


@st.cache_resource(show_spinner="Loading configuration...")
def load_config_cached(config_path):
    return load_config(config_path)


@st.cache_resource(show_spinner="Loading encoder...")
def get_encoder_cached(model_name, variant, config_path):
    return get_encoder(model_name, load_config_cached(config_path), variant=variant)


@st.cache_resource(show_spinner="Loading FAISS index...")
def get_indexer_cached(index_path):
    return EmbeddingIndexer.load(index_path)


@st.cache_resource(show_spinner="Loading report generator...")
def get_reporter_cached(config_path):
    return IncidentReporter(load_config_cached(config_path))


def first_frame(path, max_width=320):
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (max_width, int(h * scale)))
    return frame


def run_retrieval(config_path, model_name, variant, index_path, query, k):
    start = time.perf_counter()
    encoder = get_encoder_cached(model_name, variant, config_path)
    indexer = get_indexer_cached(str(index_path))
    retriever = CCTVRetriever(encoder, indexer)
    results = retriever.retrieve(query, k=k)
    latency = time.perf_counter() - start
    return results, latency


def render_results(results, columns=5):
    if not results:
        st.info("No results returned.")
        return
    cols = st.columns(min(columns, len(results)))
    for col, r in zip(cols, results):
        with col:
            thumb = first_frame(r.get("path", ""))
            if thumb is not None:
                st.image(thumb)
            else:
                st.caption("(frame unavailable)")
            st.markdown(f"**#{r['rank']}** score `{r['score']:.4f}`")
            st.caption(f"{r.get('dataset', '?')} | {r.get('label', '?')}")
            with st.expander("details"):
                st.json({k: v for k, v in r.items() if k not in ("rank", "score")}, expanded=False)


def retrieval_page():
    st.title("VL-JEPA vs CLIP - CCTV Event Retrieval")
    config_path = str(PROJECT_ROOT / "config" / "config.yaml")
    config = load_config_cached(config_path)

    index_dir = Path(config["retrieval"]["index_save_dir"])
    index_files = sorted(index_dir.glob("*.faiss")) if index_dir.is_dir() else []

    with st.sidebar:
        st.header("Controls")
        device = resolve_device(config["models"]["clip"].get("device", "auto"))
        st.caption(f"Device: **{device}** | torch-cuda: {bool(__import__('torch').cuda.is_available())}")

        model_mode = st.radio("Model", ["CLIP", "VL-JEPA", "Both (compare)"])
        variant = None
        if model_mode in ("CLIP", "Both (compare)"):
            variant = st.selectbox("CLIP variant", ["primary", "secondary"])
        if not index_files:
            st.warning(
                "No FAISS indices found. Run scripts/extract_embeddings.py and "
                "scripts/build_index.py first (see README)."
            )
        index_path = st.selectbox(
            "Index", index_files, format_func=lambda p: p.name if p else "-"
        )
        k = st.slider("Top-K", 1, 20, int(config["retrieval"].get("top_k", 5)))
        generate_report = st.checkbox("Generate LLM report", value=True)

    st.header("Query")
    query = st.text_input(
        "Natural-language query",
        value="a person running away after an assault",
    )
    run = st.button("Retrieve", type="primary", width="stretch")

    if run and index_path and query.strip():
        target_models = ["clip"] if model_mode == "CLIP" else (
            ["vljepa"] if model_mode == "VL-JEPA" else ["vljepa", "clip"]
        )
        per_model = {}
        for m in target_models:
            v = variant if m == "clip" else None
            with st.spinner(f"Retrieving with {m}..."):
                try:
                    results, latency = run_retrieval(config_path, m, v, index_path, query.strip(), k)
                    per_model[m] = (results, latency)
                except Exception as exc:
                    st.error(f"{m} retrieval failed: {exc}")

        if model_mode == "Both (compare)" and len(per_model) == 2:
            left, right = st.columns(2)
            for col, m in zip((left, right), ("vljepa", "clip")):
                if m in per_model:
                    results, latency = per_model[m]
                    with col:
                        st.subheader(f"{m.upper()} ({latency:.2f}s)")
                        render_results(results)
        else:
            for m, (results, latency) in per_model.items():
                st.subheader(f"{m.upper()} - {latency:.2f}s")
                render_results(results)

        first_results = next(iter(per_model.values()))[0] if per_model else []
        if generate_report and first_results:
            with st.spinner("Generating incident report..."):
                try:
                    reporter = get_reporter_cached(config_path)
                    report = reporter.generate_report(first_results, query.strip(), grounded=True)
                    score, _ = grounding_score(report, evidence_text(first_results))
                    st.subheader("Incident report")
                    st.write(report)
                    m1, m2 = st.columns(2)
                    m1.metric("Grounding score", f"{score:.2f}")
                    m2.metric("Hallucination rate", f"{1.0 - score:.2f}")
                except Exception as exc:
                    st.error(f"Report generation failed: {exc}")


summarizer = st.Page("video_summarizer.py", title="Video Summarizer", icon="🎬", default=True)
retrieval = st.Page(retrieval_page, title="Dataset Retrieval", icon="🔍")
st.navigation([summarizer, retrieval]).run()
