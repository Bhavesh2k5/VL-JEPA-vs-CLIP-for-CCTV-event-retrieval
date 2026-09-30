import hashlib
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

import reporting.video_summarizer as vs
from models.encoder_factory import get_encoder
from reporting.llm_reporter import IncidentReporter
from utils.common import PROJECT_ROOT, load_config


@st.cache_resource(show_spinner="Loading configuration...")
def cached_config(config_path):
    return load_config(config_path)


@st.cache_resource(show_spinner="Loading encoder...")
def cached_encoder(name, config_path):
    return get_encoder(name, cached_config(config_path), variant="primary")


@st.cache_resource(show_spinner="Loading report generator...")
def cached_reporter(config_path):
    return IncidentReporter(cached_config(config_path))


@st.cache_resource(show_spinner="Loading story narrator (first run downloads ~3 GB)...")
def cached_narrator(config_path):
    from reporting.story_narrator import StoryNarrator

    cfg = cached_config(config_path)
    return StoryNarrator(cfg["llm"].get("story_model"), cfg["llm"].get("device", "auto"))


def get_narrator(config_path):
    try:
        return cached_narrator(config_path)
    except Exception as exc:
        st.caption(f"Story narrator unavailable ({exc}); using flan-t5 fallback.")
        return cached_reporter(config_path)


@st.cache_data(show_spinner=False)
def cached_summarize(file_hash, tmp_bytes, suffix, config_path) -> dict:
    results = {}
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(tmp_bytes)
        tmp_path = tmp.name
    try:
        clip = cached_encoder("clip", config_path)
        results["clip"] = vs.summarize_video(clip, "clip", tmp_path)
        vljepa = cached_encoder("vljepa", config_path)
        results["vljepa"] = vs.summarize_video(vljepa, "vljepa", tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return results


@st.cache_data(show_spinner=False)
def cached_compare(file_hash, config_path) -> dict:
    summaries = st.session_state["summaries"]
    return vs.compare_summaries(
        summaries["clip"], summaries["vljepa"], narrator=get_narrator(config_path)
    )


@st.cache_data(show_spinner=False)
def cached_story(file_hash, model_name, config_path) -> str:
    summaries = st.session_state["summaries"]
    return vs.story_narration(get_narrator(config_path), summaries[model_name])


@st.cache_data(show_spinner=False)
def cached_answer(file_hash, question, config_path) -> dict:
    summaries = st.session_state["summaries"]
    narrator = get_narrator(config_path)
    return {
        "clip": vs.answer_question(narrator, summaries["clip"]["summary_text"], question),
        "vljepa": vs.answer_question(narrator, summaries["vljepa"]["summary_text"], question),
    }


def render_summary(result):
    st.subheader(f"{result['model'].upper()} summary")
    dom = result["overall"][:3]
    for col, (label, p) in zip(st.columns(len(dom)), dom):
        col.metric(label, f"{p * 100:.1f}%")
    top_p = result["segments"][0]["top"][0][1]
    if top_p < 0.35:
        st.warning(
            "Low classification confidence - the scene may not match the known "
            "activity categories well. Treat labels as tentative."
        )
    if result["model"] == "vljepa":
        st.caption(
            "Note: VL-JEPA here is the V-JEPA 2 visual backbone paired with a CLIP text "
            "encoder in an unaligned embedding space (zero-training fallback, repo "
            "hypothesis H1). Its activity labels are experimental - the temporal-structure "
            "lines come from V-JEPA 2 embeddings alone and are the reliable part."
        )
    st.caption(vs.build_summary_text(result))
    st.markdown("**Segment timeline**")
    cols = st.columns(min(3, len(result["segments"])))
    for i, seg in enumerate(result["segments"]):
        with cols[i % 3]:
            st.image(seg["thumbnail"], caption=f"seg {i + 1} ({vs.format_ts(seg['start'])})", width="stretch")
            top = seg["top"][0]
            st.markdown(f"**{top[0]}** `{top[1] * 100:.0f}%`")
            st.caption(", ".join(f"{l} {p * 100:.0f}%" for l, p in seg["ranked"][1:3]))


def main():
    st.title("CCTV Video Summarizer")
    st.caption(
        "Upload a video, get summaries from CLIP and VL-JEPA, ask questions about the "
        "incident, and compare both models. Answers come only from the model summaries."
    )

    config_path = str(PROJECT_ROOT / "config" / "config.yaml")

    upload = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        help="Short clips work best; VL-JEPA runs on CPU (~20s per analyzed segment).",
    )
    if upload is None:
        st.info("Upload a video to begin. Both model summaries, Q&A, and the model comparison will appear here.")
        return

    st.video(upload)
    file_hash = hashlib.sha256(upload.getvalue()).hexdigest()

    analyze = st.button("Analyze video", type="primary", width="stretch")
    if analyze:
        n_clips = vs.MAX_SEGMENTS
        st.caption(
            f"Analyzing {n_clips} segments: CLIP first (fast), then VL-JEPA "
            f"(~{n_clips * 20}s on CPU). Keep this page open."
        )
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(upload.name).suffix) as tmp:
            tmp.write(upload.getvalue())
            tmp_path = tmp.name
        tmp_bytes = upload.getvalue()
        suffix = Path(upload.name).suffix
        Path(tmp_path).unlink(missing_ok=True)
        st.session_state["summaries"] = cached_summarize(file_hash, tmp_bytes, suffix, config_path)

    if "summaries" not in st.session_state:
        st.info("Click **Analyze video** to generate both model summaries.")
        return

    results = st.session_state["summaries"]
    left, right = st.columns(2)
    with left:
        render_summary(results["clip"])
    with right:
        render_summary(results["vljepa"])

    st.divider()
    st.header("Model comparison")
    cmp = cached_compare(file_hash, config_path)
    m1, m2, m3 = st.columns(3)
    m1.metric("Top-1 segment agreement", f"{cmp['top1_agreement'] * 100:.0f}%")
    m2.metric("CLIP avg confidence", f"{cmp['avg_conf_a'] * 100:.0f}%")
    m3.metric("VL-JEPA avg confidence", f"{cmp['avg_conf_b'] * 100:.0f}%")
    st.dataframe(
        [
            {
                "window": p["window"],
                "CLIP": f"{p['a']} ({p['a_p'] * 100:.0f}%)",
                "VL-JEPA": f"{p['b']} ({p['b_p'] * 100:.0f}%)",
                "match": "yes" if p["match"] else "no",
            }
            for p in cmp["per_segment"]
        ],
        width="stretch",
        hide_index=True,
    )
    if cmp.get("llm_diff"):
        st.markdown("**LLM key differences**")
        st.write(cmp["llm_diff"])

    st.divider()
    st.header("Incident story")
    st.caption("A chronological narration of the video, generated from each model's analysis.")
    s1, s2 = st.columns(2)
    for col, model in zip((s1, s2), ("clip", "vljepa")):
        with col:
            st.markdown(f"**{model.upper()}-based narration**")
            with st.spinner(f"Writing {model.upper()} story..."):
                st.write(cached_story(file_hash, model, config_path))

    st.divider()
    st.header("Ask about the incident")
    st.caption("Answers are generated from each model's summary only.")
    question = st.text_input("Your question", placeholder="e.g. What happened in the video?")
    if st.button("Ask", type="primary", width="stretch") and question.strip():
        qa = cached_answer(file_hash, question.strip(), config_path)
        a1, a2 = st.columns(2)
        with a1:
            st.markdown("**CLIP-based answer**")
            st.info(qa["clip"])
        with a2:
            st.markdown("**VL-JEPA-based answer**")
            st.info(qa["vljepa"])


main()
