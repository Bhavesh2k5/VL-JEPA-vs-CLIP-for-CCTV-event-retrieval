from pathlib import Path
from typing import Any, Dict, List


def format_evidence_block(evidence_clips: List[Dict[str, Any]]) -> str:
    """Formats top-k retrieved video clips into a structured evidence string."""
    lines = []
    for i, clip in enumerate(evidence_clips, start=1):
        score = clip.get("similarity_score", 0.0)
        label = clip.get("label", "unknown")
        meta = clip.get("metadata", {})
        viewpoint = meta.get("viewpoint", "stationary")
        filename = meta.get("filename", Path(clip.get("path", "")).name if clip.get("path") else f"clip_{i}.mp4")
        dataset = meta.get("dataset", "CCTV")

        lines.append(
            f"Evidence Item {i}: Video File '{filename}' | Dataset: {dataset} | "
            f"Viewpoint: {viewpoint} | Semantic Class: {label} | Relevance Score: {score:.3f}"
        )
    return "\n".join(lines)


def build_grounded_prompt(query: str, evidence_clips: List[Dict[str, Any]]) -> str:
    """Builds grounded RAG prompt containing structured video evidence."""
    evidence_text = format_evidence_block(evidence_clips)
    prompt = (
        "You are an AI CCTV Surveillance Analyst. Write an objective, factual incident report based ONLY "
        "on the retrieved video evidence provided below. Do not speculate or introduce unverified details.\n\n"
        f"INVESTIGATION QUERY: {query}\n\n"
        f"RETRIEVED VIDEO EVIDENCE:\n{evidence_text}\n\n"
        "Generate a structured incident report containing exactly these four sections:\n"
        "1. INCIDENT CLASSIFICATION: State the detected event type and confidence.\n"
        "2. OBSERVED EVIDENCE: Detail the actions, camera viewpoints, and evidence files confirmed in the footage.\n"
        "3. SEVERITY ASSESSMENT: State whether the incident is Low, Medium, or High severity.\n"
        "4. ACTION REQUIRED: Recommend appropriate security protocol."
    )
    return prompt


def build_ungrounded_prompt(query: str) -> str:
    """Builds ungrounded prompt without evidence to test hallucination baseline (H3)."""
    prompt = (
        "You are an AI CCTV Surveillance Analyst. Write a surveillance incident report based on the user's query.\n\n"
        f"INVESTIGATION QUERY: {query}\n\n"
        "Generate a structured incident report containing:\n"
        "1. INCIDENT CLASSIFICATION\n"
        "2. OBSERVED EVIDENCE\n"
        "3. SEVERITY ASSESSMENT\n"
        "4. ACTION REQUIRED"
    )
    return prompt
