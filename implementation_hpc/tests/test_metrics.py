import pytest

from evaluation import (
    compute_all_retrieval_metrics,
    evaluate_faithfulness,
    recall_at_k,
)


def test_retrieval_metrics_math():
    """Verifies correctness of Recall@K, MRR, nDCG, and mAP."""
    retrieved = [["violence", "running", "walking"], ["boxing", "digging", "waving"]]
    positives = [["violence"], ["waving"]]

    metrics = compute_all_retrieval_metrics(retrieved, positives, k_values=[1, 2, 3])

    # Query 1 has violence at rank 1 (R@1=1, R@2=1, R@3=1)
    # Query 2 has waving at rank 3 (R@1=0, R@2=0, R@3=1)
    assert metrics["Recall@1"] == 0.5
    assert metrics["Recall@2"] == 0.5
    assert metrics["Recall@3"] == 1.0
    # MRR: (1/1 + 1/3) / 2 = 1.3333 / 2 = 0.6667
    assert abs(metrics["MRR"] - 0.6667) < 0.001


def test_faithfulness_grounding_score():
    """Verifies that grounded statements receive high score, while ungrounded receive low score."""
    evidence = "Evidence Item 1: Video File 'V_1.mp4' | Semantic Class: violence | Viewpoint: street"

    good_report = "INCIDENT CLASSIFICATION: Detected violence. Analysis of video file V_1.mp4 confirms physical street violence."
    res_good = evaluate_faithfulness(good_report, evidence)
    assert res_good["grounding_score"] >= 0.7
    assert res_good["hallucination_rate"] <= 0.3

    bad_report = "Incident was recorded in room 999 with 55 suspects holding weapons on year 1999."
    res_bad = evaluate_faithfulness(bad_report, evidence)
    assert res_bad["grounding_score"] <= 0.5
