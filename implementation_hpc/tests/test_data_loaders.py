import json
from pathlib import Path
import pytest
import torch

from data import build_dataset, collate_video_batch
from tests.conftest import create_dummy_video


def test_rlvs_dataset_avi_and_mp4(test_config, tmp_path):
    """Verifies that RLVS loader correctly captures both MP4 and AVI files."""
    v_dir = tmp_path / "rlvs_mock" / "Violence"
    nv_dir = tmp_path / "rlvs_mock" / "NonViolence"

    create_dummy_video(v_dir / "V_1.mp4")
    create_dummy_video(nv_dir / "NV_1.mp4")
    create_dummy_video(nv_dir / "NV_602.avi")  # The critical AVI file

    ds = build_dataset("rlvs", test_config)
    assert len(ds) == 3

    labels = ds.get_labels()
    assert "violence" in labels
    assert "non_violence" in labels

    # Test __getitem__
    item = ds[0]
    assert "video" in item
    assert item["video"].shape == (4, 3, 64, 64)


def test_ucf_arg_viewpoints(test_config, tmp_path):
    """Verifies nested viewpoint indexing and viewpoint filtering."""
    base = tmp_path / "ucf_arg_mock"
    create_dummy_video(base / "aerial_clips" / "aerial_clips" / "boxing" / "person01_01_aerial_boxing.avi")
    create_dummy_video(base / "ground_clips" / "ground_clips" / "running" / "person01_01_ground_running.avi")

    ds_all = build_dataset("ucf_arg", test_config)
    assert len(ds_all) == 2

    ds_aerial = build_dataset("ucf_arg", test_config, viewpoint_filter="aerial")
    assert len(ds_aerial) == 1
    assert ds_aerial[0]["label"] == "boxing"


def test_collate_fn_with_variable_sizes():
    """Verifies that collate_video_batch pads and stacks variable sized videos without throwing."""
    v1 = torch.rand(4, 3, 60, 60)
    v2 = torch.rand(4, 3, 72, 72)

    batch = [
        {"video": v1, "path": "p1", "label": "c1", "metadata": {}},
        {"video": v2, "path": "p2", "label": "c2", "metadata": {}},
    ]

    collated = collate_video_batch(batch)
    assert "video" in collated
    assert collated["video"].shape == (2, 4, 3, 72, 72)


def test_ucf_crime_dataset(test_config, tmp_path):
    """Verifies that UCF-Crime loader indexes anomaly folders and infers correct labels."""
    crime_dir = tmp_path / "ucf_crime_mock"
    test_config["data"]["ucf_crime"] = {"path": str(crime_dir)}

    create_dummy_video(crime_dir / "capstone_ucf_crime_p1_abuse" / "Abuse001_x264.mp4")
    create_dummy_video(crime_dir / "capstone_ucf_crime_p3_robbery" / "Robbery001_x264.mp4")
    create_dummy_video(crime_dir / "capstone_ucf_crime_normal" / "Normal_Videos_001_x264.mp4")

    ds = build_dataset("ucf_crime", test_config)
    assert len(ds) == 3

    labels = ds.get_labels()
    assert "abuse" in labels
    assert "robbery" in labels
    assert "normal" in labels
