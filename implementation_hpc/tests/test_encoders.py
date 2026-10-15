import pytest
import torch

from models import get_encoder


def test_encoder_factory_selection(test_config):
    """Verifies that encoder factory instantiates encoders and provides correct methods."""
    # Test with CPU mock config
    encoder = get_encoder("clip", test_config)
    assert hasattr(encoder, "encode_video")
    assert hasattr(encoder, "encode_text")
    assert hasattr(encoder, "get_embedding_dim")

    dim = encoder.get_embedding_dim()
    assert dim > 0

    # Test text encoding on mock text
    text_emb = encoder.encode_text(["violent fight", "normal walking"])
    assert text_emb.shape == (2, dim)

    # Verify unit-norm hypersphere
    norms = torch.norm(text_emb, p=2, dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)
