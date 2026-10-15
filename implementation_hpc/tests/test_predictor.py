import pytest
import torch

from models import BiDirectionalInfoNCELoss, CrossModalPredictor


def test_cross_modal_predictor_forward():
    """Verifies output dimensions and unit normalization of CrossModalPredictor."""
    batch_size = 4
    visual_dim = 1024
    text_dim = 768
    shared_dim = 1536

    model = CrossModalPredictor(
        visual_dim=visual_dim,
        text_dim=text_dim,
        d_model=128,
        shared_dim=shared_dim,
        num_layers=2,
        num_heads=4,
    )

    v_feats = torch.randn(batch_size, visual_dim)
    t_feats = torch.randn(batch_size, text_dim)

    pred_v, target_t = model(v_feats, t_feats)

    assert pred_v.shape == (batch_size, shared_dim)
    assert target_t.shape == (batch_size, shared_dim)

    # Verify unit-norm hypersphere projection
    v_norms = torch.norm(pred_v, p=2, dim=-1)
    t_norms = torch.norm(target_t, p=2, dim=-1)
    torch.testing.assert_close(v_norms, torch.ones_like(v_norms), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(t_norms, torch.ones_like(t_norms), atol=1e-5, rtol=1e-5)


def test_infonce_loss():
    """Verifies that InfoNCE produces higher loss on random alignment than perfect alignment."""
    criterion = BiDirectionalInfoNCELoss(temperature=0.07)

    dim = 64
    batch_size = 8

    # Identical representations (perfect alignment)
    v = torch.randn(batch_size, dim)
    v_norm = torch.nn.functional.normalize(v, p=2, dim=-1)
    loss_perfect = criterion(v_norm, v_norm)

    # Disjoint representations (random)
    t = torch.randn(batch_size, dim)
    t_norm = torch.nn.functional.normalize(t, p=2, dim=-1)
    loss_random = criterion(v_norm, t_norm)

    assert loss_perfect < loss_random
