import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class BiDirectionalInfoNCELoss(nn.Module):
    """
    Continuous Cosine Alignment + Bi-directional InfoNCE contrastive loss.
    Enforces angular alignment while preventing representation collapse.
    """

    def __init__(self, temperature: float = 0.07, cosine_weight: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.cosine_weight = cosine_weight

    def forward(self, pred_embeddings: torch.Tensor, target_embeddings: torch.Tensor) -> torch.Tensor:
        """
        pred_embeddings:   [B, D] (L2-normalized predicted embedding from video)
        target_embeddings: [B, D] (L2-normalized target embedding from text)
        """
        pred_norm = F.normalize(pred_embeddings, p=2, dim=-1)
        target_norm = F.normalize(target_embeddings, p=2, dim=-1)

        # 1. Continuous Cosine Alignment
        loss_cos = (1.0 - torch.sum(pred_norm * target_norm, dim=-1)).mean()

        # 2. Pairwise InfoNCE contrastive term
        logits = torch.matmul(pred_norm, target_norm.t()) / self.temperature  # [B, B]
        b = logits.size(0)
        labels = torch.arange(b, device=logits.device, dtype=torch.long)

        loss_v2t = F.cross_entropy(logits, labels)
        loss_t2v = F.cross_entropy(logits.t(), labels)
        loss_nce = 0.5 * (loss_v2t + loss_t2v)

        return self.cosine_weight * loss_cos + 0.1 * loss_nce


class CrossModalPredictor(nn.Module):
    """
    VL-JEPA Continuous Cross-Modal Predictor.
    Bridges visual spatiotemporal features (V-JEPA 2) with semantic text embeddings
    using continuous representation prediction in the open-vocabulary shared space.
    """

    def __init__(
        self,
        visual_dim: int = 768,
        text_dim: int = 768,
        d_model: int = 1536,
        shared_dim: int = 768,
        num_layers: int = 4,
        num_heads: int = 16,
        dim_feedforward: int = 4096,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.d_model = d_model
        self.shared_dim = shared_dim

        # Continuous Deep Spatiotemporal-to-Semantic Projector
        self.net = nn.Sequential(
            nn.Linear(visual_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, shared_dim),
        )

    def predict_video(self, visual_feats: torch.Tensor) -> torch.Tensor:
        """Projects spatiotemporal features into continuous shared semantic space."""
        if visual_feats.dim() == 3:
            visual_feats = visual_feats.mean(dim=1)
        projected = self.net(visual_feats)
        return F.normalize(projected, p=2, dim=-1)

    def project_text(self, text_feats: torch.Tensor) -> torch.Tensor:
        """Preserves pretrained open-vocabulary manifold in the continuous shared space."""
        return F.normalize(text_feats, p=2, dim=-1)

    def forward(self, visual_feats: torch.Tensor, text_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Joint forward pass returning (predicted_visual_embedding, target_text_embedding)."""
        pred_embed = self.predict_video(visual_feats)
        target_embed = self.project_text(text_feats)
        return pred_embed, target_embed
