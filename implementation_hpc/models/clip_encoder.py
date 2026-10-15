import logging
from typing import List, Union

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer, logging as hf_logging
hf_logging.set_verbosity_error()

from utils.common import clear_gpu_cache, resolve_device

logger = logging.getLogger(__name__)


def _as_tensor(x):
    if torch.is_tensor(x):
        return x
    if hasattr(x, "text_embeds") and x.text_embeds is not None:
        return x.text_embeds
    if hasattr(x, "image_embeds") and x.image_embeds is not None:
        return x.image_embeds
    if hasattr(x, "pooler_output") and x.pooler_output is not None:
        return x.pooler_output
    if hasattr(x, "last_hidden_state") and x.last_hidden_state is not None:
        return x.last_hidden_state[:, 0]
    return getattr(x, "logits", x)


class CLIPEncoder:
    """
    High-throughput CLIP visual and textual encoder optimized for RTX 4000 Ada.
    Implements mean frame pooling across temporal sequences followed by L2-normalization.
    """

    def __init__(self, config: dict, variant: str = "primary"):
        clip_cfg = config["models"]["clip"]
        model_id = clip_cfg.get(variant, clip_cfg.get("primary", "openai/clip-vit-large-patch14"))

        self.device = resolve_device(config.get("system", {}).get("device", "auto"))
        self.use_fp16 = config.get("system", {}).get("fp16", True) and self.device.type == "cuda"
        self.dtype = torch.float16 if self.use_fp16 else torch.float32

        logger.info("Initializing CLIPEncoder with checkpoint: %s (device=%s, fp16=%s)", model_id, self.device, self.use_fp16)

        self.model = CLIPModel.from_pretrained(model_id, dtype=self.dtype)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id)
        self.processor = CLIPProcessor.from_pretrained(model_id)

        self.model = self.model.to(self.device).eval()
        self.embedding_dim = self.model.projection_dim if hasattr(self.model, "projection_dim") else 768

    @torch.no_grad()
    def encode_video(self, videos: torch.Tensor) -> torch.Tensor:
        """
        Encodes video batch [B, T, C, H, W] into normalized clip embeddings [B, D].
        """
        if videos.dim() == 4:
            videos = videos.unsqueeze(0)  # [1, T, C, H, W]

        b, t, c, h, w = videos.shape
        flat_frames = videos.view(b * t, c, h, w).to(self.device, dtype=self.dtype)

        if (h, w) != (224, 224):
            flat_frames = F.interpolate(flat_frames, size=(224, 224), mode="bilinear", align_corners=False)

        frame_features = _as_tensor(self.model.get_image_features(pixel_values=flat_frames))  # [B*T, D]
        frame_features = frame_features.view(b, t, -1)  # [B, T, D]

        clip_features = frame_features.mean(dim=1)  # [B, D]
        clip_features = F.normalize(clip_features, p=2, dim=-1)
        return clip_features.cpu().float()

    @torch.no_grad()
    def encode_text(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Encodes natural-language queries into normalized text embeddings [B, D].
        """
        if isinstance(texts, str):
            texts = [texts]

        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        text_features = _as_tensor(self.model.get_text_features(**tokens))  # [B, D]
        text_features = F.normalize(text_features, p=2, dim=-1)
        return text_features.cpu().float()

    def get_embedding_dim(self) -> int:
        return self.embedding_dim
