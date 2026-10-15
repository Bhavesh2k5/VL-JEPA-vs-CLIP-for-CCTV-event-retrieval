import logging
from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, CLIPModel, CLIPTokenizer, logging as hf_logging
hf_logging.set_verbosity_error()

from data.data_utils import IMAGENET_MEAN, IMAGENET_STD, normalize_frames
from utils.common import resolve_device
from .cross_modal_predictor import CrossModalPredictor

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


class VLJEPAEncoder:
    """
    VL-JEPA video-language encoder for implementation_hpc.
    Combines Meta V-JEPA 2 ViT-L spatiotemporal video encoder with
    a trainable/trained CrossModalPredictor to project into a shared 1536-d space.
    """

    def __init__(self, config: dict, predictor_checkpoint: Optional[str] = None):
        self.config = config
        vcfg = config["models"]["vljepa"]
        pcfg = config["models"]["predictor"]

        self.device = resolve_device(config.get("system", {}).get("device", "auto"))
        self.use_fp16 = config.get("system", {}).get("fp16", True) and self.device.type == "cuda"
        self.dtype = torch.float16 if self.use_fp16 else torch.float32

        self.num_frames = int(vcfg.get("num_frames", 16))
        self.input_size = int(vcfg.get("input_size", 256))
        self.visual_dim = int(vcfg.get("visual_dim", 1024))
        self.text_dim = int(vcfg.get("text_dim", 768))
        self.shared_dim = int(pcfg.get("shared_dim", 768))

        # 1. Load Visual Backbone (V-JEPA 2 ViT-L)
        self.visual_model = self._load_visual_backbone(vcfg)

        # 2. Load Text Backbone (CLIP text encoder)
        text_id = vcfg.get("text_backbone", "openai/clip-vit-large-patch14")
        logger.info("Loading text encoder: %s", text_id)
        self.clip_model = CLIPModel.from_pretrained(text_id, dtype=self.dtype).to(self.device).eval()
        self.tokenizer = CLIPTokenizer.from_pretrained(text_id)

        # 3. Instantiate Cross-Modal Predictor
        self.predictor = CrossModalPredictor(
            visual_dim=self.visual_dim,
            text_dim=self.text_dim,
            d_model=int(pcfg.get("d_model", 2048)),
            shared_dim=self.shared_dim,
            num_layers=int(pcfg.get("num_layers", 4)),
            num_heads=int(pcfg.get("num_heads", 16)),
            dim_feedforward=int(pcfg.get("dim_feedforward", 4096)),
            dropout=float(pcfg.get("dropout", 0.1)),
        ).to(self.device).eval()

        self.has_trained_predictor = False
        if predictor_checkpoint and Path(predictor_checkpoint).exists():
            self.load_predictor(predictor_checkpoint)

    def _load_visual_backbone(self, vcfg: dict) -> nn.Module:
        """Loads V-JEPA 2 visual backbone or resilient fallback."""
        model_id = vcfg.get("visual_backbone", "qubvel-hf/vjepa2-vitl-fpc16-256-ssv2")
        logger.info("Initializing V-JEPA 2 visual backbone: %s", model_id)
        try:
            try:
                model = AutoModel.from_pretrained(model_id, dtype=self.dtype, local_files_only=True)
            except Exception:
                model = AutoModel.from_pretrained(model_id, dtype=self.dtype)
            model = model.to(self.device).eval()
            self.is_torchvision = False
            return model
        except Exception as e:
            logger.warning("Hugging Face AutoModel load for '%s' failed (%s). Falling back to PyTorch Hub / Torchvision ViT-L.", model_id, e)
            try:
                import torchvision.models as tvm
                model = tvm.vit_l_16(weights=tvm.ViT_L_16_Weights.DEFAULT)
                # Remove classification head to output pure 1024-d latent representations
                model.heads = nn.Identity()
                model = model.to(self.device, dtype=self.dtype).eval()
                self.is_torchvision = True
                return model
            except Exception as e2:
                logger.error("Failed to load secondary visual backbone: %s", e2)
                raise RuntimeError(f"Could not load visual backbone: {e2}")

    def load_predictor(self, checkpoint_path: Union[str, Path]) -> None:
        """Loads trained weights for the CrossModalPredictor."""
        cpath = Path(checkpoint_path)
        if not cpath.exists():
            raise FileNotFoundError(f"Predictor checkpoint not found: {cpath}")
        state = torch.load(cpath, map_location=self.device)
        if "model_state_dict" in state:
            state = state["model_state_dict"]

        # Adapt input visual dimension if needed
        weight_key = "0.weight" if "0.weight" in state else ("net.0.weight" if "net.0.weight" in state else None)
        if weight_key is not None:
            in_dim = state[weight_key].shape[1]
            if in_dim != self.predictor.visual_dim:
                logger.info("Adjusting predictor visual_dim (%d -> %d) to match checkpoint", self.predictor.visual_dim, in_dim)
                pcfg = self.config.get("models", {}).get("predictor", {})
                self.predictor = CrossModalPredictor(
                    visual_dim=in_dim,
                    text_dim=self.text_dim,
                    d_model=int(pcfg.get("d_model", 1536)),
                    shared_dim=self.shared_dim,
                    dropout=float(pcfg.get("dropout", 0.1)),
                ).to(self.device)

        if any(k.startswith("0.") or k.startswith("4.") for k in state.keys()):
            self.predictor.net.load_state_dict(state)
        elif any(k.startswith("net.") for k in state.keys()):
            self.predictor.load_state_dict(state)
        else:
            try:
                self.predictor.load_state_dict(state, strict=False)
            except Exception as err:
                logger.warning("Predictor state dict load partial: %s", err)

        self.predictor.eval()
        self.has_trained_predictor = True
        logger.info("Loaded trained CrossModalPredictor from: %s", cpath)

    @torch.no_grad()
    def extract_raw_visual(self, videos: torch.Tensor) -> torch.Tensor:
        """Extracts raw frozen visual features [B, 1024] from video tensor [B, T, C, H, W]."""
        if videos.dim() == 4:
            videos = videos.unsqueeze(0)
        b, t, c, h, w = videos.shape

        # Resample frames if count differs from self.num_frames
        if t != self.num_frames:
            idx = torch.linspace(0, t - 1, self.num_frames, device=videos.device).long()
            videos = videos[:, idx]
            t = self.num_frames

        if getattr(self, "is_torchvision", False):
            # Torchvision ViT-L expects 224x224
            target_size = 224
            flat_frames = videos.view(b * t, c, h, w).to(self.device, dtype=torch.float32)
            if (h, w) != (target_size, target_size):
                flat_frames = F.interpolate(flat_frames, size=(target_size, target_size), mode="bilinear", align_corners=False)
            flat_frames = normalize_frames(flat_frames, IMAGENET_MEAN, IMAGENET_STD).to(self.dtype)
            feats_per_frame = self.visual_model(flat_frames)
            if hasattr(feats_per_frame, "last_hidden_state"):
                feats_per_frame = feats_per_frame.last_hidden_state[:, 0]
            feats = feats_per_frame.view(b, t, -1).mean(dim=1)
        else:
            # V-JEPA 2 ViT-L spatiotemporal video transformer expects 256x256
            target_size = self.input_size  # 256
            flat_frames = videos.view(b * t, c, h, w).to(self.device, dtype=torch.float32)
            if (h, w) != (target_size, target_size):
                flat_frames = F.interpolate(flat_frames, size=(target_size, target_size), mode="bilinear", align_corners=False)
            flat_frames = normalize_frames(flat_frames, IMAGENET_MEAN, IMAGENET_STD).to(self.dtype)
            reshaped = flat_frames.view(b, t, c, target_size, target_size)
            out = self.visual_model(pixel_values_videos=reshaped)
            hidden = getattr(out, "last_hidden_state", None)
            if hidden is None:
                hidden = getattr(out, "pooler_output", None)
            feats = hidden.mean(dim=1) if hidden.dim() == 3 else hidden

        return feats.float()  # [B, 1024]

    @torch.no_grad()
    def extract_raw_text(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """Extracts raw text features [B, 768] using CLIP text encoder."""
        if isinstance(texts, str):
            texts = [texts]
        tokens = self.tokenizer(texts, padding=True, truncation=True, max_length=77, return_tensors="pt").to(self.device)
        feats = _as_tensor(self.clip_model.get_text_features(**tokens))
        return feats.float()

    @torch.no_grad()
    def encode_video(self, videos: torch.Tensor) -> torch.Tensor:
        """
        Encodes video tensor [B, T, C, H, W] into normalized embedding vector.
        If predictor is trained, outputs 1536-d aligned vector.
        Otherwise outputs 768-d unit vector with proper post-slice L2-normalization.
        """
        raw_vis = self.extract_raw_visual(videos)  # [B, 1024]
        if self.has_trained_predictor:
            if self.predictor.visual_dim != raw_vis.shape[-1]:
                raw_vis = raw_vis[:, :self.predictor.visual_dim]
            raw_vis = F.normalize(raw_vis, p=2, dim=-1)
            emb = self.predictor.predict_video(raw_vis.to(self.device))  # [B, 768]
        else:
            # Fallback prefix projection: slice first 768 dims then L2-normalize
            sliced = raw_vis[:, :self.text_dim]
            emb = F.normalize(sliced, p=2, dim=-1)
        return emb.cpu().float()

    @torch.no_grad()
    def encode_text(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Encodes natural-language queries into normalized embedding vector.
        If predictor is trained, outputs 1536-d aligned vector.
        Otherwise outputs 768-d unit vector.
        """
        raw_txt = self.extract_raw_text(texts)  # [B, 768]
        if self.has_trained_predictor:
            emb = self.predictor.project_text(raw_txt.to(self.device))  # [B, 1536]
        else:
            emb = F.normalize(raw_txt, p=2, dim=-1)
        return emb.cpu().float()

    def get_embedding_dim(self) -> int:
        return self.shared_dim if self.has_trained_predictor else self.text_dim
