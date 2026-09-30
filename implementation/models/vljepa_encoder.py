import logging

import torch
import torch.nn.functional as F
from transformers import CLIPTokenizer

from data.data_utils import normalize_frames
from retrieval.retrieval_utils import l2_normalize_tensor
from utils.common import resolve_device

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _as_tensor(output):
    return output if torch.is_tensor(output) else output.pooler_output


class VLJEPAEncoder:
    """VL-JEPA encoder wrapper with a V-JEPA 2 + CLIP-text fallback.

    Prefers a native vision-language VL-JEPA checkpoint when it can be loaded
    from the configured id. Otherwise loads the configured V-JEPA 2 visual
    backbone and pairs it with a CLIP text encoder. In the fallback the two
    sides live in disjoint embedding spaces, so align="truncate" projects both
    onto their shared prefix dimension. That zero-training alignment is part
    of what hypothesis H1 evaluates, not a settled property of the system.
    """

    def __init__(self, config: dict):
        mcfg = config["models"]["vljepa"]
        self.mcfg = mcfg
        self.device = resolve_device(mcfg.get("device", "auto"))
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.input_size = int(mcfg.get("input_size", config["video"].get("frame_size", 224)))
        self.num_frames = int(mcfg.get("num_frames", 16))
        self.align = str(mcfg.get("align", "truncate")).lower()
        self.visual, self.visual_dim, self.native_vl = self._load_visual(mcfg)
        self.text_model, self.text_tokenizer = self._load_text(mcfg)
        self.text_dim = self._probe_text_dim()
        if self.native_vl:
            self.embedding_dim = self.visual_dim
        elif self.align == "truncate":
            self.embedding_dim = int(min(self.visual_dim, self.text_dim))
            logger.info(
                "vljepa fallback: truncating visual(%d)/text(%d) to shared dim %d",
                self.visual_dim, self.text_dim, self.embedding_dim,
            )
        elif self.visual_dim != self.text_dim:
            raise ValueError(
                f"visual dim {self.visual_dim} != text dim {self.text_dim}; "
                "set models.vljepa.align='truncate' or use a native VL-JEPA checkpoint"
            )
        else:
            self.embedding_dim = self.visual_dim

    def _load_visual(self, mcfg):
        from transformers import AutoModel

        model_id = mcfg.get("checkpoint") or mcfg.get("visual_backbone")
        if not model_id:
            raise ValueError("models.vljepa.visual_backbone (or checkpoint) must be set")
        try:
            model = AutoModel.from_pretrained(model_id, torch_dtype=self.dtype)
        except Exception as exc:
            raise RuntimeError(
                f"cannot load visual backbone '{model_id}' ({exc}). Set "
                "models.vljepa.visual_backbone to a V-JEPA 2 checkpoint such as "
                "facebook/vjepa2-vits16-224, facebook/vjepa2-vitb16-224 or "
                "facebook/vjepa2-vitl16-224."
            ) from exc
        native = "vl-jepa" in model_id.lower() or "vljepa" in model_id.lower()
        model = model.to(self.device).eval()
        dim = self._probe_visual_dim(model)
        return model, dim, native

    def _forward_visual(self, model, x):
        try:
            return model(pixel_values_videos=x)
        except TypeError:
            return model(pixel_values=x)

    def _probe_visual_dim(self, model):
        for t in (self.num_frames, 16, 8):
            try:
                dummy = torch.zeros(
                    1, t, 3, self.input_size, self.input_size, device=self.device, dtype=self.dtype
                )
                out = self._forward_visual(model, dummy)
                hidden = getattr(out, "last_hidden_state", None)
                if hidden is None:
                    hidden = getattr(out, "pooler_output", None)
                if hidden is None:
                    continue
                dim = int(hidden.shape[-1])
                self.num_frames = t
                logger.info("vljepa: visual backbone accepts %d frames, dim %d", t, dim)
                return dim
            except Exception:
                continue
        raise RuntimeError(
            "cannot infer V-JEPA 2 output dim; adjust models.vljepa.input_size / num_frames "
            "to match the checkpoint's expected resolution and tubelet length"
        )

    def _load_text(self, mcfg):
        from transformers import CLIPModel

        text_id = mcfg.get("text_encoder", "openai/clip-vit-base-patch32")
        try:
            model = CLIPModel.from_pretrained(text_id, torch_dtype=self.dtype)
            tokenizer = CLIPTokenizer.from_pretrained(text_id)
        except Exception as exc:
            raise RuntimeError(f"cannot load text encoder '{text_id}' ({exc})") from exc
        model = model.to(self.device).eval()
        return model, tokenizer

    def _probe_text_dim(self):
        dummy = self.encode_text(["probe"], truncate=False)
        return int(dummy.shape[-1])

    def _match_frames(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[1]
        if t == self.num_frames:
            return x
        if t > self.num_frames:
            idx = torch.linspace(0, t - 1, self.num_frames, device=x.device).long()
            return x[:, idx]
        pad = x[:, -1:].repeat(1, self.num_frames - t, 1, 1, 1)
        return torch.cat([x, pad], dim=1)

    @torch.no_grad()
    def encode_video(self, videos: torch.Tensor) -> torch.Tensor:
        if videos.dim() == 4:
            videos = videos.unsqueeze(0)
        videos = self._match_frames(videos)
        b, t = videos.shape[:2]
        x = videos.reshape(b * t, *videos.shape[2:]).to(self.device, dtype=torch.float32)
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        x = normalize_frames(x, IMAGENET_MEAN, IMAGENET_STD).to(self.dtype)
        out = self._forward_visual(self.visual, x.view(b, t, 3, self.input_size, self.input_size))
        hidden = getattr(out, "last_hidden_state", None)
        if hidden is None:
            hidden = getattr(out, "pooler_output")
        emb = hidden.mean(dim=1) if hidden.dim() == 3 else hidden
        emb = l2_normalize_tensor(emb.float())
        if not self.native_vl:
            emb = emb[:, : self.embedding_dim]
        return emb.cpu()

    @torch.no_grad()
    def encode_text(self, texts, truncate: bool = True) -> torch.Tensor:
        tokens = self.text_tokenizer(
            list(texts), padding=True, truncation=True, max_length=77, return_tensors="pt"
        ).to(self.device)
        feats = _as_tensor(self.text_model.get_text_features(**tokens))
        feats = l2_normalize_tensor(feats.float())
        if truncate and not self.native_vl:
            feats = feats[:, : self.embedding_dim]
        return feats.cpu()

    def get_embedding_dim(self) -> int:
        return int(self.embedding_dim)
