import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer

from data.data_utils import normalize_frames
from retrieval.retrieval_utils import l2_normalize_tensor
from utils.common import resolve_device

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _as_tensor(output):
    return output if torch.is_tensor(output) else output.pooler_output


class CLIPEncoder:
    def __init__(self, config: dict, variant: str = "primary"):
        mcfg = config["models"]["clip"]
        self.variant = variant
        self.model_name = mcfg.get(variant) or mcfg["primary"]
        self.device = resolve_device(mcfg.get("device", "auto"))
        self.input_size = int(mcfg.get("input_size", config["video"].get("frame_size", 224)))
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = CLIPModel.from_pretrained(self.model_name, torch_dtype=self.dtype).to(self.device).eval()
        self.tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
        self.processor = CLIPImageProcessor.from_pretrained(self.model_name)

    @torch.no_grad()
    def encode_video(self, videos: torch.Tensor) -> torch.Tensor:
        if videos.dim() == 4:
            videos = videos.unsqueeze(0)
        b, t = videos.shape[:2]
        x = videos.reshape(b * t, *videos.shape[2:]).to(self.device, dtype=torch.float32)
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        x = normalize_frames(x, CLIP_MEAN, CLIP_STD).to(self.dtype)
        feats = _as_tensor(self.model.get_image_features(pixel_values=x))
        feats = feats.view(b, t, -1).mean(dim=1)
        return l2_normalize_tensor(feats.float()).cpu()

    @torch.no_grad()
    def encode_text(self, texts) -> torch.Tensor:
        tokens = self.tokenizer(
            list(texts), padding=True, truncation=True, max_length=77, return_tensors="pt"
        ).to(self.device)
        feats = _as_tensor(self.model.get_text_features(**tokens))
        return l2_normalize_tensor(feats.float()).cpu()

    def get_embedding_dim(self) -> int:
        return int(self.model.config.projection_dim)
