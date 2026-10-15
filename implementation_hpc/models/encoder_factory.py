import logging
from typing import Optional

from utils.common import clear_gpu_cache
from .clip_encoder import CLIPEncoder
from .vljepa_encoder import VLJEPAEncoder

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = {
    "clip": CLIPEncoder,
    "vljepa": VLJEPAEncoder,
}


def get_encoder(model_name: str, config: dict, predictor_checkpoint: Optional[str] = None):
    """Factory function instantiating requested vision-language encoder."""
    name_clean = model_name.lower().replace("-", "_")
    clear_gpu_cache()

    if name_clean == "clip":
        return CLIPEncoder(config=config, variant="primary")
    if name_clean == "clip_base":
        return CLIPEncoder(config=config, variant="secondary")
    if name_clean == "vljepa":
        return VLJEPAEncoder(config=config, predictor_checkpoint=predictor_checkpoint)

    raise ValueError(f"Unknown model name '{model_name}'. Choose from: {list(SUPPORTED_MODELS.keys())}")
