"""Instruction-tuned causal LM used for video story narration and Q&A.

The repo's flan-t5 reporter is fast but produces short, low-quality stories.
A small instruction-tuned chat model gives chronological, readable narrations
while still running comfortably on CPU.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.common import get_logger, resolve_device

logger = get_logger("story_narrator")

DEFAULT_STORY_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


class StoryNarrator:
    def __init__(self, model_id: str = None, device: str = "auto"):
        self.model_id = model_id or DEFAULT_STORY_MODEL
        self.device = resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, dtype=dtype
        ).to(self.device).eval()

    @torch.no_grad()
    def narrate(self, prompt: str, max_new_tokens: int = 220) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a CCTV video narrator for an incident-analysis app. "
                    "Write clear, simple English and never invent details that are "
                    "not supported by the given facts."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
