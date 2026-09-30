import logging

import torch
from transformers import AutoTokenizer

from reporting.prompt_templates import build_report_prompt
from utils.common import resolve_device

logger = logging.getLogger(__name__)


class IncidentReporter:
    def __init__(self, config: dict):
        lcfg = config["llm"]
        self.cfg = lcfg
        self.model_name = lcfg["model"]
        self.device = resolve_device(lcfg.get("device", "auto"))
        self.max_new_tokens = int(lcfg.get("max_new_tokens", 512))
        self.temperature = float(lcfg.get("temperature", 0.3))
        self.max_input_tokens = int(lcfg.get("max_input_tokens", 1024))
        quant = str(lcfg.get("quantization") or "none").lower()
        load_kwargs = {"torch_dtype": torch.float16 if self.device == "cuda" else torch.float32}
        if quant in ("4bit", "8bit"):
            try:
                from transformers import BitsAndBytesConfig

                load_kwargs.update(
                    device_map="auto",
                    quantization_config=BitsAndBytesConfig(
                        load_in_4bit=(quant == "4bit"), load_in_8bit=(quant == "8bit")
                    ),
                )
            except ImportError:
                logger.warning("bitsandbytes unavailable - loading %s unquantized", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.seq2seq = None
        self.model = self._load_model(load_kwargs)

    def _load_model(self, load_kwargs):
        from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, **load_kwargs)
            self.seq2seq = True
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)
            self.seq2seq = False
        if "device_map" not in load_kwargs:
            model = model.to(self.device)
        return model.eval()

    def generate_report(self, evidence, query: str, grounded: bool = True) -> str:
        prompt = build_report_prompt(evidence, query, grounded=grounded)
        return self._generate(prompt)

    def generate_from_prompt(self, prompt: str, max_new_tokens: int | None = None) -> str:
        return self._generate(prompt, max_new_tokens=max_new_tokens)

    @torch.no_grad()
    def _generate(self, prompt: str, max_new_tokens: int | None = None) -> str:
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_input_tokens
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        gen_kwargs = {"max_new_tokens": int(max_new_tokens or self.max_new_tokens)}
        if self.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=self.temperature)
        output = self.model.generate(**inputs, **gen_kwargs)
        text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        if not self.seq2seq and text.startswith(prompt):
            text = text[len(prompt):]
        return text.strip()
