"""
LLM Generator using self-hosted Llama 3.2 3B (Layer 4, Component 4.2).

Per ANTIGRAVITY_PROMPT.md Phase 4:
  - Model: meta-llama/Llama-3.2-3B-Instruct (4-bit quantized)
  - Framework: transformers + bitsandbytes
  - VRAM: ~3 GB (fits RTX 3050 Ti alongside the embedding model)
  - temperature=0.3 for focused, factual answers
  - max_new_tokens=512

100% OFFLINE after initial model download.
"""

import time
from typing import List, Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)


class LLMGenerator:
    """Self-hosted LLM for constrained answer generation.

    The model is loaded once in 4-bit quantization and kept in memory.
    Call ``generate(prompt)`` for single queries or ``generate_batch``
    for multiple prompts.
    """

    def __init__(
        self,
        model_name: str = "models/llama-3.2-3b",
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        use_4bit: bool = True,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

        print(f"[LLM] Loading {model_name} ...")
        t0 = time.time()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quant_config = None
        load_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.float16,
        }
        if use_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["quantization_config"] = quant_config

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, **load_kwargs,
        )
        self.model.eval()

        elapsed = time.time() - t0
        vram_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        print(f"[LLM] Model loaded in {elapsed:.1f}s  (VRAM: {vram_gb:.1f} GB)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        """Generate a single answer from a formatted prompt.

        Steps (per methodology):
            1. Tokenize prompt
            2. Generate with sampling (temp=0.3)
            3. Decode output
            4. Remove prompt echo
            5. Strip whitespace
        """
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=8192,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the prompt echo from the output
        answer = full_text[len(prompt):].strip()

        # Trim at stop boundaries (methodology spec: stop at QUERY:/CONTEXT:)
        for stop_token in ("QUERY:", "CONTEXT:", "SYSTEM:"):
            idx = answer.find(stop_token)
            if idx != -1:
                answer = answer[:idx].strip()

        return answer

    def generate_batch(self, prompts: List[str]) -> List[str]:
        """Generate answers for multiple prompts sequentially.

        Batch tokenisation is avoided because prompt lengths vary widely
        and padding wastes VRAM on a 4 GB card.
        """
        return [self.generate(p) for p in prompts]

    def get_info(self) -> dict:
        """Return model metadata."""
        return {
            "model_name": self.model_name,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "device": str(self.model.device),
            "vram_gb": round(
                torch.cuda.memory_allocated() / 1e9, 2
            ) if torch.cuda.is_available() else 0,
        }
