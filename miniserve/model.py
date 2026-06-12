"""Model + tokenizer loading. One small CPU-friendly default."""

from __future__ import annotations

import os
from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = os.environ.get("MODEL_NAME", "HuggingFaceTB/SmolLM2-135M-Instruct")


@lru_cache(maxsize=4)
def load(name: str = DEFAULT_MODEL, device: str = "cpu"):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    dtype = torch.float32 if device == "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
    model.to(device).eval()
    return model, tok
