"""Shared pytest fixtures.

Loading the model is expensive (~5s on CPU), so we load it once per session
and reuse it across tests via ``functools.lru_cache`` in miniserve.model.
"""

from __future__ import annotations

import os

# Quiet the Windows-only "HF cache uses symlinks" warning — it's about disk
# usage on Win, not correctness. Set before transformers/huggingface_hub import.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import pytest

from miniserve.model import DEFAULT_MODEL, load


@pytest.fixture(scope="session")
def model_and_tokenizer():
    return load(DEFAULT_MODEL, device="cpu")
