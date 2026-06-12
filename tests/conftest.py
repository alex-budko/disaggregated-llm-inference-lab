"""Shared pytest fixtures.

Loading the model is expensive (~5s on CPU), so we load it once per session
and reuse it across tests via ``functools.lru_cache`` in miniserve.model.
"""

from __future__ import annotations

import pytest

from miniserve.model import DEFAULT_MODEL, load


@pytest.fixture(scope="session")
def model_and_tokenizer():
    return load(DEFAULT_MODEL, device="cpu")
