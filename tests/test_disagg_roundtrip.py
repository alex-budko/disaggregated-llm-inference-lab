"""The disagg pipeline must produce the same tokens as the monolithic path.

If serialization corrupts the KV cache (wrong dtype, shape, layer ordering),
the decode worker will produce different (or nonsense) tokens. This test
exercises prefill() -> serialize -> deserialize -> decode_from() and
compares to a single in-process generate().
"""

from __future__ import annotations

import base64
import io

import torch

from miniserve.engine import decode_from, generate, prefill
from miniserve.serialization import decode_kv, encode_kv


def _roundtrip_logits(t: torch.Tensor) -> torch.Tensor:
    buf = io.BytesIO()
    torch.save(t.detach().cpu(), buf)
    return torch.load(io.BytesIO(base64.b64decode(base64.b64encode(buf.getvalue()))), weights_only=False)


def test_disagg_matches_monolithic(model_and_tokenizer):
    model, tok = model_and_tokenizer
    prompt = "List three reasons distributed systems are hard:"
    max_tokens = 16

    mono = generate(model, tok, prompt, max_tokens, use_kv_cache=True)

    pre = prefill(model, tok, prompt, use_kv_cache=True)
    kv_b64 = encode_kv(pre.past_key_values)
    past = decode_kv(kv_b64, device="cpu")
    first_logits = _roundtrip_logits(pre.first_token_logits)

    generated, _ = decode_from(
        model, tok,
        past_key_values=past,
        first_token_logits=first_logits,
        max_new_tokens=max_tokens,
    )

    assert generated == mono.output_token_ids, (
        f"disagg/mono divergence:\n  mono={mono.output_token_ids}\n  disagg={generated}"
    )
