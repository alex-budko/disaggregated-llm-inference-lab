# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A mini LLM serving system for understanding three concepts head-on:

- **KV cache** — avoid re-attending over the full prompt for every new token
- **Prefix caching** — skip prefill on repeated long-context prompts
- **Disaggregated prefill** — run the prompt forward pass and the token-by-token generation on separate workers so they don't head-of-line-block each other under concurrent load

Design intent is *hybrid*: the core engine in `miniserve/` is built from scratch on top of HuggingFace `transformers` so the mechanics are readable, while `benchmark_vllm.py` benchmarks real vLLM as the "production baseline." Numbers in any write-up should keep that framing — the toy engine teaches, vLLM is the realistic target.

## Architecture

The pedagogical heart is `miniserve/engine.py`. Every concept is a toggle on a single `generate()` function — that's deliberate, so a reader can see exactly what each optimization changes:

- `use_kv_cache=False` re-encodes the full growing sequence each decode step. The slow baseline.
- `prefix_cache=<PrefixCache>` skips prefill for any matched prompt prefix. A *full* hit also caches the final logits, so no model forward runs at all.
- `prefill()` and `decode_from()` are split entry points so a separate process can do prefill and ship the resulting KV state to a separate decode process.

Two serving topologies. Both expose `POST /v1/generate` with the same schema (`miniserve/schemas.py`):

| Topology | Processes | Default ports |
|---|---|---|
| Monolithic | `miniserve.monolithic_server` (one process holds the model + prefix cache) | 8000 |
| Disaggregated | `miniserve.prefill_server` + `miniserve.decode_server` + `miniserve.gateway` | 8001 / 8002 / 8000 |

KV-cache transport between disagg workers is `torch.save` → base64 → HTTP JSON (`miniserve/serialization.py`). That is deliberately slow; the goal is to make the *concept* legible. Real disaggregated systems (vLLM / MoonCake / NIXL) move KV blocks via NCCL / RDMA / shared GPU memory. Mention this caveat when discussing disagg-vs-monolithic numbers.

Default model is `HuggingFaceTB/SmolLM2-135M-Instruct` so the whole stack runs on CPU. Override via `MODEL_NAME` env var. `miniserve.model.load()` is `lru_cache`-d, so repeated calls in the same process reuse the loaded weights.

## Critical invariants

- The KV cache and prefix cache are mathematically equivalent to the slow baseline. `tests/test_kv_cache_equivalence.py` and `tests/test_prefix_cache.py::test_partial_hit_is_correct` enforce that — any new optimization must keep them green. Don't relax these into approximate-equality checks; if outputs diverge, the implementation is wrong.
- All disagg processes must load the same `MODEL_NAME`. KV-cache tensor shapes are model-specific; a prefill/decode mismatch crashes the decode worker on the first request.
- `_model_lock` (and `_lock` in the disagg workers) serializes calls into a single torch model — PyTorch models are not safe to call concurrently from multiple threads in one process. Don't remove these without replacing them with batched scheduling.

## Commands

```
make install                                       # pip install -r requirements.txt
make test                                          # pytest -q
pytest tests/test_prefix_cache.py::test_full_hit_skips_prefill -v   # single test
make serve-mono                                    # monolithic server on :8000
python -m miniserve.prefill_server                 # disagg prefill worker, :8001
python -m miniserve.decode_server                  # disagg decode worker, :8002
python -m miniserve.gateway                        # disagg gateway, :8000
python benchmark_vllm.py --model <name> -n 20 -c 4 # vLLM baseline (requires a running vLLM server)
```

Every server exposes `/healthz` and `/metrics` (Prometheus exposition format). The gateway's `/healthz` also probes its prefill and decode workers.

## Workflow preferences

- Use plan mode before large edits.
- Explain the architecture tradeoff before implementing.
- When adding a new experiment, update the README so the headline result is reproducible from the documented command.
