# Disaggregated LLM Inference Lab

A small LLM serving stack that demonstrates — head-on — three optimizations
that determine how modern inference engines (vLLM, SGLang, TensorRT-LLM)
behave in production:

1. **KV cache** — without it, decoding is quadratic in output length.
2. **Prefix caching** — repeated long-context prompts skip prefill almost entirely.
3. **Disaggregated prefill** — prompt processing and token generation on
   separate workers so heavy prefills don't head-of-line-block decodes.

The pedagogical core (`miniserve/`) is ~500 lines on top of HuggingFace
`transformers` — every concept is a toggle you can read, compare, and
benchmark. The same repo also benchmarks real **vLLM** as the production
baseline (`benchmark_vllm.py`), so you can honestly say *"this is what the
concept does; this is what the real system delivers."*

Default model is `HuggingFaceTB/SmolLM2-135M-Instruct` so the whole stack
runs on a CPU laptop in seconds. Set `MODEL_NAME` to use anything bigger.

---

## Concepts at a glance

### 1. KV cache

A transformer's self-attention turns the prompt `x_1..x_n` into key/value
tensors `(K, V)`. To predict token `n+1`, the model attends over all of
`K, V`. To predict token `n+2`, *naively* you'd re-attend over `n+1`
tokens, recomputing the same `K, V` you already had. The KV cache keeps
those tensors around so each new token costs **one** forward pass on
**one** new position, not on the full growing sequence.

Without KV cache: O(N²) total work to generate N tokens.
With KV cache: O(N) total work.

`benchmarks/bench_kv_cache.py` sweeps output length and plots both curves.

### 2. Prefix caching

Two requests sharing a long system prompt should not pay to prefill it
twice. Prefix caching keys the KV state by prompt-prefix token IDs and
reuses it across requests. On a *full* hit we even cache the final-token
logits, so the second request does **zero** model forward passes during
prefill.

vLLM and SGLang use a radix tree at block granularity; we use a simpler
longest-prefix exact match (`miniserve/prefix_cache.py`). Same principle,
fewer moving parts.

`benchmarks/bench_prefix_cache.py` runs 8 user queries against an 80-line
system prompt twice — cold then warm — and shows the prefill time collapse.

### 3. Disaggregated prefill

Prefill is *compute-bound* (one big GEMM over a long sequence). Decode is
*memory-bound* (many tiny matrix-vector ops over a tall KV cache). On the
same hardware they fight each other: a long prefill blocks every decode in
flight, spiking everyone's TTFT.

Disaggregated serving puts prefill on one worker and decode on another.
Each worker scales independently, and a heavy prefill no longer freezes
all in-flight decodes. The cost is shipping the KV cache between workers
— production systems use NCCL/RDMA/shared GPU memory; this repo uses
HTTP+pickle because the goal is to make the *concept* legible.

```
              Client
                │
                ▼
        ┌──────────────────┐
        │     Gateway      │       OpenAI-ish /v1/generate
        │   (port 8000)    │
        └─────┬────────────┘
              │
   ┌──────────┘   ┌──────────────────┐
   ▼              ▼                  │
┌──────────┐   ┌────────────────────┐│
│ Prefill  │──►│  Decode worker     ││
│ worker   │   │  (port 8002)       │◄┘ KV cache shipped over HTTP
│ (8001)   │   │  many tiny GEMVs   │
│ big GEMM │   └────────────────────┘
└──────────┘
```

`benchmarks/bench_disagg.py` compares TTFT distributions between the
monolithic and disagg topologies under concurrent load.

---

## Architecture

```
miniserve/
  engine.py            ← the heart: generate() with toggles, plus prefill()/decode_from()
  prefix_cache.py      ← longest-prefix exact match, LRU eviction, hit/miss stats
  serialization.py     ← KV-cache transport for the disagg path
  model.py             ← HuggingFace load() with lru_cache
  metrics.py           ← Prometheus counters / histograms
  schemas.py           ← Pydantic v2 request/response models
  monolithic_server.py ← single-process baseline (port 8000)
  prefill_server.py    ← disagg prefill worker (port 8001)
  decode_server.py     ← disagg decode worker (port 8002)
  gateway.py           ← disagg gateway (port 8000), routes prefill→decode
benchmarks/
  bench_kv_cache.py
  bench_prefix_cache.py
  bench_disagg.py
benchmark_vllm.py      ← real vLLM baseline (separate; needs a vLLM server)
tests/                 ← equivalence tests: optimizations must not change outputs
```

Both topologies expose:

- `POST /v1/generate` — `{prompt, max_tokens, use_prefix_cache}` → text + timings
- `GET  /healthz`
- `GET  /metrics` — Prometheus exposition format

---

## Quickstart

```bash
pip install -r requirements.txt
pytest -q                                    # equivalence tests pass

# Concept demos (each writes CSV + PNG to results/)
python -m benchmarks.bench_kv_cache
python -m benchmarks.bench_prefix_cache

# Live the monolithic server
python -m miniserve.monolithic_server        # :8000
curl -s -X POST localhost:8000/v1/generate \
     -H 'content-type: application/json' \
     -d '{"prompt":"Hello","max_tokens":16}'

# Disagg topology in three terminals
python -m miniserve.prefill_server           # :8001
python -m miniserve.decode_server            # :8002
python -m miniserve.gateway                  # :8000

# Or everything in Docker, with Prometheus on :9090
docker compose up --build
```

To compare topologies under concurrent load you need both running on
different ports — easiest via:

```bash
docker compose --profile mono up --build     # monolithic on :8010, disagg on :8000
python -m benchmarks.bench_disagg \
    --mono-url   http://localhost:8010 \
    --disagg-url http://localhost:8000 \
    -n 16 -c 8
```

---

## Results

> Numbers below are placeholders — run the benchmarks above and paste the
> output in. Each `python -m benchmarks.bench_*` writes a CSV next to a PNG
> in `results/`.

### KV cache (single-stream decode)

| Output tokens | Without KV cache | With KV cache | Speedup |
|---|---|---|---|
| 32 | _t.b.d._ | _t.b.d._ | _t.b.d._ |
| 64 | _t.b.d._ | _t.b.d._ | _t.b.d._ |
| 128 | _t.b.d._ | _t.b.d._ | _t.b.d._ |

### Prefix caching (8 queries sharing a long system prompt)

| Pass | Mean prefill | Mean TTFT |
|---|---|---|
| Cold (cache miss) | _t.b.d._ | _t.b.d._ |
| Warm (cache hit) | _t.b.d._ | _t.b.d._ |

### Disaggregated vs monolithic (TTFT, concurrency=8)

| Topology | p50 TTFT | p95 TTFT |
|---|---|---|
| Monolithic | _t.b.d._ | _t.b.d._ |
| Disaggregated | _t.b.d._ | _t.b.d._ |

### vLLM baseline

```
python benchmark_vllm.py --model <name> -n 50 -c 4 -o results/vllm.csv
```

---

## Honest caveats

- The toy engine is for **understanding**, not production performance.
  vLLM does paged attention, continuous batching, chunked prefill, and a
  radix-tree prefix cache — most of which are out of scope here.
- KV-cache shipping over HTTP+pickle is intentionally slow. Real disagg
  systems use NCCL / RDMA / shared GPU memory. The benchmark therefore
  highlights *queueing* benefits, not transport benefits.
- Prefix cache here is exact longest-prefix match. Real implementations
  match at fixed block boundaries via a radix tree (vLLM, SGLang).
- All servers serialize calls to the single model with a `threading.Lock`.
  Real engines schedule batches across requests; that's the actual
  source of throughput in production.

## Layout choices worth knowing

- `MODEL_NAME` env var selects the model; default keeps everything on CPU.
- `miniserve.model.load()` is `lru_cache`-d, so multiple servers in one
  process share the same weights.
- The disagg prefill worker owns the prefix cache; the decode worker is
  stateless. That mirrors what production splits actually do.
