# Convenience targets. On Windows, use `make` from Git Bash, or run the
# python commands directly from PowerShell.

MODEL ?= HuggingFaceTB/SmolLM2-135M-Instruct

.PHONY: install test bench-kv bench-prefix bench-disagg serve-mono serve-disagg up down

install:
	pip install -r requirements.txt

test:
	pytest -q

bench-kv:
	python -m benchmarks.bench_kv_cache --model $(MODEL)

bench-prefix:
	python -m benchmarks.bench_prefix_cache --model $(MODEL)

bench-disagg:
	python -m benchmarks.bench_disagg

serve-mono:
	python -m miniserve.monolithic_server

up:
	docker compose up --build

down:
	docker compose down
