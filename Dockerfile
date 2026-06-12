FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface

WORKDIR /app

# System deps kept minimal; CPU-only torch wheel is small enough.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first to avoid pulling the giant CUDA wheel
# unintentionally on hosts without a GPU.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<3" \
 && pip install -r requirements.txt

COPY miniserve ./miniserve
COPY benchmarks ./benchmarks
COPY benchmark_vllm.py ./

EXPOSE 8000 8001 8002

# Override `command:` in docker-compose to pick which server runs.
CMD ["python", "-m", "miniserve.monolithic_server"]
