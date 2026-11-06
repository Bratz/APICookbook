# The API image for the OCI VM (deploy/oci/docker-compose.yml). Local development never uses it.
# Build from the repo root:  docker build -t cookbook-api .
# Python pinned to the interpreter the test suite passed on.
FROM python:3.10.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/models/hf
WORKDIR /app

# CPU-only torch first: the default x86 wheel drags in ~3 GB of CUDA the VM has no use for.
# sentence-transformers (requirements.txt) then finds it satisfied and leaves it alone.
# Extra index, not --index-url: torch's own dependencies must come from PyPI (the PyTorch index has
# only sdists of some for arm64, and building them needs tools it doesn't carry). `+cpu` forces the CPU build.
RUN pip install --no-cache-dir torch==2.12.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ backend/
COPY frontend/ frontend/
COPY api_specs/ api_specs/
COPY collections/ collections/

# models/ holds the downloaded bge model (HF_HOME) and the RAG index cache: a volume in compose,
# so only the first start pays the download and the ~2 min embedding build.
RUN useradd --create-home --uid 10001 app && mkdir -p logs models && chown app logs models
USER app
EXPOSE 7600
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7600", "--proxy-headers", "--forwarded-allow-ips", "*"]
