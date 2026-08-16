# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Base image — slim variant keeps the image small; matches the Python
# version your CI workflow already uses (3.10)
# ---------------------------------------------------------------------------
FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffering stdout/stderr —
# standard for containers so your logs show up immediately, not buffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/fastapi_app

WORKDIR /app

# ---------------------------------------------------------------------------
# Install dependencies first, separately from copying the code.
# Docker caches each layer — as long as requirements.txt doesn't change,
# this layer is reused on every rebuild instead of reinstalling everything,
# which makes iterating on your actual code much faster to rebuild.
# ---------------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Copy only what the running app actually needs — not the whole repo
# (no notebooks/, no .dvc/, no data/raw etc. — see .dockerignore)
# ---------------------------------------------------------------------------
COPY src/ ./src/
COPY fastapi_app/ ./fastapi_app/
COPY models/ ./models/

# Run as a non-root user — basic container security practice
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

WORKDIR /app/fastapi_app

EXPOSE 8000

# Basic container-level health check — Docker will mark the container
# unhealthy if /health stops responding, useful once you deploy this
# behind an orchestrator (Docker Compose, ECS, k8s, etc.)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
