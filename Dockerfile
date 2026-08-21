FROM python:3.11-slim

WORKDIR /app

# Install system deps for building native packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv

# Copy dependency spec first (cache layer)
COPY pyproject.toml ./
RUN uv pip install --system -e . || pip install --no-cache-dir -e .

# Copy application code
COPY . .

# Generate synthetic data + train model if not present
RUN python data/synthetic_generator.py && \
    python src/tabular/train.py && \
    python src/graph/batch_score.py

# Non-root user for security
RUN useradd --create-home sentinel
USER sentinel

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
