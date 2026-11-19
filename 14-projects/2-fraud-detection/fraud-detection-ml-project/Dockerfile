# Serving image. Build after `make train`: it bakes in the current model from artifacts/.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Pinned runtime dependencies first: this layer stays cached until the lock file changes.
COPY requirements.lock ./
RUN pip install -r requirements.lock

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

# Only the current model; the runs/ history stays out of the image (see .dockerignore).
COPY artifacts/fraud_pipeline.joblib artifacts/model_metadata.json ./artifacts/

RUN useradd --create-home --uid 10001 appuser
USER appuser

ENV FRAUD_ARTIFACT_DIR=/app/artifacts \
    FRAUD_LOG_FORMAT=json
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=3)"

# One process per container; scale by running more replicas behind a load balancer.
CMD ["uvicorn", "fraud_ml.api:app", "--host", "0.0.0.0", "--port", "8000"]
