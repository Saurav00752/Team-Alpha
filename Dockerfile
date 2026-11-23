FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements-prod.txt .
RUN pip install --upgrade pip && pip wheel --wheel-dir /wheels -r requirements-prod.txt

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip && rm -rf /var/lib/apt/lists/*
WORKDIR /service
COPY requirements-prod.txt .
COPY --from=builder /wheels /wheels
RUN python3 -m pip install --no-index --find-links=/wheels -r requirements-prod.txt
COPY app ./app
RUN mkdir -p data/active_learning
EXPOSE 8000
CMD ["python3", "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]

