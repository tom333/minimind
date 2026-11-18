# Image minimind : entraînement (PyTorch CUDA) + code Dagster (études Optuna).
# Les données et les sorties ne sont PAS dans l'image : volumes du nœud montés dans /data.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.12 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/app/.venv \
    PATH=/app/.venv/bin:$PATH \
    HOME=/app \
    PYTHONUNBUFFERED=1 \
    MINIMIND_DATA_DIR=/data/corpus \
    MINIMIND_OUT_DIR=/data/out \
    MLFLOW_TRACKING_URI=http://mlflow.ia-lab.svc.cluster.local:5000

WORKDIR /app
# gcc : torch.compile (Inductor) compile des extensions à l'exécution
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ libc6-dev \
    && rm -rf /var/lib/apt/lists/*
# Couche lourde et stable (PyTorch + CUDA) séparée des dépendances légères
RUN uv venv /app/.venv && uv pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
COPY deploy/requirements.txt deploy/requirements.txt
RUN uv pip install -r deploy/requirements.txt

# Dagster retire le répertoire de travail de sys.path après avoir chargé le code : sans
# PYTHONPATH, les imports faits pendant un run (trainer.study) échouent.
ENV PYTHONPATH=/app
COPY model model
COPY dataset/__init__.py dataset/lm_dataset.py dataset/
COPY trainer trainer
COPY orchestration orchestration
RUN useradd --uid 1000 --home-dir /app dagster && chown dagster /app \
    # Le K8sRunLauncher monte dagster.yaml dans /opt/dagster/dagster_home et y écrit
    && mkdir -p /opt/dagster/dagster_home && chown -R dagster /opt/dagster

USER 1000
EXPOSE 4000
CMD ["dagster", "api", "grpc", "-m", "orchestration.definitions", "-a", "defs", "-h", "0.0.0.0", "-p", "4000"]
