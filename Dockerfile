# Image minimind : entraînement (PyTorch CUDA) + code Dagster (études Optuna, chaîne de données).
# Les dépendances viennent de pyproject.toml / uv.lock : mêmes versions que l'environnement local.
# Les données et les sorties ne sont PAS dans l'image : volumes du nœud montés dans /data.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.12 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH=/app/.venv/bin:$PATH \
    HOME=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MINIMIND_DATA_DIR=/data/corpus \
    MINIMIND_OUT_DIR=/data/out \
    MINIMIND_MLFLOW_URI=https://mlflow.tgu.ovh \
    HF_HOME=/tmp/huggingface
# PYTHONPATH : Dagster retire le répertoire de travail de sys.path après avoir chargé le code.
# MINIMIND_MLFLOW_URI : MLflow n'accepte que son nom public, et le run launcher Dagster impose
#   MLFLOW_TRACKING_URI à tous les runs avec le nom de service interne, qui est refusé.
# HF_HOME : cache Hugging Face dans le stockage éphémère du pod, hors du disque de données.

WORKDIR /app
# gcc : torch.compile (Inductor) compile des extensions à l'exécution
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ libc6-dev \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# commit du code embarqué : posé après l'installation des dépendances pour ne pas invalider cette couche
ARG GIT_COMMIT=inconnu
ENV MINIMIND_GIT_COMMIT=$GIT_COMMIT \
    MINIMIND_IMAGE=localhost:32000/minimind-dagster:$GIT_COMMIT
COPY model model
COPY dataset/__init__.py dataset/lm_dataset.py dataset/
COPY trainer trainer
COPY corpus corpus
COPY orchestration orchestration
RUN useradd --uid 1000 --home-dir /app dagster && chown dagster /app \
    # Le K8sRunLauncher monte dagster.yaml dans /opt/dagster/dagster_home et y écrit
    && mkdir -p /opt/dagster/dagster_home && chown -R dagster /opt/dagster

USER 1000
EXPOSE 4000
CMD ["dagster", "api", "grpc", "-m", "orchestration.definitions", "-a", "defs", "-h", "0.0.0.0", "-p", "4000"]
