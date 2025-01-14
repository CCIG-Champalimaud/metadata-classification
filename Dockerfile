FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

COPY uv.lock pyproject.toml README.md /app/

RUN uv sync --frozen --no-install-project --no-dev --no-cache

RUN mkdir -p /models

COPY models/xgb.standard.100.pkl /models
COPY src src
RUN uv sync --frozen --no-dev --no-cache --extra xgboost --extra serve
RUN uv clean
COPY config-api-docker.yaml config-api.yaml

# Reset the entrypoint, don't invoke `uv`
ENTRYPOINT ["uv", "run"]