# Reproducible CPU image for the Stillwar pipeline.
#
# Build:
#   docker build -t stillwar:2.0.0 .
# Run the integrity check:
#   docker run --rm stillwar:2.0.0
# Run the full pipeline (needs the raw snapshot, which is copied in):
#   docker run --rm stillwar:2.0.0 python -m stillwar all
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/tmp/uv-cache

WORKDIR /app

RUN pip install --no-cache-dir uv==0.11.8

# Dependency resolution first, so a source-only change reuses the layer.
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
COPY configs ./configs
COPY data/raw ./data/raw
COPY tests ./tests
COPY docs ./docs
COPY tools ./tools
COPY Makefile ./

RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

CMD ["python", "-m", "stillwar", "verify"]
