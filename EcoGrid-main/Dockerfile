FROM python:3.10-slim

WORKDIR /app

# Stable uv install for lock-based, reproducible sync
RUN pip install --no-cache-dir uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LORA_ADAPTER_DIR=/app/lora_adapter

# Install only locked runtime deps first for layer caching
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra train --no-install-project

# Copy source and install project itself
COPY . .
RUN uv sync --frozen --no-dev --extra train

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD /opt/venv/bin/python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/_stcore/health', timeout=3).read()" || exit 1

CMD ["/opt/venv/bin/streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=7860", "--server.headless=true"]
