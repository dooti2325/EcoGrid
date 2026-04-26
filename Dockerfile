# Stage 1: Runtime
FROM python:3.10-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Install dependencies using uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# Copy application code
COPY . .

# Expose server port (Hugging Face Spaces default)
EXPOSE 7860

# Health check
HEALTHCHECK CMD curl --fail http://localhost:7860/health || exit 1

# Run the FastAPI server via uv
CMD ["uv", "run", "--project", ".", "server", "--port", "7860", "--host", "0.0.0.0"]
