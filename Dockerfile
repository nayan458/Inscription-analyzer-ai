# -------- Stage 1: Builder --------
FROM python:3.13-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install build tools + git
RUN apt-get update && apt-get install -y \
    gcc g++ libffi-dev git && rm -rf /var/lib/apt/lists/*

# # Install build dependencies (needed for some Python packages)
# RUN apt add --no-cache gcc musl-dev libffi-dev git

# Upgrade pip and install uv
RUN pip install --upgrade pip && pip install uv

# Copy requirements
COPY requirements.txt .

ENV UV_HTTP_TIMEOUT=120
# Install dependencies into a temporary directory
COPY wheels/ /wheels/
RUN uv pip install --no-cache-dir --system /wheels/* || true
RUN uv pip install --no-cache-dir --system \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# RUN uv pip install --no-cache-dir --system -r requirements.txt


# -------- Stage 2: Runtime --------
FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local /usr/local

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the app
CMD ["uvicorn", "app_api:app", "--host", "0.0.0.0", "--port", "8000"]
