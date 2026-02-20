# PHILOS Docker image — development & training
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip \
    git curl wget \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python setup
RUN ln -sf /usr/bin/python3.10 /usr/bin/python
WORKDIR /app

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]" || pip install --no-cache-dir numpy torch gymnasium fastapi uvicorn pydantic pyyaml omegaconf tensorboard

# Copy source
COPY . .

# Install the package
RUN pip install --no-cache-dir -e .

# Default: run API server
EXPOSE 8000
CMD ["python", "scripts/launch_api.py", "--host", "0.0.0.0", "--port", "8000"]
