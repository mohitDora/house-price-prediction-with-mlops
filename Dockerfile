FROM python:3.11-slim

# Set environment variables for non-interactive operations and better caching
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install DVC as it's needed to pull data/model artifacts
# We need to install git as well for dvc to function properly with git-tracked .dvc files
# Using a single RUN command to combine apt-get and pip installs to reduce image layers
RUN apt-get update && apt-get install -y git && \
    pip install --no-cache-dir "dvc[s3]" && \
    rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml and uv.lock first to leverage Docker cache
# This ensures that if only your application code changes, but dependencies don't,
# Docker can use a cached layer for dependency installation.
COPY pyproject.toml /app/
# IMPORTANT: If you are generating and committing a uv.lock file locally (highly recommended for reproducibility),
# you must uncomment and include the next line:
COPY uv.lock /app/

# Install uv (uv itself is needed to run uv sync)
RUN pip install uv

# Install project dependencies using uv sync from pyproject.toml (and uv.lock if copied)
RUN uv sync

# Copy the rest of your application code, including .dvc/config and .dvc pointer files
COPY . /app/

# Expose the port that FastAPI will run on
EXPOSE 8000

# Set the command to run your FastAPI application using Uvicorn
# Fix: Run uvicorn as a Python module
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
