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

# Copy pyproject.toml and poetry.lock (or requirements.txt) first to leverage Docker cache
COPY pyproject.toml /app/
COPY uv.lock /app/
# If you're using poetry.lock instead of uv.lock/requirements.txt:
# COPY poetry.lock /app/

# Install uv (if you want to use uv for dependency management inside docker)
RUN pip install uv

# Install project dependencies using uv sync
# This will install all dependencies from pyproject.toml
RUN uv sync

# Copy the rest of your application code, including .dvc/config and .dvc pointer files
COPY . /app/

# Expose the port that FastAPI will run on
EXPOSE 8000

# Set the command to run your FastAPI application using Uvicorn
# The model artifacts will be pulled from S3 by DVC via load_model_artifacts at app startup
You're facing a classic Docker/Python path issue: exec: "uvicorn": executable file not found in $PATH: unknown.

This error means that when your Docker container tries to execute the uvicorn command (as specified in your CMD instruction), it can't find uvicorn in any of the directories listed in the container's PATH environment variable.

Even though uv sync installs uvicorn as part of your project's dependencies, sometimes the exact location where uvicorn's executable script is placed by uv (or pip) might not be on the default system PATH within the python:3.11-slim base image.

The Fix: Use Python's Module Execution
The most robust and recommended way to run Python scripts/executables installed by pip or uv inside a Docker container is to execute them as Python modules using python -m. This ensures that Python itself finds the module within its installed packages, regardless of the system's PATH configuration.

Here's how to fix your Dockerfile:

Dockerfile

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
