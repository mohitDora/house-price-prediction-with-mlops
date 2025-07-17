FROM python:3.11-slim

# Set environment variables for non-interactive operations and better caching
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install DVC as it's needed to pull data/model artifacts
# We need to install git as well for dvc to function properly with git-tracked .dvc files
RUN pip install --no-cache-dir dvc[s3] 
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml and poetry.lock (or requirements.txt) first to leverage Docker cache
# Use uv for dependency management inside Docker
COPY pyproject.toml /app/
# If you're using poetry.lock instead of uv.lock/requirements.txt:
# COPY poetry.lock /app/

# Install uv (if you want to use uv for dependency management inside docker)
RUN pip install uv

# Install project dependencies using uv sync
# This will install all dependencies from pyproject.toml
RUN uv sync --system 

# Copy the rest of your application code
COPY . /app/

# Expose the port that FastAPI will run on
EXPOSE 8000

# Set the command to run your FastAPI application using Uvicorn
# --host 0.0.0.0 makes the server accessible from outside the container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]