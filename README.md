# HOUSE PRICE PREDICTION MLOPS PROJECT

---

## TABLE OF CONTENTS
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Project Structure](#project-structure)
5. [Setup and Installation](#setup-and-installation)
   - [Prerequisites](#prerequisites)
   - [Local Setup](#local-setup)
   - [DVC Configuration](#dvc-configuration)
   - [MLflow Tracking Server](#mlflow-tracking-server)
6. [Usage](#usage)
   - [Running the Training Pipeline](#running-the-training-pipeline)
   - [Running the FastAPI Application Locally](#running-the-fastapi-application-locally)
   - [Making Predictions (API Usage)](#making-predictions-api-usage)
7. [Dockerization](#dockerization)
   - [Building the Docker Image](#building-the-docker-image)
   - [Running the Docker Container](#running-the-docker-container)
8. [Cloud Deployment (AWS EC2)](#cloud-deployment-aws-ec2)
9. [Contributing](#contributing)
10. [Contact](#contact)

---

## 1. PROJECT OVERVIEW
This project implements an end-to-end Machine Learning Operations (MLOps) pipeline for predicting house prices. It encompasses data versioning, experiment tracking, model training, and API deployment.

**Key Goals:**
- **Reproducibility:** Ensure data, models, and code are versioned for consistent results.
- **Experiment Tracking:** Log and compare machine learning experiments efficiently.
- **Scalable Deployment:** Serve predictions via a robust FastAPI web service.

---

## 2. FEATURES
- **Data Versioning:** Utilizes DVC (Data Version Control) to manage datasets and machine learning models, storing artifacts on Amazon S3.
- **Experiment Tracking:** Integrates MLflow for logging parameters, metrics, and models for each training run.
- **Model Training:** A modular Python pipeline for data processing and model training (Scikit-learn based).
- **FastAPI Service:** A lightweight and high-performance API built with FastAPI for serving house price predictions.
- **Dependency Management:** Uses `uv` for fast and reliable dependency resolution and environment management via `pyproject.toml` and `uv.lock`.
- **Containerization:** Packaged into a Docker image for consistent deployment across environments.
- **Cloud Deployment:** Designed for deployment on AWS EC2.

---

## 3. ARCHITECTURE
The project's architecture involves several interconnected components:

- **Git Repository:** Stores code, DVC pointers (.dvc files) for data/models, and configuration.
- **DVC (Data Version Control):** Links to actual data and model files stored in an S3 bucket (DVC Remote).
- **S3 (Amazon Simple Storage Service):** Serves as the central storage for versioned datasets and trained model artifacts.
- **Local Development Environment:** Where developers run training pipelines and test the API using Python and `uv`.
- **MLflow Tracking Server:** Records all experiment details (parameters, metrics, models) from training runs. This can be local or a remote server.
- **Docker Image:** Contains the FastAPI application code, dependencies (installed by uv), DVC, and Git. It's built to be self-contained for deployment.
- **AWS EC2 Instance (Production Environment):** Hosts the Docker container.
  - **Docker Container:** Runs the FastAPI API on port 8000. On startup, it performs a `dvc pull` to fetch the latest model artifacts from S3.

---

## 4. PROJECT STRUCTURE
```
.
├── .dockerignore                 # Specifies files to ignore when building Docker image
├── .env.example                  # Example environment variables
├── .github/                      # GitHub Actions workflows (for CI/CD - future)
├── app/                          # FastAPI application
│   ├── main.py                   # Main FastAPI app entry point
│   ├── utils.py                  # Utility functions (e.g., for model loading)
│   └── ...
├── data/                         # Raw and processed data
│   ├── raw_data.csv.dvc          # DVC pointer for raw data
│   └── processed_data.parquet.dvc# DVC pointer for processed data
├── Dockerfile                    # Docker build instructions
├── demo.py                       # Example script for API interaction
├── logs/                         # Application logs
├── models/                       # Trained models and preprocessors
│   ├── preprocessor.joblib.dvc   # DVC pointer for preprocessor
│   └── final_model.joblib.dvc    # DVC pointer for final model
├── notebooks/                    # Jupyter notebooks for EDA, experimentation
├── pyproject.toml                # Project metadata and dependencies (Poetry/uv)
├── README.md                     # This file (Markdown version)
├── requirements.txt              # Alternative for pip installation (generated/manual)
├── src/                          # Source code for ML pipeline
│   ├── __init__.py
│   ├── data_processing.py        # Scripts for data ingestion and preprocessing
│   ├── train_model.py            # Script for model training and MLflow logging
│   └── evaluate_model.py         # Script for model evaluation
├── tests/                        # Unit and integration tests
├── uv.lock                       # Lock file for `uv` managed dependencies
└── setup.py                      # Setuptools configuration for package installation
```

---

## 5. SETUP AND INSTALLATION

### Prerequisites
Before you begin, ensure you have the following installed:

- **Git:** For version control.
- **Python 3.11+:** The project uses specific Python features and dependencies.
- **uv:** Install `uv` globally or in your base environment:
  ```sh
  pip install uv
  # or
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
  Ensure `uv`'s install location is in your PATH.
- **Docker Desktop:** For containerization.
- **AWS CLI:** Configured with credentials that have access to your S3 bucket (for DVC) and ECR (for Docker images).
  ```sh
  aws configure
  ```

### Local Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/mohitDora/house-price-prediction-mlops.git
   cd house-price-prediction-mlops
   ```
2. Create and activate a virtual environment with `uv`:
   ```sh
   uv venv
   source .venv/bin/activate  # On Linux/macOS
   # .venv\Scripts\activate   # On Windows
   ```
3. Install dependencies using `uv sync`:
   ```sh
   uv sync
   ```
   This will install all dependencies listed in `pyproject.toml` and `uv.lock`.
4. Set up environment variables:
   Create a `.env` file in the project root based on `.env.example`:
   ```sh
   # .env
   AWS_ACCESS_KEY_ID="your_aws_access_key_id"
   AWS_SECRET_ACCESS_KEY="your_aws_secret_access_key"
   MONGO_URI=...
   MONGO_DB_NAME=...
   MONGO_COLLECTION_NAME=...
   ```
   **Security Note:** For production, avoid hardcoding AWS credentials. Use IAM Roles (EC2 roles) or AWS Secrets Manager.

### DVC Configuration
1. Initialize DVC (if not already done):
   ```sh
   dvc init --no-scm # If you want to use DVC without Git tracking
   # OR dvc init # If you want DVC to integrate with Git
   ```
   (The repo should already be DVC-initialized if you cloned it).
2. Configure DVC Remote (S3):
   ```sh
   dvc remote add -d s3_remote s3://<YOUR_S3_BUCKET_NAME>/dvc-cache
   ```
   Replace `<YOUR_S3_BUCKET_NAME>` with your S3 bucket. This bucket will store your DVC cache (actual data/model files).
3. Pull Data and Models:
   ```sh
   dvc pull
   ```
   This command will download the versioned data and model files from your S3 DVC remote into your `data/` and `models/` directories.

### MLflow Tracking Server
For local development, you can run an MLflow tracking server:
```sh
mlflow ui --host 0.0.0.0 --port 5000
```
Then access the MLflow UI at [http://localhost:5000](http://localhost:5000). For production, you'd host this on a dedicated server (e.g., EC2) and update `MLFLOW_TRACKING_URI` in your `.env` file.

---

## 6. USAGE

### Running the Training Pipeline
The training pipeline consists of data processing and model training.

1. Ensure your `.env` is configured and DVC remote is set up.
2. Run the training script:
   ```sh
   python src/train_model.py
   ```
   This script will perform data processing, train the model, log the experiment to MLflow, and update the DVC-tracked model artifacts.

### Running the FastAPI Application Locally
1. Ensure you have pulled the latest data and models using `dvc pull`.
2. Activate your virtual environment.
3. Run the FastAPI application:
   ```sh
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```
   The `--reload` flag is useful for local development.

   Your API will be accessible at [http://localhost:8000](http://localhost:8000). You can access the API documentation (Swagger UI) at [http://localhost:8000/docs](http://localhost:8000/docs).

### Making Predictions (API Usage)
You can test the API using `curl` or a tool like Postman/Insomnia.

**Example Prediction Request:**
```sh
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "total_sqft": 1500,
  "bath": 2,
  "bhk": 3,
  "location": "Electronic City"
}'
```

---

## 7. DOCKERIZATION
The project is containerized using Docker for consistent and reproducible deployments.

### Building the Docker Image
Ensure you are in the root directory of the project where `Dockerfile` and `.dockerignore` are located.

```sh
docker build -t house-price-predictor-api:latest .
```
This command builds the Docker image named `house-price-predictor-api` with the tag `latest`.

### Running the Docker Container
1. Ensure your `.env` file is prepared as described in Setup and Installation.
2. Run the container:
   ```sh
   docker run -d \
     --name house-price-api \
     -p 8000:8000 \
     --env-file ./.env \
     <DOCKER_USERNAME>/house-price-predictor-api:latest
   ```
   This will run the container in detached mode (`-d`), map port 8000 from the container to your host machine (`-p 8000:8000`), load environment variables from your `.env` file, and name the container `house-price-api`.
3. Verify:
   - Check running containers: `docker ps`
   - View container logs: `docker logs house-price-api`
   - Access the API: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 8. CLOUD DEPLOYMENT (AWS EC2)
This section provides a high-level guide for deploying your Dockerized FastAPI app to an AWS EC2 instance.

1. **AWS IAM Role:** Create an IAM Role for your EC2 instance with permissions for:
   - `AmazonEC2ContainerRegistryReadOnly` (if using ECR)
   - `AmazonS3ReadOnlyAccess` (or more specific bucket policies for DVC data/models)
   - `AmazonS3FullAccess` (if DVC needs to write to S3 during container startup or if you run training on EC2)
2. **EC2 Key Pair:** Generate an EC2 Key Pair for SSH access.
3. **Security Group:** Create a Security Group to allow inbound traffic on:
   - Port 22 (SSH) from your IP or a specific range.
   - Port 8000 (FastAPI) from `0.0.0.0/0` (for public access).
4. **Launch EC2 Instance:**
   - Choose a suitable AMI (e.g., Ubuntu Server, Amazon Linux).
   - Select an instance type (e.g., `t3.medium`).
   - Attach the IAM Role and Security Group created above.
   - Enable Auto-assign Public IP.
5. **SSH into EC2 and Install Docker:**
   - `ssh -i /path/to/your-key.pem ec2-user@<YOUR_EC2_PUBLIC_IP>`
   - Install Docker (refer to previous instructions for commands).
6. **Push Docker Image to ECR (Recommended):**
   - Create an ECR repository in your AWS account.
   - Tag your local Docker image and push it to ECR.
   - On EC2, authenticate Docker with ECR and pull the image.
7. **Create `.env` file on EC2:**
   - `mkdir ~/app-data`
   - `nano ~/app-data/.env` and add your production environment variables (e.g., MLflow Tracking URI, AWS region).
8. **Run Docker Container on EC2:**
   ```sh
   docker run -d \
     --name house-price-api \
     -p 8000:8000 \
     --env-file ~/app-data/.env \
     <DOCKER_USERNAME>/house-price-predictor-api:latest
   ```
9. **Verify:** Access your API at `http://<YOUR_EC2_PUBLIC_IP>:8000`.

---

## 10. CONTACT
Mohit Kumar Dora - doramohitkumar@gmail.com

**Project Link:** [https://github.com/mohtiDora/house-price-prediction-with-mlops](https://github.com/mohtiDora/house-price-prediction-with-mlops)
