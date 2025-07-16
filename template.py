import os

project_name = "src"

# Folder structure
folders = [
    ".github/workflows/",
    "data/",
    "models/",
    "notebooks/",
]

# Files to create with their relative paths
files = [
    f"{project_name}/__init__.py",
    f"{project_name}/config.py",
    f"{project_name}/logger.py",
    f"{project_name}/exception.py",
    f"{project_name}/data_ingestion.py",
    f"{project_name}/data_processing.py",
    f"{project_name}/train_model.py",
    f"{project_name}/predict_model.py",
    f"{project_name}/utils.py",
    "app.py",
    "requirements.txt",
    "Dockerfile",
    ".dockerignore",
    "setup.py",
]


def create_structure():
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Created folder: {folder}")

    for file in files:
        file_path = os.path.join(file)
        dir_name = os.path.dirname(file_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        with open(file_path, "w") as f:
            pass  # create empty file
        print(f"Created file: {file_path}")


if __name__ == "__main__":
    create_structure()
