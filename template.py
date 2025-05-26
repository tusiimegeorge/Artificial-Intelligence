import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s %(asctime)s %(filename)s]: %(message)s:')

project_name = "swahiliNewsClassifier"


def create_file_with_directories(filepath: Path) -> None:
    """
    Create directories and an empty file if they don't exist.

    Args:
        filepath (Path): Path to the file.
    """
    if filepath.parent != Path(""):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"Creating directory: {filepath.parent}")

    if not filepath.exists() or filepath.stat().st_size == 0:
        filepath.touch()
        logging.info(f"Creating empty file: {filepath}")


list_of_files = [
    ".github/workflows/.gitkeep",
    ".github/workflows/codecov.yml",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/model_training_and_evaluation.py",
    f"src/{project_name}/utilities/_init__.py",
    f"src/{project_name}/utilities/helper_functions.py",
    f"src/{project_name}/configuration/__init__.py",
    f"src/{project_name}/configuration/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/stage_01_data_ingestion.py",
    f"src/{project_name}/pipeline/stage_02_model_training_and_evaluation.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/entities.py",
    f"src/{project_name}/constants/__init__.py",
    "tests/__init__.py",
    "tests/conftest.py",
    "configuration/configuration.yaml",
    "parameters.yaml",
    "requirements.txt",
    "setup.py",
    "main.py",
    "dvc.yaml",
    "Dockerfile",
    "logs/20240608-124455.log",
    "research/01_data_ingestion.ipynb",
    "research/02_model_training.ipynb",
    "app.py",
    "autopep.py",
    ".env",
    "codecov.yml",
    "models/model_name.pth",
]

for filepath in list_of_files:
    create_file_with_directories(Path(filepath))
