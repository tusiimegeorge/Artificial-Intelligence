import os
import yaml
import json
import joblib
import base64
from box import ConfigBox
from pathlib import Path
from typing import Any, List
from box.exceptions import BoxValueError
from swahiliNewsClassifier import log


def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads a YAML file and returns its contents wrapped in a ConfigBox."""
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            if content is None:
                raise ValueError("yaml file is empty")
            log.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError as e:
        log.error(f"Error loading YAML file: {e}")
        raise ValueError("yaml file is empty") from e
    except Exception as e:
        log.error(f"Unexpected error loading YAML file: {e}")
        raise


def create_directories(paths: List[Path], verbose: bool = True) -> None:
    """Create directories specified in the list."""
    for path in paths:
        os.makedirs(path, exist_ok=True)
        if verbose:
            log.info(f"Created directory at: {path}")


def save_json(path: Path, data: dict) -> None:
    """Save JSON data to a file."""
    try:
        with open(path, 'w') as json_file:
            json.dump(data, json_file)
        log.info(f"JSON file saved at: {path}")
    except IOError as e:
        log.error(f"An error occurred while writing to the file: {e}")
        raise


def load_json(path: Path) -> ConfigBox:
    """Load JSON file data and return as a ConfigBox."""
    try:
        with open(path, 'r') as json_file:
            content = json.load(json_file)
            log.info(f"JSON file loaded successfully from: {path}")
            return ConfigBox(content)
    except FileNotFoundError as e:
        log.error(f"File not found at: {path}")
        raise
    except json.JSONDecodeError as e:
        log.error(f"Error decoding JSON file at: {path}. Reason: {e}")
        raise


def save_bin(data: Any, path: Path) -> None:
    """Save file as binary."""
    try:
        joblib.dump(value=data, filename=path)
        log.info(f"Binary file saved at: {path}")
    except Exception as e:
        log.error(f"Error saving binary file at: {path}. Reason: {e}")
        raise


def load_bin(path: Path) -> Any:
    """Load a binary file."""
    if not path.exists():
        raise FileNotFoundError(f"File not found at: {path}")

    try:
        data = joblib.load(path)
        log.info(f"Binary file loaded from: {path}")
        return data
    except Exception as e:
        log.error(f"Error loading binary file from {path}. Reason: {e}")
        raise


def get_size(path: Path) -> str:
    """Get the size of a file in kilobytes."""
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~{size_in_kb} KB"


def decode_image(image_string: str, file_name: str) -> None:
    """Decode a base64-encoded image string and save it to a file."""
    try:
        image_data = base64.b64decode(image_string)
        with open(file_name, "wb") as f:
            f.write(image_data)
        log.info(f"Image saved at: {file_name}")
    except Exception as e:
        log.error(f"Error decoding image: {e}")
        raise


def encode_image_into_base64(image_path: str) -> bytes:
    """Encode an image file into base64 format."""
    try:
        with open(image_path, "rb") as f:
            encoded_image = base64.b64encode(f.read())
        return encoded_image
    except Exception as e:
        log.error(f"Error encoding image at: {image_path}. Reason: {e}")
        raise
