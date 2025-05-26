import os
import subprocess
import logging
from typing import List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s %(asctime)s %(filename)s]: %(message)s:')


def get_all_python_files(directory: Path) -> List[Path]:
    """
    Get all Python files within a given directory and its subdirectories.

    Args:
        directory (Path): The directory to search for Python files.

    Returns:
        List[Path]: A list of file paths to Python files.
    """
    return list(directory.rglob('*.py'))


def execute_command(command: List[str]) -> subprocess.CompletedProcess:
    """
    Execute a command in the shell.

    Args:
        command (List[str]): The command and its arguments to execute.

    Returns:
        subprocess.CompletedProcess: The completed subprocess.
    """
    try:
        process = subprocess.run(command, capture_output=True, text=True)
        return process
    except FileNotFoundError as e:
        logging.error(f"Command '{command}' not found: {e}")
        raise e


def run_autopep8(file: Path) -> None:
    """
    Run autopep8 on a Python file.

    Args:
        file (Path): Path to the Python file to format.
    """
    autopep8_process = execute_command(
        ['autopep8', '--in-place', '--aggressive', '--aggressive', str(file)])
    if autopep8_process.returncode != 0:
        logging.warning(autopep8_process.stdout)


def lint_python_file(file: Path) -> None:
    """
    Lint a Python file using autopep8.

    Args:
        file (Path): Path to the Python file to lint.
    """
    run_autopep8(file)


def main():
    """
    Main function to lint all Python files in the project directory.
    """
    all_python_files = get_all_python_files(Path(os.getcwd()))

    with ThreadPoolExecutor() as executor:
        future_to_file = {
            executor.submit(
                lint_python_file,
                file): file for file in all_python_files}
        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error linting file {file}: {e}")


if __name__ == "__main__":
    main()
