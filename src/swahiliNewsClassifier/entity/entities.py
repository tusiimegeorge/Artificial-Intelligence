from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    """
    Configuration class for the data ingestion process.

    Attributes:
        root_dir (Path): The root directory where data will be stored or processed.
        train_source_URL (str): The URL from which the training data will be fetched.
        test_source_URL (str): The URL from which the test data will be fetched.
        train_data_file (Path): The local file path where the downloaded training data will be stored.
        test_data_file (Path): The local file path where the downloaded test data will be stored.
        decompressed_dir (Path): The directory where the downloaded data will be extracted.
    """
    root_dir: Path
    train_source_URL: str
    test_source_URL: str
    train_data_file: Path
    test_data_file: Path
    decompressed_dir: Path


@dataclass(frozen=True)
class ModelTrainingAndEvaluationConfig:
    """
    Configuration class for model training using ULMFiT (Universal Language Model Fine-tuning).

    Attributes:
        test_size (float): Proportion of the dataset to include in the test split. This parameter is used to split the dataset into training and validation sets.

        learning_rate_1 (float): Learning rate for training the language model learner. This is used during the fine-tuning of the pre-trained language model.

        learning_rate_2 (float): Learning rate for the first phase of classifier training. This is used in the initial phase of training the text classifier.

        learning_rate_3 (float): Learning rate for the second phase of classifier training. This is used in the second phase of training the text classifier.

        learning_rate_4 (float): Learning rate for the third phase of classifier training. This is used in the third phase of training the text classifier.

        learning_rate_5 (float): Learning rate for the fourth phase of classifier training. This is used in the final phase of training the text classifier.

        batch_size_1 (int): Batch size for language model training. This parameter defines the number of samples that will be propagated through the network at once during language model training.

        batch_size_2 (int): Batch size for text classifier training. This parameter defines the number of samples that will be propagated through the network at once during text classifier training.

        epochs_1 (int): Number of epochs for training the language model learner. This defines the number of complete passes through the training dataset.

        epochs_2 (int): Number of epochs for the first phase of classifier training. This defines the number of complete passes through the training dataset in the first phase.

        epochs_3 (int): Number of epochs for the second phase of classifier training. This defines the number of complete passes through the training dataset in the second phase.

        epochs_4 (int): Number of epochs for the third phase of classifier training. This defines the number of complete passes through the training dataset in the third phase.

        epochs_5 (int): Number of epochs for the fourth phase of classifier training. This defines the number of complete passes through the training dataset in the final phase.

        training_data (Path): Path to the training data CSV file. This file contains the text data and corresponding labels for training and validation.

        root_dir (Path): Root directory for storing model artifacts. This directory is used to save trained models, logs, and other artifacts.

        mlflow_tracking_uri (str): URI for the MLflow tracking server. This is used to log and track experiments with MLflow.

        mlflow_repo_name (str): Repository name for MLflow tracking. This is used to organize and identify different MLflow runs within the repository.

        mlflow_repo_owner (str): Owner of the MLflow repository. This is used to identify the owner of the MLflow repository.

        all_params (dict): Dictionary containing all parameters used for model training. This includes all hyperparameters and other settings for reproducibility and logging.
    """
    test_size: float
    learning_rate_1: float
    learning_rate_2: float
    learning_rate_3: float
    learning_rate_4: float
    learning_rate_5: float
    batch_size_1: int
    batch_size_2: int
    epochs_1: int
    epochs_2: int
    epochs_3: int
    epochs_4: int
    epochs_5: int
    training_data: Path
    root_dir: Path
    mlflow_tracking_uri: str
    mlflow_repo_name: str
    mlflow_repo_owner: str
    all_params: dict
