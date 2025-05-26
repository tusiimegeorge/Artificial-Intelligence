from swahiliNewsClassifier.constants import *
from swahiliNewsClassifier.utilities.helper_functions import read_yaml, create_directories
from swahiliNewsClassifier.entity.entities import DataIngestionConfig, ModelTrainingAndEvaluationConfig
from dotenv import load_dotenv
import os

load_dotenv()


class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH,
                 params_filepath=PARAMS_FILE_PATH):
        """
        Initialize ConfigurationManager with configuration and parameter files.

        Args:
            config_filepath (str): Path to the configuration YAML file.
            params_filepath (str): Path to the parameters YAML file.
        """
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Get the data ingestion configuration.

        Returns:
            DataIngestionConfig: Configuration object for data ingestion.
        """
        data_ingestion_config = self.config.data_ingestion

        create_directories([data_ingestion_config.root_dir])

        return DataIngestionConfig(
            root_dir=data_ingestion_config.root_dir,
            train_source_URL=data_ingestion_config.train_source_URL,
            test_source_URL=data_ingestion_config.test_source_URL,
            train_data_file=data_ingestion_config.train_data_file,
            test_data_file=data_ingestion_config.test_data_file,
            decompressed_dir=data_ingestion_config.decompressed_dir
        )

    def get_model_training_and_evaluation_config(
            self) -> ModelTrainingAndEvaluationConfig:
        """
        Get the model training and evaluation configuration.

        Returns:
            ModelTrainingConfig: Configuration object for model training and evaluation.
        """
        create_directories([self.config.training.root_dir])

        return ModelTrainingAndEvaluationConfig(
            root_dir=self.config.training.root_dir,
            training_data=self.config.training.training_data_path,
            test_size=self.params.TEST_SIZE,
            learning_rate_1=self.params.LEARNING_RATE_1,
            learning_rate_2=self.params.LEARNING_RATE_2,
            learning_rate_3=self.params.LEARNING_RATE_3,
            learning_rate_4=self.params.LEARNING_RATE_4,
            learning_rate_5=self.params.LEARNING_RATE_5,
            batch_size_1=self.params.BATCH_SIZE_1,
            batch_size_2=self.params.BATCH_SIZE_2,
            epochs_1=self.params.EPOCHS_1,
            epochs_2=self.params.EPOCHS_2,
            epochs_3=self.params.EPOCHS_3,
            epochs_4=self.params.EPOCHS_4,
            epochs_5=self.params.EPOCHS_5,
            mlflow_repo_name=os.getenv('MLFLOW_REPO_NAME'),
            mlflow_tracking_uri=os.getenv('MLFLOW_TRACKING_URI'),
            mlflow_repo_owner=os.getenv('MLFLOW_REPO_OWNER'),
            all_params=self.params,

        )
