from swahiliNewsClassifier.configuration.configuration import ConfigurationManager
from swahiliNewsClassifier.components.data_ingestion import DataIngestion
from swahiliNewsClassifier import log

STAGE_NAME = "Data Ingestion Stage"


class DataIngestionTrainingPipeline:
    def __init__(self):
        """
        Initialize the DataIngestionTrainingPipeline object.
        """
        self.config = ConfigurationManager()

    def main(self):
        """
        Execute the data ingestion process.
        """
        try:
            data_ingestion_config = self.config.get_data_ingestion_config()
            data_ingestion = DataIngestion(
                data_ingestion_configurations=data_ingestion_config)
            data_ingestion.download_file()
            data_ingestion.extract_zip_file()
        except Exception as e:
            log.exception(f"An error occurred during {STAGE_NAME}: {e}")
            raise e


if __name__ == '__main__':
    pipeline = DataIngestionTrainingPipeline()
    pipeline.main()
