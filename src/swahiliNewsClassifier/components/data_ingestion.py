import os
import zipfile
import gdown
from swahiliNewsClassifier.entity.entities import DataIngestionConfig
from swahiliNewsClassifier import log


class DataIngestion:
    def __init__(self, data_ingestion_configurations: DataIngestionConfig):
        """
        Initialize DataIngestion object with the provided configuration.

        Args:
            data_ingestion_configurations (DataIngestionConfig): Configuration object for data ingestion.
        """
        self.data_ingestion_configurations = data_ingestion_configurations

    def download_file(self):
        """Fetch data from a URL.

        Raises:
            Exception: If an error occurs during the download process.
        """
        os.makedirs("artifacts/data_ingestion/compressed", exist_ok=True)
        os.makedirs("artifacts/data_ingestion/decompressed", exist_ok=True)
        dataset_urls = [
            self.data_ingestion_configurations.train_source_URL,
            self.data_ingestion_configurations.test_source_URL]
        zip_download_dir = [
            self.data_ingestion_configurations.train_data_file,
            self.data_ingestion_configurations.test_data_file]

        for url, dest in zip(dataset_urls, zip_download_dir):
            try:
                log.info(f"Downloading data from {url} into file {dest}")

                file_id = url.split("/")[-2]
                prefix = "https://drive.google.com/uc?/export=download&id="
                gdown.download(prefix + file_id, dest)

                log.info(f"Downloaded data from {url} into file {dest}")
            except Exception as e:
                log.error(f"Error downloading file from {url} to {dest}")
                raise e

    def extract_zip_file(self):
        """Extract a zip file.

        This method extracts the contents of a zip file specified in the configuration
        to the directory specified in the configuration.

        Raises:
            Exception: If an error occurs during the extraction process.
        """
        zip_download_dir = [
            self.data_ingestion_configurations.train_data_file,
            self.data_ingestion_configurations.test_data_file]
        decompress_path = self.data_ingestion_configurations.decompressed_dir
        os.makedirs(decompress_path, exist_ok=True)

        for zip_file in zip_download_dir:
            try:
                with zipfile.ZipFile(zip_file, "r") as zip_ref:
                    zip_ref.extractall(decompress_path)

                log.info(
                    f"Extracted zip file {zip_file} into: {decompress_path}")
            except Exception as e:
                log.error(f"Error extracting zip file: {zip_file}")
                raise e
