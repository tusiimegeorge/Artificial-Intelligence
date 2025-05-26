from swahiliNewsClassifier.components.data_ingestion import DataIngestion, DataIngestionConfig
import pytest
from unittest.mock import patch
import os
import sys

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '../src')))


@pytest.fixture
def data_ingestion_configurations():
    return DataIngestionConfig(
        root_dir="artifacts/data_ingestion",
        train_source_URL="https://drive.google.com/file/d/15stuLDZkXNOgBUC1rnx5yXYdVPViUjNB/view?usp=sharing",
        test_source_URL="https://drive.google.com/file/d/1mjmYzMdnn_UwSEgTQ7i-cJ5WSOokt9Er/view?usp=sharing",
        train_data_file="artifacts/data_ingestion/compressed/train_data.zip",
        test_data_file="artifacts/data_ingestion/compressed/test_data.zip",
        decompressed_dir="artifacts/data_ingestion/decompressed",
    )


@pytest.fixture
def data_ingestion(data_ingestion_configurations):
    return DataIngestion(
        data_ingestion_configurations=data_ingestion_configurations)


@patch('swahiliNewsClassifier.components.data_ingestion.os.makedirs')
@patch('swahiliNewsClassifier.components.data_ingestion.gdown.download')
def test_download_file(mock_gdown_download, mock_makedirs, data_ingestion):
    mock_gdown_download.return_value = None

    data_ingestion.download_file()

    assert mock_gdown_download.call_count == 2

    mock_gdown_download.assert_any_call(
        "https://drive.google.com/uc?/export=download&id=15stuLDZkXNOgBUC1rnx5yXYdVPViUjNB",
        "artifacts/data_ingestion/compressed/train_data.zip")
    mock_gdown_download.assert_any_call(
        "https://drive.google.com/uc?/export=download&id=1mjmYzMdnn_UwSEgTQ7i-cJ5WSOokt9Er",
        "artifacts/data_ingestion/compressed/test_data.zip")


@patch('swahiliNewsClassifier.components.data_ingestion.os.makedirs')
@patch('swahiliNewsClassifier.components.data_ingestion.zipfile.ZipFile.extractall')
@patch('swahiliNewsClassifier.components.data_ingestion.zipfile.ZipFile')
def test_extract_zip_file(
        mock_zipfile,
        mock_extractall,
        mock_makedirs,
        data_ingestion):
    data_ingestion.extract_zip_file()

    assert mock_makedirs.call_count == 1
    mock_makedirs.assert_called_with(
        "artifacts/data_ingestion/decompressed", exist_ok=True)

    assert mock_zipfile.call_count == 2
    mock_zipfile.assert_any_call(
        "artifacts/data_ingestion/compressed/train_data.zip", "r")
    mock_zipfile.assert_any_call(
        "artifacts/data_ingestion/compressed/test_data.zip", "r")
    assert mock_zipfile.return_value.__enter__.return_value.extractall.call_count == 2
