from swahiliNewsClassifier.entity.entities import ModelTrainingAndEvaluationConfig
from swahiliNewsClassifier import log
import torch
import fastai
from fastai.text.all import *
import pandas as pd
import numpy as np
from functools import partial
import io
import os
from sklearn.model_selection import train_test_split
from swahiliNewsClassifier import log
import boto3
from dotenv import load_dotenv
import dagshub
import mlflow

load_dotenv()

class ModelTrainingAndEvaluation:
    def __init__(self, model_training_and_evaluation_configurations: ModelTrainingAndEvaluationConfig):
        """
        Initialize ModelTraining object with the provided configuration.

        Args:
            model_training_and_evaluation_configurations (ModelTrainingConfig): Configuration object for model training.
        """
        self.model_training_and_evaluation_configurations = model_training_and_evaluation_configurations
        self.bucket_name = "swahili-news-classifier"
        self.model_path = f"models/text_classifier_learner.pkl"
        self.s3 = boto3.client('s3', aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'), aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'), region_name=os.getenv('REGION_NAME'))

    def upload_to_s3(self) -> None:
        """
        Saves the trained text classifier onto an AWS S3 Bucket.
        """
        try:
            self.s3.upload_file(self.model_path, self.bucket_name, self.model_path)
            print(f"Successfully uploaded {self.model_path} to s3://{self.bucket_name}/{self.model_path}")
        except Exception as e:
            print(f"Failed to upload {self.model_path} to s3://{self.bucket_name}/{self.model_path}. Error: {e}")

    def load_data(self) -> pd.DataFrame:
        """
        Load the training data from the specified path.

        Returns:
            pd.DataFrame: Loaded training data.
        """
        log.info('Loading training data')
        train = pd.read_csv(self.model_training_and_evaluation_configurations.training_data)
        return train

    def prepare_data(self, train) -> 'tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]':
        """
        Prepare the data for training and validation.

        Args:
            train (pd.DataFrame): The loaded training data.

        Returns:
            tuple: A tuple containing training data (df_trn), validation data (df_val), and data for language model (df_lm).
        """
        df_trn, df_val = train_test_split(train, stratify=train['category'], test_size=self.model_training_and_evaluation_configurations.test_size, random_state=123)
        df_lm = pd.concat([df_trn, df_val], axis=0)[['content']]
        return df_trn, df_val, df_lm

    def create_dataloaders(self, df_lm) -> DataLoaders:
        """
        Create dataloaders for the language model.

        Args:
            df_lm (pd.DataFrame): Data for language model.

        Returns:
            DataLoaders: Dataloaders for the language model.
        """
        log.info('Creating Language Model Learner Dataloader')
        dblock = DataBlock(
            blocks=TextBlock.from_df('content', is_lm=True),
            get_x=ColReader('text'),
            splitter=RandomSplitter(0.1))

        dls = dblock.dataloaders(df_lm, bs=self.model_training_and_evaluation_configurations.batch_size_1)
        return dls

    def train_language_model(self, dls) -> Learner:
        """
        Train the language model.

        Args:
            dls (DataLoaders): Dataloaders for the language model.

        Returns:
            Learner: Trained language model learner.
        """
        log.info('Training Language Model Learner')
        learn = language_model_learner(dls, AWD_LSTM, drop_mult=0.3, metrics=[accuracy]).to_fp16()
        learn.lr_find()
        learn.fine_tune(self.model_training_and_evaluation_configurations.epochs_1, self.model_training_and_evaluation_configurations.learning_rate_1)

        log.info('Saving best Language Model Learner.')

        learn.save_encoder(f'language_model_learner')

        return learn

    def create_text_classifier_dataloaders(self, df_trn, dls_lm) -> DataLoaders:
        """
        Create dataloaders for the text classifier.

        Args:
            df_trn (pd.DataFrame): Training data.
            dls_lm (DataLoaders): Dataloaders for the language model to get vocabulary and sequence length.

        Returns:
            DataLoaders: Dataloaders for the text classifier.
        """
        log.info('Creating Text Classifier Learner Data Loaders')
        blocks = (TextBlock.from_df('content', seq_len=dls_lm.seq_len, vocab=dls_lm.vocab), CategoryBlock())
        dblock = DataBlock(
            blocks=blocks,
            get_x=ColReader('text'),
            get_y=ColReader('category'),
            splitter=RandomSplitter(0.2))
        
        return dblock.dataloaders(df_trn, bs=self.model_training_and_evaluation_configurations.batch_size_2)
    
    def log_to_mlflow(self, metrics: list) -> None:
        os.environ['MLFLOW_TRACKING_URI'] = self.model_training_and_evaluation_configurations.mlflow_tracking_uri

        dagshub.init(repo_owner=self.model_training_and_evaluation_configurations.mlflow_repo_owner, repo_name=self.model_training_and_evaluation_configurations.mlflow_repo_name, mlflow=True)

        with mlflow.start_run():
            mlflow.log_params(self.model_training_and_evaluation_configurations.all_params)
            mlflow.log_metric('val_loss', metrics[0])
            mlflow.log_metric('val_accuracy', metrics[1])

    def train_text_classifier(self, dls) -> None:
        """
        Train the text classifier.

        Args:
            dls (DataLoaders): Dataloaders for the text classifier.
        """

        log.info('Training Text Classifier Learner.')

        learn = text_classifier_learner(dls, AWD_LSTM, metrics=[accuracy]).to_fp16()
        learn.load_encoder(f'language_model_learner')
        learn.lr_find()
        learn.fit_one_cycle(self.model_training_and_evaluation_configurations.epochs_2, self.model_training_and_evaluation_configurations.learning_rate_2)
        learn.freeze_to(-2)
        learn.fit_one_cycle(self.model_training_and_evaluation_configurations.epochs_3, slice(1e-3/(2.6**4), self.model_training_and_evaluation_configurations.learning_rate_3))
        learn.freeze_to(-3)
        learn.fit_one_cycle(self.model_training_and_evaluation_configurations.epochs_4, slice(5e-3/(2.6**4), self.model_training_and_evaluation_configurations.learning_rate_4))
        learn.unfreeze()
        learn.fit_one_cycle(self.model_training_and_evaluation_configurations.epochs_5, slice(1e-3/(2.6**4), self.model_training_and_evaluation_configurations.learning_rate_5))
        classifier_metrics = learn.validate()
        self.log_to_mlflow(classifier_metrics)
        learn.export('models/text_classifier_learner.pkl')



    def run_pipeline(self) -> None:
        """
        Run the complete model training pipeline.
        """
        train = self.load_data()
        df_trn, df_val, df_lm = self.prepare_data(train)
        dls_lm = self.create_dataloaders(df_lm)
        lm_learner = self.train_language_model(dls_lm)
        dls_clf = self.create_text_classifier_dataloaders(df_trn, dls_lm)
        self.train_text_classifier(dls_clf)
        self.upload_to_s3()