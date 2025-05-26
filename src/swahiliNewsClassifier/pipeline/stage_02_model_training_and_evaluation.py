from swahiliNewsClassifier.configuration.configuration import ConfigurationManager
from swahiliNewsClassifier.components.model_training_and_evaluation import ModelTrainingAndEvaluation
from swahiliNewsClassifier import log

STAGE_NAME = "Model Training and Evaluation Stage"


class ModelTrainingAndEvaluationPipeline:
    def __init__(self):
        """
        Initialize the ModelTrainingAndEvaluationPipeline object.
        """
        self.config = ConfigurationManager()

    def main(self):
        """
        Execute the model training and evaluation process.
        """
        try:
            model_training_and_evaluation_config = self.config.get_model_training_and_evaluation_config()
            model_training_and_evaluation = ModelTrainingAndEvaluation(
                model_training_and_evaluation_configurations=model_training_and_evaluation_config)
            model_training_and_evaluation.run_pipeline()
        except Exception as e:
            log.exception(f"An error occurred during {STAGE_NAME}: {e}")
            raise e


if __name__ == '__main__':
    pipeline = ModelTrainingAndEvaluationPipeline()
    pipeline.main()
