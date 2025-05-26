from swahiliNewsClassifier import log
from swahiliNewsClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from swahiliNewsClassifier.pipeline.stage_02_model_training_and_evaluation import ModelTrainingAndEvaluationPipeline


def run_pipeline_stage(stage_name, pipeline_class) -> None:
    """
    Run a pipeline stage and handle logging and exceptions.

    Args:
        stage_name (str): The name of the stage to run.
        pipeline_class (class): The class of the pipeline stage to instantiate and run.

    Returns:
        None
    """
    try:
        log.info(
            f"\033[1m>>>>>>>>>>>>>>>>>>>>> {stage_name} STARTED <<<<<<<<<<<<<<<<<<<<<\033[0m")
        pipeline = pipeline_class()
        pipeline.main()
        log.info(
            f"\033[1m>>>>>>>>>>>>>>>>>>> {stage_name} COMPLETED <<<<<<<<<<<<<<<<<<<\033[0m\n")
    except Exception as e:
        log.exception(f"An error occurred during {stage_name}: {e}")
        raise e


if __name__ == '__main__':
    run_pipeline_stage("DATA INGESTION STAGE", DataIngestionTrainingPipeline)
    run_pipeline_stage(
        "MODEL TRAINING AND EVALUATION STAGE",
        ModelTrainingAndEvaluationPipeline)
