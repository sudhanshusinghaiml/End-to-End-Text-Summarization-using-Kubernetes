from src.text_summarization.config.configuration import ConfigurationManager
from src.text_summarization.components.model_trainer import ModelTraining
from src.text_summarization.logger import logging


class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_training_config()
        model_trainer_config = ModelTraining(config = model_trainer_config)
        model_trainer_config.train()