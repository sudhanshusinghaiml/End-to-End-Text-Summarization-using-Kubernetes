from src.text_summarization.config.configuration import ConfigurationManager
from src.text_summarization.components.data_transformations import DataTransformation
from src.text_summarization.logger import logging


class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.convert()