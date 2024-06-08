from src.text_summarization.config.config_manager import ConfigurationManager
from src.text_summarization.components.data_ingestion import DataIngestion
from src.text_summarization.components.data_transformations import DataTransformation
from src.text_summarization.components.data_validation import DataValiadtion
from src.text_summarization.components.model_evaluation import ModelEvaluation
from src.text_summarization.components.model_trainer import ModelTraining
from src.text_summarization.logger import logging



class DataIngestionPipeline:
    """This class contains the methods that triggers the Data Ingestion Pipeline"""
    def __init__(self):
        pass

    def main(self):
        """This method triggers the Data Ingestion Pipeline"""
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config = data_ingestion_config)
        if data_ingestion.get_file_from_url():
            logging.info('Calling download_data_from_s3...')
            if data_ingestion.download_data_from_s3():
                logging.info('Calling extract_zip_file...')
                data_ingestion.extract_zip_file()
            else:
                logging.exception('Failed to download files from S3') 
        else:
            logging.exception('Failed to get files from Open Source')


class DataValidationPipeline:
    """This class contains the methods that triggers the Data Validation Pipeline"""
    def __init__(self):
        pass

    def main(self):
        """This method triggers the Data Validation Pipeline"""
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValiadtion(config = data_validation_config)
        data_validation.validate_all_files_exist()



class DataTransformationPipeline:
    """This class contains the methods that triggers the Data Transformation Pipeline"""
    def __init__(self):
        pass

    def main(self):
        """This method triggers the Data Transformation Pipeline"""
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.convert()


class ModelTrainingPipeline:
    """This class contains the methods that triggers the Model Training Pipeline"""
    def __init__(self):
        pass

    def main(self):
        """This method triggers the Model Training Pipeline"""
        config = ConfigurationManager()
        model_trainer_config = config.get_model_training_config()
        model_trainer_config = ModelTraining(config = model_trainer_config)
        model_trainer_config.train()


class ModelEvaluationPipeline:
    """This class contains the methods that triggers the Model Evaluation Pipeline"""
    def __init__(self):
        pass

    def main(self):
        """This method triggers the Model Evaluation Pipeline"""
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation_config = ModelEvaluation(config = model_evaluation_config)
        model_evaluation_config.evaluate()
