"""
This module basically manages the configurations for each stage fo the pipeline
"""

from src.text_summarization.constants import ( TrainingArguments, 
                                              DataIngestionConstants,
                                              DataTransformationConstants,
                                              DataValidationConstants,
                                              ModelTrainingConstants,
                                              ModelEvaluationConstants)

from src.text_summarization.utils.common_utils import read_yaml, create_directories
from src.text_summarization.entity import (DataIngestionConfig, 
                                           DataValidationConfig, 
                                           DataTransformationConfig, 
                                           ModelTrainingConfig,
                                           ModelEvaluationConfig
                                          )


class ConfigurationManager:
    """This class binds the methods for all the configuration files"""
    def __init__(self):
        self.data_ingestion_const = DataIngestionConstants()
        self.data_validation_const = DataValidationConstants()
        self.data_transformation_const = DataTransformationConstants()
        self.model_training_const = ModelTrainingConstants()
        self.model_evaluation_const = ModelEvaluationConstants()
        

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """This method assigns the constants for Data Ingestion config"""

        create_directories([self.data_ingestion_const.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir = self.data_ingestion_const.DATA_INGESTION_ROOT_DIR,
            data_url = self.data_ingestion_const.DATA_URL,
            downloaded_data_file = self.data_ingestion_const.DOWNLOADED_DATA_FILE,
            unzipped_dir = self.data_ingestion_const.UNZIPPED_DIR,
            data_bucket_name = self.data_ingestion_const.DATA_BUCKET_NAME,
            data_file_name = self.data_ingestion_const.DATA_FILE_NAME
        )

        return data_ingestion_config
    
  
    def get_data_validation_config(self) -> DataValidationConfig:
        """This method assigns the constants for Data Validation config"""

        create_directories([self.data_validation_const.DATA_VALIDATION_ROOT_DIR])

        data_validation_config = DataValidationConfig(
            root_dir = self.data_validation_const.DATA_VALIDATION_ROOT_DIR,
            status_file = self.data_validation_const.DATA_VALIDATION_STATUS_FILE,
            all_required_files = self.data_validation_const.ALL_REQUIRED_FILES,
        )

        return data_validation_config
    

    def get_data_transformation_config(self) -> DataTransformationConfig:
        """This method assigns the constants for Data Transformation config"""

        create_directories([self.data_transformation_const.DATA_TRANSFORMATION_ROOT_DIR])

        data_transformation_config = DataTransformationConfig(
            root_dir = self.data_transformation_const.DATA_TRANSFORMATION_ROOT_DIR,
            data_path=self.data_transformation_const.TRANSFORMED_DATA_PATH,
            tokenizer_name = self.data_transformation_const.TOKENIZER_NAME
        )

        return data_transformation_config
    


    def get_model_training_config(self) -> ModelTrainingConfig:
        """This method assigns the constants for Model Training config"""

        config = ModelTrainingConstants()
        params = TrainingArguments()
        create_directories([config.root_dir])

        model_trainer_config = ModelTrainingConfig(
            root_dir=config.MODEL_TRAINING_ROOT_DIR,
            data_path=config.MODEL_TRAINING_DATA_PATH,
            model_ckpt = config.MODEL_CKPT,
            num_train_epochs = params.NUM_TRAIN_EPOCHS,
            warmup_steps = params.WARMUP_STEPS,
            per_device_train_batch_size = params.PER_DEVICE_TRAIN_BATCH_SIZE,
            weight_decay = params.WEIGHT_DECAY,
            logging_steps = params.LOGGING_STEPS,
            evaluation_strategy = params.EVALUATION_STRATEGY,
            eval_steps = params.EVAL_STEPS,
            save_steps = params.SAVE_STEPS,
            gradient_accumulation_steps = params.GRADIENT_ACCUMULATION_STEPS
        )

        return model_trainer_config
    


    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        """This method assigns the constants for Model Evaluation config"""

        config = ModelEvaluationConstants()
        
        create_directories([config.MODEL_EVALUATION_ROOT_DIR])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.MODEL_EVALUATION_ROOT_DIR,
            data_path=config.DATA_PATH,
            model_path = config.SAVED_MODEL_PATH,
            tokenizer_path = config.TOKENIZER_PATH,
            metric_file_name = config.METRIC_FILE_NAME
           
        )

        return model_evaluation_config
