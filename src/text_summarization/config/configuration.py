"""
This module basically manages the configurations for each stage fo the pipeline
"""

from src.text_summarization.constants import TrainingArguments
from src.text_summarization.utils.common_utils import read_yaml, create_directories
from src.text_summarization.entity.config import (DataIngestionConfig, 
                                                  DataValidationConfig, 
                                                  DataTransformationConfig, 
                                                  ModelTrainerConfig, 
                                                  ModelEvaluationConfig
                                                  )


class ConfigurationManager:
    """This class binds the methods for all the configuration files"""
    def __init__(self):
        pass


    def get_data_ingestion_config(self, config: DataIngestionConfig) -> DataIngestionConfig:

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    

    
    def get_data_validation_config(self, config: DataValidationConfig) -> DataValidationConfig:

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            ALL_REQUIRED_FILES=config.ALL_REQUIRED_FILES,
        )

        return data_validation_config
    


    def get_data_transformation_config(self, config: DataTransformationConfig) -> DataTransformationConfig:

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            tokenizer_name = config.tokenizer_name
        )

        return data_transformation_config
    


    def get_model_trainer_config(self, config: ModelTrainerConfig, params: TrainingArguments) -> ModelTrainerConfig:

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            model_ckpt = config.model_ckpt,
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
    


    def get_model_evaluation_config(self, config: ModelEvaluationConfig) -> ModelEvaluationConfig:

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            model_path = config.model_path,
            tokenizer_path = config.tokenizer_path,
            metric_file_name = config.metric_file_name
           
        )

        return model_evaluation_config
