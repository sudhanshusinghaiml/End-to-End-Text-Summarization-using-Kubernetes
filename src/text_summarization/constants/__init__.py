"""This module maps the configuration for all the constants in each pipeline"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import List

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

ARTIFACTS_ROOT: str = os.path.join("artifacts",TIMESTAMP)

@dataclass
class TrainingArguments:
  NUM_TRAIN_EPOCHS = 1
  WARMUP_STEPS = 500
  PER_DEVICE_TRAIN_BATCH_SIZE = 1
  PER_DEVICE_EVAL_BATCH_SIZE = 1
  WEIGHT_DECAY = 0.01
  LOGGING_STEPS = 10
  EVALUATION_STRATEGY = "steps"
  EVAL_STEPS = 50
  SAVE_STEPS = 1e6
  GRADIENT_ACCUMULATION_STEPS = 16



@dataclass
class DataIngestionConstants:
  DATA_INGESTION_ROOT_DIR: str = os.path.join(ARTIFACTS_ROOT,"DataIngestionArtifacts")
  DATA_FILE_NAME: str = "data.zip"
  DATA_URL: str = "https://text-summer-bucket.s3.amazonaws.com/summarizer-data.zip"
  DOWNLOADED_DATA_FILE: str = os.path.join(DATA_INGESTION_ROOT_DIR, DATA_FILE_NAME)
  UNZIPPED_DIR: str =  DATA_INGESTION_ROOT_DIR
  DATA_BUCKET_NAME: str = "text-summarization-data-06062024"



@dataclass
class DataValidationConstants:
  DATA_VALIDATION_STATUS_FILE = "status.txt"
  ALL_REQUIRED_FILES: List[str] = field(default_factory=list)
  DATA_VALIDATION_ROOT_DIR: str = os.path.join(ARTIFACTS_ROOT,"DataValidationArtifacts")
  DATA_VALIDATION_STATUS_FILE: str = os.path.join(DATA_VALIDATION_ROOT_DIR, DATA_VALIDATION_STATUS_FILE)



@dataclass
class DataTransformationConstants:
  DATA_TRANSFORMATION_ROOT_DIR: str = os.path.join(ARTIFACTS_ROOT,"DataTransformationArtifacts")
  TRANSFORMED_DATA_PATH: str = DataIngestionConstants.DATA_INGESTION_ROOT_DIR
  TOKENIZER_NAME: str = "google/pegasus-cnn_dailymail"
  MAX_INPUT_LENGTH: int = 1024
  MAX_TARGET_LENGTH: int = 128
  PREFIX: str = "Summarize: "



@dataclass
class ModelTrainingConstants:
  MODEL_TRAINING_ROOT_DIR: str = os.path.join(ARTIFACTS_ROOT, "ModelTraining")
  MODEL_TRAINING_DATA_PATH: str = DataTransformationConstants.DATA_TRANSFORMATION_ROOT_DIR
  MODEL_CKPT: str = "google/pegasus-cnn_dailymail"
  MODEL_PATH: str = os.path.join(MODEL_TRAINING_ROOT_DIR, "models")
  TOKENIZER_PATH: str = os.path.join(MODEL_TRAINING_ROOT_DIR, "tokenizer")
  


@dataclass
class ModelEvaluationConstants:
  MODEL_EVALUATION_ROOT_DIR: str = os.path.join(ARTIFACTS_ROOT, "ModelEvaluation")
  DATA_PATH: str =  ModelTrainingConstants.MODEL_TRAINING_DATA_PATH
  TRAINED_MODEL_PATH: str = ModelTrainingConstants.MODEL_PATH
  TRAINED_TOKENIZER_PATH: str =  ModelTrainingConstants.TOKENIZER_PATH
  MODEL_PREFIX: str = "models"
  TOKENIZER_PREFIX: str = "tokenizer"
  DOWNLOADED_MODEL_PATH: str = os.path.join(MODEL_EVALUATION_ROOT_DIR, MODEL_PREFIX)
  DOWNLOADED_TOKENIZER_PATH: str =  os.path.join(MODEL_EVALUATION_ROOT_DIR, TOKENIZER_PREFIX)
  METRIC_FILE_NAME: str = os.path.join(MODEL_EVALUATION_ROOT_DIR, "metrics.csv")
  MODEL_BUCKET_NAME: str = "text-summarization-models-06062024"
  MODEL_EVALUATION_STATUS_FILE = "model_evaluation.json"
  MODEL_EVALUATION_STATUS_FILE: str = os.path.join(MODEL_EVALUATION_ROOT_DIR, MODEL_EVALUATION_STATUS_FILE)



@dataclass
class ModelPusherConstants:
  REFERENCE_DIR: str = os.path.join(ARTIFACTS_ROOT, "ModelEvaluation")
  TRAINED_MODEL_PATH: str = ModelTrainingConstants.MODEL_PATH
  TRAINED_TOKENIZER_PATH: str =  ModelTrainingConstants.TOKENIZER_PATH
  MODEL_PREFIX: str = ModelEvaluationConstants.MODEL_PREFIX
  TOKENIZER_PREFIX: str = ModelEvaluationConstants.TOKENIZER_PREFIX
  DOWNLOADED_MODEL_PATH: str = ModelEvaluationConstants.DOWNLOADED_MODEL_PATH
  DOWNLOADED_TOKENIZER_PATH: str =  ModelEvaluationConstants.DOWNLOADED_TOKENIZER_PATH
  METRIC_FILE_NAME: str = ModelEvaluationConstants.METRIC_FILE_NAME
  MODEL_BUCKET_NAME: str = ModelEvaluationConstants.MODEL_BUCKET_NAME
  MODEL_EVALUATION_STATUS_FILE: str = ModelEvaluationConstants.MODEL_EVALUATION_STATUS_FILE



@dataclass
class PredictionPipelineConstants:
  PREDICTION_PIPELINE_ROOT_DIR: str = os.path.join("artifacts","PredictionPipeline")
  PREDICTION_DATA_PATH: str = os.path.join(PREDICTION_PIPELINE_ROOT_DIR, "data")
  MODEL_BUCKET_NAME: str = ModelEvaluationConstants.MODEL_BUCKET_NAME
  MODEL_PREFIX: str = ModelEvaluationConstants.MODEL_PREFIX
  TOKENIZER_PREFIX: str = ModelEvaluationConstants.TOKENIZER_PREFIX
  MODEL_FOR_PREDICTION: str = os.path.join(PREDICTION_PIPELINE_ROOT_DIR, MODEL_PREFIX)
  TOKENIZER_FOR_PREDICTION: str = os.path.join(PREDICTION_PIPELINE_ROOT_DIR, TOKENIZER_PREFIX)
