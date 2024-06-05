"""This module maps the configuration for all the constants in each pipeline"""

import os
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import List

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

ARTIFACTS_ROOT: str = os.path.join("artifacts",TIMESTAMP)

@dataclass
class TrainingArguments:
  NUM_TRAIN_EPOCHS = 1
  WARMUP_STEPS = 500
  PER_DEVICE_TRAIN_BATCH_SIZE = 1
  WEIGHT_DECAY = 0.01
  LOGGING_STEPS = 10
  EVALUATION_STRATEGY = "STEPS"
  EVAL_STEPS = 500
  SAVE_STEPS = 1E6
  GRADIENT_ACCUMULATION_STEPS = 16


@dataclass
class DataIngestionConstants:
  DATA_INGESTION_ROOT_DIR: str = os.path.join(ARTIFACTS_ROOT,"DataIngestionArtifacts")
  DATA_FILE_NAME: str = "data.zip"
  DATA_DOWNLOAD_URL: str = "https://text-summer-bucket.s3.amazonaws.com/summarizer-data.zip"
  DOWNLOADED_DATA_FILE: str = os.path.join(DATA_INGESTION_ROOT_DIR, DATA_FILE_NAME)
  UNZIPPED_DIR: str =  DATA_INGESTION_ROOT_DIR


@dataclass
class DataValidationConstants:
  DATA_VALIDATION_STATUS_FILE = "status.txt"
  ALL_REQUIRED_FILES: List[str] = ["train", "test", "validation"]
  DATA_VALIDATION_ROOT_DIR: str = os.path.join(ARTIFACTS_ROOT,"DataValidationArtifacts")
  DATA_VALID_STATUS_FILE: str = os.path.join(DATA_VALIDATION_ROOT_DIR, DATA_VALIDATION_STATUS_FILE)


@dataclass
class DataTransformationConstants:
  DATA_TRANSFORMATION_ROOT_DIR: str = os.path.join(ARTIFACTS_ROOT,"DataTransformationArtifacts")
  TRANSFORMED_DATA_PATH: str = os.path.join(DATA_TRANSFORMATION_ROOT_DIR, "samsum_dataset")
  TOKENIZER_NAME: str = "google/pegasus-cnn_dailymail"



@dataclass
class ModelTrainingConstants:
  MODEL_TRAINING_ROOT_DIR: str = os.path.join(ARTIFACTS_ROOT, "ModelTraining")
  MODEL_TRAINING_DATA_PATH: str = DataTransformationConstants.TRANSFORMED_DATA_PATH
  MODEL_CKPT: str = "google/pegasus-cnn_dailymail"


@dataclass
class ModelEvaluationConstants:
  MODEL_EVALUATION_ROOT_DIR: str = os.path.join(ARTIFACTS_ROOT, "ModelEvaluation")
  DATA_PATH: str =  ModelTrainingConstants.MODEL_TRAINING_DATA_PATH
  SAVED_MODEL_PATH: str = os.path.join(MODEL_EVALUATION_ROOT_DIR, "TrainedModel" )
  TOKENIZER_PATH: str =  os.path.join(MODEL_EVALUATION_ROOT_DIR, "Tokenizer" )
  metric_file_name: str = os.path.join(MODEL_EVALUATION_ROOT_DIR, "metrics.csv")

