"""This module includes all the configurations for each stage of pipeline"""

from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    data_url: str
    downloaded_data_file: Path
    unzipped_dir: Path
    data_bucket_name: str
    data_file_name: str


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    status_file: str
    all_required_files: list



@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    transformed_data_path: Path
    tokenizer_name: Path
    max_input_length: int
    max_target_length: int
    prefix: str



@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    data_path: Path
    model_ckpt: str
    model_path: Path
    tokenizer_path: Path
    num_train_epochs: int
    warmup_steps: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    weight_decay: float
    logging_steps: int
    evaluation_strategy: str
    eval_steps: int
    save_steps: float
    gradient_accumulation_steps: int



@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    data_path: Path
    trained_model_path: Path
    trained_tokenizer_path: Path
    model_prefix: str
    tokenizer_prefix: str
    downloaded_model_path: Path
    downloaded_tokenizer_path: Path
    metric_file_name: Path
    model_bucket_name: str
    model_status_file: str

@dataclass(frozen=True)
class ModelPusherConfig:
    reference_dir: Path
    trained_model_path: Path
    trained_tokenizer_path: Path
    model_prefix: str
    tokenizer_prefix: str
    downloaded_model_path: Path
    downloaded_tokenizer_path: Path
    metric_file_name: Path
    model_bucket_name: str
    model_status_file: str


@dataclass(frozen=True)
class PredictionPipelineConfig:
    root_dir: Path
    data_path: Path
    saved_model_path: Path
    tokenizer_path: Path
    model_bucket_name: str
    model_prefix: str
    tokenizer_prefix: str