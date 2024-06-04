"""This module maps the configuration for all the constants in each pipeline"""

from pathlib import Path
from dataclasses import dataclass


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