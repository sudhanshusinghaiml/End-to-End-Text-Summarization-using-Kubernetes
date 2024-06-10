"""This module is used for training model on custom data"""
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset, load_from_disk
import torch
import os
from src.text_summarization.entity import ModelTrainingConfig
from src.text_summarization.logger import logging
# from accelerate import Accelerator
# accelerator = Accelerator()


class ModelTraining:
    """This class encapsulates the method for training the models on custom data"""
    def __init__(self, config: ModelTrainingConfig):
        self.config = config

    
    def train(self):
        """This method is used for training the models on custom data"""
        logging.info("Inside ModelTraining.train of model_trainer module")

        logging.info(f"Validating cuda availability - {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            device = "cuda"  
        else:
            device = "cpu"

        logging.info(f"Device is set to - {device}")
        
        logging.info("Setting tokenizer, Model and data collator")
        
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        logging.info("Completed setting Tokenizer")
        
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt)
        logging.info("Completed setting Model")
        
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model = model)
        logging.info("Completed setting data collator")

        logging.info(f"Loading data from disk {self.config.data_path}")
        dataset_pt = load_from_disk(self.config.data_path)

        logging.info(dataset_pt)

        logging.info(f"Setting TrainingArguments for model training ")
        trainer_args = Seq2SeqTrainingArguments(
            output_dir = self.config.root_dir, 
            num_train_epochs = self.config.num_train_epochs,
            warmup_steps = self.config.warmup_steps,
            per_device_train_batch_size = self.config.per_device_train_batch_size,
            per_device_eval_batch_size = self.config.per_device_train_batch_size,
            weight_decay = self.config.weight_decay, 
            logging_steps = self.config.logging_steps,
            evaluation_strategy = self.config.evaluation_strategy, 
            eval_steps = self.config.eval_steps, 
            save_steps = self.config.save_steps,
            gradient_accumulation_steps = self.config.gradient_accumulation_steps
        ) 

        logging.info(f"Setting Model Parameters for Trainer ")
        trainer = Seq2SeqTrainer(
            model = model, 
            args=trainer_args,
            tokenizer = tokenizer, 
            data_collator = seq2seq_data_collator,
            train_dataset = dataset_pt["train"],
            eval_dataset = dataset_pt["validation"]
            )
        
        logging.info(f"Model Training Started....")
        trainer.train()
        logging.info(f"Model Training completed.")

        logging.info(f"Saving Trained Model - {self.config.root_dir}")
        model.save_pretrained(os.path.join(self.config.root_dir,"model"))

        logging.info(f"Saving tokenizer - {self.config.root_dir}")
        tokenizer.save_pretrained(os.path.join(self.config.root_dir,"tokenizer"))
