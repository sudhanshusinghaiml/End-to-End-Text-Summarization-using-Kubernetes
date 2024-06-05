"""This module is used for training model on custom data"""
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
import torch
import os
from src.text_summarization.entity import ModelTrainingConfig


class ModelTraining:
    """This class encapsulates the method for training the models on custom data"""
    def __init__(self, config: ModelTrainingConfig):
        self.config = config

    
    def train(self):
        """This method is used for training the models on custom data"""

        if torch.cuda.is_available():
            device = "cuda"  
        else:
            device = "cpu"
        
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model = model)
        
        #loading data 
        dataset_pt = load_from_disk(self.config.data_path)


        trainer_args = TrainingArguments(
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

        trainer = Trainer(model = model, 
                          args=trainer_args,
                          tokenizer = tokenizer, 
                          data_collator = seq2seq_data_collator,
                          train_dataset = dataset_pt["test"],
                          eval_dataset = dataset_pt["validation"]
                          )
        
        trainer.train()

        ## Save model
        model.save_pretrained(os.path.join(self.config.root_dir,"model"))

        ## Save tokenizer
        tokenizer.save_pretrained(os.path.join(self.config.root_dir,"tokenizer"))
