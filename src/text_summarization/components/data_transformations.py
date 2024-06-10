import os
from src.text_summarization.logger import logging
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
from src.text_summarization.entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)


    def convert_examples_to_features(self, example_batch):
        input_encodings = self.tokenizer(example_batch['dialogue'] , 
                                         max_length = 1024, 
                                         truncation = True 
                                        )
        
        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(example_batch['summary'], 
                                              max_length = 128, 
                                              truncation = True 
                                            )
            
        return {
            'input_ids' : input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }
    

    def load_samsum_dataset(self):
        data_files = {
            "train": os.path.join(self.config.transformed_data_path,"samsum-train.csv"), 
            "validation": os.path.join(self.config.transformed_data_path ,"samsum-validation.csv"), 
            "test": os.path.join(self.config.transformed_data_path ,"samsum-test.csv")
            }
        logging.info(f"Data files that will be loaded - {data_files}")

        dataset = load_dataset("csv", data_files= data_files)
        logging.info(f" After loading csv in dataset format- {dataset}")

        # Logging the sizes of the datasets
        logging.info(f"Train dataset size: {len(dataset['train'])}")
        logging.info(f"Text dataset size: {len(dataset['test'])}")
        logging.info(f"Validation dataset size: {len(dataset['validation'])}")

        return dataset


    def preprocess_dataset(self, examples):
        prefix = "Summarize: "
        inputs = [prefix + doc for doc in examples["dialogue"]]

        model_inputs = self.tokenizer(inputs, 
                                      max_length= self.config.max_input_length, 
                                      truncation=True)

        # Setup the tokenizer for targets
        labels = self.tokenizer(text_target = examples["summary"],
                                max_length = self.config.max_target_length, 
                                truncation=True
                                )

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    def convert(self):
        # dataset_samsum = load_from_disk(self.config.transformed_data_path)
        dataset_samsum = self.load_samsum_dataset()

        logging.info(f"Logging to asses the dataset - {dataset_samsum}")

        dataset_samsum_pt = dataset_samsum.map(self.preprocess_dataset, batched=True)

        logging.info(f"Logging to asses the dataset after pre-processing - {dataset_samsum_pt}")

        logging.info(f"Model Inputs for training datasets \n {dataset_samsum_pt['train'][:2]}")
        logging.info(f"Model Inputs for test datasets \n {dataset_samsum_pt['test'][:2]}")
        logging.info(f"Model Inputs for validation datasets \n {dataset_samsum_pt['validation'][:2]}")

        dataset_samsum_pt.save_to_disk(self.config.root_dir)

