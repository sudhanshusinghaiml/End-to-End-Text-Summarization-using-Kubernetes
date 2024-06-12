"""This module is used for Evaluating the models after training"""
import json
import os, sys
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk, load_metric
import torch
import pandas as pd
from tqdm import tqdm
from src.text_summarization.entity import ModelEvaluationConfig
from src.text_summarization.config.aws_storage_operations import S3Operations 
from src.text_summarization.logger import logging
from src.text_summarization.exception import TextSummarizerException


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.s3 = S3Operations()


    def generate_chunks(self, list_of_elements, batch_size):
        """split the dataset into smaller batches that we can process simultaneously
        Yield successive batch-sized chunks from list_of_elements."""
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i : i + batch_size]


    def save_rouge_score(self, trained_model_average_rouge_score, downloaded_model_average_rouge_score = 0)-> bool:
        try:
            logging.info(f"Trained Model Average ROUGE Score - {trained_model_average_rouge_score}")
            logging.info(f"Downloaded Model Average ROUGE Score - {downloaded_model_average_rouge_score}")
            if (trained_model_average_rouge_score >= downloaded_model_average_rouge_score) or (downloaded_model_average_rouge_score == 0):
                data = {
                    "trained_model_accepted": True
                }

            if downloaded_model_average_rouge_score > trained_model_average_rouge_score:
                data = {
                    "trained_model_accepted": False
                }
            
            with open(self.config.model_status_file, 'w') as f:
                json.dump(data, f, indent=4)

            return
        
        except Exception as error:
            logging.exception(error)
            raise TextSummarizerException(error, sys)
    
    def get_model_scores(self,
                         dataset, 
                         metric, 
                         model, 
                         tokenizer,
                         batch_size=16,
                         device="cuda" if torch.cuda.is_available() else "cpu",
                         text_column="article",
                         summary_column="highlights"):
        
        article_batches = list(self.generate_chunks(dataset[text_column], batch_size))
        target_batches = list(self.generate_chunks(dataset[summary_column], batch_size))

        for article_batch, target_batch in tqdm(
            zip(article_batches, target_batches), total=len(article_batches)):
            
            inputs = tokenizer(article_batch,
                               max_length=1024,
                               truncation=True,
                               padding="max_length",
                               return_tensors="pt"
                               )
            
            summaries = model.generate(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                length_penalty=0.8, 
                num_beams=8, 
                max_length=128
                )
            
            ''' parameter for length penalty ensures that the model does not generate sequences that are too long. '''
            
            # Finally, we decode the generated texts, 
            decoded_summaries = [
                tokenizer.decode(summary,
                                 skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False
                                 )
                for summary in summaries]
            
            # decoded_summaries = [d.replace("", " ") for d in decoded_summaries]
                        
            metric.add_batch(
                predictions = decoded_summaries,
                references = target_batch
                )
            
        #  Finally compute and return the ROUGE scores.
        score = metric.compute()
        return score
    

    def evaluate(self):
        logging.info("Inside ModelEvaluation.evaluate")
        
        logging.info(f"loading test data for model evaluation")
        dataset_pt = load_from_disk(self.config.data_path)

        logging.info("Setting ROUGE metrics for scoring")
        rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        rouge_metric = load_metric('rouge')

        logging.info("Fetching the Trained Model")  
        device = "cuda" if torch.cuda.is_available() else "cpu"
        trained_tokenizer = AutoTokenizer.from_pretrained(self.config.trained_tokenizer_path)
        trained_model = AutoModelForSeq2SeqLM.from_pretrained(self.config.trained_model_path).to(device)

        logging.info("Calculating Metric Score for Trained Model")       
        trained_model_scores = self.get_model_scores(
            dataset_pt['test'][0:10], 
            rouge_metric,
            trained_model,
            trained_tokenizer,
            batch_size = 2,
            text_column = 'dialogue',
            summary_column= 'summary'
        )

        trained_rouge_dict = dict((rn, trained_model_scores[rn].mid.fmeasure) for rn in rouge_names)

        logging.info(f"trained_rouge_dict - {trained_rouge_dict}")

        trained_model_avg_scores = sum(trained_rouge_dict.values())/ len(trained_rouge_dict.values())

        if self.s3.is_bucket_empty(self.config.model_bucket_name):
            
            self.save_rouge_score(trained_model_avg_scores)
            logging.info(f"{self.config.model_bucket_name} is empty. No Model is saved in S3 Bucket so far.")
            df = pd.DataFrame([trained_model_scores ], index = ['TrainedModelScores'])
            df.to_csv(self.config.metric_file_name, index=False)

        else:
            logging.info("Dowloading latest Models from S3 Bucket")
            self.s3.get_latest_model_from_s3(
                self.config.model_bucket_name,
                self.config.model_prefix,
                self.config.root_dir
            )

            logging.info("Dowloading latest Tokenizer from S3 Bucket")
            self.s3.get_latest_model_from_s3(
                self.config.model_bucket_name,
                self.config.tokenizer_prefix,
                self.config.root_dir
            )

            logging.info("Fetching the Downloaded Model")  
            device = "cuda" if torch.cuda.is_available() else "cpu"
            downloaded_tokenizer = AutoTokenizer.from_pretrained(self.config.downloaded_tokenizer_path)
            downloaded_model = AutoModelForSeq2SeqLM.from_pretrained(self.config.downloaded_model_path).to(device)  

            logging.info("Calculating Metric Score for Downloaded Model")       
            downloaded_model_score = self.get_model_scores(
                dataset_pt['test'][0:10], 
                rouge_metric,
                downloaded_model,
                downloaded_tokenizer,
                batch_size = 2,
                text_column = 'dialogue',
                summary_column= 'summary'
            )

            downloaded_rouge_dict = dict((rn, downloaded_model_score[rn].mid.fmeasure ) for rn in rouge_names)

            logging.info(f"downloaded_rouge_dict - {downloaded_rouge_dict}")

            downloaded_model_avg_scores = sum(downloaded_rouge_dict.values())/ len(downloaded_rouge_dict.values())

            self.save_rouge_score(trained_model_avg_scores, downloaded_model_avg_scores)
           
            df = pd.DataFrame.from_records([trained_rouge_dict, downloaded_rouge_dict], index= ['TrainedModel','DownloadedModel'])
            
            df.to_csv(self.config.metric_file_name, index=False)
