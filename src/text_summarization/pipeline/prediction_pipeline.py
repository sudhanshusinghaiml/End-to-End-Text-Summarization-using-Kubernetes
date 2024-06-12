from datetime import datetime, timedelta
import os
from src.text_summarization.config.config_manager import ConfigurationManager
from src.text_summarization.config.aws_storage_operations import S3Operations
from transformers import AutoTokenizer
from transformers import pipeline
from src.text_summarization.logger import logging
import boto3


class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_prediction_pipeline_config()
        self.s3_storage = boto3.client("s3")

    @staticmethod
    def is_modified_within_last_24_hours(file_path):
        """This method is used to check if the downloaded files are latest or not"""
        
        last_modified_time = os.path.getmtime(file_path)
        current_time = datetime.now()
        last_modified_datetime = datetime.fromtimestamp(last_modified_time)
        logging.info(f" File {file_path} last modified at {last_modified_datetime} and difference between modified and current datetime is {current_time - last_modified_datetime}")

        return (current_time - last_modified_datetime) < timedelta(hours=24)

    
    def get_latest_model_from_s3(self):
        logging.info(f"Downloading latest model from S3 Bucket")
      
        response = self.s3_storage.list_objects_v2(
            Bucket = self.config.model_bucket_name,
            Prefix = self.config.model_prefix + "/"
            )
        
        logging.info(f"Latest model response from S3 Bucket - {response}")
        
        if 'Contents' in response:
            for obj in response['Contents']:
                file_name = obj['Key']
                file_name_with_path = os.path.join(self.config.root_dir, file_name)
                os.makedirs(os.path.dirname(file_name_with_path), exist_ok=True)
                self.s3_storage.download_file(
                    self.config.model_bucket_name,
                    file_name,
                    file_name_with_path
                    )
        else:
            logging.info(f"No files found in folder 'model' of bucket {self.config.model_bucket_name}")


    def get_latest_tokenizer_from_s3(self):
        logging.info(f"Downloading latest tokenizer from S3 Bucket")
        
        response = self.s3_storage.list_objects_v2(
            Bucket = self.config.model_bucket_name,
            Prefix = self.config.tokenizer_prefix + "/"
            )
        
        logging.info(f"Latest tokenizer response from S3 Bucket - {response}")
        
        if 'Contents' in response:
            for obj in response['Contents']:
                file_name = obj['Key']
                file_name_with_path = os.path.join(self.config.root_dir, file_name)
                os.makedirs(os.path.dirname(file_name_with_path), exist_ok=True)
                self.s3_storage.download_file(
                    self.config.model_bucket_name,
                    file_name,
                    file_name_with_path
                    )
        else:
            logging.info(f"No files found in folder 'model' of bucket {self.config.model_bucket_name}")



    def predict(self, text):
        """This method is used to Summarize the texts"""

        logging.info("Inside PredictionPipeline.predict methods")

        for file in os.listdir(self.config.saved_model_path):
            status = self.is_modified_within_last_24_hours(os.path.join(self.config.saved_model_path, file))
            if status:
                logging.info("Latest Models are available in the local directory")
                logging.info("Latest Models will be downloaded after 24 hours")
            else:
                logging.info("Dowloading latest Models from S3 Bucket")
                self.get_latest_model_from_s3()

        for file in os.listdir(self.config.tokenizer_path):
            status = self.is_modified_within_last_24_hours(os.path.join(self.config.tokenizer_path, file))
            if status:
                logging.info("Latest Tokenizer is available in the local directory")
                logging.info("Latest Tokenizer will be downloaded after 24 hours")
            else:
                logging.info("Dowloading latest Tokenizer from S3 Bucket")
                self.get_latest_tokenizer_from_s3()


        logging.info(f"Tokenizer Path - {self.config.tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        
        gen_kwargs = {"length_penalty": 0.8, "num_beams":8, "max_length": 128}

        logging.info(f"Downloaded Model Path - {self.config.saved_model_path}")
        pipe = pipeline("summarization", model = self.config.saved_model_path, tokenizer=tokenizer)

        print(f"Dialogue: {text}")

        output = pipe(text, **gen_kwargs)[0]["summary_text"]
        print(f"Model Summary: {output}")

        logging.info("Completed execution of PredictionPipeline.predict methods")

        return output