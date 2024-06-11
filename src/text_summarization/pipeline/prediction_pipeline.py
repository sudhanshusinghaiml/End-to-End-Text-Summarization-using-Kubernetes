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

        self.get_latest_model_from_s3()

        self.get_latest_tokenizer_from_s3()

        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        
        gen_kwargs = {"length_penalty": 0.8, "num_beams":8, "max_length": 128}

        pipe = pipeline("summarization", model = self.config.saved_model_path, tokenizer=tokenizer)

        print("Dialogue:")
        print(text)

        output = pipe(text, **gen_kwargs)[0]["summary_text"]
        print("\nModel Summary:")
        print(output)

        logging.info("Completed execution of PredictionPipeline.predict methods")

        return output