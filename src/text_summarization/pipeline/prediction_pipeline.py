from datetime import datetime, timedelta
import os
from src.text_summarization.config.config_manager import ConfigurationManager
from src.text_summarization.config.aws_storage_operations import S3Operations
from transformers import AutoTokenizer
from transformers import pipeline
from src.text_summarization.logger import logging


class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_prediction_pipeline_config()
        self.s3 = S3Operations()

    def predict(self, text):
        """This method is used to Summarize the texts"""

        logging.info("Inside PredictionPipeline.predict methods")

        for file in os.listdir(self.config.saved_model_path):
            status = self.s3.is_modified_within_last_24_hours(os.path.join(self.config.saved_model_path, file))
            if status:
                logging.info("Latest Models are available in the local directory")
                logging.info("Latest Models will be downloaded after 24 hours")
            else:
                logging.info("Dowloading latest Models from S3 Bucket")
                self.s3.get_latest_model_from_s3(
                    self.config.model_bucket_name,
                    self.config.model_prefix,
                    file
                )

        for file in os.listdir(self.config.tokenizer_path):
            status = self.s3.is_modified_within_last_24_hours(os.path.join(self.config.tokenizer_path, file))
            if status:
                logging.info("Latest Tokenizer is available in the local directory")
                logging.info("Latest Tokenizer will be downloaded after 24 hours")
            else:
                logging.info("Dowloading latest Tokenizer from S3 Bucket")
                self.get_latest_model_from_s3(
                    self.config.model_bucket_name,
                    self.config.tokenizer_prefix,
                    file
                )


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