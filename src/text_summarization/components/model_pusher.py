"""This module is used for Pushing the best models to S3 Bucket after training"""
import pandas as pd
import sys, os
import json
from src.text_summarization.entity import ModelPusherConfig
from src.text_summarization.config.aws_storage_operations import S3Operations 
from src.text_summarization.logger import logging
from src.text_summarization.exception import TextSummarizerException

class ModelPusher:
    """This class is used to push the models to S3 Bucket"""
    def __init__(self, config: ModelPusherConfig):
        self.config = config
        self.s3 = S3Operations()


    def upload_model_files(self, directory_name, s3_prefix):
        """This method is used to upload model files to S3"""

        for file in os.listdir(directory_name):
            file_path = os.path.join(directory_name, file)
            file_name = s3_prefix +"/"+ file
            logging.info(f"File Path - {file_path}")
            logging.info(f"File Name - {file_name}")
            self.s3.upload_file(
                local_file_path = file_path,
                file_name = file_name,
                bucket_name = self.config.model_bucket_name
            )

        return 

    def initiate_model_pusher(self):
        """This method is used to push the models to S3 bucket"""
        try:
            logging.info("Inside ModelPusher.initiate_model_pusher() ")

            with open(self.config.model_status_file, 'r') as file:
                data = json.load(file)

            logging.info(f"Loaded the {self.config.model_status_file}")
            logging.info(f"trained_model_accepted is set to {data}")

            if data['trained_model_accepted']:
                logging.info("Uploading models to s3 bucket")
                self.upload_model_files(self.config.trained_model_path, self.config.model_prefix)
                logging.info("Uploaded models to s3 bucket")
                
                logging.info("Uploading tokenizer to s3 bucket")
                self.upload_model_files(self.config.trained_tokenizer_path, self.config.tokenizer_prefix) 
                logging.info("Uploaded tokenizer to s3 bucket")
            else:
                logging.info("Trained Model is not accpted as the best model")
                logging.info("Trained model will not be pushed to S3 Bucket")

            logging.info("Successfully executed ModelPusher.initiate_model_pusher()")
            
            return True
        except Exception as error:
            logging.exception(error)
            raise TextSummarizerException(error, sys) from error



    