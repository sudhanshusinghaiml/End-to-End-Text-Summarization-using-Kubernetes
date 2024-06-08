import os, sys
import urllib.request as request
import zipfile
from src.text_summarization.logger import logging
from src.text_summarization.utils.common_utils import get_size
from src.text_summarization.entity import DataIngestionConfig
from src.text_summarization.config.aws_storage_operations import S3Operations
from src.text_summarization.exception import TextSummarizerException
from pathlib import Path


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.s3_storage = S3Operations()


    def get_file_from_url(self):
        """This method is used to download the file from URL"""
        if not os.path.exists(self.config.downloaded_data_file):
            filename, headers = request.urlretrieve(
                url = self.config.data_url,
                filename = self.config.downloaded_data_file
            )
            logging.info(f"{filename} download! with following info: \n{headers}")
        else:
            logging.info(f"File already exists of size: {get_size(Path(self.config.downloaded_data_file))}")
            logging.info(f"Pushing the {filename} into Bucket - {self.config.data_bucket_name}")

            return os.path.exists(self.config.downloaded_data_file)


    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzipped_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.downloaded_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)


    def download_data_from_s3(self) -> str:
        """This method is used to download the data from s3"""
        try:
            zip_download_dir = self.config.root_dir
            os.makedirs(zip_download_dir, exist_ok=True)

            logging.info(f"Downloading data from s3 into file {zip_download_dir}")

            self.config.downloaded_data_file

            self.s3_storage.download_object(
                file_name = self.config.data_file_name,
                bucket_name = self.config.data_bucket_name,
                file_path = self.config.downloaded_data_file,
            )

            if os.path.exists(self.config.downloaded_data_file):
                logging.info(f"Downloaded data from s3 into file {self.config.downloaded_data_file}")
                return True
            else:
                return False

        except Exception as error:
            logging.exception(error)
            raise TextSummarizerException(error, sys) from error
