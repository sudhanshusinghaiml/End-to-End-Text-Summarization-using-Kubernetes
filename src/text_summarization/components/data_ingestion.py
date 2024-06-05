import os
import urllib.request as request
import zipfile
from src.text_summarization.logger import logging
from src.text_summarization.utils.common_utils import get_size
from src.text_summarization.entity import DataIngestionConfig
from pathlib import Path


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url = self.config.data_url,
                filename = self.config.downloaded_data_file
            )
            logging.info(f"{filename} download! with following info: \n{headers}")
        else:
            logging.info(f"File already exists of size: {get_size(Path(self.config.downloaded_data_file))}")  


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
