"""This module contains all the methods that will be used for S3 Operations"""

import os
import sys
from typing import List, Union
import pickle
from pandas import DataFrame, read_csv
import boto3
from botocore.exceptions import ClientError
from mypy_boto3_s3.service_resource import Bucket

from src.text_summarization.logger import logging
from src.text_summarization.exception import TextSummarizerException


class S3Operations:
    """This class encapsulates contains all the methods that will be used for S3 Operations"""

    def __init__(self):
        self.s3_client = boto3.client("s3")
        self.s3_resource = boto3.resource("s3")

    def download_object(self, key, bucket_name, filename):
        """This method is used for downloading the file from S3"""
        bucket = self.s3_resource.Bucket(bucket_name)
        bucket.download_file(Key=key, Filename=filename)

    @staticmethod
    def read_object(object_name: "boto3.resources.factory.s3.Object", 
                    decode: bool = True) -> Union[bytes, str]:
        """
        Method Name :   read_object
        Description :   This method reads the object_name object with kwargs
        Output      :   The column name is renamed
        """
        logging.info("Inside the read_object method of S3Operations class")
        try:
            body = object_name.get()["Body"].read()

            logging.info("Completed execution of the read_object method of S3Operations class")
            if decode:
                return body.decode()
            return body

        except Exception as error:
            logging.error(error)
            raise TextSummarizerException(error, sys) from error

    def get_bucket(self, bucket_name: str) -> Bucket:
        """
        Method Name :   get_bucket
        Description :   This method gets the bucket object based on the bucket_name
        Output      :   Bucket object is returned based on the bucket name
        """
        logging.info("Inside the get_bucket method of S3Operations class")
        try:
            bucket = self.s3_resource.Bucket(bucket_name)
            logging.info("Completed execution of the get_bucket method of S3Operations class")
            return bucket

        except Exception as error:
            logging.error(error)
            raise TextSummarizerException(error, sys) from error

    def is_model_present(self, bucket_name: str, s3_model_key: str) -> bool:
        """
        Method Name :   is_model_present
        Description :   This method validates whether model is present in the s3 bucket or not.
        Output      :   True or False
        """
        try:
            bucket = self.get_bucket(bucket_name)

            return any(True for _ in bucket.objects.filter(Prefix=s3_model_key))

        except Exception as error:
            logging.error(error)
            raise TextSummarizerException(error, sys) from error

    def get_first_object_else_all(self, object_list: List) -> Union[object, List[object]]:
        """This method is used to return the first or all objects"""
        return object_list[0] if len(object_list) == 1 else object_list

    def get_file_object(self, filename: str, bucket_name: str) -> Union[List[object], object]:
        """
        Method Name :   get_file_object
        Description :   This method gets the file object from bucket_name bucket based on filename
        Output      :   list of objects or object is returned based on filename
        """
        logging.info("Inside the get_file_object method of S3Operations class")
        try:
            bucket = self.get_bucket(bucket_name)
            # list_objects = [object for object in bucket.objects.filter(Prefix=filename)]
            list_objects = list(bucket.objects.filter(Prefix=filename))
            file_objs = self.get_first_object_else_all(list_objects)
            logging.info("Completed execution the get_file_object method of S3Operations class")
            return file_objs

        except Exception as error:
            logging.error(error)
            raise TextSummarizerException(error, sys) from error

    def load_model(self, model_name: str, bucket_name: str, model_dir=None) -> object:
        """
        Method Name :   load_model
        Description :   This method loads the model_name from bucket_name bucket with kwargs
        Output      :   list of objects or object is returned based on filename
        """
        logging.info("Inside the load_model method of S3Operations class")

        try:
            if model_dir is None:
                model_file = model_name
            else:
                model_file = os.path.join(model_dir, model_name)
                # model_file = model_dir + "/" + model_name

            file_object = self.get_file_object(model_file, bucket_name)
            model_object = self.read_object(file_object, decode=False)
            model = pickle.load(model_object)
            logging.info("Completed execution of load_model method of S3Operations class")
            return model

        except Exception as error:
            logging.error(error)
            raise TextSummarizerException(error, sys) from error

    def create_folder(self, folder_name: str, bucket_name: str) -> None:
        """
        Method Name :   create_folder
        Description :   This method creates a folder_name folder in bucket_name bucket
        Output      :   Folder is created in s3 bucket
        """
        logging.info("Inside the create_folder method of S3Operations class")

        try:
            self.s3_resource.Object(bucket_name, folder_name).load()

        except ClientError as error:
            if error.response["Error"]["Code"] == "404":
                folder_obj = folder_name + "/"
                self.s3_client.put_object(Bucket=bucket_name, Key=folder_obj)
            else:
                pass
            logging.info("Completed execution of the create_folder method of S3Operations class")

    def upload_file(self, from_filename: str, to_filename: str, bucket_name: str,
                    remove: bool = True) -> None:
        """
        Method Name :   upload_file
        Description :   This method uploads the from_filename file to bucket_name bucket with
                        to_filename as bucket filename
        Output      :   Folder is created in s3 bucket
        """
        logging.info("Inside the upload_file method of S3Operations class")
        try:
            logging.info(
                f"Uploading {from_filename} file to {to_filename} file in {bucket_name} bucket"
            )

            self.s3_resource.meta.client.upload_file(
                from_filename, bucket_name, to_filename
            )
            logging.info(
                f"Uploaded {from_filename} file to {to_filename} file in {bucket_name} bucket"
            )

            if remove is True:
                os.remove(from_filename)
                logging.info(f"Remove is set to {remove}, deleted the file")
            else:
                logging.info(f"Remove is set to {remove}, not deleted the file")
            logging.info("Completed execution of the upload_file method of S3Operations class")

        except Exception as error:
            logging.error(error)
            raise TextSummarizerException(error, sys) from error

    def upload_folder(self, folder_name: str, bucket_name: str) -> None:
        """
        Method Name :   upload_file
        Description :   This method uploads the from_filename file to bucket_name bucket with
                        to_filename as bucket filename
        Output      :   Folder is created in s3 bucket
        """
        logging.info("Inside the upload_folder method of S3Operations class")
        try:
            lst = os.listdir(folder_name)
            for file in lst:
                local_file = os.path.join(folder_name, file)
                dest_file = file
                self.upload_file(local_file, dest_file, bucket_name, remove=False)
            logging.info("Completed execution of the upload_folder method of S3Operations class")

        except Exception as error:
            logging.error(error)
            raise TextSummarizerException(error, sys) from error

    def upload_df_as_csv(
        self,
        data_frame: DataFrame,
        local_filename: str,
        bucket_filename: str,
        bucket_name: str,
    ) -> None:
        """
        Method Name :   upload_df_as_csv
        Description :   This method uploads the dataframe to bucket_filename csv file
                        in bucket_name bucket
        Output      :   Folder is created in s3 bucket
        """
        logging.info("Inside the upload_df_as_csv method of S3Operations class")
        try:
            data_frame.to_csv(local_filename, index=None, header=True)
            self.upload_file(local_filename, bucket_filename, bucket_name)
            logging.info("Completed execution of the upload_df_as_csv method of S3Operations class")

        except Exception as error:
            logging.error(error)
            raise TextSummarizerException(error, sys) from error

    def get_df_from_object(self, object_: object) -> DataFrame:
        """
        Method Name :   get_df_from_object
        Description :   This method gets the dataframe from the object_name object
        Output      :   Folder is created in s3 bucket
        """
        logging.info("Inside the get_df_from_object method of S3Operations class")

        try:
            content = self.read_object(object_)
            read_csv_df = read_csv(content, na_values="na")
            logging.info("Completed execution of the get_df_from_object method of S3Operations class")
            return read_csv_df

        except Exception as error:
            logging.error(error)
            raise TextSummarizerException(error, sys) from error

    def read_csv(self, filename: str, bucket_name: str) -> DataFrame:
        """
        Method Name :   get_df_from_object
        Description :   This method gets the dataframe from the object_name object
        Output      :   Folder is created in s3 bucket
        """
        logging.info("Inside the read_csv method of S3Operations class")
        try:
            csv_obj = self.get_file_object(filename, bucket_name)
            read_csv_df = self.get_df_from_object(csv_obj)
            logging.info("Completed execution of the read_csv method of S3Operations class")
            return read_csv_df

        except Exception as error:
            logging.error(error)
            raise TextSummarizerException(error, sys) from error
