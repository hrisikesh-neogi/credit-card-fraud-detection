import sys
from typing import Dict, Tuple
import os
import pandas as pd
import pickle
import yaml
import boto3

from src.constant import *
from src.exception import CustomException
from src.logger import logging


class MainUtils:
    def __init__(self) -> None:
        pass

    def read_yaml_file(self, filename: str) -> dict:
        try:
            with open(filename, "rb") as yaml_file:
                return yaml.safe_load(yaml_file)

        except Exception as e:
            raise CustomException(e, sys) from e

    def read_schema_config_file(self) -> dict:
        try:
            schema_config = self.read_yaml_file(os.path.join("config", "schema.yaml"))

            return schema_config

        except Exception as e:
            raise CustomException(e, sys) from e

    

    @staticmethod
    def save_object(file_path: str, obj: object) -> None:
        logging.info("Entered the save_object method of MainUtils class")

        try:
            with open(file_path, "wb") as file_obj:
                pickle.dump(obj, file_obj)

            logging.info("Exited the save_object method of MainUtils class")

        except Exception as e:
            raise CustomException(e, sys) from e

    

    @staticmethod
    def load_object(file_path: str) -> object:
        logging.info("Entered the load_object method of MainUtils class")

        try:
            with open(file_path, "rb") as file_obj:
                obj = pickle.load(file_obj)

            logging.info("Exited the load_object method of MainUtils class")

            return obj

        except Exception as e:
            raise CustomException(e, sys) from e
        
    @staticmethod
    def upload_file(from_filename, to_filename, bucket_name):
        try:
            s3_resource = boto3.resource("s3")

            s3_resource.meta.client.upload_file(from_filename, bucket_name, to_filename)

        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def download_model(bucket_name, bucket_file_name, dest_file_name):
        try:
            s3_client = boto3.client("s3")

            s3_client.download_file(bucket_name, bucket_file_name, dest_file_name)

            return dest_file_name

        except Exception as e:
            raise CustomException(e, sys)
        
    
