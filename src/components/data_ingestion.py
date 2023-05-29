import sys
import os
import numpy as np
import pandas as pd
from pymongo import MongoClient
from zipfile import Path
from src.constant import *
from src.exception import CustomException
from src.logger import logging

from src.utils.main_utils import MainUtils
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(artifact_folder)
    raw_data_path: str = os.path.join(data_ingestion_dir,"card_data.csv")


class DataIngestion:
    def __init__(self):

        self.data_ingestion_config = DataIngestionConfig()
        self.utils = MainUtils()

    def export_collection_as_dataframe(self,collection_name, db_name):
        try:
            mongo_client = MongoClient(MONGO_DB_URL)

            collection = mongo_client[db_name][collection_name]

            df = pd.DataFrame(list(collection.find()))

            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis=1)

            df.replace({"na": np.nan}, inplace=True)

            return df

        except Exception as e:
            raise CustomException(e, sys)

    def export_data_into_raw_data_dir(self) -> pd.DataFrame:
        """
        Method Name :   export_data_into_feature_store
        Description :   This method reads data from mongodb and saves it into artifacts. 
        
        Output      :   dataset is returned as a pd.DataFrame
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   0.1
       
        """
        try:
            logging.info(f"Exporting data from mongodb")
            raw_data_dir = self.data_ingestion_config.data_ingestion_dir
            os.makedirs(raw_data_dir, exist_ok=True)

            raw_data_path = self.data_ingestion_config.raw_data_path

            

            logging.info(f"Saving exported data into feature store file path: {raw_data_path}")
            dataset  = self.export_collection_as_dataframe(db_name=MONGO_DATABASE_NAME,
                                                           collection_name= MONGO_COLLECTION_NAME)
           
            dataset.to_csv(raw_data_path, index=False)




        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self) -> Path:
        """
            Method Name :   initiate_data_ingestion
            Description :   This method initiates the data ingestion components of training pipeline 
            
            Output      :   train set and test set are returned as the artifacts of data ingestion components
            On Failure  :   Write an exception log and then raise an exception
            
            Version     :   1.2
            Revisions   :   moved setup to cloud
        """
        logging.info("Entered initiate_data_ingestion method of Data_Ingestion class")

        try:
            self.export_data_into_raw_data_dir()

            logging.info("Got the data from mongodb")

            logging.info(
                "Exited initiate_data_ingestion method of Data_Ingestion class"
            )

            return self.data_ingestion_config.raw_data_path

        except Exception as e:
            raise CustomException(e, sys) from e
